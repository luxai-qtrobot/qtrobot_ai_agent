"""
llm_engine.py - LLMEngine: reusable glue between an OpenAI-compatible client, an
optional RobotMemory, and an optional Agent (MCP tool-calling).

Owns the round-loop mechanics (stream -> check tool_calls -> execute via Agent -> feed
results back -> repeat until a round has no more tool_calls) and memory bookkeeping, so
callers don't re-implement it per script. Does NOT decide what to do with the model's
output (speak it, log it, etc.) - that's presentation logic and stays with the caller.
ask() yields each complete sentence as it's produced, not per-round or per-token, so a
caller wanting low-latency TTS can start speaking sentence 1 while the model is still
generating sentence 2.
"""

import re

from luxai.magpie.utils import Logger

SENTENCE_END_RE = re.compile(r"[.!?]+(\s+)")


def extract_sentences(buffer: str):
    """Split complete sentences off the front of buffer; return (sentences, leftover)."""
    sentences = []
    last_end = 0
    for m in SENTENCE_END_RE.finditer(buffer):
        end = m.end()
        sentences.append(buffer[last_end:end].strip())
        last_end = end
    return sentences, buffer[last_end:]


class LLMEngine:

    def __init__(self, client, model: str, memory=None, agent=None, system_prompt: str = None):
        """
        client:        a (sync) OpenAI client.
        model:         model name passed to every completion call.
        memory:        a RobotMemory instance, or None to fall back to a plain in-memory
                       history list (seeded with system_prompt, if given).
        agent:         an Agent instance for MCP tool-calling, or None for plain chat
                       (no tools=, no tool-call handling).
        system_prompt: only used when memory is None, to seed the fallback history.
        """
        self.client = client
        self.model = model
        self.memory = memory
        self.agent = agent
        self._history = [{"role": "system", "content": system_prompt}] if (memory is None and system_prompt) else []

    async def ask(self, user_input: str):
        """Run one full user turn - streaming, tool-calling, memory bookkeeping - and
        yield each complete sentence as soon as it's produced. Nothing is returned; a
        caller not using the generator's output gets nothing meaningful from calling
        this without iterating it."""
        self._add({"role": "user", "content": user_input})

        round_num = 1
        while True:
            messages = self._get_messages()
            gen = self._stream_round(messages, label=f"round {round_num}")

            # Manually drive the sync generator (async generators can't use `yield
            # from`) so we can both forward its yields and capture its final
            # (assistant_message, tool_calls) via StopIteration.value.
            assistant_msg, tool_calls = None, None
            try:
                while True:
                    sentence = next(gen)
                    yield sentence
            except StopIteration as stop:
                assistant_msg, tool_calls = stop.value

            self._add(assistant_msg)

            if not tool_calls:
                break

            for tc in tool_calls:
                Logger.info(f"[tool call] {tc['name']}({tc['arguments']})")

            results = await self.agent.execute(tool_calls)
            for r in results:
                Logger.info(f"[tool result] {r['content']}")
                self._add({
                    "role": "tool",
                    "tool_call_id": r["tool_call_id"],
                    "content": r["content"],
                })
                for extra in r["extra_messages"]:
                    self._add(extra)

            round_num += 1

        if self.memory:
            self.memory.end_turn()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add(self, message: dict) -> None:
        if self.memory:
            self.memory.add(message)
        else:
            self._history.append(message)

    def _get_messages(self) -> list:
        return self.memory.get() if self.memory else self._history

    def _stream_round(self, messages: list, label: str = ""):
        """Sync generator: yields each complete sentence as it streams in. Returns
        (assistant_message, tool_calls) via StopIteration.value once the round ends -
        tool_calls is the flat [{'id','name','arguments'}, ...] shape Agent.execute()
        expects; assistant_message is the OpenAI-format message ready for memory/history."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.agent.schemas() if self.agent else None,
            tool_choice="auto" if self.agent else None,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        buffer = ""
        content = ""
        tool_calls_acc = {}  # index -> {"id": str, "name": str, "arguments": str}

        # Logger.debug(f"--- streaming deltas ({label}) ---")
        for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                content += delta.content
                buffer += delta.content
                sentences, buffer = extract_sentences(buffer)
                for sentence in sentences:
                    if sentence:
                        yield sentence

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    acc = tool_calls_acc.setdefault(tc.index, {"id": None, "name": None, "arguments": ""})
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function and tc.function.name:
                        acc["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        acc["arguments"] += tc.function.arguments
                    Logger.debug(f"[tool_call delta] index={tc.index} id={tc.id} "
                                 f"name={tc.function.name if tc.function else None} "
                                 f"args_fragment={tc.function.arguments if tc.function else None!r}")

        leftover = buffer.strip()
        if leftover:
            yield leftover

        # Logger.debug(f"--- end of stream ({label}); full content={content!r} ---")

        if not tool_calls_acc:
            return {"role": "assistant", "content": content}, []

        ordered = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
        assistant_msg = {
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in ordered
            ],
        }
        return assistant_msg, ordered
