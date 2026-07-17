"""
llm_engine.py - LLMEngine: reusable glue between an OpenAI-compatible client, an
optional RobotMemory, and an optional ToolEngine (MCP tool-calling).

submit()/output()/cancel(): A single background worker
thread processes one submitted input at a time - no concurrent LLM calls, 
but a round that comes back with tool_calls is never waited
on: it's "parked" (dispatched non-blocking via ToolEngine.execute_async(), NOT added to
memory yet) so the worker is immediately free to start the next submitted input rather
than sit idle for a slow tool/agent call.

When a parked round resolves, a dedicated follow-up completion - whose message array
ends exactly at the tool result, the one shape models reliably know how to continue -
produces the actual answer. That answer, together with the original tool call and its
result, then gets spliced into WorkingMemory at its true chronological position (right
after the question that triggered it, not appended wherever it happened to finish) via
RobotMemory.insert_resolved(). Its sentences flow into the same output() stream as
everything else - a caller doesn't need to know or care which submitted turn produced
what, it just drains output() in the order things become ready.

cancel() does two independent things: it stops sentences from being pushed to output()
for whatever's currently streaming live, and it calls ToolEngine.cancel_all() to mark
every currently in-flight tool call (across the live round AND any parked ones) as
unwanted - see tool_engine.py for how that actually stops (or fails to stop) the
underlying call. Either way, execute_async()'s callback comes back with the CANCELLED
sentinel instead of real results for anything that was in flight at the time - see
_resolve_parked(), which splices in a short fixed reply instead of running the normal
follow-up completion, so the model doesn't see an unanswered question sitting in
history and spontaneously retry it later.

Known, deliberate simplification: a spliced-in exchange does not get the normal
end_turn() tool-result compaction (that operates on the tail of memory, not an
arbitrary earlier position) - it stays as full, uncompacted messages. Acceptable for
now; revisit if it turns out to matter in practice.
"""

import asyncio
import queue
import re
import threading

from luxai.magpie.utils import Logger

from tool.tool_engine import CANCELLED

SENTENCE_END_RE = re.compile(r"[.!?]+(\s+)")

_SHUTDOWN = object()   # tells the worker thread to stop
_STOP = object()       # tells output() to end the stream

# Appended (never spoken) after whatever was actually said, when a round is cut short
# - see _process_turn()/_resolve_parked(). Plain, first-person, short: matches the
# system prompt's own voice in case it later surfaces in a short-term-memory summary.
# Two variants: cancel() means the user/system abandoned the topic outright;
# interrupt() (e.g. barge-in while speaking) doesn't - the topic may still matter,
# so its note deliberately avoids implying the whole thing was rejected.
CANCELLED_REPLY = "I didn't finish that - it was cancelled."
INTERRUPTED_REPLY = "I was interrupted there."


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

    def __init__(self, client, model: str, memory=None, tool_engine=None, system_prompt: str = None,
                 max_tokens: int = 800, timeout: float = 60.0):
        """
        client:        a (sync) OpenAI client - safe to share across threads, this
                       class calls it from more than one (documented httpx.Client
                       thread-safety, same assumption ShortTermMemory relies on).
        model:         model name passed to every completion call.
        memory:        a RobotMemory instance, or None to fall back to a plain in-memory
                       history list (seeded with system_prompt, if given).
        tool_engine:   a ToolEngine instance for MCP tool-calling, or None for plain chat
                       (no tools=, no tool-call handling, no parking - a round can never
                       come back with tool_calls if none were ever offered).
        system_prompt: only used when memory is None, to seed the fallback history.
        max_tokens:    hard cap on generated tokens per round - guards against a runaway
                       generation (e.g. the model looping on a reasoning/think tag and
                       never emitting real content) hanging the whole conversation.
                       Normal spoken replies are far shorter than this.
        timeout:       per-round call timeout (seconds) - guards against a genuinely
                       stalled connection. Separate concern from max_tokens: a looping
                       generation still streams bytes continuously, so it wouldn't trip
                       an idle read timeout on its own - max_tokens is what bounds that.
        """
        self.client = client
        self.model = model
        self.memory = memory
        self.tool_engine = tool_engine
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._history = [{"role": "system", "content": system_prompt}] if (memory is None and system_prompt) else []

        self._input_queue = queue.Queue()
        self._output_queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._cancel_reason = "cancelled"  # which of cancel()/interrupt() set the event
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Public - submit/output/cancel/shutdown
    # ------------------------------------------------------------------

    def submit(self, user_input: str) -> None:
        """Non-blocking - queues user_input, processed once the worker is free (one
        submitted input at a time; see class docstring for why a busy worker doesn't
        mean a stalled conversation)."""
        self._input_queue.put(user_input)

    def cancel(self) -> None:
        """The broad "stop everything" case (e.g. user says "cancel that"/"stop
        stop"): stops the turn currently streaming live from yielding any more
        output, and asks ToolEngine to stop/discard every tool call currently in
        flight - see class docstring. Also clears WSM entries for those same ids
        immediately, rather than waiting for _resolve_parked's eventual CANCELLED
        handling - a call without a cancel pairing can keep running for a while after
        this, and its "in progress" entry would otherwise stay visible (and
        misleading) for that whole window even though the user already cancelled it."""
        self._cancel_reason = "cancelled"
        self._cancel_event.set()
        if self.tool_engine:
            self._wsm_discard(self.tool_engine.cancel_all())

    def interrupt(self) -> None:
        """The narrow case (e.g. the user starts talking over TTS, detected by
        ASR/VAD): stops the turn currently streaming live from yielding any more
        output, same as cancel() - but does NOT touch in-flight tool/agent calls.
        Starting to talk doesn't mean the user has abandoned whatever's running in
        the background, just that they want to be heard now."""
        self._cancel_reason = "interrupted"
        self._cancel_event.set()

    async def output(self):
        """Async generator - yields each complete sentence as it becomes available,
        from whichever submitted turn produced it, plus None once some turn fully
        completes. A caller only interested in one submitted turn's answer (e.g.
        AgentBase, which submits exactly one input per LLMEngine instance) can stop at
        the first None; a caller draining forever (e.g. the main CLI loop, servicing
        many submit() calls over this engine's lifetime) just skips None and keeps
        going, until shutdown() ends the stream (see there for why a plain task.cancel()
        from the caller's side can't stop this cleanly on its own)."""
        while True:
            item = await asyncio.to_thread(self._output_queue.get)
            if item is _STOP:
                return
            yield item

    def shutdown(self) -> None:
        """Stops the background worker thread, and unblocks output() so it returns
        cleanly instead of hanging. Call when this engine is done being used (e.g.
        AgentBase.run() calls this in its finally block, since it builds a fresh
        LLMEngine per call and would otherwise leak one idle thread per call; the main
        CLI loop calls this on /exit).

        output() sits blocked inside `await asyncio.to_thread(queue.get)` - a plain
        `task.cancel()` on whatever's draining output() only stops *waiting* for that
        thread-pool call, it can't interrupt the blocking queue.get() already running
        in the thread pool, so cancelling that task alone hangs forever once nothing
        more is ever queued. Pushing a sentinel here instead lets output() notice and
        return on its own, same trick shutdown already uses to stop the worker."""
        self._input_queue.put(_SHUTDOWN)
        self._worker.join(timeout=5.0)
        self._output_queue.put(_STOP)

    # ------------------------------------------------------------------
    # Internal - worker thread: processes one submitted input at a time
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            user_input = self._input_queue.get()
            if user_input is _SHUTDOWN:
                return
            self._cancel_event.clear()
            self._process_turn(user_input)

    def _process_turn(self, user_input: str) -> None:
        anchor = self._add({"role": "user", "content": user_input})
        context = self._get_messages()
        assistant_msg, tool_calls = self._run_round(context, respect_cancel=True, label="round 1")

        if tool_calls is CANCELLED:
            # _stream_round noticed cancel()/interrupt() mid-stream and cut the round
            # short - assistant_msg (if not None) is exactly what was already spoken,
            # kept as-is (see _stream_round) rather than discarded, plus a short
            # closing note picked by which of cancel()/interrupt() triggered this.
            reply = CANCELLED_REPLY if self._cancel_reason == "cancelled" else INTERRUPTED_REPLY
            messages = ([assistant_msg] if assistant_msg is not None else [])
            messages.append({"role": "assistant", "content": reply})
            self._insert_resolved(anchor, messages)
            self._output_queue.put(None)
            return

        if not tool_calls or not self.tool_engine:
            self._add(assistant_msg)
            if self.memory:
                self.memory.end_turn()
            self._output_queue.put(None)  # turn complete
            return

        for tc in tool_calls:
            Logger.debug(f"[tool call] {tc['name']}({tc['arguments']})")

        self._wsm_add(tool_calls)

        # Park: this round is not committed to memory yet. context is a snapshot as of
        # right now - it must NOT include anything a later, unrelated submit() adds
        # while this one is still resolving, or the eventual synthesis call would see
        # content it shouldn't (see class docstring / the design discussion this
        # implements). Dispatching here returns immediately - the worker loop is free
        # to pick up the next queued input without waiting on this tool call at all.
        self.tool_engine.execute_async(
            tool_calls,
            lambda results: self._resolve_parked(anchor, context, assistant_msg, tool_calls, results),
        )

    def _resolve_parked(self, anchor: dict, context: list, pending_msg: dict,
                         dispatched_calls: list[dict], results: list) -> None:
        """Runs on ToolEngine.execute_async()'s own background thread, never the main
        worker thread - blocking here (a follow-up completion, another tool round) is
        fine, there's nothing else for this thread to do, and it can't delay the next
        submitted input either way.

        results is CANCELLED when this round's tool call(s) were cancelled before they
        finished (see cancel()/ToolEngine.cancel_all()) - splice in a fixed closing
        reply instead of running the follow-up completion at all, so the anchor doesn't
        sit in memory as a permanently unanswered question the model might pick back up
        on its own later."""
        if results is CANCELLED:
            self._wsm_discard(tc["id"] for tc in dispatched_calls)
            self._insert_resolved(anchor, [{"role": "assistant", "content": CANCELLED_REPLY}])
            self._output_queue.put(None)
            return

        resolved = [pending_msg] + self._tool_result_messages(results)
        # World-state note for THIS batch, consumed exactly once by the completion
        # call that's about to react to it - see _wsm_finish(). Re-set each time the
        # loop dispatches a further batch below, so a later round doesn't keep
        # re-showing an earlier round's now-stale "just finished" note.
        wsm_msgs = self._wsm_finish(dispatched_calls)

        while True:
            full_context = context + resolved + wsm_msgs
            assistant_msg, tool_calls = self._run_round(full_context, respect_cancel=False, label="parked round")
            resolved.append(assistant_msg)
            if not tool_calls:
                break
            for tc in tool_calls:
                Logger.debug(f"[tool call] {tc['name']}({tc['arguments']})")
            self._wsm_add(tool_calls)
            tool_results = self._execute_tools_blocking(tool_calls)
            if tool_results is CANCELLED:
                # This iteration's assistant_msg (just appended, with tool_calls) now
                # has no matching tool results - replace it rather than leave it
                # dangling, or the next real completion call would 400 on the orphaned
                # tool_calls/tool_result pairing.
                self._wsm_discard(tc["id"] for tc in tool_calls)
                resolved[-1] = {"role": "assistant", "content": CANCELLED_REPLY}
                break
            wsm_msgs = self._wsm_finish(tool_calls)
            resolved.extend(self._tool_result_messages(tool_results))

        self._insert_resolved(anchor, resolved)
        self._output_queue.put(None)  # turn complete

    # ------------------------------------------------------------------
    # Internal - world state (in-flight background actions, see world_state_memory.py)
    # ------------------------------------------------------------------

    def _wsm_add(self, tool_calls: list[dict]) -> None:
        """Marks each tool call as a running background action, visible in ANY
        round's context (not just the one that dispatched it) via RobotMemory.get() -
        this is what lets a concurrently-processed turn answer "what are you doing?"
        correctly instead of appearing idle. No-op without a memory/world_state tier."""
        if not self.memory:
            return
        for tc in tool_calls:
            self.memory.add_world_state(tc["id"], f"[bg action]: {tc['name']}({tc['arguments']}) - in progress")

    def _wsm_finish(self, tool_calls: list[dict]) -> list[dict]:
        """Marks each tool call finished, captures one fresh render of the whole
        current world state, and removes these entries immediately - before the
        completion call that will use it even runs, not after (see WorldStateMemory.
        finish()). Returns [] (nothing to append) if there was nothing to show, else a
        single message to append to that one call's context."""
        if not self.memory:
            return []
        updates = {tc["id"]: f"[bg action]: {tc['name']}({tc['arguments']}) - just finished" for tc in tool_calls}
        text = self.memory.finish_world_state(updates)
        return [{"role": "user", "content": text}] if text else []

    def _wsm_discard(self, tool_call_ids) -> None:
        """Clears world-state entries by id, without a "finished" narration - used
        both when a batch was cancelled rather than resolved (CANCELLED_REPLY already
        covers what the user sees there) and directly from cancel(), which only has
        raw ids back from ToolEngine.cancel_all(), not tool_call dicts. Either way,
        this just stops an entry lingering as "in progress" once it no longer is."""
        if not self.memory:
            return
        for tool_call_id in tool_call_ids:
            self.memory.remove_world_state(tool_call_id)

    def _run_round(self, messages: list, respect_cancel: bool, label: str = ""):
        """Drives _stream_round() to completion, pushing each sentence to output() as
        it arrives. Returns (assistant_message, tool_calls) - or
        (spoken_message_or_None, CANCELLED) if respect_cancel and cancel()/interrupt()
        fired mid-stream, see _stream_round()."""
        gen = self._stream_round(messages, respect_cancel=respect_cancel, label=label)
        try:
            while True:
                sentence = next(gen)
                self._output_queue.put(sentence)
        except StopIteration as stop:
            return stop.value

    def _execute_tools_blocking(self, tool_calls: list[dict]) -> list[dict]:
        """Blocking wrapper around ToolEngine.execute_async() - safe here because
        _resolve_parked already runs off the main worker thread, so blocking this
        thread doesn't freeze anything. Still goes through execute_async (not a bare
        asyncio.run()) so dispatch lands on the correct event loop - see
        ToolEngine.execute_async's own docstring for why that matters."""
        done = threading.Event()
        box = {}

        def _on_done(results):
            box["results"] = results
            done.set()

        self.tool_engine.execute_async(tool_calls, _on_done)
        done.wait()
        return box["results"]

    def _tool_result_messages(self, results: list[dict]) -> list[dict]:
        messages = []
        for r in results:
            Logger.debug(f"[tool result] {r['content']}")
            messages.append({
                "role": "tool",
                "tool_call_id": r["tool_call_id"],
                "content": r["content"],
            })
            messages.extend(r["extra_messages"])
        return messages

    # ------------------------------------------------------------------
    # Internal - memory (handles both RobotMemory and the plain-history fallback)
    # ------------------------------------------------------------------

    def _add(self, message: dict) -> dict:
        """Returns the message object itself, so callers can hold onto it as an anchor
        for a later insert_resolved()/insert_after() splice - matched by identity, not
        position, since positions shift as other turns get added."""
        if self.memory:
            self.memory.add(message)
        else:
            self._history.append(message)
        return message

    def _insert_resolved(self, anchor: dict, messages: list[dict]) -> None:
        if self.memory:
            self.memory.insert_resolved(anchor, messages)
            return
        try:
            index = next(i for i, m in enumerate(self._history) if m is anchor)
        except StopIteration:
            self._history.extend(messages)
        else:
            self._history[index + 1:index + 1] = messages

    def _get_messages(self) -> list:
        return self.memory.get() if self.memory else list(self._history)

    def _stream_round(self, messages: list, respect_cancel: bool, label: str = ""):
        """Sync generator: yields each complete sentence as it streams in. Returns
        (assistant_message, tool_calls) via StopIteration.value once the round ends -
        tool_calls is the flat [{'id','name','arguments'}, ...] shape ToolEngine.execute()
        expects; assistant_message is the OpenAI-format message ready for memory/history.

        If respect_cancel and cancel()/interrupt() fires while still streaming, stops
        pulling further chunks - checked once per raw chunk, not once per sentence, so
        it cuts in within roughly one chunk's latency rather than waiting for the
        whole response - and closes the underlying stream so the connection is
        actually released (most OpenAI-compatible backends stop generating/billing
        once the client disconnects; a `break` alone would just abandon the iterator
        without closing anything). Returns (spoken_message_or_None, CANCELLED) in that
        case - spoken_message is exactly what was already yielded as complete
        sentences (and so already sent to output()/spoken), None if nothing had been
        spoken yet; any incomplete trailing fragment still sitting in the buffer was
        never actually said, so it's dropped. The caller appends a short closing note
        rather than discarding this real, already-spoken content."""
        buffer = ""
        content = ""
        spoken = ""
        tool_calls_acc = {}  # index -> {"id": str, "name": str, "arguments": str}

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_engine.schemas() if self.tool_engine else None,
                tool_choice="auto" if self.tool_engine else None,
                stream=True,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            Logger.debug(f"--- streaming deltas ({label}) ---")
            for chunk in stream:
                if respect_cancel and self._cancel_event.is_set():
                    Logger.debug(f"--- {self._cancel_reason} mid-stream ({label}) ---")
                    try:
                        stream.close()
                    except Exception:
                        pass
                    partial = spoken.strip()
                    spoken_msg = {"role": "assistant", "content": partial} if partial else None
                    return spoken_msg, CANCELLED

                delta = chunk.choices[0].delta

                # reasoning_content isn't a declared field on ChoiceDelta (it's a
                # llama.cpp/vLLM extension, not part of OpenAI's schema) - chunks that
                # don't carry it (e.g. the first, role-only chunk) don't have the key
                # at all, so a direct delta.reasoning_content raises AttributeError.
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning and Logger.log_level == "DEBUG":
                    print(reasoning, end="", flush=True)

                if delta.content:
                    content += delta.content
                    buffer += delta.content
                    sentences, buffer = extract_sentences(buffer)
                    for sentence in sentences:
                        if sentence:
                            spoken += sentence + " "
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
        except Exception as e:
            # A genuinely stalled connection (timeout=) lands here - a runaway/looping
            # generation is instead bounded by max_tokens= and ends the stream normally,
            # so it never reaches this branch at all.
            Logger.warning(f"LLMEngine: streaming call failed ({label}): {e}")
            fallback = "Sorry, I'm having trouble responding right now."
            yield fallback
            return {"role": "assistant", "content": fallback}, []

        leftover = buffer.strip()
        if leftover:
            yield leftover

        Logger.debug(f"--- end of stream ({label}); full content={content!r} ---")

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
