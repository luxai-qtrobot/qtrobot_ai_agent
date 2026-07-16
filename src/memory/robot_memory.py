"""
robot_memory.py - RobotMemory: orchestrates working/short-term/long-term memory tiers
into one coherent context for the LLM.

Only RobotMemory knows the real token ceiling (max_tokens) - individual tiers only know
their own local, ratio-derived budgets and are otherwise self-contained; they never need
runtime awareness of each other or of the parent.
"""

from .utils import count_message_tokens, count_messages_tokens


class RobotMemory:

    def __init__(self, static: str, max_tokens: int, working, short_term=None, long_term=None):
        """
        static:     always-on content (e.g. system prompt) - never trimmed.
        max_tokens: the one absolute token ceiling in the whole system. Enforced only at
                    get() time, non-destructively - it decides what's included in a given
                    call, it never deletes anything from working/short_term's own stores.
        working:    a WorkingMemory instance (required).
        short_term: a ShortTermMemory instance, or None to skip that tier.
        long_term:  a LongTermMemory instance, or None to skip that tier - archived chat
                    history and loaded documents both live there (see
                    memory/long_term_memory.py).
        """
        self.static = static
        self.max_tokens = max_tokens
        self.working = working
        self.short_term = short_term
        self.long_term = long_term

        self.working.bind(max_tokens, on_evict=self._on_working_evict)
        if self.short_term is not None:
            self.short_term.bind(max_tokens, on_evict=self._on_short_term_evict)

    def _on_working_evict(self, messages: list[dict]) -> None:
        if self.short_term is not None:
            self.short_term.add(messages)
        # else: no short-term configured - evicted content is simply dropped.

    def _on_short_term_evict(self, messages: list[dict]) -> None:
        """Called with each raw batch right before short_term folds it into its running
        summary - an unlossy archival copy, since summarizing is inherently lossy and
        long_term is meant to be the durable record once it exists."""
        if self.long_term is not None:
            self.long_term.add(messages)
        # else: no long-term configured yet - archival copy is simply dropped.

    def add(self, message: dict) -> None:
        self.working.add(message)

    def insert_resolved(self, anchor: dict, messages: list[dict]) -> None:
        """Splices a previously-parked, now-resolved exchange (tool call + result +
        synthesized answer) into working memory at anchor's chronological position -
        see WorkingMemory.insert_after(). Facade only, so callers (LLMEngine) never
        reach into working memory directly."""
        self.working.insert_after(anchor, messages)

    def end_turn(self) -> None:
        self.working.end_turn()

    def get(self) -> list[dict]:
        """Assemble static + working's raw messages + short_term summary into the final
        list ready to hand to client.chat.completions.create(messages=...)."""
        messages = [{"role": "system", "content": self.static}]

        # Right after static, not buried mid-conversation or tacked onto the end like
        # STM's summary: this is stable/instructional (which documents exist and are
        # searchable), not a per-turn narrative, so it belongs with the other front-
        # loaded instructions the model attends to most reliably.
        doc_index = self.long_term.document_index_summary() if self.long_term is not None else ""
        if doc_index:
            messages.append({"role": "system", "content": doc_index})

        messages.extend(self.working.get())

        # Appended last, not right after static: models attend most reliably to the
        # start (system) and especially the end (most recent turns) of the context -
        # burying the summary between static and a long conversation history is exactly
        # the position least likely to actually influence the response. Not "system"
        # either: that role is meant to be a stable instruction prefix (and most servers,
        # incl. llama.cpp, reuse KV-cache across calls when it's unchanged) - a
        # fast-changing per-turn summary there would blur "instruction" vs. "context"
        # and defeat that caching. ShortTermMemory.get() already tags its own output
        # clearly as memory, not literal conversation.
        summary = self.short_term.get() if self.short_term is not None else ""
        if summary:
            messages.append({"role": "user", "content": summary})

        return self._trim_to_budget(messages)

    def flush_all(self) -> None:
        """On shutdown: archives whatever's left in working memory straight to
        long_term, skipping short_term's summarization entirely - nobody will ever
        read a fresh running summary again once the process exits, so computing one on
        the way out is pure waste, and it's an LLM call that can itself fail or hang
        right when we're trying to exit cleanly. Anything short_term was ALREADY
        summarizing before this call is still waited on below (avoid abandoning a
        thread mid-flight), just nothing new gets queued into it here."""
        remaining = self.working.get()
        if remaining and self.long_term is not None:
            self.long_term.add(remaining)
        self.working.flush(notify=False)
        if self.short_term is not None:
            self.short_term.flush()
        if self.long_term is not None:
            self.long_term.flush()

    def print(self, raw: bool = False) -> None:
        """Debug helper for a '/mem' command - pretty-prints (or raw=True dumps the exact
        JSON of) what get() would currently send to the LLM."""
        messages = self.get()

        if raw:
            import json
            print(json.dumps(messages, indent=2))
            return

        total = count_messages_tokens(messages)
        print(f"\n--- memory ({len(messages)} messages, ~{total}/{self.max_tokens} tokens) ---")
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content")

            if isinstance(content, list):
                parts = []
                for block in content:
                    if block.get("type") == "image_url":
                        parts.append(f"<image, {len(block['image_url']['url'])} chars>")
                    else:
                        parts.append(str(block))
                content_str = " ".join(parts)
            else:
                content_str = content or ""
                if len(content_str) > 300:
                    content_str = content_str[:300] + f"... <{len(content_str)} chars total>"

            extra = ""
            if msg.get("tool_calls"):
                calls = ", ".join(f"{tc['function']['name']}({tc['function']['arguments']})" for tc in msg["tool_calls"])
                extra = f"  tool_calls=[{calls}]"
            if msg.get("tool_call_id"):
                extra = f"  tool_call_id={msg['tool_call_id']}"

            print(f"[{i}] {role}: {content_str}{extra}")
        print("--- end memory ---\n")

    # ------------------------------------------------------------------
    # Read-time, non-destructive budget enforcement
    # ------------------------------------------------------------------

    def _trim_to_budget(self, messages: list[dict]) -> list[dict]:
        if count_messages_tokens(messages) <= self.max_tokens:
            return messages

        # static, plus the doc index right after it if present (see get()) - both are
        # stable/instructional, never trimmed, same as static always was.
        has_doc_index = self.long_term is not None and self.long_term.document_index_summary() != ""
        lead_count = 2 if has_doc_index else 1
        lead_msgs = messages[:lead_count]

        has_summary = self.short_term is not None and self.short_term.get() != ""
        summary_msg = messages[-1] if has_summary else None
        working_msgs = messages[lead_count:-1] if has_summary else messages[lead_count:]

        # The current, still-open turn is never trimmed - same protection WorkingMemory's
        # own eviction gives it, applied here too so the read path can't reintroduce the
        # same bug (cutting a turn's own tool result out of its own follow-up round).
        boundary = self.working.turn_boundary()
        protected = working_msgs[boundary:]
        trimmable = working_msgs[:boundary]

        budget = self.max_tokens - count_messages_tokens(lead_msgs) - count_messages_tokens(protected)

        if summary_msg is not None:
            summary_cost = count_message_tokens(summary_msg)
            if summary_cost <= budget:
                budget -= summary_cost
            else:
                summary_msg = None  # doesn't fit at all - drop it entirely

        # Keep the most recent trimmable messages that fit, dropping the oldest first.
        # Stops at the first (oldest-going-backward) message that doesn't fit, rather
        # than skipping over it to pack in older/smaller ones - a contiguous recent
        # window keeps tool_call/tool_result pairing and conversational order intact,
        # which matters more here than perfectly maximizing token utilization.
        kept = []
        for msg in reversed(trimmable):
            cost = count_message_tokens(msg)
            if cost > budget:
                break
            kept.insert(0, msg)
            budget -= cost

        result = list(lead_msgs)
        result.extend(kept)
        result.extend(protected)
        if summary_msg is not None:
            result.append(summary_msg)
        return result
