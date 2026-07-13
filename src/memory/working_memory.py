"""
working_memory.py - WorkingMemory: raw, unsummarized recent chat history.

Self-managed: image superseding and overflow eviction happen automatically on add();
turn-boundary tool-result compaction happens on end_turn(). No external caller needs to
know about or apply any of these rules - they're all internal invariants of this class.
"""

from .base import WorkingMemoryBase
from .utils import count_messages_tokens

# How much detail to keep in a compacted tool-result summary line (name(args) -> result).
# Long tool results (e.g. a full gesture list) get truncated here, not dropped entirely.
COMPACTED_RESULT_CHARS = 150


class WorkingMemory(WorkingMemoryBase):

    def __init__(self, size_ratio: float = 0.75, flush_ratio: float = 0.2):
        """
        size_ratio:  this tier's share of RobotMemory's max_tokens, resolved into an
                     absolute budget once, at bind() time.
        flush_ratio: fraction of this tier's own budget to shed on overflow (e.g. 0.2 ->
                     evict oldest messages until ~80% full), so flushes are chunky rather
                     than thrashing one message at a time at the boundary.
        """
        self.size_ratio = size_ratio
        self.flush_ratio = flush_ratio
        self.max_tokens = None      # resolved by bind()
        self._on_evict = None       # resolved by bind()

        self._messages: list[dict] = []
        self._turn_start = 0        # index into _messages where the current turn began

    def bind(self, total_tokens: int, on_evict) -> None:
        """Called once by RobotMemory: resolves size_ratio into an absolute budget and
        wires up the eviction hand-off to short-term memory."""
        self.max_tokens = int(total_tokens * self.size_ratio)
        self._on_evict = on_evict

    def add(self, message: dict) -> None:
        if self._has_image(message):
            self._supersede_images()
        self._messages.append(message)
        self._evict_if_needed()

    def get(self) -> list[dict]:
        return list(self._messages)

    def turn_boundary(self) -> int:
        """Index into get()'s returned list where the current, still-open turn begins.
        Messages at or after this index must never be evicted or trimmed away - RobotMemory
        uses this to protect them in its own read-time budget trim too."""
        return self._turn_start

    def end_turn(self) -> None:
        """Compact this turn's own tool_call/tool_result exchanges in place, now that the
        turn is fully resolved and no longer needs full detail for live in-turn reasoning
        (the LLM already used the raw result to answer - only future turns need less)."""
        turn_messages = self._messages[self._turn_start:]
        self._messages[self._turn_start:] = self._compact_tool_exchanges(turn_messages)
        self._turn_start = len(self._messages)

    def flush(self) -> None:
        """Unconditionally hand off everything to short-term, bypassing the normal
        overflow trigger. Called on shutdown so nothing is lost just because the session
        ended before working memory happened to overflow."""
        if self._messages and self._on_evict:
            self._on_evict(self._messages)
        self._messages = []
        self._turn_start = 0

    # ------------------------------------------------------------------
    # Internal invariants
    # ------------------------------------------------------------------

    @staticmethod
    def _has_image(message: dict) -> bool:
        content = message.get("content")
        return isinstance(content, list) and any(b.get("type") == "image_url" for b in content)

    def _supersede_images(self) -> None:
        """Only the most recent image is cognitively useful - drop any prior ones the
        instant a new one arrives, regardless of memory pressure."""
        for msg in self._messages:
            if self._has_image(msg):
                msg["content"] = [b for b in msg["content"] if b.get("type") != "image_url"]
        kept = [m for m in self._messages if m.get("content")]
        self._turn_start -= (len(self._messages) - len(kept))
        self._turn_start = max(0, self._turn_start)
        self._messages = kept

    def _evict_if_needed(self) -> None:
        if self.max_tokens is None or count_messages_tokens(self._messages) <= self.max_tokens:
            return

        target = int(self.max_tokens * (1 - self.flush_ratio))
        evicted = []
        # Never evict anything from the current, still-open turn - only messages strictly
        # before _turn_start are eligible. If clearing all of those still isn't enough,
        # working memory just runs over its own soft budget until end_turn() resolves it;
        # RobotMemory's own read-time trim (also turn_boundary()-aware) is the hard backstop.
        while self._turn_start > 0 and count_messages_tokens(self._messages) > target:
            evicted.append(self._messages.pop(0))
            self._turn_start -= 1

        if evicted and self._on_evict:
            self._on_evict(evicted)

    def _compact_tool_exchanges(self, messages: list[dict]) -> list[dict]:
        """Replace each assistant tool_calls + matching tool-result messages with one
        short summary line ('name(args) -> result'), long results truncated. Everything
        else (plain text turns, image messages) passes through untouched."""
        results = {
            m["tool_call_id"]: str(m.get("content") or "")
            for m in messages if m.get("role") == "tool"
        }

        compacted = []
        for msg in messages:
            if msg.get("role") == "tool":
                continue  # folded into the summary line below

            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                actions = []
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    args = tc["function"]["arguments"]
                    result = results.get(tc["id"], "")
                    if len(result) > COMPACTED_RESULT_CHARS:
                        result = result[:COMPACTED_RESULT_CHARS] + f"...<{len(result)} chars>"
                    actions.append(f"{name}({args}) -> {result}")
                compacted.append({"role": "assistant", "content": "[tool actions] " + "; ".join(actions)})
                if msg.get("content"):
                    compacted.append({"role": "assistant", "content": msg["content"]})
            else:
                compacted.append(msg)

        return compacted
