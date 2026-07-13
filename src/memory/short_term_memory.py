"""
short_term_memory.py - ShortTermMemory: a single running narrative summary of content
evicted out of working memory.

Deliberately simple: one plain-prose summary string, updated in place every time a new
batch arrives ("fold this new excerpt into the summary, keep it concise"). No fact list,
no tags, no separate condense step - staying short is just part of every update.

Summarization is an LLM call and can take seconds - it always runs on a background
thread, tracked internally, so it never blocks the live conversation. Unlike asyncio
tasks, a real OS thread can't be starved by a long blocking call elsewhere in the
process (e.g. the live streaming chat call) - it just runs. Content that's been evicted
from working memory but not yet folded into the summary is kept in a 'pending' holding
area and is still included in get() as raw text, so nothing the user said is ever
silently missing from context, even mid-summarization.

Takes a plain (sync) OpenAI client - the same one the live conversation uses is fine to
share, since the SDK's client is safe to call from multiple threads at once.
"""

import threading

from luxai.magpie.utils import Logger

from .base import ShortTermMemoryBase
from .utils import raw_excerpt

DEFAULT_SUMMARY_PROMPT = (
    "You are a memory summarizer for a robot's conversation. Update the running summary "
    "below to also incorporate the new conversation excerpt. Write it as short bullet "
    "points - one short sentence each, describing what happened (including any tool use) "
    "- no long paragraphs, no commentary. Keep it under about {max_tokens} tokens; drop "
    "stale or minor detail if needed to stay within that.\n\n"
    "Current summary:\n{summary}\n\n"
    "New excerpt:\n{excerpt}\n\n"
    "Return ONLY the updated summary as bullet points."
)


class ShortTermMemory(ShortTermMemoryBase):

    def __init__(self, llm, model: str, size_ratio: float = 0.25, flush_ratio: float = 0.2,
                 retries: int = 1, timeout: float = 30.0):
        """
        llm:         a (sync) OpenAI client - background summarization calls it from a
                     worker thread; safe to share with the client the live chat uses.
        model:       model name passed to every summarization call.
        size_ratio:  this tier's share of RobotMemory's max_tokens, resolved at bind() -
                     used as a soft target size hint in the summarization prompt.
        flush_ratio: unused - there's nothing to evict here (see class docstring); kept
                     only for constructor symmetry with WorkingMemory.
        retries:     extra attempts for a failed/hung summarization call.
        timeout:     per-call timeout (seconds), so a stuck call fails loudly instead of
                     wedging a batch in 'pending' forever.
        """
        self.llm = llm
        self.model = model
        self.size_ratio = size_ratio
        self.flush_ratio = flush_ratio
        self.retries = retries
        self.timeout = timeout
        self.max_tokens = None      # resolved by bind()
        self._on_evict = None       # resolved by bind()

        self._lock = threading.Lock()
        self._summary = ""
        self._pending: list[tuple[list[dict], threading.Thread]] = []

    def bind(self, total_tokens: int, on_evict=None) -> None:
        """on_evict, if given, receives each raw evicted batch (messages) right before it
        gets folded into the summary - an uncondensed archival copy for long-term memory,
        since folding into a running summary is inherently lossy."""
        self.max_tokens = int(total_tokens * self.size_ratio)
        self._on_evict = on_evict

    def add(self, messages: list[dict]) -> None:
        """Called by working memory on eviction. Never blocks - archives the raw batch
        (if a callback is set), starts a background summarization thread, and returns
        immediately."""
        if self._on_evict:
            self._on_evict(messages)

        thread = threading.Thread(target=self._summarize, args=(messages,), daemon=True)
        with self._lock:
            self._pending.append((messages, thread))
        thread.start()

    def get(self) -> str:
        """Current running summary, plus raw text for anything still being processed - so
        nothing is ever missing from context, just sometimes less compact mid-transition.
        Clearly tagged as memory, not literal conversation."""
        with self._lock:
            summary = self._summary
            pending_batches = [messages for messages, thread in self._pending if thread.is_alive()]

        parts = [summary] if summary else []
        parts.extend(raw_excerpt(m) for m in pending_batches)

        if not parts:
            return ""
        body = "\n".join(parts)
        return f"[Memory - background information the robot recalls]:\n{body}"

    def flush(self) -> None:
        """Join all in-flight summarization threads - called by RobotMemory.flush_all() on
        shutdown so nothing pending is lost when the process exits."""
        with self._lock:
            threads = [thread for _, thread in self._pending]
        for thread in threads:
            thread.join()

    # ------------------------------------------------------------------
    # Background summarization (runs on a worker thread, started from add())
    # ------------------------------------------------------------------

    def _summarize(self, messages: list[dict]) -> None:
        excerpt = raw_excerpt(messages)
        with self._lock:
            current_summary = self._summary
        prompt = DEFAULT_SUMMARY_PROMPT.format(
            summary=current_summary or "(none yet)",
            excerpt=excerpt,
            max_tokens=self.max_tokens or 200,
        )
        Logger.debug(f"[ShortTermMemory] summarizing:\n{excerpt}")

        updated = self._ask(prompt)

        with self._lock:
            if updated:
                Logger.debug(f"[ShortTermMemory] summary updated: {updated}")
                self._summary = updated
            else:
                Logger.debug("[ShortTermMemory] summarization returned nothing - keeping previous summary")
            self._pending = [(m, t) for m, t in self._pending if t is not threading.current_thread()]

    def _ask(self, prompt: str) -> str:
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    timeout=self.timeout,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                last_error = e
                Logger.debug(f"[ShortTermMemory] summarization attempt {attempt + 1} failed: {e}")
        Logger.debug(f"[ShortTermMemory] summarization failed after {self.retries + 1} attempts: {last_error}")
        return ""
