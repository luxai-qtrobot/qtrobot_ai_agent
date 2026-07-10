"""
long_term_memory.py - LongTermMemory: durable, embedding-searchable archive that sits
below short-term memory in the working -> short-term -> long-term flow, plus a
separate ingestion path for loaded reference documents.

Both archived chat history and loaded documents live in the same vector store,
distinguished only by a `kind` tag (CHAT_KIND / DOCUMENT_KIND) - filtering on it in
search() is what lets search_memory() and search_documents() (tool/memory_tools.py)
be two distinct LLM-facing tools backed by one index, per the design discussion.

Embedding is fastembed (ONNX, CPU, in-process) - no extra server, no torch, and
doesn't touch the llama.cpp chat server at all. Kept off the hot path: add() always
returns immediately and does the actual embed+store on a background thread, the same
pattern ShortTermMemory uses for its own summarization calls.

Vector search is plain NumPy brute-force cosine similarity - fine at the scale this is
built for; swap in hnswlib/chromadb later only if that stops being true.
"""

import threading
import time
from dataclasses import dataclass

import numpy as np
from fastembed import TextEmbedding

from luxai.magpie.utils import Logger

from .message_utils import raw_excerpt

CHAT_KIND = "chat"
DOCUMENT_KIND = "document"

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Structured relative-time buckets accepted by search()'s time_hint - deliberately not
# freeform NL parsing (no extra dependency, and this is what models naturally reach
# for when a user says "yesterday" / "last week" anyway).
TIME_HINTS = ("today", "yesterday", "this_week", "last_week", "this_month", "earlier")


@dataclass
class _Record:
    text: str
    kind: str
    meta: dict
    timestamp: float


class LongTermMemory:

    def __init__(self, model_name: str = DEFAULT_MODEL, chunk_chars: int = 1000, chunk_overlap: int = 100):
        """
        model_name:    fastembed text embedding model.
        chunk_chars:   max characters per chunk when splitting a document for indexing.
        chunk_overlap: characters shared between consecutive chunks, so a fact sitting
                       right on a chunk boundary isn't cut in half in every chunk.
        """
        self._embedder = TextEmbedding(model_name=model_name)
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap

        self._lock = threading.Lock()
        self._vectors: np.ndarray = None  # (N, D), L2-normalized rows
        self._records: list[_Record] = []
        self._pending: list[threading.Thread] = []
        self._documents: dict[str, str] = {}  # source name -> brief description

    # ------------------------------------------------------------------
    # Chat archival - wired via RobotMemory._on_short_term_evict
    # ------------------------------------------------------------------

    def add(self, messages: list[dict]) -> None:
        """Archive a raw batch evicted from short-term memory. Never blocks - embeds and
        stores on a background thread, same pattern as ShortTermMemory.add()."""
        text = raw_excerpt(messages)
        if not text:
            return
        self._store_async(text, CHAT_KIND, {})

    # ------------------------------------------------------------------
    # Document ingestion - chunking + indexing only. Loading/parsing actual files
    # (pdf, docx, ...) is a separate, not-yet-built concern; callers hand this
    # already-extracted text.
    # ------------------------------------------------------------------

    def add_document(self, source: str, text: str, summary: str = "") -> None:
        """source:  short identifier (e.g. filename) - also shown in search results.
        text:    full extracted document text, chunked internally before indexing.
        summary: one-line description surfaced via document_index_summary(), so the
                 LLM knows this document exists and roughly what it covers without
                 spending a search_documents() call just to discover that."""
        with self._lock:
            self._documents[source] = summary or (text[:200].strip() + "...")

        for chunk in self._chunk_text(text):
            self._store_async(chunk, DOCUMENT_KIND, {"source": source})

    def document_index_summary(self) -> str:
        """Always-on list of loaded documents (name + brief description) - lets the LLM
        know search_documents() is worth calling, and roughly what it'll find, without a
        tool round trip just to discover that. Empty string if nothing is loaded."""
        with self._lock:
            documents = dict(self._documents)
        if not documents:
            return ""
        lines = [f"- {name}: {summary}" for name, summary in documents.items()]
        return "[Documents loaded and searchable via search_documents()]:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5, kind: str = None, time_hint: str = None) -> list[dict]:
        """Embeds query, cosine-ranks against the store, optionally filtered by kind
        (CHAT_KIND/DOCUMENT_KIND) and/or a structured time_hint bucket (see TIME_HINTS -
        chat records only; documents have no meaningful "when"). Returns
        [{"text", "kind", "meta", "when"}, ...] most-similar first - "when" is rendered
        relative to now (e.g. "3 days ago"), not a raw timestamp, since that's what the
        model can actually reason with."""
        with self._lock:
            vectors = self._vectors
            records = list(self._records)

        if vectors is None or not records:
            return []

        mask = np.ones(len(records), dtype=bool)
        if kind is not None:
            mask &= np.array([r.kind == kind for r in records])
        if time_hint is not None:
            start, end = self._time_range(time_hint)
            mask &= np.array([start <= r.timestamp < end for r in records])

        if not mask.any():
            return []

        query_vec = self._embed([query])[0]
        candidates = [r for r, keep in zip(records, mask) if keep]
        sims = vectors[mask] @ query_vec
        order = np.argsort(-sims)[:top_k]

        return [
            {
                "text": candidates[i].text,
                "kind": candidates[i].kind,
                "meta": candidates[i].meta,
                # Only chat has a meaningful "when" - a document's embed-time timestamp
                # says nothing about the document itself, so don't present it as one.
                "when": self._relative_time(candidates[i].timestamp) if candidates[i].kind == CHAT_KIND else None,
            }
            for i in order
        ]

    def flush(self) -> None:
        """Join all in-flight embed+store threads - called by RobotMemory.flush_all() on
        shutdown so nothing pending is lost when the process exits."""
        with self._lock:
            threads = list(self._pending)
        for thread in threads:
            thread.join()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _store_async(self, text: str, kind: str, meta: dict) -> None:
        thread = threading.Thread(target=self._store, args=(text, kind, meta), daemon=True)
        with self._lock:
            self._pending.append(thread)
        thread.start()

    def _store(self, text: str, kind: str, meta: dict) -> None:
        vec = self._embed([text])[0]
        record = _Record(text=text, kind=kind, meta=meta, timestamp=time.time())
        with self._lock:
            self._vectors = vec.reshape(1, -1) if self._vectors is None else np.vstack([self._vectors, vec])
            self._records.append(record)
            self._pending = [t for t in self._pending if t is not threading.current_thread()]
        Logger.debug(f"[LongTermMemory] stored {kind} record ({len(text)} chars)")

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self._embedder.embed(texts)))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-8, None)

    def _chunk_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = self._snap_to_word_boundary(text, min(start + self.chunk_chars, len(text)))
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = self._snap_to_word_boundary(text, end - self.chunk_overlap)
        return chunks

    @staticmethod
    def _snap_to_word_boundary(text: str, pos: int, lookahead: int = 50) -> int:
        """Nudges a chunk-split position forward to the next whitespace, so a chunk
        never starts or ends mid-word (e.g. "QTrobot" split into "Q" + "Trobot").
        Bounded lookahead so one unusually long unbroken run of text can't blow up a
        chunk - falls back to cutting right at pos if no whitespace is found nearby."""
        if pos <= 0 or pos >= len(text) or text[pos].isspace():
            return pos
        limit = min(pos + lookahead, len(text))
        for i in range(pos, limit):
            if text[i].isspace():
                return i
        return pos

    @staticmethod
    def _time_range(hint: str) -> tuple:
        now = time.localtime()
        today_start = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, -1))
        day = 86400

        if hint == "today":
            return today_start, today_start + day
        if hint == "yesterday":
            return today_start - day, today_start
        if hint == "this_week":
            return today_start - now.tm_wday * day, today_start + day
        if hint == "last_week":
            start = today_start - (now.tm_wday + 7) * day
            return start, start + 7 * day
        month_start = time.mktime((now.tm_year, now.tm_mon, 1, 0, 0, 0, 0, 0, -1))
        if hint == "this_month":
            return month_start, today_start + day
        return 0.0, month_start  # "earlier" (or unrecognized) - everything before this month

    @staticmethod
    def _relative_time(timestamp: float) -> str:
        delta = time.time() - timestamp
        if delta < 3600:
            return f"{max(1, int(delta // 60))} minute(s) ago"
        if delta < 86400:
            return f"{int(delta // 3600)} hour(s) ago"
        days = int(delta // 86400)
        if days == 1:
            return "yesterday"
        if days < 7:
            return f"{days} days ago"
        if days < 35:
            return f"{days // 7} week(s) ago"
        if days < 365:
            return f"{days // 30} month(s) ago"
        return f"{days // 365} year(s) ago"
