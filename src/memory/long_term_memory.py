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

Vector search is plain NumPy brute-force cosine similarity, followed by a cross-encoder
rerank pass (see search()) - fine at the scale this is built for; swap in hnswlib/
chromadb for stage 1 later only if that stops being true.
"""

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from luxai.magpie.utils import Logger

from .utils import raw_excerpt

CHAT_KIND = "chat"
DOCUMENT_KIND = "document"

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANK_POOL_SIZE = 20  # candidates handed to the reranker before cutting down to top_k

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

    def __init__(self, model_name: str = DEFAULT_MODEL, rerank_model: str = DEFAULT_RERANK_MODEL,
                 chunk_chars: int = 1000, chunk_overlap: int = 100):
        """
        model_name:    fastembed text embedding model (stage 1: cheap recall).
        rerank_model:  fastembed cross-encoder model (stage 2: precise reranking of the
                       stage-1 candidate pool - see search()).
        chunk_chars:   max characters per chunk when splitting a document for indexing.
        chunk_overlap: characters shared between consecutive chunks, so a fact sitting
                       right on a chunk boundary isn't cut in half in every chunk.
        """
        self._embedder = TextEmbedding(model_name=model_name)
        self._reranker = TextCrossEncoder(model_name=rerank_model)
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap

        self._lock = threading.Lock()
        self._vectors: np.ndarray = None  # (N, D), L2-normalized rows
        self._records: list[_Record] = []
        self._pending: list[threading.Thread] = []
        self._documents: list[str] = []  # distinct summaries, dedup'd - see add_document()

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

    def add_document(self, text: str, summary: str = "", meta: dict = None) -> None:
        """text:    full extracted document text, chunked internally before indexing.
        summary: one-line description surfaced via document_index_summary(), so the LLM
                 knows what's available before spending a search_documents() call to
                 find out. Distinct summaries only - loading many files under the same
                 shared summary (e.g. via DirectoryReader) collapses to one line.
        meta:    optional free-form per-chunk metadata (e.g. {"source": "file.txt",
                 "path": ..., "modified": ...} - see document_reader.py), passed through
                 unchanged to every chunk's record. search_documents() results show
                 whatever's present (e.g. "source", for citation) - nothing is required."""
        resolved_summary = summary or (text[:200].strip() + "...")
        with self._lock:
            if resolved_summary not in self._documents:
                self._documents.append(resolved_summary)

        meta = meta or {}
        for chunk in self._chunk_text(text):
            self._store_async(chunk, DOCUMENT_KIND, meta)

    def document_index_summary(self) -> str:
        """Always-on list of distinct document summaries - lets the LLM know
        search_documents() is worth calling, and roughly what it'll find, without a tool
        round trip just to discover that. Empty string if nothing is loaded."""
        with self._lock:
            summaries = list(self._documents)
        if not summaries:
            return ""
        lines = [f"- {s}" for s in summaries]
        return "[Documents loaded and searchable via search_documents()]:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5, kind: str = None, time_hint: str = None) -> list[dict]:
        """Two-stage retrieve-then-rerank, optionally filtered by kind
        (CHAT_KIND/DOCUMENT_KIND) and/or a structured time_hint bucket (see TIME_HINTS -
        chat records only; documents have no meaningful "when"). Stage 1 is a cheap
        cosine pass over the whole (filtered) store that casts a wide net - fast, but
        easily fooled by chunks that just share vocabulary with the query. Stage 2 runs
        a cross-encoder over that shortlist, scoring query+chunk jointly for real
        relevance, and only that final ranking decides what's returned - too slow to run
        over the whole store, cheap over a short pool. Returns
        [{"text", "kind", "meta", "when"}, ...] most-relevant first - "when" is rendered
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

        pool_size = min(RERANK_POOL_SIZE, len(candidates))
        pool = [candidates[i] for i in np.argsort(-sims)[:pool_size]]

        rerank_scores = list(self._reranker.rerank(query, [r.text for r in pool]))
        ranked = [pool[i] for i in np.argsort(-np.array(rerank_scores))[:top_k]]

        return [
            {
                "text": r.text,
                "kind": r.kind,
                "meta": r.meta,
                # Only chat has a meaningful "when" - a document's embed-time timestamp
                # says nothing about the document itself, so don't present it as one.
                "when": self._relative_time(r.timestamp) if r.kind == CHAT_KIND else None,
            }
            for r in ranked
        ]

    def flush(self) -> None:
        """Join all in-flight embed+store threads - called by RobotMemory.flush_all() on
        shutdown so nothing pending is lost when the process exits."""
        with self._lock:
            threads = list(self._pending)
        for thread in threads:
            thread.join()

    # ------------------------------------------------------------------
    # Persistence - chat history only. Documents are never persisted: they're cheap to
    # reindex from source files every run (see document_reader.py), and re-embedding on
    # load (rather than trusting stored vectors) means an old save file can never go
    # silently stale if the embedding model is ever changed later.
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Writes archived chat records as plain, human-readable JSON (text/timestamp/
        meta - no vectors), so the file doubles as a readable chat log a user can open
        or hand-edit, not just a technical cache."""
        with self._lock:
            records = [
                {"text": r.text, "timestamp": r.timestamp, "meta": r.meta}
                for r in self._records if r.kind == CHAT_KIND
            ]
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        Logger.info(f"[LongTermMemory] saved {len(records)} chat record(s) to {path}")

    def load(self, path: str) -> None:
        """Reindexes chat history written by save() - every record is re-embedded fresh
        rather than trusting old vectors (see class docstring above for why). No-op if
        the file doesn't exist yet (e.g. first run)."""
        in_path = Path(path)
        if not in_path.exists():
            return
        records = json.loads(in_path.read_text(encoding="utf-8"))
        if not records:
            return

        vectors = self._embed([r["text"] for r in records])
        with self._lock:
            for r, vec in zip(records, vectors):
                self._records.append(_Record(text=r["text"], kind=CHAT_KIND, meta=r["meta"], timestamp=r["timestamp"]))
            self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        Logger.info(f"[LongTermMemory] loaded {len(records)} chat record(s) from {path}")

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
