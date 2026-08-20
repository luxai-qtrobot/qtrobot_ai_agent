"""Persistent semantic search over conversations and reference documents."""

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from luxai.magpie.utils import Logger

CHAT_KIND = "chat"
DOCUMENT_KIND = "document"

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANK_POOL_SIZE = 20
TIME_HINTS = ("today", "yesterday", "this_week", "last_week", "this_month", "earlier")


@dataclass
class _Record:
    text: str
    kind: str
    meta: dict[str, Any]
    timestamp: float


class LongTermMemory:
    """Embedding-searchable chat history and in-memory document index.

    Chat messages are persisted as JSON Lines at ``chat_history_path``. The file is
    loaded and re-embedded automatically when the memory is constructed. Documents
    are intentionally reloaded and reindexed by the application on every run.
    """

    def __init__(
        self,
        chat_history_path: str | Path,
        model_name: str = DEFAULT_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        chunk_chars: int = 1000,
        chunk_overlap: int = 100,
    ) -> None:
        self.chat_history_path = Path(chat_history_path)
        self._embedder = TextEmbedding(model_name=model_name)
        self._reranker = TextCrossEncoder(model_name=rerank_model)
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap

        self._lock = threading.Lock()
        self._vectors: np.ndarray | None = None
        self._records: list[_Record] = []
        self._documents: list[str] = []

        self._load_chat_history()
        self._store_queue: queue.Queue[_Record] = queue.Queue()
        self._store_worker = threading.Thread(
            target=self._run_store_worker,
            daemon=True,
            name="ltm-store",
        )
        self._store_worker.start()

    def add_message(self, role: str, text: str) -> None:
        """Persist and index one final user or assistant message in the background."""
        role = role.strip()
        text = text.strip()
        if not role or not text:
            return
        self._store_async(text, CHAT_KIND, {"role": role}, timestamp=time.time())

    def add_document(
        self,
        text: str,
        summary: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Chunk and index extracted reference-document text in the background."""
        text = text.strip()
        if not text:
            return

        resolved_summary = summary or (text[:200].strip() + "...")
        with self._lock:
            if resolved_summary not in self._documents:
                self._documents.append(resolved_summary)

        for chunk in self._chunk_text(text):
            self._store_async(chunk, DOCUMENT_KIND, dict(meta or {}))

    def document_index_summary(self) -> str:
        """Return a prompt-ready inventory of loaded document collections."""
        with self._lock:
            summaries = list(self._documents)
        if not summaries:
            return ""
        lines = "\n".join(f"- {summary}" for summary in summaries)
        return f"[Documents loaded and searchable via search_documents()]:\n{lines}"

    def wait_for_documents(self) -> None:
        """Wait until all documents submitted during startup are ready to search."""
        self._store_queue.join()

    def search(
        self,
        query: str,
        top_k: int = 5,
        kind: str | None = None,
        time_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve and rerank records, optionally filtering by kind and time."""
        with self._lock:
            vectors = self._vectors
            records = list(self._records)

        if vectors is None or not records:
            return []

        mask = np.ones(len(records), dtype=bool)
        if kind is not None:
            mask &= np.array([record.kind == kind for record in records])
        if time_hint is not None:
            start, end = self._time_range(time_hint)
            mask &= np.array([start <= record.timestamp < end for record in records])

        if not mask.any():
            return []

        query_vector = self._embed([query])[0]
        candidates = [record for record, keep in zip(records, mask) if keep]
        similarities = vectors[mask] @ query_vector

        pool_size = min(RERANK_POOL_SIZE, len(candidates))
        pool = [candidates[index] for index in np.argsort(-similarities)[:pool_size]]
        rerank_scores = list(
            self._reranker.rerank(query, [record.text for record in pool])
        )
        ranked = [
            pool[index]
            for index in np.argsort(-np.asarray(rerank_scores))[:top_k]
        ]

        return [
            {
                "text": record.text,
                "kind": record.kind,
                "meta": record.meta,
                "when": (
                    self._relative_time(record.timestamp)
                    if record.kind == CHAT_KIND
                    else None
                ),
            }
            for record in ranked
        ]

    def _load_chat_history(self) -> None:
        if not self.chat_history_path.exists():
            return

        records: list[_Record] = []
        with self.chat_history_path.open("r", encoding="utf-8") as history_file:
            for line_number, line in enumerate(history_file, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    role = str(item["role"]).strip()
                    text = str(item["text"]).strip()
                    timestamp = float(item["timestamp"])
                    if role and text:
                        records.append(
                            _Record(
                                text=text,
                                kind=CHAT_KIND,
                                meta={"role": role},
                                timestamp=timestamp,
                            )
                        )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    Logger.warning(
                        f"[LongTermMemory] ignoring invalid chat-history line "
                        f"{line_number}: {exc}"
                    )

        if not records:
            return

        vectors = self._embed([record.text for record in records])
        with self._lock:
            self._records.extend(records)
            self._vectors = vectors
        Logger.info(
            f"[LongTermMemory] loaded {len(records)} chat message(s) from "
            f"{self.chat_history_path}"
        )

    def _store_async(
        self,
        text: str,
        kind: str,
        meta: dict[str, Any],
        timestamp: float | None = None,
    ) -> None:
        self._store_queue.put(
            _Record(
                text=text,
                kind=kind,
                meta=meta,
                timestamp=timestamp if timestamp is not None else time.time(),
            )
        )

    def _run_store_worker(self) -> None:
        while True:
            record = self._store_queue.get()
            try:
                self._store(record)
            except Exception as exc:
                Logger.error(
                    f"[LongTermMemory] failed to store {record.kind} record: {exc}"
                )
            finally:
                self._store_queue.task_done()

    def _store(self, record: _Record) -> None:
        if record.kind == CHAT_KIND:
            self._append_chat_record(record)

        vector = self._embed([record.text])[0]
        with self._lock:
            self._vectors = (
                vector.reshape(1, -1)
                if self._vectors is None
                else np.vstack([self._vectors, vector])
            )
            self._records.append(record)
        Logger.debug(
            f"[LongTermMemory] stored {record.kind} record "
            f"({len(record.text)} chars)"
        )

    def _append_chat_record(self, record: _Record) -> None:
        item = {
            "role": record.meta["role"],
            "text": record.text,
            "timestamp": record.timestamp,
        }
        encoded = json.dumps(item, ensure_ascii=False)
        with self._lock:
            self.chat_history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.chat_history_path.open("a", encoding="utf-8") as history_file:
                history_file.write(encoded + "\n")

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(list(self._embedder.embed(texts)))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-8, None)

    def _chunk_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = self._snap_to_word_boundary(
                text,
                min(start + self.chunk_chars, len(text)),
            )
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = self._snap_to_word_boundary(text, end - self.chunk_overlap)
        return chunks

    @staticmethod
    def _snap_to_word_boundary(text: str, position: int, lookahead: int = 50) -> int:
        if position <= 0 or position >= len(text) or text[position].isspace():
            return position
        limit = min(position + lookahead, len(text))
        for index in range(position, limit):
            if text[index].isspace():
                return index
        return position

    @staticmethod
    def _time_range(hint: str) -> tuple[float, float]:
        now = time.localtime()
        today_start = time.mktime(
            (now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, -1)
        )
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
        month_start = time.mktime(
            (now.tm_year, now.tm_mon, 1, 0, 0, 0, 0, 0, -1)
        )
        if hint == "this_month":
            return month_start, today_start + day
        return 0.0, month_start

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
