from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from memory import CHAT_KIND, LongTermMemory


class _Embedding:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def embed(self, texts):
        for text in texts:
            lowered = text.lower()
            yield np.asarray(
                [
                    1.0,
                    float(len(text) + 1),
                    float(lowered.count("tea") * 10 + lowered.count("robot") + 1),
                ],
                dtype=np.float32,
            )


class _Reranker:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def rerank(self, query, documents):
        query_words = set(query.lower().split())
        return [
            float(len(query_words.intersection(document.lower().split())))
            for document in documents
        ]


class LongTermMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._embedding_patch = patch(
            "memory.long_term_memory.TextEmbedding",
            _Embedding,
        )
        self._reranker_patch = patch(
            "memory.long_term_memory.TextCrossEncoder",
            _Reranker,
        )
        self._embedding_patch.start()
        self._reranker_patch.start()

    def tearDown(self) -> None:
        self._reranker_patch.stop()
        self._embedding_patch.stop()

    def test_messages_persist_as_jsonl_and_reload_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "chat.jsonl"
            memory = LongTermMemory(history_path)

            memory.add_message("user", "I prefer tea.")
            memory.add_message("assistant", "I will remember that.")
            memory.wait_for_documents()

            records = [
                json.loads(line)
                for line in history_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {(record["role"], record["text"]) for record in records},
                {
                    ("user", "I prefer tea."),
                    ("assistant", "I will remember that."),
                },
            )

            reloaded = LongTermMemory(history_path)
            results = reloaded.search("tea", kind=CHAT_KIND, top_k=2)

            self.assertEqual(len(results), 2)
            self.assertEqual(
                {result["meta"]["role"] for result in results},
                {"user", "assistant"},
            )

    def test_blank_messages_and_documents_are_not_written_to_chat_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "chat.jsonl"
            memory = LongTermMemory(history_path)

            memory.add_message("user", "   ")
            memory.add_document(
                "QTrobot is a social robot.",
                summary="QTrobot manual",
                meta={"source": "manual.txt"},
            )
            memory.wait_for_documents()

            self.assertFalse(history_path.exists())
            self.assertIn("QTrobot manual", memory.document_index_summary())

    def test_background_message_storage_preserves_jsonl_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "chat.jsonl"
            memory = LongTermMemory(history_path)
            expected = [f"message {index}" for index in range(20)]

            for text in expected:
                memory.add_message("user", text)
            memory.wait_for_documents()

            stored = [
                json.loads(line)["text"]
                for line in history_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(stored, expected)


if __name__ == "__main__":
    unittest.main()
