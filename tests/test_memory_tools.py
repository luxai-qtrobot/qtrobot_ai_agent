from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from memory import CHAT_KIND, DOCUMENT_KIND
from tool.providers.memory_tools import MEMORY_TOOL_INSTRUCTIONS, MemoryTools


class _Memory:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        if kwargs.get("kind") == CHAT_KIND:
            return [
                {
                    "text": "The user likes tea.",
                    "meta": {"role": "user"},
                    "when": "yesterday",
                }
            ]
        return [
            {
                "text": "QTrobot has two articulated arms.",
                "meta": {"source": "manual.pdf"},
                "when": None,
            }
        ]


class MemoryToolsTests(unittest.TestCase):
    def test_search_memory_uses_chat_filter_and_time_hint(self) -> None:
        memory = _Memory()
        tools = MemoryTools(memory)

        result = tools.search_memory("What does the user like?", "yesterday")

        self.assertEqual(
            memory.calls,
            [
                {
                    "query": "What does the user like?",
                    "kind": CHAT_KIND,
                    "time_hint": "yesterday",
                }
            ],
        )
        self.assertIn("(yesterday)", result)
        self.assertIn("[user]", result)
        self.assertIn("The user likes tea.", result)

    def test_search_documents_uses_document_filter_and_formats_source(self) -> None:
        memory = _Memory()
        tools = MemoryTools(memory)

        result = tools.search_documents("How many arms does QTrobot have?")

        self.assertEqual(
            memory.calls,
            [
                {
                    "query": "How many arms does QTrobot have?",
                    "kind": DOCUMENT_KIND,
                }
            ],
        )
        self.assertIn("[manual.pdf]", result)

    def test_empty_search_result_is_clear(self) -> None:
        class EmptyMemory:
            def search(self, *args, **kwargs):
                return []

        self.assertEqual(
            MemoryTools(EmptyMemory()).search_memory("unknown"),
            "No matching results found.",
        )

    def test_usage_instructions_match_s2s_owned_recent_history(self) -> None:
        self.assertIn("recent conversation", MEMORY_TOOL_INSTRUCTIONS)
        self.assertIn("search_memory", MEMORY_TOOL_INSTRUCTIONS)
        self.assertIn("search_documents", MEMORY_TOOL_INSTRUCTIONS)
        self.assertNotIn("working memory", MEMORY_TOOL_INSTRUCTIONS.lower())
        self.assertNotIn("short-term", MEMORY_TOOL_INSTRUCTIONS.lower())


if __name__ == "__main__":
    unittest.main()
