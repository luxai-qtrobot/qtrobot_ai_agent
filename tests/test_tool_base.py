from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tool.tool_base import ToolBase


class _Tool(ToolBase):
    def register(self, schema) -> None:
        pass


class ToolBaseTests(unittest.TestCase):
    def test_event_sink_is_optional(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "has no event sink"):
            _Tool().emit_event({"type": "test"})

    def test_background_provider_can_emit_an_event(self) -> None:
        events: list[dict] = []
        tool = _Tool(events.append)

        tool.emit_event({"type": "test", "payload": "done"})

        self.assertEqual(events, [{"type": "test", "payload": "done"}])


if __name__ == "__main__":
    unittest.main()
