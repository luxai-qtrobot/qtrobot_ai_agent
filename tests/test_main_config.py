from __future__ import annotations

import sys
import unittest
from pathlib import Path

from luxai.magpie.frames import DictFrame


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import (
    BACKGROUND_EVENT_INSTRUCTIONS,
    DEFAULT_ASSISTANT_INSTRUCTIONS,
    MEMORY_TOOL_INSTRUCTIONS,
    _log_event,
    _session_config,
)


class _Memory:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def add_message(self, role: str, text: str) -> None:
        self.messages.append((role, text))


class MainSessionConfigTests(unittest.TestCase):
    def test_default_instructions_include_background_event_contract(self) -> None:
        session = _session_config("Aiden", None, [])

        self.assertIn(DEFAULT_ASSISTANT_INSTRUCTIONS, session["instructions"])
        self.assertIn(BACKGROUND_EVENT_INSTRUCTIONS, session["instructions"])
        self.assertIn(MEMORY_TOOL_INSTRUCTIONS, session["instructions"])
        self.assertIn("reminder.due", session["instructions"])
        self.assertIn("search_web", session["instructions"])

    def test_custom_instructions_are_preserved_with_required_contract(self) -> None:
        session = _session_config("Aiden", "Be playful.", [])

        self.assertTrue(session["instructions"].startswith("Be playful."))
        self.assertIn(BACKGROUND_EVENT_INSTRUCTIONS, session["instructions"])

    def test_document_inventory_is_added_to_session_instructions(self) -> None:
        inventory = "[Documents loaded and searchable via search_documents()]:\n- QTrobot manual"

        session = _session_config("Aiden", None, [], inventory)

        self.assertIn(inventory, session["instructions"])
        self.assertEqual(session["instructions"].count(inventory), 1)

    def test_logged_transcripts_are_forwarded_to_long_term_memory(self) -> None:
        memory = _Memory()

        _log_event(
            DictFrame(
                gid="session-1",
                value={
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "  Hello robot.  ",
                },
            ),
            memory,
        )
        _log_event(
            DictFrame(
                gid="session-1",
                value={
                    "type": "response.output_audio_transcript.done",
                    "transcript": "  Hello!  ",
                },
            ),
            memory,
        )

        self.assertEqual(
            memory.messages,
            [("user", "Hello robot."), ("assistant", "Hello!")],
        )

    def test_empty_transcript_is_not_stored(self) -> None:
        memory = _Memory()

        _log_event(
            DictFrame(
                gid="session-1",
                value={
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "   ",
                },
            ),
            memory,
        )

        self.assertEqual(memory.messages, [])


if __name__ == "__main__":
    unittest.main()
