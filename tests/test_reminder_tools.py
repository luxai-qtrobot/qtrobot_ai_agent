from __future__ import annotations

import math
import queue
import sys
import time
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from luxai.magpie.schema import McpSchema
from tool import ReminderTools
from tool.providers.reminder_tools import MAX_REMINDER_MESSAGE_LENGTH


class ReminderToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events: queue.Queue[dict] = queue.Queue()
        self.tools = ReminderTools(self.events.put)

    def tearDown(self) -> None:
        self.tools.cleanup()

    def test_registers_all_reminder_methods(self) -> None:
        schema = McpSchema(name="test-tools")

        self.tools.register(schema)

        tool_names = {
            tool["name"] for tool in schema._mcp_tools_list()["tools"]
        }
        self.assertEqual(
            tool_names,
            {"set_reminder", "list_reminders", "cancel_reminder"},
        )

    def test_multiple_reminders_return_contract_and_emit_due_events(self) -> None:
        first = self.tools.set_reminder("take medicine.", 0.04)
        second = self.tools.set_reminder("drink water.", 0.08)

        self.assertEqual(first["status"], "completed")
        self.assertTrue(first["task_id"].startswith("reminder_"))
        self.assertIn("take medicine.", first["summary"])
        self.assertGreater(first["remaining_seconds"], 0)
        self.assertEqual(
            [item["task_id"] for item in self.tools.list_reminders()],
            [first["task_id"], second["task_id"]],
        )

        events = [
            self.events.get(timeout=0.5),
            self.events.get(timeout=0.5),
        ]
        self.assertEqual(
            events,
            [
                {
                    "type": "reminder.due",
                    "id": first["task_id"],
                    "payload": "Remind the user to take medicine.",
                    "due_at": first["due_at"],
                },
                {
                    "type": "reminder.due",
                    "id": second["task_id"],
                    "payload": "Remind the user to drink water.",
                    "due_at": second["due_at"],
                },
            ],
        )
        self.assertEqual(self.tools.list_reminders(), [])

    def test_cancelled_reminder_does_not_fire(self) -> None:
        reminder = self.tools.set_reminder("do not fire", 0.04)

        result = self.tools.cancel_reminder(reminder["task_id"])
        time.sleep(0.07)

        self.assertIn("Cancelled reminder", result)
        self.assertIn(reminder["task_id"], result)
        self.assertTrue(self.events.empty())
        self.assertEqual(self.tools.list_reminders(), [])

    def test_unknown_task_id_returns_normal_result(self) -> None:
        result = self.tools.cancel_reminder("reminder_missing")

        self.assertIn("No active reminder", result)
        self.assertIn("reminder_missing", result)

    def test_invalid_reminders_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            self.tools.set_reminder(" ", 1)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self.tools.set_reminder(
                "x" * (MAX_REMINDER_MESSAGE_LENGTH + 1),
                1,
            )

        for delay in (False, True, 0, -1, math.nan, math.inf, -math.inf):
            with self.subTest(delay=delay):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite number greater than zero",
                ):
                    self.tools.set_reminder("test", delay)

    def test_cleanup_is_idempotent_and_prevents_future_events(self) -> None:
        self.tools.set_reminder("discard me", 0.04)

        self.tools.cleanup()
        self.tools.cleanup()
        time.sleep(0.07)

        self.assertTrue(self.events.empty())
        self.assertEqual(self.tools.list_reminders(), [])
        with self.assertRaisesRegex(RuntimeError, "stopping"):
            self.tools.set_reminder("too late", 1)


if __name__ == "__main__":
    unittest.main()
