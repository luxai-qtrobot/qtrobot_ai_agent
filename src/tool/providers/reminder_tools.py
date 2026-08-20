"""In-memory reminders exposed as local MCP tools."""

from __future__ import annotations

import math
import threading
import time
import uuid
from datetime import datetime
from typing import Any

from luxai.magpie.schema import McpSchema
from luxai.magpie.utils import Logger

from ..tool_base import ToolBase


MAX_REMINDER_MESSAGE_LENGTH = 1_000


class ReminderTools(ToolBase):
    """Schedule multiple reminders and emit each one when it becomes due."""

    def __init__(self, event_sink) -> None:
        super().__init__(event_sink)
        self._condition = threading.Condition()
        self._reminders: dict[str, dict[str, Any]] = {}
        self._stopping = False
        self._scheduler = threading.Thread(
            target=self._run_scheduler,
            name="qtrobot-reminder-scheduler",
            daemon=True,
        )
        self._scheduler.start()

    def register(self, schema: McpSchema) -> None:
        schema.method()(self.set_reminder)
        schema.method()(self.list_reminders)
        schema.method()(self.cancel_reminder)

    def set_reminder(self, message: str, delay_seconds: float) -> dict[str, Any]:
        """Set a reminder or timer to fire after a positive number of seconds.

        ``message`` should describe what to remind the user to do. Use this for
        requests such as "remind me in one minute to make a call" or "set a
        timer for 20 seconds".
        """
        message = message.strip()
        if not message:
            raise ValueError("Reminder message cannot be empty")
        if len(message) > MAX_REMINDER_MESSAGE_LENGTH:
            raise ValueError(
                "Reminder message cannot exceed "
                f"{MAX_REMINDER_MESSAGE_LENGTH} characters"
            )

        if isinstance(delay_seconds, bool):
            raise ValueError(
                "delay_seconds must be a finite number greater than zero"
            )
        try:
            delay = float(delay_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "delay_seconds must be a finite number greater than zero"
            ) from exc
        if not math.isfinite(delay) or delay <= 0:
            raise ValueError(
                "delay_seconds must be a finite number greater than zero"
            )

        task_id = f"reminder_{uuid.uuid4().hex[:10]}"
        reminder = {
            "task_id": task_id,
            "message": message,
            "due_at": time.time() + delay,
            "deadline": time.monotonic() + delay,
        }

        with self._condition:
            if self._stopping:
                raise RuntimeError("Reminder scheduler is stopping")
            self._reminders[task_id] = reminder
            self._condition.notify()

        public_reminder = self._public_reminder(reminder)
        result = {
            "status": "completed",
            "task_id": task_id,
            "summary": (
                f"Reminder set for {public_reminder['due_at']}: {message}"
            ),
            "due_at": public_reminder["due_at"],
            "remaining_seconds": public_reminder["remaining_seconds"],
        }
        Logger.info(
            f"Reminder scheduled: {task_id} at {result['due_at']} ({message})"
        )
        return result

    def list_reminders(self) -> list[dict[str, Any]]:
        """List all active reminders and timers, ordered by due time."""
        with self._condition:
            reminders = sorted(
                self._reminders.values(),
                key=lambda reminder: reminder["deadline"],
            )
            return [self._public_reminder(reminder) for reminder in reminders]

    def cancel_reminder(self, task_id: str) -> str:
        """Cancel one active reminder or timer by its task ID."""
        with self._condition:
            reminder = self._reminders.pop(task_id, None)
            if reminder is None:
                return f"No active reminder found with task ID {task_id}."
            self._condition.notify()

        Logger.info(f"Reminder cancelled: {task_id}")
        return f"Cancelled reminder {task_id}: {reminder['message']}"

    def cleanup(self) -> None:
        """Stop the scheduler and discard all reminders not yet due."""
        with self._condition:
            self._stopping = True
            self._reminders.clear()
            self._condition.notify_all()

        if threading.current_thread() is not self._scheduler:
            self._scheduler.join(timeout=2.0)

    def _run_scheduler(self) -> None:
        while True:
            due: list[dict[str, Any]] = []
            with self._condition:
                while not self._stopping:
                    if not self._reminders:
                        self._condition.wait()
                        continue

                    now = time.monotonic()
                    next_deadline = min(
                        reminder["deadline"]
                        for reminder in self._reminders.values()
                    )
                    wait_seconds = next_deadline - now
                    if wait_seconds > 0:
                        self._condition.wait(timeout=wait_seconds)
                        continue

                    due_task_ids = [
                        task_id
                        for task_id, reminder in self._reminders.items()
                        if reminder["deadline"] <= now
                    ]
                    due = [
                        self._reminders.pop(task_id)
                        for task_id in due_task_ids
                    ]
                    break

                if self._stopping:
                    return

            for reminder in sorted(due, key=lambda item: item["deadline"]):
                self._emit_due_event(reminder)

    def _emit_due_event(self, reminder: dict[str, Any]) -> None:
        event = {
            "type": "reminder.due",
            "id": reminder["task_id"],
            "payload": f"Remind the user to {reminder['message']}",
            "due_at": self._format_timestamp(reminder["due_at"]),
        }
        Logger.info(
            f"Reminder due: {reminder['task_id']} ({reminder['message']})"
        )
        try:
            self.emit_event(event)
        except Exception as exc:
            Logger.error(
                f"Could not emit reminder {reminder['task_id']}: {exc}"
            )

    @classmethod
    def _public_reminder(
        cls,
        reminder: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "task_id": reminder["task_id"],
            "message": reminder["message"],
            "due_at": cls._format_timestamp(reminder["due_at"]),
            "remaining_seconds": max(
                0.0,
                round(reminder["deadline"] - time.monotonic(), 3),
            ),
        }

    @staticmethod
    def _format_timestamp(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp).astimezone().isoformat(
            timespec="seconds"
        )
