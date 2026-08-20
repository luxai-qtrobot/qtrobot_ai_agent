from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tool_call_coordinator import ToolCallCoordinator


class _Client:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def send_event(self, event) -> None:
        self.events.append(dict(event))


class _ControlledEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.gates: dict[str, asyncio.Event] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.cancel_count = 0
        self.cancelled_calls: list[str] = []

    async def execute(self, calls):
        call = dict(calls[0])
        call_id = call["id"]
        self.calls.append(call)
        gate = self.gates.setdefault(call_id, asyncio.Event())
        try:
            await gate.wait()
        except asyncio.CancelledError:
            self.cancelled_calls.append(call_id)
            raise
        return [
            self.results.get(
                call_id,
                {"tool_call_id": call_id, "output": f"result {call_id}", "images": []},
            )
        ]

    def cancel_all(self) -> None:
        self.cancel_count += 1


class _ImmediateEngine:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def execute(self, calls):
        self.calls.extend(dict(call) for call in calls)
        return [self.result]


class _ShutdownAwareEngine:
    def __init__(self) -> None:
        self.execute_started = asyncio.Event()
        self.cancel_count = 0
        self.wait_started = asyncio.Event()
        self.cancel_release = asyncio.Event()

    async def execute(self, calls):
        self.execute_started.set()
        await asyncio.Event().wait()

    def cancel_all(self) -> None:
        self.cancel_count += 1

    async def wait_for_cancellations(self) -> None:
        self.wait_started.set()
        await self.cancel_release.wait()


def _response_created(response_id: str, metadata=None) -> dict[str, Any]:
    return {
        "type": "response.created",
        "response": {"id": response_id, "metadata": metadata or {}},
    }


def _output_added(call_id: str, index: int, response_id: str = "response_1"):
    return {
        "type": "response.output_item.added",
        "response_id": response_id,
        "output_index": index,
        "item": {
            "type": "function_call",
            "call_id": call_id,
            "name": "lookup",
            "arguments": "",
        },
    }


def _tool_call(
    call_id: str,
    index: int = 0,
    response_id: str = "response_1",
) -> dict[str, Any]:
    return {
        "type": "response.function_call_arguments.done",
        "response_id": response_id,
        "output_index": index,
        "call_id": call_id,
        "name": "lookup",
        "arguments": f'{{"index":{index}}}',
    }


def _function_call(call_id: str, index: int = 0) -> dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": "lookup",
        "arguments": f'{{"index":{index}}}',
    }


def _response_done(
    response_id: str = "response_1",
    *,
    status: str = "completed",
    output=(),
) -> dict[str, Any]:
    return {
        "type": "response.done",
        "response": {
            "id": response_id,
            "status": status,
            "output": list(output),
        },
    }


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


class ToolCallCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_result_before_origin_response_done(self):
        client = _Client()
        engine = _ImmediateEngine(
            {"tool_call_id": "call_0", "output": "ready", "images": []}
        )
        coordinator = ToolCallCoordinator(client, engine)

        coordinator.handle_event(_response_created("response_1"))
        coordinator.handle_event(_output_added("call_0", 0))
        coordinator.handle_event(_tool_call("call_0", 0))

        await _wait_until(lambda: len(client.events) == 1)
        self.assertEqual(client.events[0]["type"], "conversation.item.create")
        self.assertEqual(client.events[0]["item"]["output"], "ready")

        coordinator.handle_event(
            _response_done(output=[_function_call("call_0", 0)])
        )
        await _wait_until(lambda: len(client.events) == 2)
        self.assertEqual(client.events[1]["type"], "response.create")
        await coordinator.close()

    async def test_executes_immediately_deduplicates_and_commits_in_output_order(self):
        client = _Client()
        engine = _ControlledEngine()
        coordinator = ToolCallCoordinator(client, engine)

        coordinator.handle_event(_response_created("response_1"))
        coordinator.handle_event(_output_added("call_0", 0))
        coordinator.handle_event(_output_added("call_1", 1))
        coordinator.handle_event(_tool_call("call_1", 1))
        coordinator.handle_event(_tool_call("call_1", 1))
        coordinator.handle_event(_tool_call("call_0", 0))
        await _wait_until(lambda: len(engine.calls) == 2)

        engine.gates["call_1"].set()
        coordinator.handle_event(
            _response_done(
                output=[_function_call("call_0", 0), _function_call("call_1", 1)]
            )
        )
        await asyncio.sleep(0)
        self.assertEqual(client.events, [])

        engine.gates["call_0"].set()
        await _wait_until(lambda: len(client.events) == 3)

        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(
            [event["item"]["call_id"] for event in client.events[:2]],
            ["call_0", "call_1"],
        )
        self.assertEqual(
            [event["type"] for event in client.events],
            [
                "conversation.item.create",
                "conversation.item.create",
                "response.create",
            ],
        )
        self.assertEqual(
            sum(event["type"] == "response.create" for event in client.events), 1
        )
        await coordinator.close()

    async def test_cancelled_origin_stops_executor_and_discards_result(self):
        client = _Client()
        engine = _ControlledEngine()
        coordinator = ToolCallCoordinator(client, engine)

        coordinator.handle_event(_response_created("response_1"))
        coordinator.handle_event(_tool_call("call_0"))
        await _wait_until(lambda: len(engine.calls) == 1)
        coordinator.handle_event(_response_done(status="cancelled"))

        await _wait_until(lambda: engine.cancelled_calls == ["call_0"])
        self.assertEqual(engine.cancel_count, 1)
        self.assertEqual(client.events, [])
        await coordinator.close()

    async def test_close_waits_for_executor_cancellation_dispatch(self):
        client = _Client()
        engine = _ShutdownAwareEngine()
        coordinator = ToolCallCoordinator(client, engine)

        coordinator.handle_event(_response_created("response_1"))
        coordinator.handle_event(_tool_call("call_0"))
        await engine.execute_started.wait()

        closing = asyncio.create_task(coordinator.close())
        await engine.wait_started.wait()
        self.assertFalse(closing.done())
        self.assertEqual(engine.cancel_count, 1)

        engine.cancel_release.set()
        await asyncio.wait_for(closing, timeout=1.0)

    async def test_canonical_image_precedes_function_output_and_links_it(self):
        client = _Client()
        engine = _ImmediateEngine(
            {
                "tool_call_id": "call_camera",
                "output": "I captured the drawing.",
                "images": [
                    {"mime_type": "image/jpeg", "data": "YWJj"},
                    {"mime_type": "image/png", "data": "ZGVm"},
                ],
            }
        )
        coordinator = ToolCallCoordinator(client, engine)

        coordinator.handle_event(
            _response_done(output=[_function_call("call_camera")])
        )
        await _wait_until(lambda: len(client.events) == 3)

        image_event, output_event, create_event = client.events
        image_item = image_event["item"]
        self.assertTrue(image_item["id"].startswith("msg_"))
        self.assertEqual(image_item["role"], "user")
        self.assertEqual(len(image_item["content"]), 2)
        self.assertEqual(
            output_event["previous_item_id"], image_item["id"]
        )
        self.assertNotIn("previous_item_id", output_event["item"])
        self.assertEqual(output_event["item"]["call_id"], "call_camera")
        self.assertEqual(output_event["item"]["output"], "I captured the drawing.")
        self.assertEqual(create_event["type"], "response.create")
        await coordinator.close()

    async def test_responses_style_image_uses_short_string_acknowledgement(self):
        client = _Client()
        engine = _ImmediateEngine(
            {
                "tool_call_id": "call_camera",
                "output": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/jpeg;base64,YWJj",
                        "detail": "high",
                    }
                ],
            }
        )
        coordinator = ToolCallCoordinator(client, engine)

        coordinator.handle_event(
            _response_done(output=[_function_call("call_camera")])
        )
        await _wait_until(lambda: len(client.events) == 3)

        self.assertEqual(
            client.events[1]["item"]["output"], "Image captured and attached."
        )
        self.assertEqual(client.events[0]["item"]["content"][0]["detail"], "high")
        await coordinator.close()

    async def test_follow_up_waits_while_a_newer_response_is_active(self):
        client = _Client()
        engine = _ControlledEngine()
        coordinator = ToolCallCoordinator(client, engine)

        coordinator.handle_event(_response_created("response_1"))
        coordinator.handle_event(_tool_call("call_0"))
        await _wait_until(lambda: len(engine.calls) == 1)
        coordinator.handle_event(
            _response_done(output=[_function_call("call_0")])
        )
        coordinator.handle_event(_response_created("response_2"))
        engine.gates["call_0"].set()

        await _wait_until(lambda: len(client.events) == 1)
        self.assertEqual(client.events[0]["type"], "conversation.item.create")
        await asyncio.sleep(0.02)
        self.assertEqual(len(client.events), 1)

        coordinator.handle_event(_response_done("response_2"))
        await _wait_until(lambda: len(client.events) == 2)
        self.assertEqual(client.events[1]["type"], "response.create")
        await coordinator.close()

    async def test_non_collision_follow_up_error_is_fatal(self):
        client = _Client()
        engine = _ImmediateEngine(
            {"tool_call_id": "call_0", "output": "ready", "images": []}
        )
        coordinator = ToolCallCoordinator(client, engine)

        coordinator.handle_event(
            _response_done(output=[_function_call("call_0", 0)])
        )
        await _wait_until(
            lambda: any(event["type"] == "response.create" for event in client.events)
        )
        create = next(
            event for event in client.events if event["type"] == "response.create"
        )
        coordinator.handle_event(
            {
                "type": "error",
                "error": {
                    "event_id": create["event_id"],
                    "type": "invalid_request_error",
                    "message": "bad follow-up",
                },
            }
        )

        with self.assertRaisesRegex(RuntimeError, "bad follow-up"):
            await coordinator.wait_for_failure()
        await coordinator.close()

    async def test_background_event_waits_for_active_response(self):
        client = _Client()
        engine = _ImmediateEngine(
            {"tool_call_id": "unused", "output": "unused", "images": []}
        )
        coordinator = ToolCallCoordinator(client, engine)

        coordinator.handle_event(_response_created("response_1"))
        coordinator.queue_background_event(
            {
                "type": "reminder.due",
                "id": "reminder_123",
                "payload": "Remind the user to take their medicine.",
            }
        )
        await asyncio.sleep(0.02)
        self.assertEqual(client.events, [])

        coordinator.handle_event(_response_done("response_1"))
        await _wait_until(lambda: len(client.events) == 2)

        item_event, create_event = client.events
        self.assertEqual(item_event["type"], "conversation.item.create")
        self.assertEqual(item_event["item"]["role"], "user")
        text = item_event["item"]["content"][0]["text"]
        self.assertIn('type="reminder.due"', text)
        self.assertIn('id="reminder_123"', text)
        self.assertIn("take their medicine", text)
        self.assertEqual(create_event["type"], "response.create")
        await coordinator.close()

    async def test_background_events_share_one_follow_up_and_escape_payload(self):
        client = _Client()
        engine = _ImmediateEngine(
            {"tool_call_id": "unused", "output": "unused", "images": []}
        )
        coordinator = ToolCallCoordinator(client, engine)

        coordinator.queue_background_event(
            {
                "type": "reminder.due",
                "id": "reminder_1",
                "payload": 'Say "hello"\nnow.',
            }
        )
        coordinator.queue_background_event(
            {
                "type": "reminder.due",
                "id": "reminder_2",
                "payload": "Second reminder.",
            }
        )

        await _wait_until(lambda: len(client.events) == 3)
        self.assertEqual(
            [event["type"] for event in client.events],
            [
                "conversation.item.create",
                "conversation.item.create",
                "response.create",
            ],
        )
        first_text = client.events[0]["item"]["content"][0]["text"]
        self.assertIn(r'payload="Say \"hello\"\nnow."', first_text)
        await coordinator.close()

    async def test_background_event_waits_for_pending_tool_output(self):
        client = _Client()
        engine = _ControlledEngine()
        coordinator = ToolCallCoordinator(client, engine)

        coordinator.handle_event(_response_created("response_1"))
        coordinator.handle_event(_tool_call("call_0"))
        await _wait_until(lambda: len(engine.calls) == 1)
        coordinator.handle_event(
            _response_done(output=[_function_call("call_0")])
        )
        coordinator.queue_background_event(
            {
                "type": "reminder.due",
                "id": "reminder_123",
                "payload": "Remind the user to stretch.",
            }
        )
        await asyncio.sleep(0.02)
        self.assertEqual(client.events, [])

        engine.gates["call_0"].set()
        await _wait_until(lambda: len(client.events) == 3)
        self.assertEqual(client.events[0]["item"]["type"], "function_call_output")
        self.assertEqual(client.events[1]["item"]["role"], "user")
        self.assertEqual(client.events[2]["type"], "response.create")
        await coordinator.close()

    async def test_background_event_does_not_interrupt_user_speech(self):
        client = _Client()
        engine = _ImmediateEngine(
            {"tool_call_id": "unused", "output": "unused", "images": []}
        )
        coordinator = ToolCallCoordinator(client, engine)

        coordinator.handle_event({"type": "input_audio_buffer.speech_started"})
        coordinator.queue_background_event(
            {
                "type": "reminder.due",
                "id": "reminder_123",
                "payload": "Remind the user to stretch.",
            }
        )
        coordinator.handle_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "Tell me a story.",
            }
        )
        await asyncio.sleep(0.02)
        self.assertEqual(client.events, [])

        coordinator.handle_event(_response_created("response_1"))
        coordinator.handle_event(_response_done("response_1"))
        await _wait_until(lambda: len(client.events) == 2)
        self.assertEqual(client.events[0]["type"], "conversation.item.create")
        self.assertEqual(client.events[1]["type"], "response.create")
        await coordinator.close()


if __name__ == "__main__":
    unittest.main()
