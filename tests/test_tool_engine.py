from __future__ import annotations

import base64
import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tool.tool_engine import CANCELLED, CANCEL_ALL_TOOL_NAME, ToolEngine
from tool.tool_output import as_function_call_output, as_input_image_item


JPEG_BASE64 = base64.b64encode(b"\xff\xd8test-jpeg\xff\xd9").decode("ascii")


class _Client:
    async def list_tools(self):
        return [
            SimpleNamespace(
                name="get_datetime",
                description="Get local time.",
                inputSchema={"type": "object", "properties": {}},
            ),
            SimpleNamespace(
                name="hidden_tool",
                description="Hidden.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    async def call_tool(self, name, arguments):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="now")],
            structured_content=None,
        )


class ToolEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_whitelist_and_execution(self) -> None:
        engine = ToolEngine(
            {"local": _Client()},
            whitelists={"local": {"get_datetime"}},
        )
        await engine.discover()

        self.assertEqual(
            [schema["name"] for schema in engine.schemas()],
            ["get_datetime", CANCEL_ALL_TOOL_NAME],
        )
        result = (
            await engine.execute(
                [{"call_id": "call-1", "name": "get_datetime", "arguments": "{}"}]
            )
        )[0]
        self.assertEqual(result["output"], "now")
        self.assertEqual(result["images"], [])

    async def test_image_is_separate_from_string_function_output(self) -> None:
        engine = ToolEngine({})
        result = engine._normalize_result(
            "call-image",
            SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text=(
                            '{"mimeType":"image/jpeg","data":"'
                            + JPEG_BASE64
                            + '"}'
                        ),
                    )
                ],
                structured_content=None,
            ),
        )

        self.assertEqual(result["output"], "Image captured.")
        self.assertEqual(len(result["images"]), 1)
        function_item = as_function_call_output(result)
        image_item = as_input_image_item(result["images"][0])
        self.assertIsInstance(function_item["output"], str)
        self.assertEqual(function_item["call_id"], "call-image")
        self.assertEqual(image_item["content"][0]["type"], "input_image")
        self.assertTrue(
            image_item["content"][0]["image_url"].startswith(
                "data:image/jpeg;base64,"
            )
        )

    async def test_tracked_cancellation_calls_paired_cancel_tool(self) -> None:
        class BlockingClient:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.release = asyncio.Event()
                self.cancel_calls = 0

            async def list_tools(self):
                return [
                    SimpleNamespace(
                        name="gesture_file_play",
                        description="Play a gesture.",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    SimpleNamespace(
                        name="gesture_cancel",
                        description="Cancel a gesture.",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                ]

            async def call_tool(self, name, arguments):
                if name == "gesture_cancel":
                    self.cancel_calls += 1
                    self.release.set()
                else:
                    self.started.set()
                    await self.release.wait()
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="done")],
                    structured_content=None,
                )

        client = BlockingClient()
        engine = ToolEngine(
            {"robot": client},
            whitelists={
                "robot": {"gesture_file_play": "gesture_cancel"},
            },
        )
        await engine.discover()
        execution = asyncio.create_task(
            engine.execute_tracked(
                [
                    {
                        "call_id": "gesture-1",
                        "name": "gesture_file_play",
                        "arguments": "{}",
                    }
                ]
            )
        )
        await client.started.wait()

        cancelled_ids = engine.cancel_all()
        result = await asyncio.wait_for(execution, timeout=1.0)

        self.assertEqual(cancelled_ids, {"gesture-1"})
        self.assertIs(result, CANCELLED)
        self.assertEqual(client.cancel_calls, 1)

    async def test_cancel_control_tool_does_not_cancel_itself(self) -> None:
        engine = ToolEngine({})
        await engine.discover()

        result = await engine.execute_tracked(
            [
                {
                    "call_id": "cancel-1",
                    "name": CANCEL_ALL_TOOL_NAME,
                    "arguments": "{}",
                }
            ]
        )

        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["tool_call_id"], "cancel-1")
        self.assertIn("No actions", result[0]["output"])

    async def test_wait_for_cancellations_drains_paired_cancel_call(self) -> None:
        class DelayedCancelClient:
            def __init__(self) -> None:
                self.action_started = asyncio.Event()
                self.action_release = asyncio.Event()
                self.cancel_started = asyncio.Event()
                self.cancel_release = asyncio.Event()

            async def list_tools(self):
                return [
                    SimpleNamespace(
                        name="gesture_file_play",
                        description="Play a gesture.",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    SimpleNamespace(
                        name="gesture_cancel",
                        description="Cancel a gesture.",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                ]

            async def call_tool(self, name, arguments):
                if name == "gesture_cancel":
                    self.cancel_started.set()
                    await self.cancel_release.wait()
                    self.action_release.set()
                else:
                    self.action_started.set()
                    await self.action_release.wait()
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="done")],
                    structured_content=None,
                )

        client = DelayedCancelClient()
        engine = ToolEngine(
            {"robot": client},
            whitelists={
                "robot": {"gesture_file_play": "gesture_cancel"},
            },
        )
        await engine.discover()
        execution = asyncio.create_task(
            engine.execute_tracked(
                [
                    {
                        "call_id": "gesture-wait",
                        "name": "gesture_file_play",
                        "arguments": "{}",
                    }
                ]
            )
        )
        await client.action_started.wait()
        engine.cancel_all()
        await client.cancel_started.wait()

        drain = asyncio.create_task(engine.wait_for_cancellations())
        await asyncio.sleep(0)
        self.assertFalse(drain.done())

        client.cancel_release.set()
        await asyncio.wait_for(drain, timeout=1.0)
        self.assertIs(await asyncio.wait_for(execution, timeout=1.0), CANCELLED)


if __name__ == "__main__":
    unittest.main()
