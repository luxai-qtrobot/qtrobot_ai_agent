from __future__ import annotations

import asyncio
import os
import sys
import threading
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import AgentBase, AgentRegistry, WebSearchAgent
from agents.web_search.tools import (
    FETCH_TIMEOUT_SECONDS,
    MAX_FETCH_CHARACTERS,
    MAX_RESULTS,
    SEARCH_TIMEOUT_SECONDS,
    WebSearchTools,
)
from luxai.magpie.schema import McpSchema
from tool import LocalToolServer
from tool.tool_base import ToolBase


def _completion(*, content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls or [],
                )
            )
        ]
    )


def _tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeAsyncOpenAI:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


class _ParallelTools(ToolBase):
    def __init__(self) -> None:
        super().__init__()
        self._barrier = threading.Barrier(2, timeout=2.0)
        self._lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0

    def register(self, schema: McpSchema) -> None:
        schema.method()(self.first)
        schema.method()(self.second)

    def first(self, value: int) -> str:
        """Return the first test value."""
        return self._run("first", value)

    def second(self, value: int) -> str:
        """Return the second test value."""
        return self._run("second", value)

    def _run(self, name: str, value: int) -> str:
        with self._lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            self._barrier.wait()
            return f"{name}:{value}"
        finally:
            with self._lock:
                self.active -= 1


class _TestAgent(AgentBase):
    def register(self, schema: McpSchema) -> None:
        return None


class AgentBaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_schema_parallel_tool_loop_and_bounded_completion(self):
        endpoint = f"inproc://test-agent-{uuid.uuid4().hex}"
        tools = _ParallelTools()
        server = LocalToolServer([tools], endpoint=endpoint)
        client = _FakeAsyncOpenAI(
            [
                _completion(
                    tool_calls=[
                        _tool_call("call-1", "first", '{"value":1}'),
                        _tool_call("call-2", "second", '{"value":2}'),
                    ]
                ),
                _completion(content="Answer with https://example.test/source"),
            ]
        )
        agent = _TestAgent(
            client,
            "test-model",
            {"first": None, "second": None},
            "Test instructions",
            endpoint=endpoint,
            max_rounds=1,
            max_tokens=321,
        )
        try:
            answer = await agent.run("test query")
        finally:
            server.terminate(timeout=1.0)

        self.assertEqual(answer, "Answer with https://example.test/source")
        self.assertEqual(tools.maximum_active, 2)
        first_call, second_call = client.completions.calls
        self.assertEqual(first_call["max_tokens"], 321)
        self.assertTrue(first_call["parallel_tool_calls"])
        self.assertEqual(
            first_call["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertEqual(
            {schema["function"]["name"] for schema in first_call["tools"]},
            {"first", "second"},
        )
        self.assertTrue(
            all(set(schema) == {"type", "function"} for schema in first_call["tools"])
        )
        tool_messages = [
            message
            for message in second_call["messages"]
            if message["role"] == "tool"
        ]
        self.assertEqual(
            {message["tool_call_id"] for message in tool_messages},
            {"call-1", "call-2"},
        )


class _ControlledWebSearchAgent(WebSearchAgent):
    def __init__(self, event_sink, owner_loop):
        super().__init__(
            client=None,
            model="test-model",
            owner_loop=owner_loop,
            event_sink=event_sink,
        )
        self.release = asyncio.Event()
        self.two_started = asyncio.Event()
        self.one_started = asyncio.Event()
        self.running = 0
        self.cancelled = 0

    async def run(self, query: str) -> str:
        if query == "failure":
            raise RuntimeError("secret backend detail")
        if query.startswith("hold"):
            self.running += 1
            self.one_started.set()
            if self.running >= 2:
                self.two_started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        await asyncio.sleep(0)
        return f"Result for {query}: https://example.test/{query}"


class WebSearchAgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.events = []
        self.event_ready = asyncio.Event()

        def sink(event):
            self.events.append(event)
            self.event_ready.set()

        self.agent = _ControlledWebSearchAgent(
            sink,
            asyncio.get_running_loop(),
        )

    async def asyncTearDown(self) -> None:
        await self.agent.close()

    async def test_returns_started_immediately_then_emits_completion(self):
        result = self.agent.search_web("current robotics news")

        self.assertEqual(result["status"], "started")
        self.assertTrue(result["task_id"].startswith("search_"))
        self.assertEqual(
            result["summary"],
            "Searching the web for: current robotics news",
        )
        self.assertEqual(self.events, [])

        await asyncio.wait_for(self.event_ready.wait(), timeout=1.0)
        self.assertEqual(
            self.events,
            [
                {
                    "type": "search.done",
                    "id": result["task_id"],
                    "payload": (
                        "Result for current robotics news: "
                        "https://example.test/current robotics news"
                    ),
                }
            ],
        )

    async def test_failure_event_hides_backend_exception(self):
        result = self.agent.search_web("failure")

        await asyncio.wait_for(self.event_ready.wait(), timeout=1.0)
        event = self.events[0]
        self.assertEqual(event["type"], "search.failed")
        self.assertEqual(event["id"], result["task_id"])
        self.assertIn("could not be completed", event["payload"])
        self.assertNotIn("secret backend detail", event["payload"])

    async def test_parallel_searches_and_shutdown_cancellation(self):
        first = self.agent.search_web("hold-one")
        second = self.agent.search_web("hold-two")
        await asyncio.wait_for(self.agent.two_started.wait(), timeout=1.0)
        self.assertEqual(self.agent.running, 2)

        await self.agent.close()
        await self.agent.close()

        self.assertEqual(self.agent.cancelled, 2)
        self.assertEqual(self.events, [])
        self.assertNotEqual(first["task_id"], second["task_id"])
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            self.agent.search_web("too late")

    def test_registers_only_public_search_tool(self):
        schema = McpSchema(name="web-search-test")
        self.agent.register(schema)
        names = {item["name"] for item in schema._mcp_tools_list()["tools"]}
        self.assertEqual(names, {"search_web"})


class WebSearchToolsTests(unittest.TestCase):
    def test_tavily_is_limited_and_has_timeout(self):
        tools = WebSearchTools(api_key="fake-key")

        class FakeTavily:
            def __init__(self):
                self.kwargs = None

            def search(self, query, **kwargs):
                self.kwargs = kwargs
                return {
                    "results": [
                        {"title": str(i), "url": f"https://e/{i}", "content": "s"}
                        for i in range(8)
                    ]
                }

        tools._client = FakeTavily()
        results = tools.search_web_api("query")

        self.assertEqual(len(results), MAX_RESULTS)
        self.assertEqual(
            tools._client.kwargs,
            {"max_results": MAX_RESULTS, "timeout": SEARCH_TIMEOUT_SECONDS},
        )

    def test_fetch_rejects_schemes_and_caps_extracted_text(self):
        tools = WebSearchTools(api_key="fake-key")
        self.assertIn("only valid HTTP", tools.fetch_url("file:///etc/passwd"))

        response = SimpleNamespace(
            text="html",
            raise_for_status=lambda: None,
        )
        with patch("agents.web_search.tools.requests.get", return_value=response) as get:
            with patch(
                "agents.web_search.tools.trafilatura.extract",
                return_value="x" * (MAX_FETCH_CHARACTERS + 50),
            ):
                result = tools.fetch_url("https://example.test/page")

        get.assert_called_once_with(
            "https://example.test/page",
            timeout=FETCH_TIMEOUT_SECONDS,
        )
        self.assertEqual(result[:MAX_FETCH_CHARACTERS], "x" * MAX_FETCH_CHARACTERS)
        self.assertTrue(result.endswith("[content truncated]"))


class AgentRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_enabled_registry_exposes_agent_and_closes_idempotently(self):
        registry = AgentRegistry(
            client=None,
            model="test-model",
            owner_loop=asyncio.get_running_loop(),
            event_sink=lambda _event: None,
            api_key="fake-key",
            endpoint=f"inproc://test-registry-{uuid.uuid4().hex}",
        )

        self.assertEqual(len(registry.as_tools()), 1)
        self.assertIsInstance(registry.as_tools()[0], WebSearchAgent)
        await registry.close()
        await registry.close()
        registry.cleanup()

    async def test_missing_tavily_key_disables_agent_cleanly(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TAVILY_API_KEY", None)
            registry = AgentRegistry(
                client=None,
                model="test-model",
                owner_loop=asyncio.get_running_loop(),
                event_sink=lambda _event: None,
            )

        self.assertEqual(registry.as_tools(), [])
        await registry.close()
        await registry.close()
        registry.cleanup()


if __name__ == "__main__":
    unittest.main()
