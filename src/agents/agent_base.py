"""Reusable base for stateless, tool-using background agents."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Mapping
from typing import Any

from fastmcp import Client
from luxai.magpie.adapters.mcp import McpTransport
from luxai.magpie.transport import ZMQRpcRequester

from tool.tool_base import ToolBase
from tool.tool_engine import CANCEL_ALL_TOOL_NAME, ToolEngine


AGENT_TOOLS_ENDPOINT = "inproc://qtrobot-s2s-agent-tools"
DEFAULT_COMPLETION_EXTRA_BODY: dict[str, Any] = {
    "chat_template_kwargs": {"enable_thinking": False}
}


class _AgentToolsRequester(ZMQRpcRequester):
    """Close control sockets created by MAGPIE's MCP worker threads.

    The generic requester stores those sockets in thread-local state.  Agent
    runs create and close requesters frequently, so retaining each worker's
    socket until process exit would leak resources.  At close time all calls
    have completed and it is safe to close the tracked sockets together.
    """

    def __init__(self, endpoint: str) -> None:
        self._agent_ctrl_sockets: set[Any] = set()
        self._agent_ctrl_lock = threading.Lock()
        super().__init__(endpoint)

    def _get_ctrl_push(self) -> Any:
        socket = super()._get_ctrl_push()
        with self._agent_ctrl_lock:
            self._agent_ctrl_sockets.add(socket)
        return socket

    def _transport_close(self) -> None:
        super()._transport_close()
        with self._agent_ctrl_lock:
            sockets = tuple(self._agent_ctrl_sockets)
            self._agent_ctrl_sockets.clear()
        for socket in sockets:
            if not socket.closed:
                socket.close(linger=0)


class AgentBase(ToolBase):
    """Run one isolated Chat Completions tool loop per query.

    Internal tools remain behind a private in-process MCP endpoint.  Each run
    creates its own client and message list, so simultaneous agent jobs do not
    share chat state.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        whitelist: Mapping[str, str | None],
        system_prompt: str,
        *,
        endpoint: str = AGENT_TOOLS_ENDPOINT,
        max_rounds: int = 8,
        max_tokens: int = 800,
        timeout: float = 60.0,
        completion_extra_body: Mapping[str, Any] | None = None,
        event_sink=None,
    ) -> None:
        super().__init__(event_sink)
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least one")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least one")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        self.client = client
        self.model = model
        self.whitelist = dict(whitelist)
        self.system_prompt = system_prompt
        self.endpoint = endpoint
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.completion_extra_body = dict(
            DEFAULT_COMPLETION_EXTRA_BODY
            if completion_extra_body is None
            else completion_extra_body
        )

    async def run(self, query: str) -> str:
        """Run a bounded, stateless agent loop and return its final answer."""
        requester = _AgentToolsRequester(self.endpoint)
        try:
            async with Client(McpTransport(requester)) as mcp_client:
                tool_engine = ToolEngine(
                    {"agent": mcp_client},
                    whitelists={"agent": self.whitelist},
                )
                await tool_engine.discover()
                chat_tools = self._chat_schemas(tool_engine.schemas())
                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ]

                # ``max_rounds`` bounds tool-execution rounds. Allow one final
                # completion after the last permitted tool batch so a valid
                # answer is not discarded at the boundary.
                for round_index in range(self.max_rounds + 1):
                    completion = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            tools=chat_tools,
                            tool_choice="auto",
                            parallel_tool_calls=True,
                            max_tokens=self.max_tokens,
                            extra_body=self.completion_extra_body,
                        ),
                        timeout=self.timeout,
                    )
                    message = self._completion_message(completion)
                    tool_calls = self._tool_calls(message)
                    content = self._field(message, "content")

                    if not tool_calls:
                        answer = str(content or "").strip()
                        if not answer:
                            raise RuntimeError(
                                "Background agent returned an empty answer"
                            )
                        return answer

                    if round_index == self.max_rounds:
                        break

                    messages.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": [call["chat"] for call in tool_calls],
                        }
                    )
                    # ToolEngine.execute uses asyncio.gather, so independent
                    # calls emitted in one model turn run in parallel.
                    results = await tool_engine.execute(
                        [call["engine"] for call in tool_calls]
                    )
                    messages.extend(
                        {
                            "role": "tool",
                            "tool_call_id": result["tool_call_id"],
                            "content": result["output"],
                        }
                        for result in results
                    )

            raise RuntimeError(
                f"Background agent exceeded its {self.max_rounds}-round limit"
            )
        finally:
            requester.close()

    @staticmethod
    def _chat_schemas(
        schemas: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert S2S's flat function schema to Chat Completions format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema.get("description", schema["name"]),
                    "parameters": schema.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
            for schema in schemas
            if schema.get("name") != CANCEL_ALL_TOOL_NAME
        ]

    @classmethod
    def _completion_message(cls, completion: Any) -> Any:
        choices = cls._field(completion, "choices") or []
        if not choices:
            raise RuntimeError("Background agent completion had no choices")
        message = cls._field(choices[0], "message")
        if message is None:
            raise RuntimeError("Background agent completion had no message")
        return message

    @classmethod
    def _tool_calls(cls, message: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in cls._field(message, "tool_calls") or []:
            call_id = str(cls._field(item, "id") or "")
            function = cls._field(item, "function")
            name = str(cls._field(function, "name") or "")
            arguments = cls._field(function, "arguments") or "{}"
            if not call_id or not name:
                raise RuntimeError("Background agent returned a malformed tool call")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, separators=(",", ":"))
            normalized.append(
                {
                    "chat": {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    },
                    "engine": {
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )
        return normalized

    @staticmethod
    def _field(value: Any, name: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)
