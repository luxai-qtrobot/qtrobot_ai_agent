"""Construction and lifecycle owner for isolated agents."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from typing import Any

from luxai.magpie.utils import Logger

from tool.local_tool_server import LocalToolServer
from tool.providers.user_tools import UserTools
from tool.tool_base import ToolBase

from .agent_base import AGENT_TOOLS_ENDPOINT
from .object_search import ObjectSearchAgent, ObjectSearchTools
from .web_search import WebSearchAgent, WebSearchTools


class AgentRegistry:
    """Own private agent tools and expose only each agent's public tool."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        owner_loop: asyncio.AbstractEventLoop,
        event_sink: Callable[[dict[str, Any]], None],
        user_tools: UserTools,
        web_search_enabled: bool = False,
        api_key: str | None = None,
        endpoint: str = AGENT_TOOLS_ENDPOINT,
        completion_extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self._agents: list[Any] = []
        self._agent_tool_server: LocalToolServer | None = None
        self._lifecycle_lock = threading.Lock()
        self._server_stopped = False

        object_tools = ObjectSearchTools(user_tools)
        private_tools: list[ToolBase] = [object_tools]
        self._agents.append(
            ObjectSearchAgent(
                client,
                model,
                owner_loop=owner_loop,
                tools=object_tools,
                event_sink=event_sink,
                endpoint=endpoint,
                completion_extra_body=completion_extra_body,
            )
        )
        self.web_search_available = False

        if web_search_enabled:
            try:
                web_tools = WebSearchTools(api_key=api_key)
            except ValueError as exc:
                Logger.warning(f"Web search disabled: {exc}")
            else:
                private_tools.append(web_tools)
                self._agents.append(
                    WebSearchAgent(
                        client,
                        model,
                        owner_loop=owner_loop,
                        event_sink=event_sink,
                        endpoint=endpoint,
                        completion_extra_body=completion_extra_body,
                    )
                )
                self.web_search_available = True

        self._agent_tool_server = LocalToolServer(
            private_tools,
            endpoint=endpoint,
        )

    def as_tools(self) -> list[ToolBase]:
        """Return public agent-as-tool providers for the application's server."""
        return list(self._agents)

    async def close(self) -> None:
        """Close all agents, then stop their private MCP server."""
        await asyncio.gather(
            *(agent.close() for agent in self._agents),
            return_exceptions=True,
        )
        self._stop_server()

    def cleanup(self) -> None:
        """Idempotent synchronous fallback for partial startup and shutdown."""
        for agent in self._agents:
            agent.cleanup()
        self._stop_server()

    def _stop_server(self) -> None:
        with self._lifecycle_lock:
            if self._server_stopped:
                return
            self._server_stopped = True
            server = self._agent_tool_server
            self._agent_tool_server = None
        if server is not None:
            server.terminate(timeout=1.0)
