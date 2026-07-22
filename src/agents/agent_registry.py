"""
agent_registry.py - AgentRegistry: the one place that knows how every agent is built -
which internal tools it needs, which server they live on, and which client/model
backs it. Keeps that wiring out of the caller (cli_chat_example.py), which only ever
sees as_tools() (ToolBase instances to expose on the main LLM's tool server) and
cleanup().

client/model passed into __init__ are just the default backend for agents that don't
need something different - nothing stops constructing an individual agent below with
its own client/model (a different llama-server, a different routed model name, even a
different hosted API) if that agent specifically needs one.
"""

from luxai.magpie.utils import Logger

from tool.local_tool_server import LocalToolServer

from .agent_base import AGENT_TOOLS_ENDPOINT
from .web_search.agent import WebSearchAgent
from .web_search.tools import WebSearchTools


class AgentRegistry:

    def __init__(self, client, model: str):
        agent_tools = []
        self._agents = []

        try:
            agent_tools.append(WebSearchTools())
        except ValueError as e:
            Logger.warning(f"Web search disabled: {e}")
        else:
            self._agents.append(WebSearchAgent(client=client, model=model, endpoint=AGENT_TOOLS_ENDPOINT))

        # new agent: try/except-construct its tools.py provider above (same pattern
        # as WebSearchTools), append its Agent(...) here only if that succeeded.

        self._agent_tool_server = LocalToolServer(agent_tools, endpoint=AGENT_TOOLS_ENDPOINT)

    def as_tools(self) -> list:
        """Every agent, as a ToolBase instance ready to register on the main LLM's
        tool server - the only thing about an agent the caller needs to know."""
        return self._agents

    def cleanup(self) -> None:
        self._agent_tool_server.terminate(timeout=1.0)
