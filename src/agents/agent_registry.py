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

from tool.local_tool_server import LocalToolServer

from .agent_base import AGENT_TOOLS_ENDPOINT
from .web_search.agent import WebSearchAgent
from .web_search.tools import WebSearchTools


class AgentRegistry:

    def __init__(self, client, model: str):
        self._agent_tool_server = LocalToolServer([WebSearchTools()], endpoint=AGENT_TOOLS_ENDPOINT)
        self._agents = [
            WebSearchAgent(client=client, model=model, endpoint=AGENT_TOOLS_ENDPOINT),
            # new agent: add its tools.py provider above, its Agent(...) here.
        ]

    def as_tools(self) -> list:
        """Every agent, as a ToolBase instance ready to register on the main LLM's
        tool server - the only thing about an agent the caller needs to know."""
        return self._agents

    def cleanup(self) -> None:
        self._agent_tool_server.terminate(timeout=1.0)
