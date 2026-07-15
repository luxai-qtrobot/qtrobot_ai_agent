"""
agent.py - WebSearchAgent: the web-search sub-agent, exposed to the main conversation's
LLM as a single search_web tool ("agent as tool"). Its internal tools (WebSearchTools,
tools.py) live on the shared agent-tools server built by the caller (see
AgentBase/AGENT_TOOLS_ENDPOINT), not here.
"""

import asyncio
from pathlib import Path

from luxai.magpie.schema import McpSchema

from agents.agent_base import AGENT_TOOLS_ENDPOINT, AgentBase

INSTRUCTIONS_PATH = Path(__file__).parent / "instructions.txt"
WEB_SEARCH_WHITELIST = {"search_web_api", "fetch_url"}


class WebSearchAgent(AgentBase):

    def __init__(self, client, model: str, endpoint: str = AGENT_TOOLS_ENDPOINT):
        super().__init__(
            client=client, model=model, endpoint=endpoint,
            whitelist=WEB_SEARCH_WHITELIST,
            system_prompt=INSTRUCTIONS_PATH.read_text(encoding="utf-8"),
        )

    def register(self, schema: McpSchema) -> None:
        schema.method()(self.search_web)

    def search_web(self, query: str) -> str:
        """Search the web for information and return a synthesized, cited answer."""
        return asyncio.run(self.run(query))
