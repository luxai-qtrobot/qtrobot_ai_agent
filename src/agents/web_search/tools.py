"""
tools.py - WebSearchTools: the web-search sub-agent's own internal tools, backed by a
real search provider (Tavily) and real page fetching (trafilatura). Registered onto
the shared agent-tools server (see cli_chat_example.py) and reached only through this
agent's own scoped ToolEngine (via AgentBase's whitelist) - never exposed to the main
conversation's ToolEngine.
"""

import os

import requests
import trafilatura
from tavily import TavilyClient

from luxai.magpie.schema import McpSchema

from tool.tool_base import ToolBase

MAX_RESULTS = 5
FETCH_TIMEOUT_SECONDS = 10


class WebSearchTools(ToolBase):

    def __init__(self, api_key: str = None):
        """api_key: Tavily API key. Falls back to the TAVILY_API_KEY environment
        variable if not given - never hardcode a real key in source."""
        super().__init__()
        api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "WebSearchTools needs a Tavily API key - pass api_key= or set "
                "the TAVILY_API_KEY environment variable."
            )
        self._client = TavilyClient(api_key=api_key)

    def register(self, schema: McpSchema) -> None:
        schema.method()(self.search_web_api)
        schema.method()(self.fetch_url)

    def search_web_api(self, query: str) -> list[dict]:
        """Run a web search and return raw results (title, url, snippet)."""
        response = self._client.search(query, max_results=MAX_RESULTS)
        return [
            {"title": r["title"], "url": r["url"], "snippet": r["content"]}
            for r in response.get("results", [])
        ]

    def fetch_url(self, url: str) -> str:
        """Fetch a URL and return its extracted main text content.

        Uses requests (not trafilatura.fetch_url) for the actual network call
        specifically for its timeout= - trafilatura.fetch_url has no reliable way to
        bound how long it blocks, and a hung fetch on the tool server's shared,
        fixed-size worker pool (see ServerNode/LocalToolServer) was observed starving
        every other tool call queued behind it for minutes. trafilatura.extract()
        itself only parses already-downloaded HTML, so splitting fetch from extract
        loses nothing."""
        try:
            response = requests.get(url, timeout=FETCH_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as e:
            return f"Could not fetch {url}: {e}"
        text = trafilatura.extract(response.text)
        return text or f"No readable content found at {url}."
