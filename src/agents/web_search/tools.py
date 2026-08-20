"""Private Tavily and page extraction tools for the web-search agent."""

from __future__ import annotations

import os
from urllib.parse import urlparse

import requests
import trafilatura
from luxai.magpie.schema import McpSchema
from tavily import TavilyClient

from tool.tool_base import ToolBase


MAX_RESULTS = 5
SEARCH_TIMEOUT_SECONDS = 15
FETCH_TIMEOUT_SECONDS = 10
MAX_FETCH_CHARACTERS = 20_000


class WebSearchTools(ToolBase):
    """Network tools exposed only to :class:`WebSearchAgent`."""

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "WebSearchTools needs a Tavily API key; pass api_key or set "
                "TAVILY_API_KEY"
            )
        self._client = TavilyClient(api_key=api_key)

    def register(self, schema: McpSchema) -> None:
        schema.method()(self.search_web_api)
        schema.method()(self.fetch_url)

    def search_web_api(self, query: str) -> list[dict[str, str]]:
        """Search the web and return up to five titles, URLs, and snippets."""
        response = self._client.search(
            query,
            max_results=MAX_RESULTS,
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
        return [
            {
                "title": str(result.get("title") or ""),
                "url": str(result.get("url") or ""),
                "snippet": str(result.get("content") or ""),
            }
            for result in response.get("results", [])[:MAX_RESULTS]
        ]

    def fetch_url(self, url: str) -> str:
        """Fetch an HTTP(S) page and return bounded extracted main text."""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Could not fetch URL: only valid HTTP and HTTPS URLs are allowed."
        try:
            response = requests.get(url, timeout=FETCH_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            return f"Could not fetch {url}: {exc}"

        text = trafilatura.extract(response.text)
        if not text:
            return f"No readable content found at {url}."
        if len(text) > MAX_FETCH_CHARACTERS:
            return text[:MAX_FETCH_CHARACTERS] + "\n[content truncated]"
        return text
