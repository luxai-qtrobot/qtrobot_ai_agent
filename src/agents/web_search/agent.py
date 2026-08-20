"""A background web-search agent exposed as one public MCP tool."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from luxai.magpie.schema import McpSchema
from luxai.magpie.utils import Logger

from ..agent_base import AGENT_TOOLS_ENDPOINT, AgentBase


INSTRUCTIONS_PATH = Path(__file__).with_name("instructions.txt")
WEB_SEARCH_WHITELIST = {"search_web_api": None, "fetch_url": None}
MAX_QUERY_CHARACTERS = 2_000


class WebSearchAgent(AgentBase):
    """Schedule independent searches and publish their eventual result."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        owner_loop: asyncio.AbstractEventLoop,
        event_sink,
        endpoint: str = AGENT_TOOLS_ENDPOINT,
        completion_extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            client,
            model,
            WEB_SEARCH_WHITELIST,
            INSTRUCTIONS_PATH.read_text(encoding="utf-8"),
            endpoint=endpoint,
            completion_extra_body=completion_extra_body,
            event_sink=event_sink,
        )
        self._owner_loop = owner_loop
        self._jobs: set[concurrent.futures.Future[None]] = set()
        self._jobs_lock = threading.Lock()
        self._closed = False

    def register(self, schema: McpSchema) -> None:
        schema.method()(self.search_web)

    def search_web(self, query: str) -> dict[str, str]:
        """Start a web search; its cited answer will arrive as a background event."""
        query = query.strip()
        if not query:
            raise ValueError("Search query cannot be empty")
        if len(query) > MAX_QUERY_CHARACTERS:
            raise ValueError(
                f"Search query cannot exceed {MAX_QUERY_CHARACTERS} characters"
            )

        task_id = f"search_{uuid.uuid4().hex[:12]}"
        with self._jobs_lock:
            if self._closed:
                raise RuntimeError("Web search agent is shutting down")
            future = asyncio.run_coroutine_threadsafe(
                self._run_and_emit(task_id, query),
                self._owner_loop,
            )
            self._jobs.add(future)
        future.add_done_callback(self._forget_job)

        Logger.info(f"Web search started: {task_id} ({query})")
        return {
            "status": "started",
            "task_id": task_id,
            "summary": f"Searching the web for: {query}",
        }

    async def _run_and_emit(self, task_id: str, query: str) -> None:
        try:
            answer = await self.run(query)
        except asyncio.CancelledError:
            Logger.info(f"Web search cancelled: {task_id}")
            raise
        except Exception as exc:
            Logger.error(f"Web search failed: {task_id}: {exc}")
            self._emit_event(
                {
                    "type": "search.failed",
                    "id": task_id,
                    "payload": (
                        "The web search could not be completed. Tell the user "
                        f"the search for {query!r} failed and they can try again."
                    ),
                }
            )
        else:
            Logger.info(f"Web search completed: {task_id}")
            self._emit_event(
                {"type": "search.done", "id": task_id, "payload": answer}
            )

    def _emit_event(self, event: dict[str, str]) -> None:
        with self._jobs_lock:
            if self._closed:
                return
        try:
            self.emit_event(event)
        except Exception as exc:
            Logger.error(
                f"Could not emit web search event {event['id']}: {exc}"
            )

    def _forget_job(
        self,
        future: concurrent.futures.Future[None],
    ) -> None:
        with self._jobs_lock:
            self._jobs.discard(future)

    async def close(self) -> None:
        """Cancel and await every active background search."""
        with self._jobs_lock:
            self._closed = True
            jobs = tuple(self._jobs)
        for job in jobs:
            job.cancel()
        if jobs:
            await asyncio.gather(
                *(asyncio.wrap_future(job) for job in jobs),
                return_exceptions=True,
            )

    def cleanup(self) -> None:
        """Idempotent synchronous cancellation for ToolBase lifecycle use."""
        with self._jobs_lock:
            self._closed = True
            jobs = tuple(self._jobs)
        for job in jobs:
            job.cancel()
