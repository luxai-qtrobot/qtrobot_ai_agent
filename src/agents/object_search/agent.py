"""Object-search agent exposed to the main conversation as one tool."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from luxai.magpie.schema import McpSchema

from ..agent_base import AGENT_TOOLS_ENDPOINT, AgentBase
from .tools import ObjectSearchTools, SEARCH_TIMEOUT_SECONDS


INSTRUCTIONS_PATH = Path(__file__).with_name("instructions.txt")
OBJECT_SEARCH_WHITELIST = {
    "get_image": None,
    "look_at_pixel": "cancel_look_at_pixel",
    "point_at_pixel": "cancel_point_at_pixel",
}
MAX_DESCRIPTION_CHARACTERS = 500


class ObjectSearchAgent(AgentBase):
    """Run one bounded visual search and return its final view to S2S."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        owner_loop: asyncio.AbstractEventLoop,
        tools: ObjectSearchTools,
        endpoint: str = AGENT_TOOLS_ENDPOINT,
        completion_extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            client,
            model,
            OBJECT_SEARCH_WHITELIST,
            INSTRUCTIONS_PATH.read_text(encoding="utf-8"),
            endpoint=endpoint,
            max_rounds=7,
            max_tokens=350,
            parallel_tool_calls=False,
            completion_extra_body=completion_extra_body,
        )
        self._owner_loop = owner_loop
        self._tools = tools
        self._search_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active: concurrent.futures.Future[str] | None = None
        self._closed = False

    def register(self, schema: McpSchema) -> None:
        schema.method()(self.search_object)
        schema.method()(self.cancel_search_object)

    def cancellations(self) -> dict[str, str]:
        return {"search_object": "cancel_search_object"}

    def search_object(self, object_description: str) -> dict[str, str]:
        """Search QTrobot's visible surroundings for an object or landmark."""
        description = object_description.strip()
        if not description:
            raise ValueError("object_description cannot be empty")
        if len(description) > MAX_DESCRIPTION_CHARACTERS:
            raise ValueError(
                "object_description cannot exceed "
                f"{MAX_DESCRIPTION_CHARACTERS} characters"
            )
        if not self._search_lock.acquire(blocking=False):
            raise RuntimeError("QTrobot is already searching for another object")

        future: concurrent.futures.Future[str] | None = None
        keep_final_view = False
        try:
            self._tools.reset()
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Object-search agent is shutting down")
                future = asyncio.run_coroutine_threadsafe(
                    self.run(description),
                    self._owner_loop,
                )
                self._active = future

            try:
                answer = future.result(timeout=SEARCH_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError as exc:
                self._cancel_robot_actions()
                future.cancel()
                raise RuntimeError("Object search timed out") from exc
            except Exception:
                self._cancel_robot_actions()
                raise

            result = {"answer": answer}
            image = self._tools.latest_image()
            if image is not None:
                result.update(image)
                self._tools.hold_final_view()
                keep_final_view = True
            return result
        finally:
            if not keep_final_view:
                self._tools.release_view()
            with self._state_lock:
                if self._active is future:
                    self._active = None
            self._search_lock.release()

    def cancel_search_object(self) -> str:
        """Cancel the active object search and release robot motion."""
        with self._state_lock:
            future = self._active
        self._cancel_robot_actions()
        if future is not None:
            future.cancel()
        return "Object search cancelled."

    async def close(self) -> None:
        with self._state_lock:
            self._closed = True
            future = self._active
        self._cancel_robot_actions()
        if future is not None:
            future.cancel()
            await asyncio.gather(
                asyncio.wrap_future(future),
                return_exceptions=True,
            )

    def cleanup(self) -> None:
        with self._state_lock:
            self._closed = True
            future = self._active
        self._cancel_robot_actions()
        if future is not None:
            future.cancel()

    def _cancel_robot_actions(self) -> None:
        self._tools.cancel_look_at_pixel()
        self._tools.cancel_point_at_pixel()
