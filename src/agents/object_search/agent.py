"""Background object-search agent exposed as one S2S tool."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from luxai.magpie.schema import McpSchema
from luxai.magpie.utils import Logger

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
    """Start one bounded visual search and publish its eventual finding."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        owner_loop: asyncio.AbstractEventLoop,
        tools: ObjectSearchTools,
        event_sink: Callable[[dict[str, Any]], None],
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
            event_sink=event_sink,
        )
        self._owner_loop = owner_loop
        self._tools = tools
        self._search_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active: concurrent.futures.Future[None] | None = None
        self._active_task_id: str | None = None
        self._search_cancelled = False
        self._closed = False

    def register(self, schema: McpSchema) -> None:
        schema.method()(self.search_object)
        schema.method()(self.cancel_search_object)

    def cancellations(self) -> dict[str, str]:
        return {"search_object": "cancel_search_object"}

    def search_object(self, object_description: str) -> dict[str, str]:
        """Start an object search; its finding arrives as a background event."""
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

        task_id = f"object_search_{uuid.uuid4().hex[:12]}"
        try:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Object-search agent is shutting down")
                self._search_cancelled = False
                future = asyncio.run_coroutine_threadsafe(
                    self._run_and_emit(task_id, description),
                    self._owner_loop,
                )
                self._active = future
                self._active_task_id = task_id
            future.add_done_callback(self._finish_search)
        except Exception:
            self._search_lock.release()
            raise

        Logger.info(f"Object search started: {task_id} ({description})")
        return {
            "status": "started",
            "task_id": task_id,
            "summary": f"Searching QTrobot's surroundings for: {description}",
        }

    async def _run_and_emit(self, task_id: str, description: str) -> None:
        keep_final_view = False
        try:
            self._tools.reset()
            answer = await asyncio.wait_for(
                self.run(description),
                timeout=SEARCH_TIMEOUT_SECONDS,
            )
            self._raise_if_cancelled(task_id)
            if self._tools.latest_image() is not None:
                self._tools.hold_final_view()
                self._raise_if_cancelled(task_id)
                keep_final_view = True

            Logger.info(f"Object search completed: {task_id}")
            self._emit_event(
                {
                    "type": "object_search.done",
                    "id": task_id,
                    "payload": answer,
                }
            )
        except asyncio.CancelledError:
            self._cancel_robot_actions()
            Logger.info(f"Object search cancelled: {task_id}")
            raise
        except Exception as exc:
            self._cancel_robot_actions()
            Logger.error(f"Object search failed: {task_id}: {exc}")
            self._emit_event(
                {
                    "type": "object_search.failed",
                    "id": task_id,
                    "payload": (
                        f"The search for {description!r} could not be completed."
                    ),
                }
            )
        finally:
            if not keep_final_view:
                self._tools.release_view()

    def cancel_search_object(self) -> str:
        """Cancel the active object search and release robot motion."""
        with self._state_lock:
            future = self._active
            if future is None:
                return "No object search is active."
            self._search_cancelled = True
        self._cancel_robot_actions()
        future.cancel()
        return "Object-search cancellation requested."

    async def close(self) -> None:
        with self._state_lock:
            self._closed = True
            self._search_cancelled = True
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
            self._search_cancelled = True
            future = self._active
        self._cancel_robot_actions()
        if future is not None:
            future.cancel()

    def _emit_event(self, event: dict[str, str]) -> None:
        with self._state_lock:
            if (
                self._closed
                or self._search_cancelled
                or self._active_task_id != event["id"]
            ):
                return
        try:
            self.emit_event(event)
        except Exception as exc:
            Logger.error(
                f"Could not emit object-search event {event['id']}: {exc}"
            )

    def _finish_search(
        self,
        future: concurrent.futures.Future[None],
    ) -> None:
        release_lock = False
        with self._state_lock:
            if self._active is future:
                self._active = None
                self._active_task_id = None
                self._search_cancelled = False
                release_lock = True
        if release_lock:
            self._search_lock.release()

    def _raise_if_cancelled(self, task_id: str) -> None:
        with self._state_lock:
            cancelled = (
                self._closed
                or self._search_cancelled
                or self._active_task_id != task_id
            )
        if cancelled:
            raise RuntimeError("Object search cancelled")

    def _cancel_robot_actions(self) -> None:
        self._tools.cancel_look_at_pixel()
        self._tools.cancel_point_at_pixel()
