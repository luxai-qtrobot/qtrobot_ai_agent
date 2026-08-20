"""Discover and execute tools exposed by one or more MCP servers."""

from __future__ import annotations

import asyncio
import base64
import binascii
import concurrent.futures
import json
import threading
from collections.abc import Collection, Mapping
from typing import Any

from fastmcp import Client
from fastmcp.exceptions import ToolError
from luxai.magpie.utils import Logger

from .tool_output import ToolCallResult, ToolImage


CANCELLED = object()
CANCEL_ALL_TOOL_NAME = "cancel_all_actions"

_CANCEL_ALL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": CANCEL_ALL_TOOL_NAME,
    "description": (
        "Cancel every action currently in progress. Use this when the user "
        "asks to stop current work."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


ToolWhitelist = Mapping[str, str | None] | Collection[str]


class ToolEngine:
    """Merge MCP sources into one flat S2S tool schema and dispatch table."""

    def __init__(
        self,
        sources: Mapping[str, Client],
        whitelists: Mapping[str, ToolWhitelist] | None = None,
        *,
        retries: int = 0,
    ) -> None:
        if retries < 0:
            raise ValueError("retries cannot be negative")
        self._sources = dict(sources)
        self._whitelists: dict[str, dict[str, str | None]] = {}
        for source, whitelist in (whitelists or {}).items():
            if isinstance(whitelist, Mapping):
                self._whitelists[source] = dict(whitelist)
            else:
                self._whitelists[source] = {name: None for name in whitelist}
        self._retries = retries
        self._tools: dict[str, tuple[Client, str, str | None]] = {}
        self._schemas: list[dict[str, Any]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending: dict[str, str] = {}
        self._pending_lock = threading.Lock()
        self._cancelled_ids: set[str] = set()
        self._cancellation_futures: dict[
            tuple[int, str], concurrent.futures.Future[None]
        ] = {}
        self._cancellation_futures_lock = threading.Lock()

    async def discover(self) -> None:
        """Discover tools and build the flat schemas accepted by S2S."""
        self._loop = asyncio.get_running_loop()
        tools: dict[str, tuple[Client, str, str | None]] = {}
        schemas: list[dict[str, Any]] = []

        for source_name, client in self._sources.items():
            discovered = list(await client.list_tools())
            whitelist = self._whitelists.get(source_name)
            if whitelist is not None:
                available = {tool.name for tool in discovered}
                missing = sorted(set(whitelist) - available)
                cancel_names = {name for name in whitelist.values() if name}
                missing_cancel_tools = sorted(cancel_names - available)
                if missing:
                    Logger.warning(
                        f"ToolEngine: {source_name} is missing tools: {missing}"
                    )
                if missing_cancel_tools:
                    Logger.warning(
                        f"ToolEngine: {source_name} is missing cancel tools: "
                        f"{missing_cancel_tools}"
                    )
                discovered = [tool for tool in discovered if tool.name in whitelist]

            for tool in discovered:
                if tool.name == CANCEL_ALL_TOOL_NAME:
                    raise ValueError(
                        f"Tool name {CANCEL_ALL_TOOL_NAME!r} is reserved by ToolEngine"
                    )
                if tool.name in tools:
                    other_source = tools[tool.name][1]
                    raise ValueError(
                        f"Tool {tool.name!r} is defined by both "
                        f"{other_source!r} and {source_name!r}"
                    )
                cancel_name = whitelist.get(tool.name) if whitelist is not None else None
                tools[tool.name] = (client, source_name, cancel_name)
                schemas.append(
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description or tool.name,
                        "parameters": tool.inputSchema
                        or {"type": "object", "properties": {}},
                    }
                )

        schemas.append(dict(_CANCEL_ALL_SCHEMA))
        self._tools = tools
        self._schemas = schemas
        Logger.info(
            f"ToolEngine: discovered {len(tools)} MCP tools: {sorted(tools)}"
        )

    def schemas(self) -> list[dict[str, Any]]:
        """Return copies of the function schemas for ``session.update``."""
        return [dict(schema) for schema in self._schemas]

    async def execute(self, tool_calls: Collection[Mapping[str, Any]]) -> list[ToolCallResult]:
        """Execute independent tool calls concurrently."""
        return list(
            await asyncio.gather(*(self._run_one(call) for call in tool_calls))
        )

    async def execute_tracked(
        self,
        tool_calls: Collection[Mapping[str, Any]],
    ) -> list[ToolCallResult] | object:
        """Execute calls while making them visible to :meth:`cancel_all`."""
        calls = list(tool_calls)
        call_ids = [self._call_id(call) for call in calls]
        with self._pending_lock:
            for call, call_id in zip(calls, call_ids):
                self._pending[call_id] = str(call.get("name") or "")

        try:
            results = await self.execute(calls)
        finally:
            with self._pending_lock:
                was_cancelled = any(
                    call_id in self._cancelled_ids for call_id in call_ids
                )
                for call_id in call_ids:
                    self._pending.pop(call_id, None)
                    self._cancelled_ids.discard(call_id)

        return CANCELLED if was_cancelled else results

    def cancel_all(self) -> set[str]:
        """Suppress pending results and request paired robot cancellations."""
        with self._pending_lock:
            pending = dict(self._pending)
            self._cancelled_ids.update(pending)

        loop = self._loop
        if loop is None:
            return set(pending)

        dispatched: set[tuple[int, str]] = set()
        for tool_name in pending.values():
            client, _source, cancel_name = self._tools.get(
                tool_name,
                (None, "", None),
            )
            if client is None or cancel_name is None:
                continue
            dispatch_key = (id(client), cancel_name)
            if dispatch_key in dispatched:
                continue
            dispatched.add(dispatch_key)
            with self._cancellation_futures_lock:
                existing = self._cancellation_futures.get(dispatch_key)
                if existing is not None and not existing.done():
                    continue
                future = asyncio.run_coroutine_threadsafe(
                    self._dispatch_cancel(client, cancel_name),
                    loop,
                )
                self._cancellation_futures[dispatch_key] = future
            # Register outside the lock: add_done_callback() may invoke the
            # callback immediately when a very fast cancel has already ended.
            future.add_done_callback(
                lambda completed, key=dispatch_key: (
                    self._forget_cancellation(key, completed)
                )
            )

        return set(pending)

    async def wait_for_cancellations(self) -> None:
        """Wait until all paired cancel calls scheduled so far have finished.

        This must complete before the owning MCP clients are closed. Calls are
        normally made from the same event loop on which :meth:`discover` ran;
        cross-loop callers are forwarded to that owning loop.
        """
        owner_loop = self._loop
        if owner_loop is None:
            return
        running_loop = asyncio.get_running_loop()
        if running_loop is not owner_loop:
            forwarded = asyncio.run_coroutine_threadsafe(
                self.wait_for_cancellations(),
                owner_loop,
            )
            await asyncio.wrap_future(forwarded)
            return

        while True:
            with self._cancellation_futures_lock:
                futures = tuple(self._cancellation_futures.values())
            if not futures:
                return
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in futures),
                return_exceptions=True,
            )

    def _forget_cancellation(
        self,
        key: tuple[int, str],
        completed: concurrent.futures.Future[None],
    ) -> None:
        with self._cancellation_futures_lock:
            if self._cancellation_futures.get(key) is completed:
                self._cancellation_futures.pop(key, None)

    async def _dispatch_cancel(self, client: Client, cancel_name: str) -> None:
        try:
            await client.call_tool(cancel_name, {})
        except Exception as exc:
            Logger.warning(
                f"ToolEngine: cancel tool {cancel_name!r} failed: {exc}"
            )

    async def _run_one(self, call: Mapping[str, Any]) -> ToolCallResult:
        call_id = self._call_id(call)
        name = str(call.get("name") or "")

        if name == CANCEL_ALL_TOOL_NAME:
            # The control call must not cancel its own confirmation result.
            with self._pending_lock:
                self._pending.pop(call_id, None)
            cancelled_ids = self.cancel_all()
            if cancelled_ids:
                output = "Cancellation requested for all actions currently in progress."
            else:
                output = "No actions are currently in progress."
            return {"tool_call_id": call_id, "output": output, "images": []}

        if name not in self._tools:
            return self._error_result(call_id, f"tool {name!r} does not exist")

        raw_arguments = call.get("arguments") or {}
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                return self._error_result(call_id, f"invalid arguments JSON: {exc}")
        else:
            arguments = raw_arguments
        if not isinstance(arguments, Mapping):
            return self._error_result(call_id, "arguments must be a JSON object")

        client, _source, _cancel_name = self._tools[name]
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                result = await client.call_tool(name, dict(arguments))
            except ToolError as exc:
                return self._error_result(call_id, str(exc))
            except Exception as exc:
                last_error = exc
                Logger.debug(
                    f"ToolEngine: {name!r} attempt {attempt + 1} failed: {exc}"
                )
            else:
                # A completed remote call is never repeated just because its
                # returned content is malformed.
                try:
                    return self._normalize_result(call_id, result)
                except Exception as exc:
                    return self._error_result(call_id, f"invalid tool result: {exc}")

        return self._error_result(
            call_id,
            f"call failed after {self._retries + 1} attempt(s): {last_error}",
        )

    def _normalize_result(self, call_id: str, result: Any) -> ToolCallResult:
        text_parts: list[str] = []
        images: list[ToolImage] = []
        seen_images: set[tuple[str, str]] = set()

        def add_image(mime_type: Any, data: Any) -> None:
            validated = self._validate_image_payload(mime_type, data)
            key = (validated["mime_type"], validated["data"])
            if key not in seen_images:
                seen_images.add(key)
                images.append(validated)

        for block in getattr(result, "content", ()) or ():
            block_type = getattr(block, "type", "unknown")
            if block_type == "text":
                image = self._extract_image_envelope(getattr(block, "text", ""))
                if image is None:
                    text_parts.append(str(getattr(block, "text", "")))
                else:
                    add_image(*image)
            elif block_type == "image":
                add_image(
                    getattr(block, "mimeType", None),
                    getattr(block, "data", None),
                )
            else:
                text_parts.append(f"Unsupported MCP content type: {block_type}.")

        structured = getattr(result, "structured_content", None)
        if structured is None:
            structured = getattr(result, "structuredContent", None)
        if structured is not None:
            image = self._extract_image_envelope(structured)
            if image is not None:
                add_image(*image)
            elif not text_parts:
                text_parts.append(
                    json.dumps(structured, separators=(",", ":"), ensure_ascii=False)
                )

        text = "\n".join(part for part in text_parts if part)
        if not text:
            text = "Image captured." if images else "Done."
        return {"tool_call_id": call_id, "output": text, "images": images}

    @classmethod
    def _extract_image_envelope(cls, value: Any) -> tuple[str, str] | None:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(value, Mapping):
            return None
        if "mimeType" not in value or "data" not in value:
            return None
        image = cls._validate_image_payload(value.get("mimeType"), value.get("data"))
        return image["mime_type"], image["data"]

    @staticmethod
    def _validate_image_payload(mime_type: Any, data: Any) -> ToolImage:
        if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
            raise ValueError("MCP image content has an invalid MIME type")
        if not isinstance(data, str) or not data:
            raise ValueError("MCP image content has no base64 data")
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("MCP image content contains invalid base64 data") from exc
        if not decoded:
            raise ValueError("MCP image content decoded to an empty image")
        return {"mime_type": mime_type, "data": data}

    @staticmethod
    def _error_result(call_id: str, message: str) -> ToolCallResult:
        return {"tool_call_id": call_id, "output": f"Error: {message}", "images": []}

    @staticmethod
    def _call_id(call: Mapping[str, Any]) -> str:
        call_id = str(call.get("call_id") or call.get("id") or "")
        if not call_id:
            raise ValueError("tool call has no call_id")
        return call_id
