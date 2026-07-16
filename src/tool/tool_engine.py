"""
tool_engine.py - ToolEngine: merges tools from one or more MCP clients (robot,
user-defined, ...) into a single OpenAI-compatible tool schema, with concurrent,
error-safe dispatch.

The tool schema list and the dispatch table are built from the same discovery pass,
so the LLM can never be offered a tool we can't actually route a call to.

Cancellation: a whitelist entry can pair a tool with its own cancel tool (e.g.
gesture_file_play -> gesture_file_cancel, matching how the robot SDK's own
ActionHandle.cancel() already works - a separate RPC to a cancel_service_name, with
the original call expected to resolve on its own once the action honours the stop).
cancel_all() dispatches that pairing for every currently in-flight call and marks its
id cancelled either way; execute_async() then discards results for cancelled ids
(passing the CANCELLED sentinel to the callback instead) rather than trying to abort
the in-flight MCP request itself - real MCP cancellation notifications don't reach our
transport (McpTransport drops every id-less JSON-RPC message), so a real stop can only
ever come from the paired tool, never from the dispatch layer.
"""

import asyncio
import json
import threading

from fastmcp import Client
from fastmcp.exceptions import ToolError
from luxai.magpie.utils import Logger

# Sentinel passed to an execute_async() callback instead of real results when every
# tool_call_id in that batch was cancelled before it finished - see cancel_all().
CANCELLED = object()


class ToolEngine:

    def __init__(self, sources: dict, whitelists: dict = None, retries: int = 1):
        """
        sources:    {source_name: fastmcp.Client}, e.g. {"robot": robot_client, "user": user_client}
        whitelists: optional {source_name: {tool_name: cancel_tool_name | None}} - only
                    expose these tools from that source; cancel_tool_name, if given, is
                    dispatched by cancel_all() to request an early, best-effort stop.
        retries:    extra attempts for transport-level failures (not for tool-reported errors)
        """
        self._sources = sources
        self._whitelists = whitelists or {}
        self._retries = retries
        self._tools: dict[str, tuple[Client, str, str | None]] = {}   # name -> (client, source_name, cancel_name)
        self._schemas: list[dict] = []
        self._loop = None   # resolved by discover() - the loop self._sources' MCP
                             # clients are bound to; execute_async() dispatches back
                             # onto this exact loop, never a fresh one, since fastmcp
                             # Client sessions can't be used from a different loop.

        # In-flight bookkeeping for cancel_all() - tool_call_id -> tool_name, populated
        # right before dispatch and cleared once that id's batch finishes either way.
        self._pending: dict[str, str] = {}
        self._pending_lock = threading.Lock()
        self._cancelled_ids: set[str] = set()

    async def discover(self) -> None:
        self._loop = asyncio.get_running_loop()
        tools: dict[str, tuple[Client, str, str | None]] = {}
        schemas: list[dict] = []

        for source_name, client in self._sources.items():
            mcp_tools = await client.list_tools()
            whitelist = self._whitelists.get(source_name)
            if whitelist:
                mcp_tools = [t for t in mcp_tools if t.name in whitelist]

            for tool in mcp_tools:
                if tool.name in tools:
                    other_source = tools[tool.name][1]
                    raise ValueError(
                        f"Tool name collision: '{tool.name}' is defined by both "
                        f"'{other_source}' and '{source_name}'"
                    )
                cancel_name = whitelist.get(tool.name) if whitelist else None
                tools[tool.name] = (client, source_name, cancel_name)
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or tool.name,
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    },
                })

        self._tools = tools
        self._schemas = schemas
        Logger.info(f"ToolEngine: discovered {len(tools)} tools: {sorted(tools)}")

    def schemas(self) -> list[dict]:
        return self._schemas

    def print_schemas(self, raw: bool = False) -> None:
        """Pretty-print every discovered/merged tool schema - name, source, description,
        parameters - for inspection (e.g. from a '/tools' debug command). raw=True dumps
        the exact JSON list as sent to the LLM instead."""
        if raw:
            print(json.dumps(self._schemas, indent=2))
            return

        print(f"\n--- tools ({len(self._schemas)}) ---")
        for schema in self._schemas:
            fn = schema["function"]
            source = self._tools[fn["name"]][1]
            print(f"[{source}] {fn['name']}: {fn['description']}")
            print(f"    parameters: {json.dumps(fn['parameters'])}")
        print("--- end tools ---\n")

    async def execute(self, tool_calls: list[dict]) -> list[dict]:
        """tool_calls: [{'id', 'name', 'arguments'(json str)}, ...], run concurrently.
        Returns [{'tool_call_id', 'content', 'extra_messages'}, ...] - never raises;
        failures come back as an error string in 'content' for the LLM to see."""
        return list(await asyncio.gather(*(self._run_one(tc) for tc in tool_calls)))

    def execute_async(self, tool_calls: list[dict], callback) -> None:
        """Non-blocking version of execute() - callable from any thread, including one
        with no event loop of its own (e.g. LLMEngine's worker thread). Lets a caller
        move on immediately instead of waiting on a slow tool/agent call - this is the
        one place that owns *how* tool dispatch avoids blocking, so callers never need
        their own threading for it.

        Dispatches the actual work back onto the loop self._sources' MCP clients were
        opened on (captured by discover()) via run_coroutine_threadsafe - a fresh loop
        (e.g. asyncio.run() on a new thread) can't touch those clients at all, they
        belong to the loop that created them. Once done, callback(results) runs on yet
        another fresh thread, not the event loop's own thread - callback may itself do
        blocking work (e.g. a follow-up completion), which would otherwise freeze the
        whole event loop, MCP transport included.

        Registers every tool_call_id as pending before dispatch and clears it once this
        batch finishes, so cancel_all() (possibly called from a different thread while
        this is in flight) knows what it can target. If any id in this batch was
        cancelled by the time it finishes, callback gets the CANCELLED sentinel instead
        of real results - the call itself was never force-stopped (only its paired
        cancel tool, if any, was asked to stop it), this just discards a now-unwanted
        answer rather than acting on it."""
        ids = [tc["id"] for tc in tool_calls]
        with self._pending_lock:
            for tc in tool_calls:
                self._pending[tc["id"]] = tc["name"]

        future = asyncio.run_coroutine_threadsafe(self.execute(tool_calls), self._loop)

        def _finish(f):
            with self._pending_lock:
                cancelled = any(i in self._cancelled_ids for i in ids)
                for i in ids:
                    self._pending.pop(i, None)
                    self._cancelled_ids.discard(i)
            results = CANCELLED if cancelled else f.result()
            threading.Thread(target=callback, args=(results,), daemon=True).start()

        future.add_done_callback(_finish)

    def cancel_all(self) -> None:
        """Marks every currently in-flight tool call cancelled - execute_async() will
        discard its result (see CANCELLED) once it finishes, regardless of whether it
        could actually be stopped early. For calls whose tool has a paired cancel tool
        (see whitelists), also dispatches that cancel tool as a real, best-effort
        request to stop early - fire-and-forget, its own result is irrelevant here.
        Calls without a pairing just keep running to completion in the background;
        wasted computation, but no different from letting any other unwanted work
        finish - never force-killed."""
        with self._pending_lock:
            pending = dict(self._pending)
            self._cancelled_ids.update(pending)

        for tool_call_id, tool_name in pending.items():
            client, _, cancel_name = self._tools.get(tool_name, (None, None, None))
            if client is not None and cancel_name is not None:
                asyncio.run_coroutine_threadsafe(self._dispatch_cancel(client, cancel_name), self._loop)

    async def _dispatch_cancel(self, client: Client, cancel_name: str) -> None:
        try:
            await client.call_tool(cancel_name, {})
        except Exception as e:
            Logger.debug(f"ToolEngine: cancel tool '{cancel_name}' failed: {e}")

    async def _run_one(self, tc: dict) -> dict:
        tool_call_id = tc["id"]
        name = tc["name"]

        if name not in self._tools:
            return self._error_result(tool_call_id, f"tool '{name}' does not exist")

        try:
            args = json.loads(tc.get("arguments") or "{}")
        except json.JSONDecodeError as e:
            return self._error_result(tool_call_id, f"invalid arguments JSON: {e}")

        client, _, _ = self._tools[name]

        last_error = None
        for attempt in range(self._retries + 1):
            try:
                result = await client.call_tool(name, args)
                return self._to_result(tool_call_id, result)
            except ToolError as e:
                # A real, reported tool failure (e.g. bad gesture name) - not transient, don't retry.
                return self._error_result(tool_call_id, str(e))
            except Exception as e:
                last_error = e
                Logger.debug(f"ToolEngine: '{name}' attempt {attempt + 1} failed: {e}")

        return self._error_result(tool_call_id, f"call failed after {self._retries + 1} attempts: {last_error}")

    def _to_result(self, tool_call_id: str, result) -> dict:
        text_parts = []
        extra_messages = []
        for block in result.content:
            if block.type == "text":
                image = self._extract_image_envelope(block.text)
                if image:
                    extra_messages.append(self._image_message(image["mimeType"], image["data"]))
                else:
                    text_parts.append(block.text)
            elif block.type == "image":
                # Native MCP image content block. McpSchema-based tools can't produce this
                # today (see _extract_image_envelope), but other MCP sources might.
                extra_messages.append(self._image_message(block.mimeType, block.data))
            else:
                text_parts.append(str(block))

        content = "\n".join(text_parts) if text_parts else ("Image captured." if extra_messages else "Done.")
        return {"tool_call_id": tool_call_id, "content": content, "extra_messages": extra_messages}

    @staticmethod
    def _extract_image_envelope(text: str):
        """McpSchema always wraps tool results as a text block (see mcp_schema.py's
        _mcp_tools_call), so a tool like get_image() can't return a native MCP image
        content block - it JSON-serializes a {'mimeType', 'data'} dict into text instead.
        Detect that shape here and treat it as an image rather than literal text."""
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(obj, dict) and "data" in obj and "mimeType" in obj:
            return obj
        return None

    @staticmethod
    def _image_message(mime_type: str, data: str) -> dict:
        return {
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{data}"},
            }],
        }

    def _error_result(self, tool_call_id: str, message: str) -> dict:
        return {"tool_call_id": tool_call_id, "content": f"Error: {message}", "extra_messages": []}
