"""
agent.py - Agent: merges tools from one or more MCP clients (robot, user-defined, ...)
into a single OpenAI-compatible tool schema, with concurrent, error-safe dispatch.

The tool schema list and the dispatch table are built from the same discovery pass,
so the LLM can never be offered a tool we can't actually route a call to.
"""

import asyncio
import json

from fastmcp import Client
from fastmcp.exceptions import ToolError
from luxai.magpie.utils import Logger


class Agent:

    def __init__(self, sources: dict, whitelists: dict = None, retries: int = 1):
        """
        sources:    {source_name: fastmcp.Client}, e.g. {"robot": robot_client, "user": user_client}
        whitelists: optional {source_name: {tool_name, ...}} - only expose these tools from that source
        retries:    extra attempts for transport-level failures (not for tool-reported errors)
        """
        self._sources = sources
        self._whitelists = whitelists or {}
        self._retries = retries
        self._tools: dict[str, tuple[Client, str]] = {}   # name -> (client, source_name)
        self._schemas: list[dict] = []

    async def discover(self) -> None:
        tools: dict[str, tuple[Client, str]] = {}
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
                tools[tool.name] = (client, source_name)
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
        Logger.info(f"Agent: discovered {len(tools)} tools: {sorted(tools)}")

    def schemas(self) -> list[dict]:
        return self._schemas

    async def execute(self, tool_calls: list[dict]) -> list[dict]:
        """tool_calls: [{'id', 'name', 'arguments'(json str)}, ...], run concurrently.
        Returns [{'tool_call_id', 'content', 'extra_messages'}, ...] - never raises;
        failures come back as an error string in 'content' for the LLM to see."""
        return list(await asyncio.gather(*(self._run_one(tc) for tc in tool_calls)))

    async def _run_one(self, tc: dict) -> dict:
        tool_call_id = tc["id"]
        name = tc["name"]

        if name not in self._tools:
            return self._error_result(tool_call_id, f"tool '{name}' does not exist")

        try:
            args = json.loads(tc.get("arguments") or "{}")
        except json.JSONDecodeError as e:
            return self._error_result(tool_call_id, f"invalid arguments JSON: {e}")

        client, _ = self._tools[name]

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
                Logger.debug(f"Agent: '{name}' attempt {attempt + 1} failed: {e}")

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
