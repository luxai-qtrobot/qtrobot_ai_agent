"""Transport-neutral representation of completed MCP calls."""

from __future__ import annotations

from typing import TypedDict


class ToolImage(TypedDict):
    mime_type: str
    data: str


class ToolCallResult(TypedDict):
    tool_call_id: str
    output: str
    images: list[ToolImage]


def as_function_call_output(result: ToolCallResult) -> dict[str, str]:
    """Build the string-only function output item accepted by S2S Realtime."""
    return {
        "type": "function_call_output",
        "call_id": result["tool_call_id"],
        "output": result["output"],
    }


def as_input_image_item(image: ToolImage) -> dict:
    """Build the Realtime user item used to supply a tool-returned image."""
    return {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_image",
                "image_url": (
                    f"data:{image['mime_type']};base64,{image['data']}"
                ),
                "detail": "auto",
            }
        ],
    }


def summarize_tool_result(result: ToolCallResult) -> str:
    """Describe a result for logs without exposing encoded image data."""
    compact = " ".join(result["output"].split())
    if len(compact) > 240:
        compact = compact[:237] + "..."
    if result["images"]:
        kinds = ", ".join(image["mime_type"] for image in result["images"])
        return f"{compact} ({len(result['images'])} image(s): {kinds})"
    return compact
