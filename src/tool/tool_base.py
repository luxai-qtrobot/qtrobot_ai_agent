"""Contract implemented by providers hosted in the local MCP server."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from luxai.magpie.schema import McpSchema


class ToolBase(ABC):
    def __init__(
        self,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.event_sink = event_sink

    def emit_event(self, event: dict[str, Any]) -> None:
        """Publish an application event from a background-capable tool."""
        if self.event_sink is None:
            raise RuntimeError(f"{type(self).__name__} has no event sink")
        self.event_sink(event)

    @abstractmethod
    def register(self, schema: McpSchema) -> None:
        """Register this provider's methods on ``schema``."""

    def cleanup(self) -> None:
        """Release resources owned by the provider."""
