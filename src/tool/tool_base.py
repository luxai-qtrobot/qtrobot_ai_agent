"""
tool_base.py - ToolBase: contract for a tool provider hosted by LocalToolServer
(local_tool_server.py).

Each provider owns a related group of tool methods (e.g. camera/datetime, memory
retrieval) and registers them onto a schema shared with the other providers -
LocalToolServer is the one thing that actually owns the MCP server/responder/endpoint,
so a provider never touches transport/threading concerns itself.
"""

from abc import ABC, abstractmethod

from luxai.magpie.schema import McpSchema


class ToolBase(ABC):

    @abstractmethod
    def register(self, schema: McpSchema) -> None:
        """Register this provider's tool methods onto schema, e.g.
        schema.method()(self.my_tool)."""
        ...

    def cleanup(self) -> None:
        """Called once by LocalToolServer.cleanup() on shutdown - override if this
        provider holds a resource that needs closing (e.g. an open camera reader).
        No-op by default."""
        pass
