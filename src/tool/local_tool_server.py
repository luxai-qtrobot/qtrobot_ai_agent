"""
local_tool_server.py - LocalToolServer: a single in-process MCP server (one
McpSchema, one ZMQRpcResponder, one ServerNode) that hosts tool methods contributed
by any number of ToolBase providers (UserTools, MemoryTools, ...), so adding a new
local tool group never means standing up another ServerNode/responder/endpoint pair.
See tool_base.py for the provider contract.
"""

from luxai.magpie.nodes import ServerNode
from luxai.magpie.schema import McpSchema
from luxai.magpie.transport import ZMQRpcResponder
from luxai.magpie.utils import Logger

from .tool_base import ToolBase

LOCAL_TOOLS_ENDPOINT = "inproc://local-tools"


class LocalToolServer(ServerNode):

    def __init__(self, providers: list[ToolBase], endpoint: str = LOCAL_TOOLS_ENDPOINT):
        self._providers = providers
        schema = McpSchema(name="local-tools", version="1.0.0")
        for provider in providers:
            provider.register(schema)

        # Providers must be registered above before this runs - BaseNode.__init__()
        # starts the request-handling thread as its very last step.
        responder = ZMQRpcResponder(endpoint, schema=schema)
        super().__init__(name="local-tool-server", responder=responder)

    def cleanup(self) -> None:
        for provider in self._providers:
            try:
                provider.cleanup()
            except Exception as e:
                Logger.warning(f"{self.name}: error cleaning up {provider}: {e}")
        super().cleanup()
