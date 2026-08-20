"""One in-process MCP server shared by application-owned tool providers."""

from collections.abc import Iterable

from luxai.magpie.nodes import ServerNode
from luxai.magpie.schema import McpSchema
from luxai.magpie.transport import ZMQRpcResponder
from luxai.magpie.utils import Logger

from .tool_base import ToolBase


LOCAL_TOOLS_ENDPOINT = "inproc://qtrobot-s2s-local-tools"


class LocalToolServer(ServerNode):
    def __init__(
        self,
        providers: Iterable[ToolBase],
        endpoint: str = LOCAL_TOOLS_ENDPOINT,
    ) -> None:
        self._providers = tuple(providers)
        schema = McpSchema(name="local-tools", version="1.0.0")
        for provider in self._providers:
            provider.register(schema)

        # ServerNode starts request handling during construction, so all methods
        # must be registered before its responder is passed to the base class.
        responder = ZMQRpcResponder(endpoint, schema=schema)
        super().__init__(name="local-tool-server", responder=responder)

    def cleanup(self) -> None:
        for provider in self._providers:
            try:
                provider.cleanup()
            except Exception as exc:
                Logger.warning(
                    f"{self.name}: error cleaning up "
                    f"{type(provider).__name__}: {exc}"
                )
        super().cleanup()
