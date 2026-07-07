"""
user_agents.py - User-defined tools, served as a local MCP server over ZMQ inproc.

UserAgents *is* the server node: it holds an already-connected Robot instance
(connected + plugin-enabled by the caller), registers its own bound methods as MCP tools
(schemas derived from each method's signature + docstring via McpSchema, no hand-written
JSON schema), and manages its own responder/thread lifecycle via ServerNode. Add new user
tools here as methods, registered in __init__ before super().__init__() runs.
"""

import base64
from datetime import datetime

import numpy as np
from simplejpeg import encode_jpeg

from luxai.magpie.nodes import ServerNode
from luxai.magpie.schema import McpSchema
from luxai.magpie.transport import ZMQRpcResponder

USER_TOOLS_ENDPOINT = "inproc://user-tools"


class UserAgents(ServerNode):

    def __init__(self, robot):
        self.robot = robot
        self.schema = McpSchema(name="user-tools", version="1.0.0")

        self.schema.method()(self.get_time)
        self.schema.method()(self.add_numbers)
        self.schema.method()(self.get_image)

        # Tools must be registered above before this runs - BaseNode.__init__() starts
        # the request-handling thread as its very last step.
        responder = ZMQRpcResponder(USER_TOOLS_ENDPOINT, schema=self.schema)
        super().__init__(name="user-tools-server", responder=responder)

    def get_time(self) -> str:
        """Get the current wall-clock time."""
        return datetime.now().strftime("%H:%M:%S")

    def add_numbers(self, a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    def get_image(self) -> dict:
        """Capture a single color image from the robot's camera, for visual questions
        like 'what do you see?'."""
        reader = self.robot.camera.stream.open_color_reader()
        try:
            frame = reader.read(timeout=3.0)
        finally:
            reader.close()

        # open_color_reader() is documented as always producing BGR - no need to trust
        # frame.pixel_format, which carries RealSense's raw format string ("BGR8").
        image = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, frame.channels)
        jpeg_bytes = encode_jpeg(image, quality=80, colorspace="BGR")

        # McpSchema always wraps tool results as a text block (see mcp_schema.py's
        # _mcp_tools_call) - there's no native "image" content type yet. This dict gets
        # JSON-serialized into that text block; Agent._to_result() knows to unpack this
        # specific {mimeType, data} shape back into a real image for the LLM.
        return {"mimeType": "image/jpeg", "data": base64.b64encode(jpeg_bytes).decode("ascii")}
