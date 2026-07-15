"""
user_tools.py - User-defined tools: get_datetime, get_image.

A ToolBase provider, not its own MCP server - it registers its bound methods (schemas
derived from each method's signature + docstring via McpSchema, no hand-written JSON
schema) onto a schema handed to it by a shared LocalToolServer (local_tool_server.py),
so adding a new tool group never means standing up another ServerNode/responder/
endpoint. Add new user tools here as methods, registered in register().
"""

import base64
from datetime import datetime

import numpy as np
from simplejpeg import encode_jpeg

from luxai.magpie.schema import McpSchema
from luxai.magpie.utils import Logger

from .tool_base import ToolBase


class UserTools(ToolBase):

    def __init__(self, robot):
        self.robot = robot        
        self._camera_reader = self.robot.camera.stream.open_color_reader()


    def register(self, schema: McpSchema) -> None:
        schema.method()(self.get_datetime)
        schema.method()(self.get_image)

    def get_datetime(self) -> str:
        """Get the current date and time."""
        return datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")

    def get_image(self) -> dict:
        """Capture a single color image from the robot's camera, for visual questions
        like 'what do you see?'."""        
        frame = self._camera_reader.read(timeout=2.0)

        # open_color_reader() is documented as always producing BGR - no need to trust
        # frame.pixel_format, which carries RealSense's raw format string ("BGR8").
        image = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, frame.channels)
        jpeg_bytes = encode_jpeg(image, quality=80, colorspace="BGR")

        # McpSchema always wraps tool results as a text block (see mcp_schema.py's
        # _mcp_tools_call) - there's no native "image" content type yet. This dict gets
        # JSON-serialized into that text block; ToolEngine._to_result() knows to unpack
        # this specific {mimeType, data} shape back into a real image for the LLM.
        Logger.info(f"get_image: captured ({frame.width}, {frame.height})!")
        return {"mimeType": "image/jpeg", "data": base64.b64encode(jpeg_bytes).decode("ascii")}

    def cleanup(self) -> None:
        """Close the camera reader (if ever opened) - called by LocalToolServer.cleanup()
        for every provider it holds."""
        if self._camera_reader is not None:
            try:
                self._camera_reader.close()
            except Exception as e:
                Logger.warning(f"user-tools: error closing camera reader: {e}")
            self._camera_reader = None
