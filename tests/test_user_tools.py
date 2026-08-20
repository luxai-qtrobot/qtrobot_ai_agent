from __future__ import annotations

import base64
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from simplejpeg import decode_jpeg, encode_jpeg, is_jpeg


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from luxai.magpie.schema import McpSchema
from tool.providers.user_tools import CAMERA_READ_TIMEOUT, UserTools


class _Reader:
    def __init__(self, frame) -> None:
        self.frame = frame
        self.timeouts: list[float] = []
        self.close_count = 0

    def read(self, *, timeout: float):
        self.timeouts.append(timeout)
        return self.frame

    def close(self) -> None:
        self.close_count += 1


class _ColorStream:
    def __init__(self, reader: _Reader) -> None:
        self.reader = reader
        self.queue_sizes: list[int] = []

    def open_color_reader(self, *, queue_size: int):
        self.queue_sizes.append(queue_size)
        return self.reader


def _make_tools(frame=None) -> tuple[UserTools, _Reader, _ColorStream]:
    if frame is None:
        pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        frame = SimpleNamespace(
            width=3,
            height=2,
            channels=3,
            data=pixels.tobytes(),
        )
    reader = _Reader(frame)
    stream = _ColorStream(reader)
    robot = SimpleNamespace(camera=SimpleNamespace(stream=stream))
    return UserTools(robot), reader, stream


class UserToolsTests(unittest.TestCase):
    def test_registers_datetime_and_image(self) -> None:
        tools, _, _ = _make_tools()
        schema = McpSchema(name="test-tools")
        tools.register(schema)

        datetime_result = schema._mcp_tools_call(
            name="get_datetime",
            arguments={},
        )
        image_result = schema._mcp_tools_call(name="get_image", arguments={})

        self.assertFalse(datetime_result["isError"])
        self.assertFalse(image_result["isError"])
        image_envelope = json.loads(image_result["content"][0]["text"])
        self.assertEqual(image_envelope["mimeType"], "image/jpeg")
        self.assertTrue(image_envelope["data"])
        tools.cleanup()

    def test_raw_camera_frame_is_encoded_and_reader_cleanup_is_idempotent(self) -> None:
        tools, reader, stream = _make_tools()

        result = tools.get_image()

        jpeg = base64.b64decode(result["data"], validate=True)
        self.assertEqual(result["mimeType"], "image/jpeg")
        self.assertTrue(is_jpeg(jpeg))
        self.assertEqual(decode_jpeg(jpeg, colorspace="BGR").shape, (2, 3, 3))
        self.assertEqual(stream.queue_sizes, [1])
        self.assertEqual(reader.timeouts, [CAMERA_READ_TIMEOUT])

        tools.cleanup()
        tools.cleanup()
        self.assertEqual(reader.close_count, 1)

    def test_valid_small_jpeg_is_not_reencoded(self) -> None:
        pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        jpeg = encode_jpeg(pixels, quality=80, colorspace="BGR")
        frame = SimpleNamespace(
            width=3,
            height=2,
            channels=0,
            format="jpeg",
            data=jpeg,
        )
        tools, _, _ = _make_tools(frame)

        result = tools.get_image()

        self.assertEqual(base64.b64decode(result["data"]), jpeg)
        tools.cleanup()

    def test_large_image_is_resized_without_changing_aspect_ratio(self) -> None:
        pixels = np.zeros((864, 1536, 3), dtype=np.uint8)
        jpeg = encode_jpeg(pixels, quality=80, colorspace="BGR")
        frame = SimpleNamespace(
            width=1536,
            height=864,
            channels=0,
            format="image/jpeg",
            data=jpeg,
        )
        tools, _, _ = _make_tools(frame)

        result = tools.get_image()

        resized = decode_jpeg(base64.b64decode(result["data"]), colorspace="BGR")
        self.assertEqual(resized.shape, (360, 640, 3))
        tools.cleanup()

    def test_malformed_raw_frame_is_rejected(self) -> None:
        frame = SimpleNamespace(
            width=3,
            height=2,
            channels=3,
            data=b"too short",
        )
        tools, _, _ = _make_tools(frame)

        with self.assertRaisesRegex(RuntimeError, "expected 18 bytes"):
            tools.get_image()
        tools.cleanup()


if __name__ == "__main__":
    unittest.main()
