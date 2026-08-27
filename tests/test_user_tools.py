from __future__ import annotations

import base64
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
from simplejpeg import decode_jpeg, encode_jpeg, is_jpeg


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from luxai.magpie.schema import McpSchema
from tool.providers.user_tools import ARM_JOINTS, CAMERA_READ_TIMEOUT, UserTools


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
    def test_visual_search_temporarily_owns_head_and_pointing_controls(self) -> None:
        tools, _, _ = _make_tools()
        robot = tools._robot
        robot.kinematics = Mock()
        robot.talking_behavior = Mock()
        robot.talking_behavior.get_source_config.return_value = {
            "head_motion": True,
            "arm_motion": True,
        }
        robot.motor = Mock()
        attention = Mock()
        attention.paused.return_value = False
        tools._human_attention = attention

        look_handle = Mock()
        look_handle.result.return_value = True
        point_handle = Mock()
        point_handle.result.return_value = True
        robot.kinematics.look_at_pixel_async.return_value = look_handle
        robot.kinematics.aim_at_pixel_async.return_value = point_handle

        tools.hold_current_gaze(60)
        image = tools.look_at_pixel(420, 240, hold_seconds=60)
        tools.point_at_pixel(1, 1, hold_seconds=60)
        tools.cancel_point_at_pixel()
        tools.cancel_look_at_pixel()

        self.assertEqual(image["mimeType"], "image/jpeg")
        attention.pause.assert_called_once_with()
        attention.resume.assert_called_once_with()
        robot.kinematics.look_at_pixel_async.assert_called_once_with(
            424,
            240,
            depth=1.0,
            only_gaze=False,
            velocity=60.0,
        )
        robot.kinematics.aim_at_pixel_async.assert_called_once_with(
            424,
            479,
            depth=1.0,
            velocity=60.0,
        )
        robot.kinematics.look_at_point.assert_called_once_with(
            1.0,
            0.0,
            0.6,
            only_gaze=False,
            velocity=60.0,
        )
        robot.talking_behavior.set_source_config.assert_has_calls(
            [
                call("media_fg", head_motion=False),
                call("media_fg", arm_motion=False),
                call("media_fg", arm_motion=True),
                call("media_fg", head_motion=True),
            ]
        )
        self.assertEqual(
            robot.motor.home.call_args_list,
            [call(joint) for joint in ARM_JOINTS],
        )
        tools.cleanup()

    def test_gesture_returns_home_after_successful_completion(self) -> None:
        tools, _, _ = _make_tools()
        handle = Mock()
        gesture = Mock()
        gesture.play_file_async.return_value = handle
        motor = Mock()
        tools._robot = SimpleNamespace(gesture=gesture, motor=motor)

        result = tools.gesture_file_play("QT/bye")

        self.assertEqual(result, "Gesture 'QT/bye' started.")
        gesture.play_file_async.assert_called_once_with("QT/bye")
        motor.home_all.assert_not_called()

        completed = handle.add_done_callback.call_args.args[0]
        completed(handle)

        handle.result.assert_called_once_with()
        motor.home_all.assert_called_once_with()
        tools.cleanup()

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
        self.assertEqual(resized.shape, (472, 840, 3))
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
