"""Small application-owned tools exposed through the local MCP server."""

import base64
import threading
from datetime import datetime

import numpy as np
from PIL import Image
from simplejpeg import decode_jpeg, encode_jpeg, is_jpeg

from luxai.magpie.schema import McpSchema
from luxai.magpie.utils import Logger
from luxai.robot.core import Robot

from ..tool_base import ToolBase


CAMERA_READ_TIMEOUT = 2.0
CAMERA_JPEG_QUALITY = 80
CAMERA_IMAGE_MAX_SIZE = (640, 480)


def _resize_to_camera_limit(image: np.ndarray) -> np.ndarray:
    """Shrink an image to the configured bounding box without cropping."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"Invalid decoded color image shape: {image.shape!r}")

    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid decoded color image size: {width}x{height}")

    max_width, max_height = CAMERA_IMAGE_MAX_SIZE
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image

    target_size = (
        max(1, min(max_width, round(width * scale))),
        max(1, min(max_height, round(height * scale))),
    )
    resized = Image.fromarray(image).resize(target_size, Image.Resampling.LANCZOS)
    return np.ascontiguousarray(np.asarray(resized))


class UserTools(ToolBase):
    """Date/time and camera tools owned by this application."""

    def __init__(self, robot: Robot) -> None:
        super().__init__()
        self._camera_lock = threading.Lock()
        self._camera_reader = robot.camera.stream.open_color_reader(queue_size=1)

    def register(self, schema: McpSchema) -> None:
        schema.method()(self.get_datetime)
        schema.method()(self.get_image)

    def get_datetime(self) -> str:
        """Get the current local date and time."""
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def get_image(self) -> dict[str, str]:
        """Capture the current view from QTrobot's color camera.

        Use this when answering requires seeing the robot's present physical
        surroundings.
        """
        with self._camera_lock:
            reader = self._camera_reader
            if reader is None:
                raise RuntimeError("The QTrobot camera reader is closed")

            frame = reader.read(timeout=CAMERA_READ_TIMEOUT)
            if frame is None:
                raise RuntimeError("Timed out waiting for a QTrobot camera frame")

            width = int(frame.width)
            height = int(frame.height)
            frame_format = getattr(frame, "format", "")
            format_values = (
                frame_format,
                getattr(frame_format, "name", ""),
                getattr(frame_format, "value", ""),
            )
            jpeg_format = any(
                str(value).lower().rsplit(".", 1)[-1]
                in {"jpeg", "jpg", "image/jpeg"}
                for value in format_values
            )

            if jpeg_format:
                jpeg = bytes(frame.data)
                if not is_jpeg(jpeg):
                    raise RuntimeError(
                        "QTrobot camera frame declares JPEG format but does not "
                        "contain a valid JPEG image"
                    )
                try:
                    image = decode_jpeg(jpeg, colorspace="BGR")
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not decode QTrobot camera JPEG: {exc}"
                    ) from exc
            else:
                channels = int(frame.channels)
                if width <= 0 or height <= 0 or channels != 3:
                    raise RuntimeError(
                        "Invalid QTrobot color frame: "
                        f"{width}x{height} with {channels} channels"
                    )

                expected_bytes = width * height * channels
                if len(frame.data) != expected_bytes:
                    raise RuntimeError(
                        "Invalid QTrobot color frame size: "
                        f"expected {expected_bytes} bytes, got {len(frame.data)}"
                    )

                # QTrobot color frames are interleaved BGR.
                image = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                    height,
                    width,
                    channels,
                )

            resized_image = _resize_to_camera_limit(image)
            if not jpeg_format or resized_image is not image:
                jpeg = encode_jpeg(
                    resized_image,
                    quality=CAMERA_JPEG_QUALITY,
                    colorspace="BGR",
                )
            height, width = resized_image.shape[:2]

        Logger.info(
            f"Camera image captured: {width}x{height}, {len(jpeg)} JPEG bytes"
        )
        return {
            "mimeType": "image/jpeg",
            "data": base64.b64encode(jpeg).decode("ascii"),
        }

    def cleanup(self) -> None:
        with self._camera_lock:
            reader = self._camera_reader
            if reader is None:
                return
            try:
                reader.close()
            except Exception as exc:
                Logger.warning(f"Could not close camera reader: {exc}")
            finally:
                self._camera_reader = None
