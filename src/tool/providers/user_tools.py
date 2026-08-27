"""Small application-owned tools exposed through the local MCP server."""

import base64
import threading
from datetime import datetime
from typing import Any, Callable

import numpy as np
from PIL import Image
from simplejpeg import decode_jpeg, encode_jpeg, is_jpeg

from luxai.magpie.schema import McpSchema
from luxai.magpie.utils import Logger
from luxai.robot.core import ActionHandle, Robot

from ..tool_base import ToolBase


CAMERA_READ_TIMEOUT = 2.0
CAMERA_JPEG_QUALITY = 90
CAMERA_IMAGE_MAX_SIZE = (840, 480)
KINEMATICS_CAMERA_SIZE = (848, 480)
DEFAULT_LOOK_HOLD_SECONDS = 15.0
DEFAULT_POINT_HOLD_SECONDS = 8.0
MAX_HOLD_SECONDS = 120.0
KINEMATICS_ACTION_TIMEOUT = 25.0
KINEMATICS_VELOCITY = 60.0
FRONT_LOOK_TARGET = (1.0, 0.0, 0.6)
ARM_JOINTS = (
    "LeftShoulderPitch",
    "LeftShoulderRoll",
    "LeftElbowRoll",
    "RightShoulderPitch",
    "RightShoulderRoll",
    "RightElbowRoll",
)


class _TimedControlLease:
    """Acquire robot control once and release it after an extendable timeout."""

    def __init__(
        self,
        acquire: Callable[[], None],
        release: Callable[[], None],
    ) -> None:
        self._acquire = acquire
        self._release = release
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None
        self._generation = 0
        self._active = False

    def hold(self) -> None:
        with self._lock:
            self._generation += 1
            self._cancel_timer()
            if self._active:
                return
            self._acquire()
            self._active = True

    def release_after(self, seconds: float) -> None:
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._cancel_timer()
            timer = threading.Timer(
                seconds,
                self._release_if_current,
                args=(generation,),
            )
            timer.daemon = True
            self._timer = timer
            timer.start()

    def release(self) -> None:
        with self._lock:
            self._generation += 1
            self._cancel_timer()
            self._release_locked()

    def _release_if_current(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._timer = None
            self._release_locked()

    def _release_locked(self) -> None:
        if not self._active:
            return
        try:
            self._release()
        finally:
            self._active = False

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


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
    """Application-owned date/time, camera, and expressive robot tools."""

    def __init__(self, robot: Robot, human_attention: Any = None) -> None:
        super().__init__()
        self._robot = robot
        self._human_attention = human_attention
        self._camera_lock = threading.Lock()
        self._camera_reader = robot.camera.stream.open_color_reader(queue_size=1)
        self._last_image_geometry = CAMERA_IMAGE_MAX_SIZE
        self._look_call_lock = threading.Lock()
        self._point_call_lock = threading.Lock()
        self._actions_lock = threading.Lock()
        self._active_look: ActionHandle | None = None
        self._active_point: ActionHandle | None = None
        self._attention_was_paused = False
        self._head_motion_was_enabled: bool | None = None
        self._arm_motion_was_enabled: bool | None = None
        self._head_lease = _TimedControlLease(
            self._acquire_head_control,
            self._release_head_control,
        )
        self._arm_lease = _TimedControlLease(
            self._acquire_arm_control,
            self._release_arm_control,
        )

    def register(self, schema: McpSchema) -> None:
        schema.method()(self.get_datetime)
        schema.method()(self.get_image)
        schema.method()(self.look_at_pixel)
        schema.method()(self.cancel_look_at_pixel)
        schema.method()(self.point_at_pixel)
        schema.method()(self.cancel_point_at_pixel)
        schema.method()(self.face_emotion_show)
        schema.method()(self.gesture_file_play)

    def cancellations(self) -> dict[str, str]:
        return {
            "look_at_pixel": "cancel_look_at_pixel",
            "point_at_pixel": "cancel_point_at_pixel",
        }

    def get_datetime(self) -> str:
        """Get the current local date and time."""
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def face_emotion_show(self, emotion: str) -> str:
        """Start showing a facial emotion without waiting for it to finish."""
        handle = self._robot.face.show_emotion_async(emotion)
        self._log_action_failure(handle, f"facial emotion {emotion!r}")
        return f"Facial emotion {emotion!r} started."

    def gesture_file_play(self, gesture: str) -> str:
        """Start playing a gesture without waiting for it to finish."""
        handle = self._robot.gesture.play_file_async(gesture)

        def completed(action: ActionHandle) -> None:
            try:
                action.result()
            except Exception as exc:
                Logger.warning(f"QTrobot gesture {gesture!r} failed: {exc}")

            try:
                self._robot.motor.home_all()
            except Exception as exc:
                Logger.warning(
                    f"Could not return QTrobot home after gesture {gesture!r}: {exc}"
                )

        handle.add_done_callback(completed)
        return f"Gesture {gesture!r} started."

    def look_at_pixel(
        self,
        u: int,
        v: int,
        hold_seconds: float = DEFAULT_LOOK_HOLD_SECONDS,
    ) -> dict[str, str]:
        """Look at a pixel in the latest 840x480 view and return a fresh image.

        Pixel (0, 0) is the upper-left corner. The head remains directed at the
        new view for ``hold_seconds`` before returning forward and restoring
        normal human attention.
        """
        hold_seconds = self._validated_hold(hold_seconds)
        with self._look_call_lock:
            self._head_lease.hold()
            handle: ActionHandle | None = None
            try:
                target_u, target_v = self._kinematics_pixel(u, v)
                handle = self._robot.kinematics.look_at_pixel_async(
                    target_u,
                    target_v,
                    depth=1.0,
                    only_gaze=False,
                    velocity=KINEMATICS_VELOCITY,
                )
                self._set_active_action("look", handle)
                completed = handle.result(timeout=KINEMATICS_ACTION_TIMEOUT)
                if completed is False:
                    raise RuntimeError("QTrobot could not complete the look action")
                image = self.get_image()
            except Exception:
                if handle is not None:
                    self._cancel_active_action("look")
                self._head_lease.release()
                raise
            finally:
                if handle is not None:
                    self._clear_active_action("look", handle)

            self._head_lease.release_after(hold_seconds)
            return image

    def cancel_look_at_pixel(self) -> str:
        """Cancel the current pixel-look action and restore normal attention."""
        self._cancel_active_action("look")
        self._head_lease.release()
        return "Pixel look cancelled."

    def point_at_pixel(
        self,
        u: int,
        v: int,
        hold_seconds: float = DEFAULT_POINT_HOLD_SECONDS,
    ) -> str:
        """Point toward a pixel in the latest view for a limited time."""
        hold_seconds = self._validated_hold(hold_seconds)
        with self._point_call_lock:
            self._arm_lease.hold()
            handle: ActionHandle | None = None
            try:
                target_u, target_v = self._kinematics_pixel(u, v)
                handle = self._robot.kinematics.aim_at_pixel_async(
                    target_u,
                    target_v,
                    depth=1.0,
                    velocity=KINEMATICS_VELOCITY,
                )
                self._set_active_action("point", handle)
                completed = handle.result(timeout=KINEMATICS_ACTION_TIMEOUT)
                if completed is False:
                    raise RuntimeError("QTrobot could not complete the pointing action")
            except Exception:
                if handle is not None:
                    self._cancel_active_action("point")
                self._arm_lease.release()
                raise
            finally:
                if handle is not None:
                    self._clear_active_action("point", handle)

            self._arm_lease.release_after(hold_seconds)
            return (
                f"QTrobot is pointing at pixel ({int(u)}, {int(v)}) for "
                f"{hold_seconds:g} seconds."
            )

    def cancel_point_at_pixel(self) -> str:
        """Cancel the current pointing action and return the arms home."""
        self._cancel_active_action("point")
        self._arm_lease.release()
        return "Pixel pointing cancelled."

    def hold_current_gaze(self, seconds: float) -> None:
        """Keep the current head target while a visual task is running."""
        seconds = self._validated_hold(seconds)
        self._head_lease.hold()
        self._head_lease.release_after(seconds)

    @staticmethod
    def _log_action_failure(handle: ActionHandle, description: str) -> None:
        def completed(action: ActionHandle) -> None:
            try:
                action.result()
            except Exception as exc:
                Logger.warning(f"QTrobot {description} failed: {exc}")

        handle.add_done_callback(completed)

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
            self._last_image_geometry = (width, height)

        Logger.info(
            f"Camera image captured: {width}x{height}, {len(jpeg)} JPEG bytes"
        )
        return {
            "mimeType": "image/jpeg",
            "data": base64.b64encode(jpeg).decode("ascii"),
        }

    @staticmethod
    def _validated_hold(value: float) -> float:
        try:
            seconds = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("hold_seconds must be a number") from exc
        if not 1.0 <= seconds <= MAX_HOLD_SECONDS:
            raise ValueError(
                f"hold_seconds must be between 1 and {MAX_HOLD_SECONDS:g}"
            )
        return seconds

    def _kinematics_pixel(self, u: int, v: int) -> tuple[int, int]:
        if isinstance(u, bool) or isinstance(v, bool):
            raise ValueError("u and v must be integer pixel coordinates")
        try:
            u = int(u)
            v = int(v)
        except (TypeError, ValueError) as exc:
            raise ValueError("u and v must be integer pixel coordinates") from exc

        with self._camera_lock:
            width, height = self._last_image_geometry
        if not 0 <= u < width or not 0 <= v < height:
            raise ValueError(
                f"pixel must be inside the latest {width}x{height} image"
            )

        target_width, target_height = KINEMATICS_CAMERA_SIZE
        target_u = round(u * (target_width - 1) / max(1, width - 1))
        target_v = round(v * (target_height - 1) / max(1, height - 1))
        return target_u, target_v

    def _set_active_action(self, kind: str, handle: ActionHandle) -> None:
        with self._actions_lock:
            if kind == "look":
                self._active_look = handle
            else:
                self._active_point = handle

    def _clear_active_action(self, kind: str, handle: ActionHandle) -> None:
        with self._actions_lock:
            if kind == "look" and self._active_look is handle:
                self._active_look = None
            elif kind == "point" and self._active_point is handle:
                self._active_point = None

    def _cancel_active_action(self, kind: str) -> None:
        with self._actions_lock:
            handle = self._active_look if kind == "look" else self._active_point
        if handle is None or handle.done():
            return
        try:
            handle.cancel(timeout=2.0)
        except Exception as exc:
            Logger.warning(f"Could not cancel QTrobot {kind} action: {exc}")

    def _acquire_head_control(self) -> None:
        attention = self._human_attention
        self._attention_was_paused = bool(
            attention is not None and attention.paused()
        )
        if attention is not None and not self._attention_was_paused:
            attention.pause()

        self._head_motion_was_enabled = None
        try:
            config = self._robot.talking_behavior.get_source_config("media_fg")
            self._head_motion_was_enabled = bool(config.get("head_motion"))
            if self._head_motion_was_enabled:
                self._robot.talking_behavior.set_source_config(
                    "media_fg",
                    head_motion=False,
                )
        except Exception as exc:
            Logger.warning(f"Could not suspend talking head motion: {exc}")

    def _release_head_control(self) -> None:
        try:
            self._robot.kinematics.look_at_point(
                *FRONT_LOOK_TARGET,
                only_gaze=False,
                velocity=KINEMATICS_VELOCITY,
            )
        except Exception as exc:
            Logger.warning(f"Could not return QTrobot's gaze forward: {exc}")
        try:
            if self._head_motion_was_enabled is not None:
                self._robot.talking_behavior.set_source_config(
                    "media_fg",
                    head_motion=self._head_motion_was_enabled,
                )
        except Exception as exc:
            Logger.warning(f"Could not restore talking head motion: {exc}")
        finally:
            attention = self._human_attention
            if attention is not None and not self._attention_was_paused:
                attention.resume()

    def _acquire_arm_control(self) -> None:
        self._arm_motion_was_enabled = None
        try:
            config = self._robot.talking_behavior.get_source_config("media_fg")
            self._arm_motion_was_enabled = bool(config.get("arm_motion"))
            if self._arm_motion_was_enabled:
                self._robot.talking_behavior.set_source_config(
                    "media_fg",
                    arm_motion=False,
                )
        except Exception as exc:
            Logger.warning(f"Could not suspend talking arm motion: {exc}")

    def _release_arm_control(self) -> None:
        for joint in ARM_JOINTS:
            try:
                self._robot.motor.home(joint)
            except Exception as exc:
                Logger.warning(f"Could not home QTrobot joint {joint}: {exc}")
        try:
            if self._arm_motion_was_enabled is not None:
                self._robot.talking_behavior.set_source_config(
                    "media_fg",
                    arm_motion=self._arm_motion_was_enabled,
                )
        except Exception as exc:
            Logger.warning(f"Could not restore talking arm motion: {exc}")

    def cleanup(self) -> None:
        self._cancel_active_action("look")
        self._cancel_active_action("point")
        self._head_lease.release()
        self._arm_lease.release()
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
