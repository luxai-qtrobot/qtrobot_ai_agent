"""Private camera and pixel-kinematics tools for object search."""

from __future__ import annotations

import threading

from luxai.magpie.schema import McpSchema

from tool.providers.user_tools import (
    DEFAULT_LOOK_HOLD_SECONDS,
    UserTools,
)
from tool.tool_base import ToolBase


SEARCH_TIMEOUT_SECONDS = 110.0


class ObjectSearchTools(ToolBase):
    """Share the application's camera and motion tools with the private agent."""

    def __init__(self, user_tools: UserTools) -> None:
        super().__init__()
        self._user_tools = user_tools
        self._image_lock = threading.Lock()
        self._latest_image: dict[str, str] | None = None

    def register(self, schema: McpSchema) -> None:
        schema.method()(self.get_image)
        schema.method()(self.look_at_pixel)
        schema.method()(self.cancel_look_at_pixel)
        schema.method()(self.point_at_pixel)
        schema.method()(self.cancel_point_at_pixel)

    def get_image(self) -> dict[str, str]:
        """Capture QTrobot's current 840x480 camera view."""
        self._user_tools.hold_current_gaze(SEARCH_TIMEOUT_SECONDS)
        return self._remember(self._user_tools.get_image())

    def look_at_pixel(
        self,
        u: int,
        v: int,
        hold_seconds: float = SEARCH_TIMEOUT_SECONDS,
    ) -> dict[str, str]:
        """Center a pixel from the latest image and return the resulting view."""
        return self._remember(
            self._user_tools.look_at_pixel(u, v, hold_seconds)
        )

    def cancel_look_at_pixel(self) -> str:
        """Cancel active head movement and release the gaze hold."""
        return self._user_tools.cancel_look_at_pixel()

    def point_at_pixel(
        self,
        u: int,
        v: int,
        hold_seconds: float = 15.0,
    ) -> str:
        """Point toward a confident object's center in the latest image."""
        return self._user_tools.point_at_pixel(u, v, hold_seconds)

    def cancel_point_at_pixel(self) -> str:
        """Cancel active pointing and return the arms home."""
        return self._user_tools.cancel_point_at_pixel()

    def cancellations(self) -> dict[str, str]:
        return self._user_tools.cancellations()

    def latest_image(self) -> dict[str, str] | None:
        with self._image_lock:
            return dict(self._latest_image) if self._latest_image else None

    def reset(self) -> None:
        """Forget imagery from the previous search."""
        with self._image_lock:
            self._latest_image = None

    def hold_final_view(self) -> None:
        """Keep the final search view briefly after the agent returns."""
        self._user_tools.hold_current_gaze(DEFAULT_LOOK_HOLD_SECONDS)

    def release_view(self) -> None:
        """Release head control immediately."""
        self._user_tools.cancel_look_at_pixel()

    def _remember(self, image: dict[str, str]) -> dict[str, str]:
        with self._image_lock:
            self._latest_image = dict(image)
        return image
