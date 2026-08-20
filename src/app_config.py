"""Paramify configuration and live-setting callbacks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from paramify.web import ParamifyWeb


ApplySetting = Callable[[str, Any], None]


class AppConfig(ParamifyWeb):
    """Forward live Paramify changes to the running application."""

    def __init__(self, config: str, **kwargs: Any) -> None:
        self._apply_setting: ApplySetting | None = None
        super().__init__(config, **kwargs)

    def bind(self, apply_setting: ApplySetting) -> None:
        self._apply_setting = apply_setting

    def unbind(self) -> None:
        self._apply_setting = None

    def _apply(self, name: str, value: Any) -> None:
        if self._apply_setting is not None:
            self._apply_setting(name, value)

    def on_robot_volume_set(self, value: float) -> None:
        self._apply("volume", value)

    def on_s2s_voice_set(self, value: str) -> None:
        self._apply("voice", value)

    def on_assistant_instructions_set(self, value: str) -> None:
        self._apply("instructions", value)
