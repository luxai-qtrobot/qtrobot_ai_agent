"""Minimal QTrobot microphone and speaker adapters for realtime PCM audio."""

from __future__ import annotations

import asyncio
import contextlib

from luxai.magpie.frames import AudioFrameRaw
from luxai.magpie.utils import Logger
from luxai.magpie.utils.common import get_uinque_id
from luxai.robot.core import Robot


MIC_SUBSCRIBER_QUEUE_SIZE = 10
MIC_ASYNC_QUEUE_SIZE = 50
SPEAKER_WRITER_QUEUE_SIZE = 100


class RobotMicSource:
    """Expose QTrobot's native microphone frames through an asyncio queue."""

    def __init__(self, robot: Robot, loop: asyncio.AbstractEventLoop) -> None:
        self._robot = robot
        self._loop = loop
        self._queue: asyncio.Queue[AudioFrameRaw] = asyncio.Queue(
            maxsize=MIC_ASYNC_QUEUE_SIZE
        )
        self._subscription = None

    def start(self) -> None:
        if self._subscription is not None:
            return
        self._subscription = self._robot.microphone.stream.on_int_audio_ch0(
            self._on_audio,
            queue_size=MIC_SUBSCRIBER_QUEUE_SIZE,
        )
        Logger.info("QTrobot microphone streaming started (mono PCM16, 16 kHz).")

    def _on_audio(self, frame: AudioFrameRaw) -> None:
        if not frame.data:
            return
        # Detach the queued frame from the SDK callback thread. The S2S client
        # will add its own session GID/ID without mutating this source frame.
        queued_frame = AudioFrameRaw(
            gid=frame.gid,
            id=frame.id,
            channels=frame.channels,
            sample_rate=frame.sample_rate,
            bit_depth=frame.bit_depth,
            format=frame.format,
            data=bytes(frame.data),
        )
        try:
            self._loop.call_soon_threadsafe(self._enqueue, queued_frame)
        except RuntimeError:
            # The asyncio loop is already closing.
            pass

    def _enqueue(self, frame: AudioFrameRaw) -> None:
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(frame)

    async def read(self) -> AudioFrameRaw:
        return await self._queue.get()

    def stop(self) -> None:
        if self._subscription is None:
            return
        with contextlib.suppress(Exception):
            self._subscription.cancel()
        self._subscription = None
        Logger.info("QTrobot microphone streaming stopped.")


class RobotSpeakerSink:
    """Play audio frames using one QTrobot foreground GID per response."""

    def __init__(self, robot: Robot) -> None:
        self._robot = robot
        self._writer = None
        self._response_id: str | None = None
        self._gid: str | int | None = None
        self._frame_id = 0

    def start(self) -> None:
        if self._writer is not None:
            return
        self._writer = self._robot.media.stream.open_fg_audio_stream_writer(
            queue_size=SPEAKER_WRITER_QUEUE_SIZE
        )
        Logger.info("QTrobot foreground audio stream opened.")

    def write(self, frame: AudioFrameRaw, response_id: str) -> None:
        if not isinstance(frame, AudioFrameRaw):
            raise TypeError(f"frame must be AudioFrameRaw, got {type(frame).__name__}")
        if not frame.data:
            return
        if self._writer is None:
            raise RuntimeError("QTrobot speaker stream is not open")

        if self._response_id != response_id:
            self.end_response()
            self._begin_response(response_id)
        self._write_frame(frame)

    def _begin_response(self, response_id: str) -> None:
        self._response_id = response_id
        self._gid = get_uinque_id()
        self._frame_id = 0
        Logger.debug(
            f"QTrobot speaker response started: response={response_id}, "
            f"gid={self._gid}"
        )

    def end_response(self, response_id: str | None = None) -> None:
        """Queue the empty end frame behind all audio for this response."""
        if self._response_id is None:
            return
        if response_id is not None and response_id != self._response_id:
            return

        finished_response = self._response_id
        finished_gid = self._gid
        self._write_frame(AudioFrameRaw(data=b""))
        self._response_id = None
        self._gid = None
        self._frame_id = 0
        Logger.debug(
            f"QTrobot speaker response ended: response={finished_response}, "
            f"gid={finished_gid}"
        )

    def interrupt(self) -> None:
        """Immediately discard queued foreground playback during barge-in."""
        try:
            self._robot.media.cancel_fg_audio_stream()
        except Exception as exc:
            Logger.warning(f"QTrobot speaker cancellation failed: {exc}")
        self.end_response()

    def _write_frame(self, source: AudioFrameRaw) -> None:
        if self._writer is None or self._gid is None:
            return

        self._frame_id += 1
        frame = AudioFrameRaw(
            gid=self._gid,
            id=self._frame_id,
            channels=source.channels,
            sample_rate=source.sample_rate,
            bit_depth=source.bit_depth,
            format=source.format,
            data=bytes(source.data),
        )
        self._writer.write(frame)

    def stop(self) -> None:
        if self._writer is None:
            return
        self.interrupt()
        with contextlib.suppress(Exception):
            self._writer.close()
        self._writer = None
        Logger.info("QTrobot foreground audio stream closed.")
