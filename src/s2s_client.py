"""Reusable native MAGPIE client for the LuxAI S2S service.

The client owns service discovery, the four MAGPIE streams, and one S2S
session. It deliberately knows nothing about microphones, speakers, robots,
or application policy. Applications feed :class:`AudioFrameRaw` objects and
consume the audio and event output streams independently.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from luxai.magpie.frames import AudioFrameRaw, DictFrame, Frame
from luxai.magpie.transport import ZMQRpcRequester, ZmqStreamReader, ZmqStreamWriter
from luxai.magpie.utils import Logger
from luxai.magpie.utils.common import get_uinque_id


AUDIO_INPUT_TOPIC = "/s2s/audio/input"
AUDIO_OUTPUT_TOPIC = "/s2s/audio/output"
EVENT_INPUT_TOPIC = "/s2s/events/input"
EVENT_OUTPUT_TOPIC = "/s2s/events/output"

CLIENT_AUDIO_READER_QUEUE_SIZE = 256
CLIENT_EVENT_READER_QUEUE_SIZE = 4096
CLIENT_INPUT_WRITER_QUEUE_SIZE = 0

DEFAULT_DISCOVERY_TIMEOUT_SECONDS = 5.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_SESSION_TIMEOUT_SECONDS = 10.0
SESSION_UPDATE_RETRY_SECONDS = 2.0
SESSION_CLOSE_TIMEOUT_SECONDS = 2.0
SESSION_CLOSE_RETRY_SECONDS = 0.25
READER_POLL_SECONDS = 0.25

FrameId = str | int
_EVENT_STREAM_END = object()


@dataclass
class S2SAudioFrame(AudioFrameRaw):
    """Assistant PCM correlated with one S2S client and response.

    This is the currently deployed ZMQ wire format. A frame containing data is
    an audio chunk. An empty frame ends ``response_key``; when ``cancelled`` is
    true, the application should also discard buffered playback.
    """

    client_gid: FrameId | None = None
    response_key: str = ""
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class _ZmqStreamRoute:
    topic: str
    endpoint: str
    direction: str
    frame_type: str
    delivery: str
    queue_size: int


@dataclass(frozen=True, slots=True)
class _S2SStreamRoutes:
    rpc_endpoint: str
    node_id: str | None
    audio_input: _ZmqStreamRoute
    audio_output: _ZmqStreamRoute
    event_input: _ZmqStreamRoute
    event_output: _ZmqStreamRoute


def _tcp_endpoint_host(endpoint: str, *, label: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "tcp" or parsed.hostname is None:
        raise ValueError(f"{label} must be a tcp://HOST:PORT endpoint, got {endpoint!r}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} has an invalid TCP port: {endpoint!r}") from exc
    if port is None or parsed.path or parsed.query or parsed.fragment:
        raise ValueError(f"{label} must be a tcp://HOST:PORT endpoint, got {endpoint!r}")
    return parsed.hostname


def _replace_wildcard_host(endpoint: str, host: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "tcp" or parsed.hostname is None:
        raise ValueError(f"S2S stream endpoint must use tcp://HOST:PORT: {endpoint!r}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"S2S stream endpoint has an invalid port: {endpoint!r}") from exc
    if port is None or parsed.path or parsed.query or parsed.fragment:
        raise ValueError(f"S2S stream endpoint must use tcp://HOST:PORT: {endpoint!r}")
    if parsed.hostname not in {"*", "0.0.0.0", "::"}:
        return endpoint
    rendered_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"tcp://{rendered_host}:{port}"


def _resolve_node_endpoint(node_id: str, timeout: float) -> str:
    from luxai.magpie.discovery import ZconfDiscovery

    Logger.info(f"Resolving S2S node {node_id!r} with Zeroconf...")
    with ZconfDiscovery() as discovery:
        info = discovery.resolve_node(node_id, timeout=timeout)
        if info is None:
            raise TimeoutError(
                f"S2S node {node_id!r} was not discovered within {timeout:g} seconds"
            )
        host = discovery.pick_best_ip(info)
        if not host:
            raise RuntimeError(f"S2S node {node_id!r} advertised no usable address")
        rendered_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
        return f"tcp://{rendered_host}:{info.port}"


def _read_system_descriptor(endpoint: str, timeout: float) -> Mapping[str, Any]:
    requester = ZMQRpcRequester(endpoint, name="s2s-system-descriptor")
    try:
        reply = requester.call({"name": "", "args": {}}, timeout=timeout)
    finally:
        requester.close()

    if not isinstance(reply, Mapping):
        raise RuntimeError(
            f"S2S descriptor RPC returned {type(reply).__name__}, expected a mapping"
        )
    if reply.get("status") is not True:
        error = reply.get("error") or reply.get("response") or "unknown server error"
        raise RuntimeError(f"S2S descriptor RPC failed: {error}")
    descriptor = reply.get("response")
    if not isinstance(descriptor, Mapping):
        raise RuntimeError("S2S descriptor RPC response has no descriptor mapping")
    return descriptor


def _parse_stream_route(topic: object, value: object, default_host: str) -> _ZmqStreamRoute:
    if not isinstance(topic, str) or not topic:
        raise ValueError(f"S2S descriptor has an invalid stream topic: {topic!r}")
    if not isinstance(value, Mapping):
        raise ValueError(f"S2S stream {topic!r} descriptor must be a mapping")

    direction = value.get("direction")
    frame_type = value.get("frame_type")
    if direction not in {"in", "out"}:
        raise ValueError(f"S2S stream {topic!r} has invalid direction {direction!r}")
    if not isinstance(frame_type, str) or not frame_type:
        raise ValueError(f"S2S stream {topic!r} has no frame_type")

    transports = value.get("transports")
    if not isinstance(transports, Mapping):
        raise ValueError(f"S2S stream {topic!r} has no transports mapping")
    transport = transports.get("zmq")
    if not isinstance(transport, Mapping):
        raise ValueError(f"S2S stream {topic!r} does not advertise ZMQ")

    endpoint = transport.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError(f"S2S stream {topic!r} has no ZMQ endpoint")
    endpoint = _replace_wildcard_host(endpoint, default_host)

    delivery = transport.get("delivery", "reliable")
    if delivery not in {"reliable", "latest"}:
        raise ValueError(f"S2S stream {topic!r} has invalid delivery {delivery!r}")
    queue_size = transport.get("queue_size", 10)
    if isinstance(queue_size, bool) or not isinstance(queue_size, int) or queue_size < 0:
        raise ValueError(f"S2S stream {topic!r} has invalid queue_size {queue_size!r}")

    return _ZmqStreamRoute(
        topic=topic,
        endpoint=endpoint,
        direction=direction,
        frame_type=frame_type,
        delivery=delivery,
        queue_size=queue_size,
    )


def _select_route(
    routes: list[_ZmqStreamRoute],
    *,
    topic: str,
    direction: str,
    frame_type: str,
) -> _ZmqStreamRoute:
    matches = [route for route in routes if route.topic == topic]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {topic!r} stream in the S2S descriptor; "
            f"found {len(matches)}"
        )
    route = matches[0]
    if route.direction != direction or route.frame_type != frame_type:
        raise ValueError(
            f"S2S stream {topic!r} must be {direction!r} {frame_type}, got "
            f"{route.direction!r} {route.frame_type}"
        )
    return route


def _discover_streams(
    *, endpoint: str | None, node_id: str | None, timeout: float
) -> _S2SStreamRoutes:
    if (endpoint is None) == (node_id is None):
        raise ValueError("Provide exactly one of endpoint or node_id")
    if timeout <= 0:
        raise ValueError("S2S discovery timeout must be greater than zero")

    rpc_endpoint = endpoint or _resolve_node_endpoint(str(node_id), timeout)
    default_host = _tcp_endpoint_host(rpc_endpoint, label="S2S base endpoint")
    Logger.info(f"Reading S2S system descriptor from {rpc_endpoint}...")
    descriptor = _read_system_descriptor(rpc_endpoint, timeout)
    streams = descriptor.get("stream")
    if not isinstance(streams, Mapping):
        raise ValueError("S2S system descriptor has no stream mapping")

    parsed = [_parse_stream_route(topic, value, default_host) for topic, value in streams.items()]
    routes = _S2SStreamRoutes(
        rpc_endpoint=rpc_endpoint,
        node_id=str(descriptor.get("node_id")) if descriptor.get("node_id") else None,
        audio_input=_select_route(
            parsed, topic=AUDIO_INPUT_TOPIC, direction="in", frame_type="AudioFrameRaw"
        ),
        audio_output=_select_route(
            parsed, topic=AUDIO_OUTPUT_TOPIC, direction="out", frame_type="S2SAudioFrame"
        ),
        event_input=_select_route(
            parsed, topic=EVENT_INPUT_TOPIC, direction="in", frame_type="DictFrame"
        ),
        event_output=_select_route(
            parsed, topic=EVENT_OUTPUT_TOPIC, direction="out", frame_type="DictFrame"
        ),
    )
    Logger.info(
        f"Discovered S2S node {routes.node_id or '<unnamed>'}: "
        f"audio in={routes.audio_input.endpoint}, audio out={routes.audio_output.endpoint}, "
        f"events in={routes.event_input.endpoint}, events out={routes.event_output.endpoint}"
    )
    return routes


def _decode_frame(raw: object) -> Frame:
    if isinstance(raw, Frame):
        return raw
    if isinstance(raw, dict):
        return Frame.from_dict(raw)
    raise TypeError(f"Expected a MAGPIE frame, got {type(raw).__name__}")


class S2SClient:
    """One native MAGPIE S2S session.

    Waiting operations are asynchronous; immediate MAGPIE writes are
    synchronous. Only one consumer may iterate each output stream at a time.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        node_id: str | None = None,
        discovery_timeout: float = DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        if (endpoint is None) == (node_id is None):
            raise ValueError("Provide exactly one of endpoint or node_id")
        if discovery_timeout <= 0:
            raise ValueError("discovery_timeout must be greater than zero")
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be greater than zero")

        self._endpoint = endpoint
        self._node_id = node_id
        self._discovery_timeout = float(discovery_timeout)
        self._connect_timeout = float(connect_timeout)

        self._routes: _S2SStreamRoutes | None = None
        self._audio_input_writer: ZmqStreamWriter | None = None
        self._event_input_writer: ZmqStreamWriter | None = None
        self._audio_output_reader: ZmqStreamReader | None = None
        self._event_output_reader: ZmqStreamReader | None = None

        self._session_gid: FrameId = get_uinque_id()
        self._audio_frame_id = 0
        self._event_frame_id = 0
        self._connected = False
        self._session_ready = asyncio.Event()
        self._session_error: str | None = None
        self._session_close_accepted = asyncio.Event()
        # Event traffic is low-volume but semantically important (including
        # future tool calls), so the public queue never drops events.
        self._event_queue: asyncio.Queue[DictFrame | object] = asyncio.Queue()
        self._event_reader_task: asyncio.Task[None] | None = None
        self._event_reader_error: Exception | None = None
        self._audio_iterator_active = False
        self._event_iterator_active = False
        self._event_stream_ended = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def session_ready(self) -> bool:
        return self._session_ready.is_set() and self._session_error is None

    @property
    def session_gid(self) -> FrameId:
        return self._session_gid

    async def __aenter__(self) -> S2SClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    async def connect(self) -> None:
        """Discover the service, open all streams, and start event ingestion."""

        if self._connected:
            return
        self._routes = await asyncio.to_thread(
            _discover_streams,
            endpoint=self._endpoint,
            node_id=self._node_id,
            timeout=self._discovery_timeout,
        )
        self._reset_session_state()

        try:
            self._open_streams()
            # Direct ZMQ writers must be connected and subsequently used from
            # the same thread. Normally this takes only a fraction of a second.
            self._wait_for_input_streams()
            self._connected = True
            self._event_reader_task = asyncio.create_task(
                self._event_reader_loop(), name="s2s-magpie-events"
            )
        except BaseException:
            await self._close_streams()
            self._routes = None
            raise

    async def update_session(
        self,
        session: Mapping[str, Any],
        *,
        timeout: float = DEFAULT_SESSION_TIMEOUT_SECONDS,
    ) -> None:
        """Send ``session.update`` and wait for ``session.updated``."""

        self._require_connected()
        if not isinstance(session, Mapping):
            raise TypeError("session must be a mapping")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        self._session_ready.clear()
        self._session_error = None
        event = {"type": "session.update", "session": dict(session)}
        deadline = time.monotonic() + timeout

        while True:
            self.send_event(event)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"S2S session was not configured within {timeout:g} seconds"
                )
            try:
                await asyncio.wait_for(
                    self._session_ready.wait(),
                    timeout=min(SESSION_UPDATE_RETRY_SECONDS, remaining),
                )
            except asyncio.TimeoutError:
                continue

            if self._session_error is not None:
                raise RuntimeError(
                    f"S2S session configuration failed: {self._session_error}"
                )
            return

    def send_audio(self, frame: AudioFrameRaw) -> None:
        """Write one microphone frame without mutating the caller's frame."""

        self._require_ready()
        if not isinstance(frame, AudioFrameRaw):
            raise TypeError(f"frame must be AudioFrameRaw, got {type(frame).__name__}")
        if not frame.data:
            return
        if frame.format.upper() != "PCM" or frame.bit_depth != 16 or frame.channels != 1:
            raise ValueError(
                "S2S input requires mono PCM16; "
                f"received {frame.format}, {frame.channels} channel(s), {frame.bit_depth}-bit"
            )
        if frame.sample_rate <= 0:
            raise ValueError("AudioFrameRaw.sample_rate must be greater than zero")

        writer = self._audio_input_writer
        routes = self._routes
        if writer is None or routes is None:
            raise RuntimeError("S2S audio input stream is not open")

        self._audio_frame_id += 1
        outbound = AudioFrameRaw(
            gid=self._session_gid,
            id=self._audio_frame_id,
            channels=frame.channels,
            sample_rate=frame.sample_rate,
            bit_depth=frame.bit_depth,
            format=frame.format,
            data=bytes(frame.data),
        )
        writer.write(outbound.to_dict(), routes.audio_input.topic)

    def send_event(self, event: Mapping[str, Any]) -> None:
        """Write one control event to the active S2S session."""

        self._require_connected()
        if not isinstance(event, Mapping):
            raise TypeError("event must be a mapping")
        writer = self._event_input_writer
        routes = self._routes
        if writer is None or routes is None:
            raise RuntimeError("S2S event input stream is not open")

        self._event_frame_id += 1
        writer.write(
            DictFrame(
                gid=self._session_gid,
                id=self._event_frame_id,
                value=dict(event),
            ).to_dict(),
            routes.event_input.topic,
        )

    def cancel_response(self) -> None:
        self.send_event({"type": "response.cancel"})

    async def receive_audio(self) -> AsyncIterator[S2SAudioFrame]:
        """Yield assistant audio, terminal, and cancellation frames."""

        self._require_connected()
        if self._audio_iterator_active:
            raise RuntimeError("receive_audio() already has an active consumer")
        self._audio_iterator_active = True
        try:
            while self._connected:
                reader = self._audio_output_reader
                if reader is None:
                    return
                try:
                    result = await asyncio.to_thread(reader.read, READER_POLL_SECONDS)
                except TimeoutError:
                    continue
                except Exception:
                    if not self._connected:
                        return
                    raise
                if result is None:
                    continue
                raw, _topic = result
                frame = _decode_frame(raw)
                if not isinstance(frame, S2SAudioFrame):
                    Logger.warning(
                        "Ignoring non-S2SAudioFrame on S2S audio output: "
                        f"{type(frame).__name__}"
                    )
                    continue
                if frame.client_gid != self._session_gid:
                    Logger.debug(
                        f"Ignoring S2S audio for stale client gid={frame.client_gid}"
                    )
                    continue
                if not self.session_ready:
                    continue
                yield frame
        finally:
            self._audio_iterator_active = False

    async def receive_events(self) -> AsyncIterator[DictFrame]:
        """Yield session and Realtime-compatible S2S event frames."""

        self._require_connected()
        if self._event_iterator_active:
            raise RuntimeError("receive_events() already has an active consumer")
        self._event_iterator_active = True
        try:
            while True:
                item = await self._event_queue.get()
                if item is _EVENT_STREAM_END:
                    if self._event_reader_error is not None and self._connected:
                        raise RuntimeError(
                            f"S2S event stream failed: {self._event_reader_error}"
                        ) from self._event_reader_error
                    return
                if isinstance(item, DictFrame):
                    yield item
        finally:
            self._event_iterator_active = False

    async def close(self) -> None:
        """Cancel output, request session closure, and release all streams."""

        if not self._connected:
            await self._close_streams()
            return

        event_task = self._event_reader_task
        try:
            with contextlib.suppress(Exception):
                self.cancel_response()

            deadline = time.monotonic() + SESSION_CLOSE_TIMEOUT_SECONDS
            while (
                event_task is not None
                and not event_task.done()
                and not self._session_close_accepted.is_set()
            ):
                with contextlib.suppress(Exception):
                    self.send_event({"type": "magpie.session.close"})
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(
                        self._session_close_accepted.wait(),
                        timeout=min(SESSION_CLOSE_RETRY_SECONDS, remaining),
                    )
                except asyncio.TimeoutError:
                    continue

            if event_task is None or event_task.done():
                with contextlib.suppress(Exception):
                    self.send_event({"type": "magpie.session.close"})
        finally:
            self._connected = False
            if event_task is not None and not event_task.done():
                event_task.cancel()
                await asyncio.gather(event_task, return_exceptions=True)
            self._event_reader_task = None
            self._end_event_stream()
            await self._close_streams()
            self._routes = None
            self._session_ready.clear()

    def _reset_session_state(self) -> None:
        self._session_gid = get_uinque_id()
        self._audio_frame_id = 0
        self._event_frame_id = 0
        self._session_ready = asyncio.Event()
        self._session_error = None
        self._session_close_accepted = asyncio.Event()
        self._event_queue = asyncio.Queue()
        self._event_reader_error = None
        self._event_stream_ended = False

    def _open_streams(self) -> None:
        routes = self._routes
        if routes is None:
            raise RuntimeError("S2S stream routes have not been discovered")

        self._audio_output_reader = ZmqStreamReader(
            routes.audio_output.endpoint,
            topic=routes.audio_output.topic,
            queue_size=CLIENT_AUDIO_READER_QUEUE_SIZE,
            bind=False,
            delivery=routes.audio_output.delivery,
        )
        self._event_output_reader = ZmqStreamReader(
            routes.event_output.endpoint,
            topic=routes.event_output.topic,
            queue_size=CLIENT_EVENT_READER_QUEUE_SIZE,
            bind=False,
            delivery=routes.event_output.delivery,
        )
        self._audio_input_writer = ZmqStreamWriter(
            routes.audio_input.endpoint,
            queue_size=CLIENT_INPUT_WRITER_QUEUE_SIZE,
            bind=False,
            delivery=routes.audio_input.delivery,
        )
        self._event_input_writer = ZmqStreamWriter(
            routes.event_input.endpoint,
            queue_size=CLIENT_INPUT_WRITER_QUEUE_SIZE,
            bind=False,
            delivery=routes.event_input.delivery,
        )

    def _wait_for_input_streams(self) -> None:
        if self._audio_input_writer is None or self._event_input_writer is None:
            raise RuntimeError("S2S input streams are not open")
        deadline = time.monotonic() + self._connect_timeout
        audio_ready = self._audio_input_writer.wait_connect(self._connect_timeout)
        remaining = max(0.0, deadline - time.monotonic())
        events_ready = self._event_input_writer.wait_connect(remaining)
        if not (audio_ready and events_ready):
            raise TimeoutError(
                "MAGPIE S2S input streams did not connect within "
                f"{self._connect_timeout:g} seconds"
            )

    async def _event_reader_loop(self) -> None:
        reader = self._event_output_reader
        if reader is None:
            return
        try:
            while self._connected:
                try:
                    result = await asyncio.to_thread(reader.read, READER_POLL_SECONDS)
                except TimeoutError:
                    continue
                if result is None:
                    continue
                raw, _topic = result
                frame = _decode_frame(raw)
                if not isinstance(frame, DictFrame):
                    Logger.warning(
                        "Ignoring non-DictFrame on S2S event output: "
                        f"{type(frame).__name__}"
                    )
                    continue
                if frame.gid != self._session_gid:
                    Logger.debug(f"Ignoring S2S event for stale client gid={frame.gid}")
                    continue
                self._handle_internal_event(frame)
                self._enqueue_event(frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._connected:
                self._event_reader_error = exc
                if not self._session_ready.is_set():
                    self._session_error = str(exc)
                    self._session_ready.set()
                Logger.error(f"S2S event reader failed: {exc}")
        finally:
            self._end_event_stream()

    def _handle_internal_event(self, frame: DictFrame) -> None:
        event = frame.value
        if not isinstance(event, Mapping):
            return
        event_type = event.get("type")
        if event_type == "session.updated":
            self._session_ready.set()
        elif event_type == "magpie.session.closing":
            self._session_close_accepted.set()
        elif event_type == "magpie.session.closed":
            self._session_close_accepted.set()
        elif event_type == "error" and not self._session_ready.is_set():
            error = event.get("error")
            if isinstance(error, Mapping):
                message = str(error.get("message") or error)
            else:
                message = str(error or event)
            self._session_error = message
            self._session_ready.set()

    def _enqueue_event(self, frame: DictFrame) -> None:
        self._event_queue.put_nowait(frame)

    def _end_event_stream(self) -> None:
        if self._event_stream_ended:
            return
        self._event_stream_ended = True
        self._event_queue.put_nowait(_EVENT_STREAM_END)

    async def _close_streams(self) -> None:
        writers = (self._audio_input_writer, self._event_input_writer)
        readers = (self._audio_output_reader, self._event_output_reader)
        self._audio_input_writer = None
        self._event_input_writer = None
        self._audio_output_reader = None
        self._event_output_reader = None

        for writer in writers:
            if writer is not None:
                with contextlib.suppress(Exception):
                    writer.close()
        await asyncio.gather(
            *(asyncio.to_thread(reader.close) for reader in readers if reader is not None),
            return_exceptions=True,
        )

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("S2SClient is not connected")

    def _require_ready(self) -> None:
        self._require_connected()
        if not self.session_ready:
            raise RuntimeError("S2S session is not configured")


__all__ = ["S2SAudioFrame", "S2SClient"]
