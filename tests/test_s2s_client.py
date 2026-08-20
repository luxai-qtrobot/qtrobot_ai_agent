from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import s2s_client as module
from luxai.magpie.frames import AudioFrameRaw, DictFrame


class _Writer:
    def __init__(self) -> None:
        self.writes: list[tuple[dict, str]] = []

    def write(self, frame: dict, topic: str) -> None:
        self.writes.append((frame, topic))


class _Reader:
    def __init__(self, frames: list[object]) -> None:
        self.frames = list(frames)

    def read(self, timeout: float):
        if not self.frames:
            raise TimeoutError
        return self.frames.pop(0), module.AUDIO_OUTPUT_TOPIC


def _routes() -> module._S2SStreamRoutes:
    def route(topic: str, direction: str, frame_type: str) -> module._ZmqStreamRoute:
        return module._ZmqStreamRoute(
            topic=topic,
            endpoint="tcp://127.0.0.1:1",
            direction=direction,
            frame_type=frame_type,
            delivery="reliable",
            queue_size=0,
        )

    return module._S2SStreamRoutes(
        rpc_endpoint="tcp://127.0.0.1:1",
        node_id="test-s2s",
        audio_input=route(module.AUDIO_INPUT_TOPIC, "in", "AudioFrameRaw"),
        audio_output=route(module.AUDIO_OUTPUT_TOPIC, "out", "S2SAudioFrame"),
        event_input=route(module.EVENT_INPUT_TOPIC, "in", "DictFrame"),
        event_output=route(module.EVENT_OUTPUT_TOPIC, "out", "DictFrame"),
    )


def _connected_client() -> module.S2SClient:
    client = module.S2SClient(endpoint="tcp://127.0.0.1:1")
    client._routes = _routes()
    client._session_gid = "session-1"
    client._connected = True
    client._session_ready.set()
    return client


class S2SClientValidationTests(unittest.TestCase):
    def test_requires_exactly_one_location(self) -> None:
        with self.assertRaises(ValueError):
            module.S2SClient()
        with self.assertRaises(ValueError):
            module.S2SClient(
                endpoint="tcp://127.0.0.1:50960",
                node_id="luxai-s2s-magpie",
            )

    def test_wildcard_endpoint_uses_descriptor_host(self) -> None:
        self.assertEqual(
            module._replace_wildcard_host("tcp://*:50961", "192.168.3.10"),
            "tcp://192.168.3.10:50961",
        )


class S2SClientAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_audio_reenvelopes_without_mutating_source(self) -> None:
        client = _connected_client()
        writer = _Writer()
        client._audio_input_writer = writer
        source = AudioFrameRaw(
            gid="microphone-gid",
            id=42,
            channels=1,
            sample_rate=16_000,
            bit_depth=16,
            data=b"\x01\x02",
        )

        client.send_audio(source)

        self.assertEqual(source.gid, "microphone-gid")
        self.assertEqual(source.id, 42)
        self.assertEqual(len(writer.writes), 1)
        payload, topic = writer.writes[0]
        self.assertEqual(topic, module.AUDIO_INPUT_TOPIC)
        self.assertEqual(payload["gid"], "session-1")
        self.assertEqual(payload["id"], 1)
        self.assertEqual(payload["data"], b"\x01\x02")

    async def test_update_ack_is_also_visible_to_event_consumer(self) -> None:
        client = _connected_client()
        writer = _Writer()
        client._event_input_writer = writer

        async def acknowledge() -> None:
            await asyncio.sleep(0)
            frame = DictFrame(
                gid="session-1",
                value={"type": "session.updated", "session": {"type": "realtime"}},
            )
            client._handle_internal_event(frame)
            client._enqueue_event(frame)

        ack = asyncio.create_task(acknowledge())
        await client.update_session({"type": "realtime"}, timeout=0.5)
        await ack

        sent, topic = writer.writes[0]
        self.assertEqual(topic, module.EVENT_INPUT_TOPIC)
        self.assertEqual(sent["value"]["type"], "session.update")

        events = client.receive_events()
        received = await anext(events)
        await events.aclose()
        self.assertEqual(received.value["type"], "session.updated")

    async def test_receive_audio_filters_stale_client_and_keeps_terminals(self) -> None:
        client = _connected_client()
        stale = module.S2SAudioFrame(
            gid="audio-old",
            client_gid="another-session",
            response_key="old",
            data=b"old",
        )
        audio = module.S2SAudioFrame(
            gid="audio-1",
            client_gid="session-1",
            response_key="response-1",
            data=b"pcm",
        )
        done = module.S2SAudioFrame(
            gid="audio-1",
            client_gid="session-1",
            response_key="response-1",
            data=b"",
        )
        cancelled = module.S2SAudioFrame(
            gid="audio-1",
            client_gid="session-1",
            response_key="response-1",
            data=b"",
            cancelled=True,
        )
        client._audio_output_reader = _Reader(
            [stale.to_dict(), audio.to_dict(), done.to_dict(), cancelled.to_dict()]
        )

        frames = client.receive_audio()
        self.assertEqual((await anext(frames)).data, b"pcm")
        self.assertEqual((await anext(frames)).data, b"")
        self.assertTrue((await anext(frames)).cancelled)
        await frames.aclose()


if __name__ == "__main__":
    unittest.main()
