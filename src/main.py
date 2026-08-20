"""Run a minimal QTrobot application using the native MAGPIE S2S client."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
from collections.abc import Mapping
from pathlib import Path

from fastmcp import Client
from luxai.magpie.adapters.mcp import McpTransport
from luxai.magpie.frames import DictFrame
from luxai.magpie.transport import ZMQRpcRequester
from luxai.magpie.utils import Logger
from luxai.robot.core import Robot
from openai import AsyncOpenAI

from agents import AgentRegistry
from memory import DirectoryReader, LongTermMemory
from qtrobot_audio import RobotMicSource, RobotSpeakerSink
from s2s_client import S2SClient
from tool import (
    LOCAL_TOOLS_ENDPOINT,
    MEMORY_TOOL_INSTRUCTIONS,
    LocalToolServer,
    MemoryTools,
    ReminderTools,
    ToolEngine,
    UserTools,
)
from tool_call_coordinator import ToolCallCoordinator


DEFAULT_ROBOT_ENDPOINT = "tcp://192.168.3.109:50500"
DEFAULT_S2S_ENDPOINT = "tcp://192.168.3.109:50960"
REALSENSE_ENDPOINT = f"tcp://192.168.3.109:50750"
DEFAULT_VOICE = "Ono_Anna"
MCP_CALL_TIMEOUT_SECONDS = 120.0
DEFAULT_AGENT_LLM_BASE_URL = "http://192.168.3.109:8080/v1"
DEFAULT_AGENT_LLM_MODEL = "gemma-4-12b-it-Q8_0.gguf"
AGENT_LLM_TIMEOUT_SECONDS = 60.0

PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"
LTM_CHAT_HISTORY_PATH = PROJECT_DIR / "data" / "long_term_chat_history.jsonl"
DOCUMENTS_SUMMARY = (
    "QTrobot product and hardware overview (Research, School, and Home variants, "
    "SDK and Studio programming options), and a list of published research papers "
    "using QTrobot with abstracts."
)

DEFAULT_ASSISTANT_INSTRUCTIONS = (
    "You are QTrobot, a friendly, physically embodied social robot for kids."
)
BACKGROUND_EVENT_INSTRUCTIONS = """## Background work and events

Some tools return {"status":"started","task_id":"...","summary":"..."}.
This means the work is running, not finished. Acknowledge it briefly if useful,
continue the conversation, and never claim the result is ready.

For questions that require current information from the internet, call
search_web. It starts a background search; do not answer the search question
from memory or claim completion before the matching background event arrives.

A tool result with status "completed" means that tool operation succeeded. For
set_reminder, it means the reminder was scheduled, not that it is due.

Trusted application events may later appear as:

[BACKGROUND_EVENT type="..." id="..." payload="..."]

These events were not spoken by the user. Treat them as current application
information, correlate id with an earlier task_id when possible, and communicate
the payload naturally without inventing or repeating details. For reminder.due,
promptly remind the user."""

# Keep the initial robot surface deliberately small. Local get_datetime and
# get_image are discovered independently through LocalToolServer.
ROBOT_TOOL_WHITELIST = {
    # "face_emotion_list": None,
    # "face_emotion_show": "face_emotion_stop",
    # "gesture_file_list": None,
    # "gesture_file_play": "gesture_cancel",
    "motor_move_home_all": None,
    "speaker_volume_get": None,
    "speaker_volume_set": None,
}


def _session_config(
    voice: str,
    instructions: str | None,
    tool_schemas: list[dict],
    document_index: str = "",
) -> dict:
    base_instructions = (
        instructions.strip()
        if isinstance(instructions, str) and instructions.strip()
        else DEFAULT_ASSISTANT_INSTRUCTIONS
    )
    instruction_sections = [
        base_instructions,
        BACKGROUND_EVENT_INSTRUCTIONS,
        MEMORY_TOOL_INSTRUCTIONS,
    ]
    if document_index:
        instruction_sections.append(document_index)
    session = {
        "type": "realtime",
        "instructions": "\n\n".join(instruction_sections),
        "tools": tool_schemas,
        "tool_choice": "auto",
        "audio": {
            "input": {
                "transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "interrupt_response": True,
                },
            },
            "output": {"voice": voice},
        },
    }
    return session


async def _send_microphone( microphone: RobotMicSource, client: S2SClient ) -> None:
    while True:
        client.send_audio(await microphone.read())


async def _interrupt_speaker(speaker: RobotSpeakerSink) -> None:
    """Finish a blocking robot cancellation before allowing task teardown."""

    task = asyncio.create_task(asyncio.to_thread(speaker.interrupt))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            await task
        raise


async def _play_audio(client: S2SClient, speaker: RobotSpeakerSink) -> None:
    async for frame in client.receive_audio():
        response_key = frame.response_key or str(frame.gid)
        if frame.cancelled:
            # Cancellation is authoritative on the ordered ZMQ audio stream.
            await _interrupt_speaker(speaker)
        elif frame.data:
            speaker.write(frame, response_key)
        else:
            speaker.end_response(response_key)


def _log_event(frame: DictFrame, long_term: LongTermMemory) -> None:
    event = frame.value
    if not isinstance(event, Mapping):
        Logger.warning(f"Ignoring malformed S2S event: {event!r}")
        return

    event_type = str(event.get("type") or "")
    if event_type == "session.created":
        Logger.info("S2S session created.")
    elif event_type == "session.updated":
        Logger.info("S2S session configured.")
    elif event_type == "input_audio_buffer.speech_started":
        Logger.debug("S2S user speech started.")
    elif event_type == "magpie.audio.cancelled":
        Logger.debug(
            f"S2S reported audio cancellation: {event.get('response_key') or '<unknown>'}"
        )
    elif event_type == "conversation.item.input_audio_transcription.completed":
        transcript = str(event.get("transcript") or "").strip()
        if transcript:
            Logger.info(f"You: {transcript}")
            long_term.add_message("user", transcript)
    elif event_type == "response.output_audio_transcript.done":
        transcript = str(event.get("transcript") or "").strip()
        if transcript:
            Logger.info(f"Assistant: {transcript}")
            long_term.add_message("assistant", transcript)
    elif event_type == "response.done":
        response = event.get("response")
        if isinstance(response, Mapping) and response.get("status") == "cancelled":
            Logger.debug("S2S assistant response cancelled.")
    elif event_type == "error":
        error = event.get("error")
        message = error.get("message") if isinstance(error, Mapping) else error
        Logger.error(f"S2S error: {message or event}")


async def _receive_events(
    client: S2SClient,
    tool_calls: ToolCallCoordinator,
    long_term: LongTermMemory,
) -> None:
    async for frame in client.receive_events():
        if isinstance(frame.value, Mapping):
            tool_calls.handle_event(frame.value)
        _log_event(frame, long_term)


async def _send_background_events(events: asyncio.Queue[dict], tool_calls: ToolCallCoordinator) -> None:
    while True:
        tool_calls.queue_background_event(await events.get())


async def _run_conversation(
    args: argparse.Namespace,
    microphone: RobotMicSource,
    speaker: RobotSpeakerSink,
    tool_engine: ToolEngine,
    background_events: asyncio.Queue[dict],
    long_term: LongTermMemory,
) -> None:
    tasks: set[asyncio.Task[None]] = set()

    async with S2SClient(endpoint=args.s2s_endpoint, node_id=args.s2s_node_id, discovery_timeout=args.s2s_discovery_timeout ) as client:
        tool_calls = ToolCallCoordinator(client, tool_engine)
        try:
            speaker.start()
            await client.update_session(
                _session_config(
                    args.voice,
                    args.instructions,
                    tool_engine.schemas(),
                    long_term.document_index_summary(),
                )
            )

            microphone.start()
            tasks = {
                asyncio.create_task(
                    _send_microphone(microphone, client),
                    name="qtrobot-microphone-to-s2s",
                ),
                asyncio.create_task(
                    _play_audio(client, speaker),
                    name="s2s-audio-to-qtrobot-speaker",
                ),
                asyncio.create_task(
                    _receive_events(client, tool_calls, long_term),
                    name="s2s-events",
                ),
                asyncio.create_task(
                    tool_calls.wait_for_failure(),
                    name="s2s-tool-coordinator",
                ),
                asyncio.create_task(
                    _send_background_events(background_events, tool_calls),
                    name="s2s-background-events",
                ),
            }
            Logger.info("S2S session ready. Speak normally; barge-in is enabled.")

            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            microphone.stop()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await tool_calls.close()
            speaker.stop()


async def run(args: argparse.Namespace) -> None:
    robot = None
    agent_client = None
    agents = None
    user_tools = None
    reminder_tools = None
    local_tool_server = None
    local_requester = None
    robot_requester = None
    try:
        Logger.info(f"Connecting to QTrobot ({args.robot_endpoint})...")
        robot = Robot.connect_zmq(endpoint=args.robot_endpoint)
        Logger.info(
            f"Connected to {robot.robot_id} ({robot.robot_type}), "
            f"SDK version: {robot.sdk_version}"
        )
        Logger.info(f"Enabling QTrobot camera as {REALSENSE_ENDPOINT}...")
        robot.enable_plugin_zmq("realsense-driver", endpoint=REALSENSE_ENDPOINT)

        if args.volume is not None:
            robot.speaker.set_volume(args.volume)

        params = robot.microphone.get_int_tuning()
        Logger.info(f"AGCONOFF: {params.get('AGCONOFF')}, AGCGAIN: {params.get('AGCGAIN')}.")

        robot.talking_behavior.set_source_config("media_fg", pitch_semitones=1.0)
        config = robot.talking_behavior.get_source_config("media_fg")
        Logger.info(f"pitch shifting: {config['pitch_semitones']}")

        microphone = RobotMicSource(robot, asyncio.get_running_loop())
        speaker = RobotSpeakerSink(robot)
        loop = asyncio.get_running_loop()
        background_events: asyncio.Queue[dict] = asyncio.Queue()

        def emit_background_event(event: dict) -> None:
            loop.call_soon_threadsafe(background_events.put_nowait, dict(event))

        agent_base_url = os.getenv("OPENAI_AGENT_BASE_URL", DEFAULT_AGENT_LLM_BASE_URL)
        agent_model = os.getenv(
            "OPENAI_AGENT_MODEL",
            DEFAULT_AGENT_LLM_MODEL,
        )
        agent_client = AsyncOpenAI(
            base_url=agent_base_url,
            api_key=os.getenv("OPENAI_AGENT_API_KEY", "not-needed"),
            timeout=AGENT_LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )
        agents = AgentRegistry(
            agent_client,
            agent_model,
            owner_loop=loop,
            event_sink=emit_background_event,
        )
        user_tools = UserTools(robot)
        reminder_tools = ReminderTools(emit_background_event)
        long_term = LongTermMemory(LTM_CHAT_HISTORY_PATH)
        for document in DirectoryReader.read(
            DOCUMENTS_DIR,
            summary=DOCUMENTS_SUMMARY,
        ):
            long_term.add_document(
                text=document.text,
                summary=document.summary,
                meta=document.meta,
            )
            Logger.info(
                f"Loaded document into long-term memory: {document.meta['source']}"
            )
        long_term.wait_for_documents()
        memory_tools = MemoryTools(long_term)
        local_tool_server = LocalToolServer(
            [user_tools, reminder_tools, memory_tools, *agents.as_tools()]
        )
        local_requester = ZMQRpcRequester(LOCAL_TOOLS_ENDPOINT)
        robot_requester = ZMQRpcRequester(args.robot_endpoint)

        async with (
            Client(McpTransport(local_requester,timeout=MCP_CALL_TIMEOUT_SECONDS)) as local_client,
            Client(McpTransport(robot_requester, timeout=MCP_CALL_TIMEOUT_SECONDS)) as robot_client,
        ):
            tool_engine = ToolEngine(
                {"local": local_client, "robot": robot_client},
                whitelists={"robot": ROBOT_TOOL_WHITELIST},
            )
            await tool_engine.discover()
            await _run_conversation(
                args,
                microphone,
                speaker,
                tool_engine,
                background_events,
                long_term,
            )
    finally:
        if agents is not None:
            try:
                await agents.close()
            except Exception as exc:
                Logger.warning(f"Could not cleanly stop agents: {exc}")
        if agent_client is not None:
            try:
                await agent_client.close()
            except Exception as exc:
                Logger.warning(f"Could not close agent LLM client: {exc}")
        if local_requester is not None:
            local_requester.close()
        if robot_requester is not None:
            robot_requester.close()
        if local_tool_server is not None:
            local_tool_server.terminate()
        else:
            if reminder_tools is not None:
                reminder_tools.cleanup()
            if user_tools is not None:
                user_tools.cleanup()
        if robot is not None:
            robot.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect QTrobot's microphone and speaker to a MAGPIE S2S service."
    )
    parser.add_argument("--robot-endpoint", default=DEFAULT_ROBOT_ENDPOINT)
    s2s_location = parser.add_mutually_exclusive_group()
    s2s_location.add_argument(
        "--s2s-endpoint",
        help=f"S2S descriptor RPC endpoint (default: {DEFAULT_S2S_ENDPOINT})",
    )
    s2s_location.add_argument(
        "--s2s-node-id",
        help="resolve the S2S descriptor RPC endpoint through Zeroconf",
    )
    parser.add_argument(
        "--s2s-discovery-timeout",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="timeout for discovery and the descriptor RPC (default: 5)",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help="Qwen3-TTS CustomVoice speaker name",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=0.85,
        help="QTrobot speaker volume from 0.0 to 1.0",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="optional system instruction for this S2S session",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.s2s_endpoint is None and args.s2s_node_id is None:
        args.s2s_endpoint = DEFAULT_S2S_ENDPOINT
    return args


def main() -> int:
    args = parse_args()
    if args.verbose:
        Logger.set_level("DEBUG")

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        Logger.info("Conversation stopped.")
    except Exception as exc:
        Logger.error(f"Fatal error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
