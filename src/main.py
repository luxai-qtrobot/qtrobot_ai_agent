"""Run a minimal QTrobot application using the native MAGPIE S2S client."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
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
from app_config import AppConfig
from behaviors import HumanAttentionBehavior
from memory import DirectoryReader, LongTermMemory
from qtrobot_audio import RobotMicSource, RobotSpeakerSink
from s2s import S2SClient, ToolCallCoordinator
from s2s._internal_instructions import (
    BACKGROUND_EVENT_INSTRUCTIONS,
    CAMERA_INSTRUCTIONS,
    EMBODIED_INTERACTION_INSTRUCTIONS,
    WEB_SEARCH_INSTRUCTIONS,
)
from tool import (
    LOCAL_TOOLS_ENDPOINT,
    LocalToolServer,
    MemoryTools,
    ReminderTools,
    ToolEngine,
    UserTools,
    memory_tool_instructions,
)


HUMAN_IDLE_ATTENTION_TIMEOUT   = 8.0
MCP_CALL_TIMEOUT_SECONDS       = 120.0
DEFAULT_AGENT_LLM_BASE_URL     = "http://127.0.0.1:8080/v1"
DEFAULT_AGENT_LLM_MODEL        = "Qwen3.5-9B-Q8_0.gguf"
AGENT_LLM_TIMEOUT_SECONDS      = 60.0
RUNTIME_SETTING_TIMEOUT_SECONDS = 15.0

# Keep the initial robot surface deliberately small. Local get_datetime and
# get_image are discovered independently through LocalToolServer.
ROBOT_TOOL_WHITELIST = {
    "face_emotion_list": None,
    "gesture_file_list": None,    
    "speaker_volume_get": None,
    "speaker_volume_set": None,
}


def _resolve_project_path(value: object) -> Path:
    project_dir = Path(__file__).resolve().parent.parent
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else project_dir / path


def _session_config(
    voice: str,
    instructions: str,
    tool_schemas: list[dict],
    document_index: str = "",
    *,
    web_search_enabled: bool = False,
    memory_enabled: bool = False,
    documents_enabled: bool = False,
) -> dict:
    base_instructions = instructions.strip()
    if not base_instructions:
        raise ValueError("assistant.instructions must not be empty")

    instruction_sections = [
        base_instructions,
        BACKGROUND_EVENT_INSTRUCTIONS,
        CAMERA_INSTRUCTIONS,
        EMBODIED_INTERACTION_INSTRUCTIONS,
    ]
    if web_search_enabled:
        instruction_sections.append(WEB_SEARCH_INSTRUCTIONS)
    retrieval_instructions = memory_tool_instructions(
        memory_enabled=memory_enabled,
        documents_enabled=documents_enabled,
    )
    if retrieval_instructions:
        instruction_sections.append(retrieval_instructions)
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


async def _send_microphone(
    microphone: RobotMicSource,
    client: S2SClient,
    interaction_paused: asyncio.Event,
) -> None:
    while True:
        frame = await microphone.read()
        if not interaction_paused.is_set():
            client.send_audio(frame)


async def _interrupt_speaker(speaker: RobotSpeakerSink) -> None:
    """Finish a blocking robot cancellation before allowing task teardown."""

    task = asyncio.create_task(asyncio.to_thread(speaker.interrupt))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            await task
        raise


async def _play_audio(
    client: S2SClient,
    speaker: RobotSpeakerSink,
    interaction_paused: asyncio.Event,
) -> None:
    async for frame in client.receive_audio():
        if interaction_paused.is_set():
            continue
        response_key = frame.response_key or str(frame.gid)
        if frame.cancelled:
            # Cancellation is authoritative on the ordered ZMQ audio stream.
            await _interrupt_speaker(speaker)
        elif frame.data:
            speaker.write(frame, response_key)
        else:
            speaker.end_response(response_key)


def _log_event(frame: DictFrame, long_term: LongTermMemory | None) -> None:
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
            if long_term is not None:
                long_term.add_message("user", transcript)
    elif event_type == "response.output_audio_transcript.done":
        transcript = str(event.get("transcript") or "").strip()
        if transcript:
            Logger.info(f"Assistant: {transcript}")
            if long_term is not None:
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
    long_term: LongTermMemory | None,
) -> None:
    async for frame in client.receive_events():
        if isinstance(frame.value, Mapping):
            tool_calls.handle_event(frame.value)
        _log_event(frame, long_term)


async def _send_background_events(events: asyncio.Queue[dict], tool_calls: ToolCallCoordinator) -> None:
    while True:
        tool_calls.queue_background_event(await events.get())


async def _run_conversation(
    config: AppConfig,
    robot: Robot,
    microphone: RobotMicSource,
    speaker: RobotSpeakerSink,
    tool_engine: ToolEngine,
    background_events: asyncio.Queue[dict],
    long_term: LongTermMemory | None,
    document_index: str,
    web_search_available: bool,
    memory_enabled: bool,
    documents_enabled: bool,
) -> None:
    parameters = config.parameters
    tasks: set[asyncio.Task[None]] = set()

    async with S2SClient(endpoint=str(parameters.s2s.endpoint)) as client:
        tool_calls = ToolCallCoordinator(client, tool_engine)
        loop = asyncio.get_running_loop()
        interaction_paused = asyncio.Event()
        if bool(parameters.paused):
            interaction_paused.set()

        async def set_paused(paused: bool) -> None:
            if paused:
                interaction_paused.set()
                await _interrupt_speaker(speaker)
                Logger.info("Interaction paused.")
            else:
                interaction_paused.clear()
                Logger.info("Interaction resumed.")

        def apply_setting(name: str, value: object) -> None:
            if name == "volume":
                robot.speaker.set_volume(float(value) / 100.0)
                return
            if name == "paused":
                future = asyncio.run_coroutine_threadsafe(
                    set_paused(bool(value)),
                    loop,
                )
                future.result(timeout=RUNTIME_SETTING_TIMEOUT_SECONDS)
                return
            if name not in {"voice", "instructions"}:
                raise ValueError(f"Unsupported live setting: {name}")

            future = asyncio.run_coroutine_threadsafe(
                client.update_session(
                    _session_config(
                        str(parameters.s2s.voice),
                        str(parameters.assistant.instructions),
                        tool_engine.schemas(),
                        document_index,
                        web_search_enabled=web_search_available,
                        memory_enabled=memory_enabled,
                        documents_enabled=documents_enabled,
                    )
                ),
                loop,
            )
            future.result(timeout=RUNTIME_SETTING_TIMEOUT_SECONDS)

        try:
            speaker.start()
            await client.update_session(
                _session_config(
                    str(parameters.s2s.voice),
                    str(parameters.assistant.instructions),
                    tool_engine.schemas(),
                    document_index,
                    web_search_enabled=web_search_available,
                    memory_enabled=memory_enabled,
                    documents_enabled=documents_enabled,
                )
            )
            config.bind(apply_setting)

            microphone.start()
            tasks = {
                asyncio.create_task(
                    _send_microphone(microphone, client, interaction_paused),
                    name="qtrobot-microphone-to-s2s",
                ),
                asyncio.create_task(
                    _play_audio(client, speaker, interaction_paused),
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
            config.unbind()
            microphone.stop()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await tool_calls.close()
            speaker.stop()


async def run(config: AppConfig) -> None:
    parameters = config.parameters
    robot = None
    agent_client = None
    agents = None
    user_tools = None
    reminder_tools = None
    local_tool_server = None
    local_requester = None
    robot_requester = None
    human_attention = None
    long_term = None
    try:
        robot_endpoint = str(parameters.robot.endpoint)
        Logger.info(f"Connecting to QTrobot ({robot_endpoint})...")
        robot = Robot.connect_zmq(endpoint=robot_endpoint)
        Logger.info(
            f"Connected to {robot.robot_id} ({robot.robot_type}), "
            f"SDK version: {robot.sdk_version}"
        )
        camera_endpoint = str(parameters.camera.endpoint)
        Logger.info(f"Enabling QTrobot camera as {camera_endpoint}...")
        robot.enable_plugin_zmq("realsense-driver", endpoint=camera_endpoint)

        robot.speaker.set_volume(float(parameters.robot.volume) / 100.0)

        robot.microphone.set_int_tuning(name="AGCONOFF", value=0.0)
        robot.microphone.set_int_tuning(name="AGCGAIN", value=15.0)
        params = robot.microphone.get_int_tuning()
        Logger.info(f"AGCONOFF: {params.get('AGCONOFF')}, AGCGAIN: {params.get('AGCGAIN')}.")

        robot.talking_behavior.set_source_config(
            "media_fg",
            pitch_semitones=float(parameters.robot.pitch_semitones),
        )
        source_config = robot.talking_behavior.get_source_config("media_fg")
        Logger.info(f"pitch shifting: {source_config['pitch_semitones']}")

        robot.motor.home_all()

        if bool(parameters.human_attention.enabled):
            human_attention = HumanAttentionBehavior(
                robot,
                detector_endpoint=str(parameters.human_attention.detector_endpoint),
                idle_attention_timeout=HUMAN_IDLE_ATTENTION_TIMEOUT,
                look_velocity=60,
            )

        microphone = RobotMicSource(robot, asyncio.get_running_loop())
        speaker = RobotSpeakerSink(robot)
        loop = asyncio.get_running_loop()
        background_events: asyncio.Queue[dict] = asyncio.Queue()

        def emit_background_event(event: dict) -> None:
            loop.call_soon_threadsafe(background_events.put_nowait, dict(event))

        if bool(parameters.web_search.enabled):
            agent_base_url = os.getenv(
                "OPENAI_AGENT_BASE_URL",
                DEFAULT_AGENT_LLM_BASE_URL,
            )
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
                api_key=str(parameters.web_search.api_key).strip() or None,
            )

        user_tools = UserTools(robot)
        reminder_tools = ReminderTools(emit_background_event)
        memory_enabled = bool(parameters.memory.enabled)
        documents_enabled = bool(parameters.documents.enabled)
        documents_available = False
        memory_tools = None
        document_index = ""
        if memory_enabled or documents_enabled:
            history_path = (
                _resolve_project_path(parameters.memory.history_path)
                if memory_enabled
                else None
            )
            long_term = LongTermMemory(history_path)
            if documents_enabled:
                documents_directory = _resolve_project_path(
                    parameters.documents.directory
                )
                documents = DirectoryReader.read(
                    documents_directory,
                    summary=str(parameters.documents.summary),
                )
                documents_available = bool(documents)
                for document in documents:
                    long_term.add_document(
                        text=document.text,
                        summary=document.summary,
                        meta=document.meta,
                    )
                    Logger.info(
                        "Loaded document into long-term memory: "
                        f"{document.meta['source']}"
                    )
                if documents_available:
                    long_term.wait_for_documents()
                    document_index = long_term.document_index_summary()
            if memory_enabled or documents_available:
                memory_tools = MemoryTools(
                    long_term,
                    memory_enabled=memory_enabled,
                    documents_enabled=documents_available,
                )

        providers = [user_tools, reminder_tools]
        if memory_tools is not None:
            providers.append(memory_tools)
        if agents is not None:
            providers.extend(agents.as_tools())
        web_search_available = bool(agents is not None and agents.as_tools())
        local_tool_server = LocalToolServer(providers)
        local_requester = ZMQRpcRequester(LOCAL_TOOLS_ENDPOINT)
        robot_requester = ZMQRpcRequester(robot_endpoint)

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
                config,
                robot,
                microphone,
                speaker,
                tool_engine,
                background_events,
                long_term if memory_enabled else None,
                document_index,
                web_search_available,
                memory_enabled,
                documents_available,
            )
    finally:
        if human_attention is not None:
            human_attention.terminate(
                timeout=HUMAN_IDLE_ATTENTION_TIMEOUT + 1.0
            )
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
            robot.motor.home_all()
            robot.close()


def main() -> int:
    try:
        if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
            raise ValueError("Usage: python main.py <path-to-config.yaml> [options]")
        config_path = Path(sys.argv[1]).expanduser()
        config = AppConfig(str(config_path))
        Logger.set_level(str(config.parameters.log_level))
        Logger.info(f"QTrobot AI Agent starting with {config_path}")
        asyncio.run(run(config))
    except KeyboardInterrupt:
        Logger.info("Conversation stopped.")
    except Exception as exc:
        Logger.error(f"{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
