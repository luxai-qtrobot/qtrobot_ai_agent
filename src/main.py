"""
main.py - QTrobot AI agent, real ASR/TTS: ASR (Parakeet, robot SDK) -> LLMEngine
(memory + tool-calling) -> TTS (robot SDK), with barge-in support.

Requirements:
    pip install openai luxai-magpie[mcp,video] tiktoken fastembed

Usage (from this file's directory):
    python main.py
"""

import asyncio
from pathlib import Path

from openai import OpenAI

from fastmcp import Client
from luxai.magpie.adapters.mcp import McpTransport
from luxai.magpie.transport import ZMQRpcRequester
from luxai.magpie.utils import Logger
from luxai.robot.core import Robot

from tool.tool_engine import ToolEngine
from tool.local_tool_server import LOCAL_TOOLS_ENDPOINT, LocalToolServer
from tool.user_tools import UserTools
from tool.memory_tools import MemoryTools
from llm.llm_engine import LLMEngine
from speak import Speaker
from memory.document_reader import DirectoryReader
from memory.long_term_memory import LongTermMemory
from memory.robot_memory import RobotMemory
from memory.short_term_memory import ShortTermMemory
from memory.working_memory import WorkingMemory
from memory.world_state_memory import WorldStateMemory
from agents.agent_registry import AgentRegistry

ROBOT_IP = "192.168.3.111"
ROBOT_ENDPOINT = f"tcp://{ROBOT_IP}:50500"
PARAKEET_ENDPOINT = f"tcp://{ROBOT_IP}:50860"
REALSENSE_ENDPOINT = f"tcp://{ROBOT_IP}:50750"

# tool_name -> cancel_tool_name | None - see ToolEngine.cancel_all(). Left None for
# now (fill in the robot's real cancel-service tool names, e.g. gesture_file_play ->
# gesture_file_cancel, when this connection is actually re-enabled).
ROBOT_TOOL_WHITELIST = {
    "tts_engine_languages_get": None,    
    "face_emotion_list": None,
    "face_emotion_show": None,
    "face_emotion_stop": None,
    "gesture_cancel": None,
    "gesture_file_list": None,
    "gesture_file_play": None,
    "motor_move_home_all": None,
    "speaker_volume_get": None,
    "speaker_volume_set": None,
}

LLM_API_BASE = "http://192.168.3.111:8080/v1"
# LLM_API_BASE = "https://api.anthropic.com/v1/"
LLM_MODEL = "gemma-4-12B-it-Q8_0.gguf"
# LLM_MODEL = "Qwen3VL-8B-Instruct-Q8_0.gguf"
# LLM_MODEL = "claude-sonnet-4-6"
MEMORY_MAX_TOKENS = 31000  # 31k

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"
LTM_CHAT_HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "long_term_chat_history.json"
SYSTEM_PROMPT = '''
You are QTrobot, a friendly social robot.
You are the robot itself, never an AI assistant or software controlling the robot.
Speak naturally as QTrobot. Keep spoken responses brief and conversational. Prefer one short sentence. Avoid long explanations, lists, repetition, emojis, and unnecessary details.
'''


DOCUMENTS_SUMMARY = (
    "QTrobot product/hardware overview (Research, School, Home variants, SDK/Studio "
    "programming options), and a list of 40+ published research papers using QTrobot "
    "with abstracts (autism therapy, education, healthcare, assistive robotics)."
)


async def main():
    robot = Robot.connect_zmq(endpoint=ROBOT_ENDPOINT)
    Logger.info(f"Connected to {robot.robot_id} ({robot.robot_type}), SDK version: {robot.sdk_version}")
    robot.enable_plugin_zmq("realsense-driver", endpoint=REALSENSE_ENDPOINT)
    robot.speaker.set_volume(0.8)
    robot.motor.home_all()
    robot.tts.set_default_engine("azure")
    
    tuning_params = { "AGCONOFF": 0, "AGCGAIN": 30, }
    for name, value in tuning_params.items():        
        ok = robot.microphone.set_int_tuning(name=name, value=value)
        Logger.info(f"set_int_tuning({name}, {value}) result: {ok}")


    client = OpenAI(base_url=LLM_API_BASE, api_key="empty")

    long_term = LongTermMemory()
    long_term.load(LTM_CHAT_HISTORY_PATH)
    for doc in DirectoryReader.read(DOCUMENTS_DIR, summary=DOCUMENTS_SUMMARY):
        long_term.add_document(text=doc.text, summary=doc.summary, meta=doc.meta)
        Logger.info(f"Loaded document into long-term memory: {doc.meta['source']}")

    memory = RobotMemory(
        static=SYSTEM_PROMPT,
        max_tokens=MEMORY_MAX_TOKENS,
        working=WorkingMemory(size_ratio=0.75, flush_ratio=0.2),
        short_term=ShortTermMemory(llm=client, model=LLM_MODEL, size_ratio=0.25, flush_ratio=0.2),
        long_term=long_term,
        world_state=WorldStateMemory(),
    )

    agents = AgentRegistry(client, LLM_MODEL)
    local_tool_server = LocalToolServer([
        UserTools(robot),
        MemoryTools(long_term),
        *agents.as_tools(),
    ])
    local_requester = ZMQRpcRequester(LOCAL_TOOLS_ENDPOINT)
    robot_requester = ZMQRpcRequester(ROBOT_ENDPOINT)

    try:
        async with (
            Client(McpTransport(local_requester, timeout=120.0)) as local_client,
            Client(McpTransport(robot_requester)) as robot_client,
        ):
            tool_engine = ToolEngine(
                sources={"local": local_client, "robot": robot_client},
                whitelists={"robot": ROBOT_TOOL_WHITELIST},
            )
            await tool_engine.discover()

            llm_engine = LLMEngine(client=client, model=LLM_MODEL, memory=memory, tool_engine=tool_engine)
            speaker = Speaker(robot, enable_barge_in=True, on_barge_in=llm_engine.interrupt, engine="azure")

            def on_parakeet_speech(speech):
                data = speech.value or {}
                text = data.get("text", "")
                if not text:
                    return
                if not data.get("is_final"):
                    # Interim result: the user has started talking - treat as
                    # barge-in, but wait for the final transcript to submit anything.
                    speaker.on_interim_speech(text)
                    return
                if speaker.is_echo(text):
                    Logger.debug(f"Ignoring self-echo: {text!r}")
                    return
                Logger.info(f"You said: {text}")
                llm_engine.submit(text)

            robot.enable_plugin_local("asr-parakeet")
            robot.asr.configure_parakeet(
                endpoint=PARAKEET_ENDPOINT,
                language="en",
                use_vad=True,
                silence_timeout=0.3,
                max_buffer_seconds=20.0,
                continuous_mode=True,
            )
            robot.asr.stream.on_parakeet_speech(on_parakeet_speech)
            # robot.asr.stream.on_parakeet_event(lambda event: Logger.debug(event.value))

            Logger.info("QTrobot conversation ready. Listening for speech... (Ctrl+C to stop)")
            try:
                async for item in llm_engine.output():
                    if item is None:
                        continue
                    try:
                        Logger.info(f"tts: {item}")
                        await speaker.talk(item)
                    except Exception as e:
                        Logger.error(f"TTS error: {e}")
            except KeyboardInterrupt:
                Logger.info("Interrupted by user.")
            finally:
                llm_engine.shutdown()
    finally:
        local_requester.close()
        robot_requester.close()
        local_tool_server.terminate(timeout=1.0)
        agents.cleanup()
        robot.close()
        memory.flush_all()
        long_term.save(LTM_CHAT_HISTORY_PATH)


if __name__ == "__main__":
    Logger.set_level("DEBUG")
    asyncio.run(main())
