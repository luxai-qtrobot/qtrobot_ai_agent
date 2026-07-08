"""
tool_call_simple_memory.py - Same tool-calling chat loop as tool_call_simple.py, but
using RobotMemory (working + short-term tiers) and LLMEngine instead of a plain history
list and inline round-loop, to exercise the memory system + reusable engine end-to-end.

No ASR/TTS - terminal in, terminal out. "[SPEAK] ..." stands in for TTS.

Requirements:
    pip install openai luxai-magpie[mcp,video] tiktoken

Usage (from this file's directory):
    python tool_call_simple_memory.py
"""

import asyncio

from openai import OpenAI

from fastmcp import Client
from luxai.magpie.adapters.mcp import McpTransport
from luxai.magpie.transport import ZMQRpcRequester
from luxai.magpie.utils import Logger
from luxai.robot.core import Robot

from agents.agent import Agent
from agents.user_agents import USER_TOOLS_ENDPOINT, UserAgents
from llm_engine import LLMEngine
from memory.robot_memory import RobotMemory
from memory.short_term_memory import ShortTermMemory
from memory.working_memory import WorkingMemory

ROBOT_IP = "192.168.3.111"
ROBOT_ENDPOINT = f"tcp://{ROBOT_IP}:50500"

ROBOT_TOOL_WHITELIST = {
    "tts_engine_languages_get",
    "tts_engines_list",
    "face_emotion_list",
    "face_emotion_show",
    "face_emotion_stop",
    "gesture_cancel",
    "gesture_file_list",
    "gesture_file_play",
    "motor_move_home_all",
    "speaker_volume_get",
    "speaker_volume_set",
}

LLM_API_BASE = "http://192.168.3.111:8080/v1"
LLM_MODEL = "gemma-4-12B-it-Q8_0.gguf"

MEMORY_MAX_TOKENS = 31000  # 31k

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools when they let you answer "
    "more accurately. Otherwise just reply in plain text. "
    "When you call a tool, you can include a short spoken reply in the same response "
    "(e.g. 'Sure, one moment!') acknowledging what you're doing where needed. "
    "Strip .xml from gesture names before calling gesture_file_play."
)


async def main():
    robot = Robot.connect_zmq(endpoint=ROBOT_ENDPOINT)
    Logger.info(f"Connected to {robot.robot_id} ({robot.robot_type}), SDK version: {robot.sdk_version}")
    robot.enable_plugin_zmq("realsense-driver", endpoint=f"tcp://{ROBOT_IP}:50750")

    user_agents = UserAgents(robot)
    user_requester = ZMQRpcRequester(USER_TOOLS_ENDPOINT)
    robot_requester = ZMQRpcRequester(ROBOT_ENDPOINT)

    # Single sync client, shared between the live streaming chat and short_term's
    # background summarization thread - the SDK's client is safe to call from multiple
    # threads at once, no need for a separate async client.
    client = OpenAI(base_url=LLM_API_BASE, api_key="not-needed")

    # Constructed before the try block (like the connections above) so it's always
    # defined by the time `finally` runs, even if something below fails early.
    memory = RobotMemory(
        static=SYSTEM_PROMPT,
        max_tokens=MEMORY_MAX_TOKENS,
        working=WorkingMemory(size_ratio=0.75, flush_ratio=0.2),
        short_term=ShortTermMemory(llm=client, model=LLM_MODEL, size_ratio=0.25, flush_ratio=0.2),
    )

    try:
        async with (
            Client(McpTransport(user_requester)) as user_client,
            Client(McpTransport(robot_requester)) as robot_client,
        ):
            agent = Agent(
                sources={"user": user_client, "robot": robot_client},
                whitelists={"robot": ROBOT_TOOL_WHITELIST},
            )
            await agent.discover()
            agent.print_schemas(raw=False)

            engine = LLMEngine(client=client, model=LLM_MODEL, memory=memory, agent=agent)

            Logger.info("Tool-calling demo ready. Try: 'what time is it?', 'what is 12 plus 30?', "
                        "'what do you see?', or 'play the bye gesture'. '/mem' shows memory "
                        "('/mem raw' for the exact JSON sent to the LLM).")
            while True:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break
                if user_input == "/mem":
                    memory.print()
                    continue
                if user_input == "/mem raw":
                    memory.print(raw=True)
                    continue

                async for text in engine.ask(user_input):
                    Logger.info(f"[SPEAK] {text}")
    finally:
        user_requester.close()
        robot_requester.close()
        user_agents.terminate(timeout=1.0)
        robot.close()
        memory.flush_all()


if __name__ == "__main__":
    Logger.set_level("DEBUG")
    asyncio.run(main())
