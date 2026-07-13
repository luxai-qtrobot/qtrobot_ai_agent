"""
tool_call_simple_memory.py - Same tool-calling chat loop as tool_call_simple.py, but
using RobotMemory (working + short-term + long-term tiers) and LLMEngine instead of a
plain history list and inline round-loop, to exercise the memory system + reusable
engine end-to-end. search_memory/search_documents (tool/memory_tools.py) are both
backed by the same LongTermMemory instance - see the system prompt below for how the
model is guided to use them.

No ASR/TTS - terminal in, terminal out. "[SPEAK] ..." stands in for TTS.

Requirements:
    pip install openai luxai-magpie[mcp,video] tiktoken fastembed

Usage (from this file's directory):
    python tool_call_simple_memory.py
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
from memory.document_reader import DirectoryReader
from memory.long_term_memory import LongTermMemory
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

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"
LTM_CHAT_HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "long_term_chat_history.json"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools when they let you answer "
    "more accurately. Otherwise just reply in plain text. "
    "When you call a tool, you can include a short spoken reply in the same response "
    "(e.g. 'Sure, one moment!') acknowledging what you're doing where needed. "
    "Strip .xml from gesture names before calling gesture_file_play. "
    "You have background information available automatically in your context (recent "
    "conversation, and a running summary of older conversation) - you don't need a tool "
    "for that. Use search_memory only when the user references something further back "
    "than what's already in your context (e.g. 'what did I tell you last week?') and "
    "the summary doesn't already cover it. Use search_documents only when the user asks "
    "about something that might be in one of the loaded documents listed in your "
    "context - if no documents are listed, don't call it."
)


DOCUMENTS_SUMMARY = (
    "QTrobot product/hardware overview (Research, School, Home variants, SDK/Studio "
    "programming options), and a list of 40+ published research papers using QTrobot "
    "with abstracts (autism therapy, education, healthcare, assistive robotics)."
)


async def main():
    robot = Robot.connect_zmq(endpoint=ROBOT_ENDPOINT)
    Logger.info(f"Connected to {robot.robot_id} ({robot.robot_type}), SDK version: {robot.sdk_version}")
    robot.enable_plugin_zmq("realsense-driver", endpoint=f"tcp://{ROBOT_IP}:50750")


    # Single llm api client
    client = OpenAI(base_url=LLM_API_BASE, api_key="not-needed")

    # Robot memmory
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
    )

    local_tool_server = LocalToolServer([
        UserTools(robot), 
        MemoryTools(long_term)
        ])
    local_requester = ZMQRpcRequester(LOCAL_TOOLS_ENDPOINT)
    robot_requester = ZMQRpcRequester(ROBOT_ENDPOINT)

    try:
        async with (
            Client(McpTransport(local_requester)) as local_client,
            Client(McpTransport(robot_requester)) as robot_client,
        ):
            tool_engine = ToolEngine(
                sources={"local": local_client, "robot": robot_client},
                whitelists={"robot": ROBOT_TOOL_WHITELIST},
            )
            await tool_engine.discover()
            tool_engine.print_schemas(raw=False)

            engine = LLMEngine(client=client, model=LLM_MODEL, memory=memory, tool_engine=tool_engine)

            Logger.info("Tool-calling demo ready. Try: 'what time is it?', 'what do you see?', "
                        "'play the bye gesture', or (after a few turns) 'what did we just talk "
                        "about?' to exercise search_memory. '/mem' shows memory ('/mem raw' for "
                        "the exact JSON sent to the LLM).")
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
        local_requester.close()
        robot_requester.close()
        local_tool_server.terminate(timeout=1.0)
        robot.close()
        memory.flush_all()
        long_term.save(LTM_CHAT_HISTORY_PATH)


if __name__ == "__main__":
    Logger.set_level("DEBUG")
    asyncio.run(main())
