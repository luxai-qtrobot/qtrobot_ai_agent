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
from memory.world_state_memory import WorldStateMemory
from agents.agent_registry import AgentRegistry

ROBOT_IP = "192.168.3.111"
ROBOT_ENDPOINT = f"tcp://{ROBOT_IP}:50500"

# tool_name -> cancel_tool_name | None - see ToolEngine.cancel_all(). Left None for
# now (fill in the robot's real cancel-service tool names, e.g. gesture_file_play ->
# gesture_file_cancel, when this connection is actually re-enabled).
ROBOT_TOOL_WHITELIST = {
    "tts_engine_languages_get": None,
    "tts_engines_list": None,
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
#LLM_API_BASE = "https://api.anthropic.com/v1/"
LLM_MODEL = "gemma-4-12B-it-Q8_0.gguf"
# LLM_MODEL = "claude-sonnet-4-6"
MEMORY_MAX_TOKENS = 31000  # 31k

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"
LTM_CHAT_HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "long_term_chat_history.json"

SYSTEM_PROMPT = (
    "You are QTrobot, a friendly social robot. You are the robot itself, not an "
    "assistant controlling it. Use the available tools to perceive the environment, "
    "perform robot actions, search memory, and search user documents. Use tools when "
    "they improve accuracy or are needed to perform the user's request. Otherwise, "
    "reply in plain text. "

    "Keep every spoken response short and natural. Prefer one brief sentence. Avoid "
    "long sentences, explanations, lists, emojis, repetition, and unnecessary details. "

    "Before calling any tool that may take noticeable time or performs a visible robot "
    "action, you MUST first produce a short spoken acknowledgment in the same assistant "
    "turn, before the tool call. Examples include playing a gesture, moving, taking or "
    "analyzing a camera image, searching documents, and searching memory. Examples of "
    "acknowledgments are: 'Sure, one moment.' 'Let me check.' or 'I will play a happy "
    "gesture.' Never wait until after the action to acknowledge it. "

    "An acknowledgment by itself is never a complete response - if you say you will "
    "check, look up, play, or perform something, the matching tool call must be in "
    "that same turn too. Never stop right after the acknowledgment with no tool call. "

    "After a successful action, do not repeat what you already announced. Only speak "
    "again if the result needs to be reported, the action failed, or the user needs "
    "additional information. "

    "You automatically have access to the recent conversation and a summary of older "
    "conversation. Use search_memory only when the needed information is not already "
    "in that context. Use search_documents only when the answer may be in a loaded "
    "document. If no documents are loaded, do not call search_documents. "

    "A message starting with '[World state ...]' near the end of the conversation is "
    "not something the user said - it is your own live situational awareness: "
    "background actions you are currently running ('[bg action]') and things "
    "currently true about your environment ('[state]'). Use it to answer questions "
    "about what you are doing or what is around you, and to avoid contradicting "
    "yourself about an action that already finished or was stopped. If it is absent, "
    "nothing notable is currently happening in the background. "

    "Never mention tools, APIs, prompts, hidden context, or internal implementation "
    "unless the user explicitly asks. Respond as QTrobot, not as an AI assistant."
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
    
    # setup 
    robot.speaker.set_volume(0.7)
    robot.motor.home_all()


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
        world_state=WorldStateMemory(),
    )

    # enable agents 
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
            # tool_engine.print_schemas(raw=False)

            llm_engine = LLMEngine(client=client, model=LLM_MODEL, memory=memory, tool_engine=tool_engine)

            async def read_input_loop():
                """Runs input() on a worker thread (via asyncio.to_thread) so it never
                blocks output_loop below - submit()/cancel() are both non-blocking, so
                typing a new line while a previous one is still resolving just works,
                no special-casing needed here."""
                while True:
                    user_input = (await asyncio.to_thread(input, "You: ")).strip()
                    if not user_input:
                        continue
                    if user_input == "/exit":
                        return
                    if user_input == "/cancel":
                        llm_engine.cancel()
                        Logger.info("Cancelled current response.")
                        continue
                    if user_input == "/mem":
                        memory.print()
                        continue
                    if user_input == "/mem raw":
                        memory.print(raw=True)
                        continue
                    llm_engine.submit(user_input)

            async def output_loop():
                """Drains output() forever - sentences arrive from whichever submitted
                turn produced them, in the order they become ready; None (a completed
                turn) is just skipped, this loop never stops on its own."""
                async for item in llm_engine.output():
                    if item is not None:
                        Logger.info(f"[SPEAK] {item}")
                        # handle = robot.tts.say_text_async(item, voice="Rosie")

            Logger.info("Tool-calling demo ready. Try: 'what time is it?', 'what do you see?', "
                        "'/mem' shows memory ('/mem raw' for raw json memory), '/cancel' cancels "
                        "the current response, '/exit' close the chat.")

            output_task = asyncio.create_task(output_loop())
            try:
                await read_input_loop()
            finally:
                llm_engine.shutdown()
                await output_task
    finally:
        local_requester.close()
        robot_requester.close()
        local_tool_server.terminate(timeout=1.0)
        agents.cleanup()
        robot.close()
        memory.flush_all()
        long_term.save(LTM_CHAT_HISTORY_PATH)


if __name__ == "__main__":
    # Logger.set_level("DEBUG")
    asyncio.run(main())
