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
    "reply in plain text. Respond as QTrobot, not as an AI assistant. "

    "Keep every spoken response short and natural. Prefer one brief sentence. Avoid "
    "long sentences, explanations, lists, emojis, repetition, and unnecessary details."
)


DOCUMENTS_SUMMARY = (
    "QTrobot product/hardware overview (Research, School, Home variants, SDK/Studio "
    "programming options), and a list of 40+ published research papers using QTrobot "
    "with abstracts (autism therapy, education, healthcare, assistive robotics)."
)


class QTrobotAIAgent:
    """Owns the entire agent: robot connection, memory tiers, tool discovery, agents,
    the LLM engine, and the ASR/TTS conversation loop - one class, everything self-
    contained, nothing left for a caller to wire up or tear down itself.

    Lifecycle is explicit, not a context manager - construct, setup() (connects
    everything), run() (the conversation loop), cleanup() (tears everything down,
    reverse order of setup()) - construct/setup/run/cleanup, the same shape as a
    typical robot demo node."""

    def __init__(self):
        # Populated by setup() - nothing is connected/built until then.
        self.robot = None
        self.client = None
        self.long_term = None
        self.memory = None
        self.agents = None
        self.local_tool_server = None
        self.tool_engine = None
        self.llm_engine = None
        self._local_requester = None
        self._robot_requester = None
        self._local_client = None
        self._robot_client = None
        self._speaking_handle = None

    # ------------------------------------------------------------------
    # Lifecycle: setup / run / cleanup
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Connects the robot, builds memory/tools/agents, starts the LLM engine and
        ASR listening. Call once before run()."""
        self.robot = Robot.connect_zmq(endpoint=ROBOT_ENDPOINT)
        Logger.info(f"Connected to {self.robot.robot_id} ({self.robot.robot_type}), "
                    f"SDK version: {self.robot.sdk_version}")
        self.robot.enable_plugin_zmq("realsense-driver", endpoint=REALSENSE_ENDPOINT)
        self.robot.speaker.set_volume(0.7)
        self.robot.motor.home_all()

        self.client = OpenAI(base_url=LLM_API_BASE, api_key="not-needed")

        self.long_term = LongTermMemory()
        self.long_term.load(LTM_CHAT_HISTORY_PATH)
        for doc in DirectoryReader.read(DOCUMENTS_DIR, summary=DOCUMENTS_SUMMARY):
            self.long_term.add_document(text=doc.text, summary=doc.summary, meta=doc.meta)
            Logger.info(f"Loaded document into long-term memory: {doc.meta['source']}")

        self.memory = RobotMemory(
            static=SYSTEM_PROMPT,
            max_tokens=MEMORY_MAX_TOKENS,
            working=WorkingMemory(size_ratio=0.75, flush_ratio=0.2),
            short_term=ShortTermMemory(llm=self.client, model=LLM_MODEL, size_ratio=0.25, flush_ratio=0.2),
            long_term=self.long_term,
            world_state=WorldStateMemory(),
        )

        self.agents = AgentRegistry(self.client, LLM_MODEL)
        self.local_tool_server = LocalToolServer([
            UserTools(self.robot),
            MemoryTools(self.long_term),
            *self.agents.as_tools(),
        ])
        self._local_requester = ZMQRpcRequester(LOCAL_TOOLS_ENDPOINT)
        self._robot_requester = ZMQRpcRequester(ROBOT_ENDPOINT)

        # Entered/exited manually (not `async with`) - setup()/cleanup() own their
        # lifecycle explicitly, same as everything else this class builds.
        self._local_client = Client(McpTransport(self._local_requester, timeout=120.0))
        self._robot_client = Client(McpTransport(self._robot_requester))
        await self._local_client.__aenter__()
        await self._robot_client.__aenter__()

        self.tool_engine = ToolEngine(
            sources={"local": self._local_client, "robot": self._robot_client},
            whitelists={"robot": ROBOT_TOOL_WHITELIST},
        )
        await self.tool_engine.discover()

        self.llm_engine = LLMEngine(client=self.client, model=LLM_MODEL,
                                     memory=self.memory, tool_engine=self.tool_engine)
        print(self.memory.static)
        self._start_listening()
        Logger.info("QTrobot conversation ready. Listening for speech... (Ctrl+C to stop)")

    def start(self) -> None:
        """Synchronous entry point - the only method meant to be called from outside
        an event loop. Runs the whole lifecycle (setup, listen until Ctrl+C,
        cleanup) via asyncio.run()."""
        asyncio.run(self._main())

    async def _main(self) -> None:
        await self.setup()
        try:
            await self.listen()
        except KeyboardInterrupt:
            Logger.info("Interrupted by user.")
        finally:
            await self.cleanup()

    async def listen(self) -> None:
        """Drains LLMEngine.output() forever, speaking each sentence, until
        cleanup() shuts the engine down and ends the stream.

        await asyncio.to_thread(handle.wait) - not a bare handle.wait() - matters
        here: ActionHandle.wait() is a plain blocking call, not a coroutine, so
        calling it directly would stall this whole event loop (and every other task
        on it, including ASR-driven submit()s) for as long as TTS is speaking."""
        async for item in self.llm_engine.output():
            if item is None:
                continue
            try:
                handle = self.robot.tts.say_text_async(item, voice="Rosie")
                self._speaking_handle = handle
                await asyncio.to_thread(handle.wait)
            except Exception as e:
                Logger.error(f"TTS error: {e}")

    async def cleanup(self) -> None:
        """Tears everything down, reverse order of setup(). Every step is
        independently guarded, so this is safe to call even if setup() didn't fully
        complete."""
        if self.llm_engine is not None:
            self.llm_engine.shutdown()
        if self._local_client is not None:
            await self._local_client.__aexit__(None, None, None)
        if self._robot_client is not None:
            await self._robot_client.__aexit__(None, None, None)
        if self._local_requester is not None:
            self._local_requester.close()
        if self._robot_requester is not None:
            self._robot_requester.close()
        if self.local_tool_server is not None:
            self.local_tool_server.terminate(timeout=1.0)
        if self.agents is not None:
            self.agents.cleanup()
        if self.robot is not None:
            self.robot.close()
        if self.memory is not None:
            self.memory.flush_all()
        if self.long_term is not None:
            self.long_term.save(LTM_CHAT_HISTORY_PATH)

    # ------------------------------------------------------------------
    # ASR - listening setup and callbacks
    # ------------------------------------------------------------------

    def _start_listening(self) -> None:
        self.robot.enable_plugin_local("asr-parakeet")
        self.robot.asr.configure_parakeet(
            endpoint=PARAKEET_ENDPOINT,
            language="en",
            use_vad=True,
            silence_timeout=0.3,
            max_buffer_seconds=20.0,
            continuous_mode=True,
        )
        self.robot.asr.stream.on_parakeet_speech(self._on_parakeet_speech)
        self.robot.asr.stream.on_parakeet_event(self._on_parakeet_event)

    def _on_parakeet_speech(self, speech) -> None:
        data = speech.value or {}
        text = data.get("text", "")
        if not text:
            return

        if not data.get("is_final"):
            # Interim result: the user has started talking - treat as barge-in, but
            # don't submit anything yet, wait for the final transcript.
            self._trigger_barge_in("interim speech")
            return

        Logger.info(f"You said: {text}")
        self.llm_engine.submit(text)

    def _on_parakeet_event(self, event) -> None:
        Logger.debug(event.value)

    def _trigger_barge_in(self, source: str) -> None:
        """Stops the robot's own output when the user starts talking over it - see
        LLMEngine.interrupt() for why this is deliberately narrower than cancel()
        (doesn't touch in-flight tool/agent calls). No lock around
        _speaking_handle: a plain attribute is enough - a single reference
        assignment is already atomic, and ActionHandle's own done()/cancel() are
        themselves safe to call from any thread."""
        handle = self._speaking_handle
        if handle is None or handle.done():
            return  # nothing playing right now
        Logger.info(f"Barge-in ({source}): cancelling TTS.")
        handle.cancel()
        self.llm_engine.interrupt()


if __name__ == "__main__":
    # Logger.set_level("DEBUG")
    QTrobotAIAgent().start()
