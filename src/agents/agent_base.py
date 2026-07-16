"""
agent_base.py - AgentBase: base class for a sub-agent exposed to the main LLM as a
tool ("agent as tool"). A subclass IS a ToolBase provider (so LocalToolServer can host
it exactly like UserTools/MemoryTools) that also carries the machinery to run its own
isolated tool-calling loop - a fresh, working-memory-only RobotMemory + LLMEngine per
run() call, connected to a scoped subset of the shared agent-tools server (see
AGENT_TOOLS_ENDPOINT below).

Subclasses call super().__init__(...) with their own whitelist/system_prompt, then
implement register() (still required, per ToolBase) to expose whichever of their own
methods should call self.run().

Each run() call builds fresh memory/engine state - a sub-agent handles one isolated
task per call, so there's nothing to carry over between calls, and fresh state also
means concurrent calls (e.g. the orchestrator firing two tool calls in one round) never
share mutable memory. The tool connection/discovery is redone every call too, rather
than cached - simpler and correct; the in-process handshake+discover() cost is small
next to the nested LLM call's own latency, and caching it safely would need a
persistent background event loop (async resources can't outlive the loop that created
them), which isn't worth the complexity unless this is shown to actually matter.
"""

from fastmcp import Client
from luxai.magpie.adapters.mcp import McpTransport
from luxai.magpie.transport import ZMQRpcRequester

from tool.tool_base import ToolBase
from tool.tool_engine import ToolEngine
from llm.llm_engine import LLMEngine
from memory.robot_memory import RobotMemory
from memory.working_memory import WorkingMemory

# Shared server for every sub-agent's internal tools (one thread/socket total, not one
# per agent) - built once by the caller (LocalToolServer([...],
# endpoint=AGENT_TOOLS_ENDPOINT)); each agent connects here with its own whitelist.
# Agents connect to it internally on their own - the caller just keeps it alive and
# cleans it up, the same way it does for any other tool source.
AGENT_TOOLS_ENDPOINT = "inproc://agent-tools"


class AgentBase(ToolBase):

    def __init__(self, client, model: str, whitelist: dict, system_prompt: str,
                 endpoint: str = AGENT_TOOLS_ENDPOINT,
                 memory_max_tokens: int = 4000, max_tokens: int = 800, timeout: float = 60.0):
        """
        client:            a (sync) OpenAI client - shared with the main conversation's client.
        model:             model name passed to every completion call.
        whitelist:         {tool_name: cancel_tool_name | None} on the agent-tools server
                           belonging to this agent - scopes this agent's ToolEngine down
                           to just its own tools (see ToolEngine.cancel_all()).
        system_prompt:     this agent's own instructions, independent of the main
                           conversation's system prompt.
        endpoint:          agent-tools server endpoint - this agent's own tools live
                           there among other agents' tools.
        memory_max_tokens: this agent's own working-memory budget - much smaller than
                           the main conversation's, since a sub-agent only needs to
                           hold one task's tool round-trips, not a running conversation.
        max_tokens:        passed through to LLMEngine - bounds a runaway generation,
                           same reasoning as the main engine's cap.
        timeout:           passed through to LLMEngine.
        """
        super().__init__()
        self.client = client
        self.model = model
        self.endpoint = endpoint
        self.whitelist = whitelist
        self.system_prompt = system_prompt
        self.memory_max_tokens = memory_max_tokens
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def run(self, query: str) -> str:
        """Runs one full agent task to completion and returns the final assembled
        answer - collects every sentence LLMEngine.output() yields into one string
        (stopping at the first None, since this engine only ever gets one submit()),
        since an agent isn't streaming to anything itself, it just returns one result
        to whoever called it (one of this agent's own exposed tool methods)."""
        requester = ZMQRpcRequester(self.endpoint)
        engine = None
        try:
            async with Client(McpTransport(requester)) as mcp_client:
                tool_engine = ToolEngine(sources={"agent": mcp_client}, whitelists={"agent": self.whitelist})
                await tool_engine.discover()

                memory = RobotMemory(
                    static=self.system_prompt,
                    max_tokens=self.memory_max_tokens,
                    working=WorkingMemory(size_ratio=1.0, flush_ratio=0.2),
                )
                engine = LLMEngine(
                    client=self.client, model=self.model, memory=memory, tool_engine=tool_engine,
                    max_tokens=self.max_tokens, timeout=self.timeout,
                )
                engine.submit(query)

                parts = []
                async for item in engine.output():
                    if item is None:
                        break
                    parts.append(item)
                return " ".join(parts)
        finally:
            requester.close()
            if engine is not None:
                engine.shutdown()
