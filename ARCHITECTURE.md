# Architecture

This document describes how QTrobot's AI agent brain is put together: the
non-blocking conversation engine, how tools and MCP fit in, the agent-as-tool
pattern, and interruption/cancellation. For memory specifically (Static / WRM /
STM / LTM / WSM), see [MEMORY.md](MEMORY.md).

## Overview

```
                         ┌────────────────────────────┐
                         │         LLMEngine           │
User input ──submit()──▶ │  (one worker thread, serial │──output()──▶ TTS/terminal
                         │   turn processing, non-      │
 cancel()/interrupt() ──▶│   blocking tool dispatch)    │
                         └──────────────┬───────────────┘
                                        │ reads/writes
                                        ▼
                         ┌────────────────────────────┐
                         │        RobotMemory          │
                         │  Static + WRM + STM + LTM    │
                         │        + WSM (get())         │
                         └────────────────────────────┘
                                        │
                                        │ tool_calls
                                        ▼
                         ┌────────────────────────────┐
                         │         ToolEngine           │
                         │  merges tool sources, dispatch│
                         │  concurrently, cancel_all()   │
                         └───────┬───────────┬──────────┘
                                 │           │
                     inproc MCP │           │ inproc MCP
                                 ▼           ▼
                     ┌───────────────┐  ┌─────────────────┐
                     │LocalToolServer│  │  Agent-as-tool   │
                     │(UserTools,    │  │ (AgentBase, e.g. │
                     │ MemoryTools,  │  │  WebSearchAgent) │
                     │ agents...)    │  │  own nested      │
                     └───────────────┘  │  LLMEngine+Tools │
                                        └─────────────────┘
```

Everything is glued together through two things: **RobotMemory** (what the
model sees) and **ToolEngine** (what the model can do), both owned by a single
**LLMEngine** per conversation. Agents are just another kind of tool provider
that happens to run its own nested LLMEngine underneath.

---

## The non-blocking conversation engine (`llm/llm_engine.py`)

### Why non-blocking matters

A tool or agent call (a web search, an LLM-backed sub-agent) can take many
seconds. A conversational robot can't go silent and unresponsive for that
whole window - the user should be able to say something else, get an answer,
and have the original request's answer arrive whenever it's ready, spoken in
whatever order things actually finish.

### The design

`LLMEngine` runs **one background worker thread** that processes submitted
inputs strictly serially - never two LLM completions in flight from the same
engine at once. The trick is what happens when a round comes back wanting to
call a tool:

- A round with **no tool calls** is a plain text reply: added to memory,
  turn ends, worker moves on.
- A round **with tool calls** is never waited on. It's **parked**: dispatched
  non-blocking via `ToolEngine.execute_async()`, and the worker is immediately
  free to pick up the next submitted input. The tool call's `assistant`
  message (with its `tool_calls`) is *not* added to memory yet - it's held in
  a closure until the call resolves.

When a parked call resolves, a dedicated follow-up completion - whose message
array ends exactly at the tool result, the shape models are actually trained
to continue from - produces the real answer. That answer, together with the
original tool call and its result, is then **spliced into working memory at
its true chronological position** (right after the question that triggered
it, via `WorkingMemory.insert_after()`) - not appended wherever it happened to
finish. So even if a search launched first resolves *last*, it still reads in
the transcript as answering the question it was asked, in order, while
whatever was asked and answered in the meantime sits in its correct place
too. Verified against real interleaved conversations (see the git history /
session notes for live traces).

All output - live or from a resolved parked call - flows into the same
`output()` async generator. A caller doesn't need to know or care which
submitted turn produced what; it just drains sentences in the order they
become ready.

### API surface

- `submit(text)` - non-blocking, queues input.
- `output()` - async generator yielding sentences as they're ready, plus
  `None` once a given turn fully completes.
- `cancel()` - the broad "stop everything" case (explicit user command, e.g.
  "cancel that"): stops the live round's output *and* asks `ToolEngine` to
  stop/discard every in-flight tool call.
- `interrupt()` - the narrow case (e.g. the user starts talking over TTS,
  detected by ASR/VAD barge-in): stops the live round's output only. Does
  **not** touch background tool/agent work - starting to talk doesn't mean
  the user abandoned whatever's running.
- `shutdown()` - stops the worker and unblocks `output()` cleanly (a plain
  `task.cancel()` on the draining side can't unblock a `queue.get()` already
  running in a thread-pool thread - `shutdown()` pushes sentinel values
  instead).

### Concurrency model, precisely

- One worker thread per `LLMEngine`, processing `_input_queue` serially.
- A parked call's resolution runs on **its own background thread**
  (`ToolEngine.execute_async()`'s callback thread) - never the worker thread,
  so a slow follow-up completion or nested tool round never blocks anything
  else.
- Cross-thread → asyncio-loop dispatch (`ToolEngine.execute_async`) uses
  `asyncio.run_coroutine_threadsafe()` onto the loop the MCP clients were
  actually opened on - never a fresh `asyncio.run()`, since `fastmcp.Client`
  sessions are bound to their originating loop.

Neither the OpenAI Agents SDK's human-in-the-loop docs nor LlamaIndex
Workflows support "answer a new message while an older tool call is still
pending" as a first-class pattern (a LlamaIndex GitHub issue asking for
exactly this was closed as not-planned) - this was built from scratch here.

---

## Tools & MCP integration (`tool/`)

### `ToolBase` - the provider contract

Any group of related tool methods (camera, memory retrieval, robot actions,
an agent) implements `ToolBase`:

```python
class ToolBase(ABC):
    def register(self, schema: McpSchema) -> None: ...  # schema.method()(self.my_tool)
    def cleanup(self) -> None: ...                        # optional, no-op default
```

That's the whole contract. A provider never touches transport or threading -
it just registers plain methods.

### `LocalToolServer` - one server, many providers

```python
LocalToolServer([UserTools(robot), MemoryTools(long_term), *agents.as_tools()])
```

A single in-process MCP server (one `McpSchema`, one `ZMQRpcResponder`, one
`ServerNode`, from `luxai.magpie`) hosts tool methods from **any number** of
providers. Adding a new local tool group never means standing up another
server/endpoint pair - this is the "beautiful, seamless" local-server pattern:
magpie gives a full MCP responder over a ZMQ `inproc://` socket for free,
and every local tool - robot actions, memory search, agents - rides the same
one.

### `ToolEngine` - merging sources, dispatching calls

`ToolEngine` connects to one or more MCP sources (the local server, the real
robot's own MCP backend, an agent's scoped server) as a `fastmcp.Client`,
merges their tool schemas into one OpenAI-compatible `tools=` list, and
dispatches calls concurrently via `asyncio.gather`. Two sources can be
wired in side by side - `{"local": local_client, "robot": robot_client}` -
and the LLM never knows the difference; both are just tools.

`execute_async()` is the non-blocking dispatch primitive everything above
relies on: callable from any thread (including the worker thread, which has
no event loop of its own), it schedules the actual MCP call onto the loop
those clients were opened on, and runs the completion callback on a fresh
thread so a slow follow-up (another LLM call) never blocks the event loop.

### Cancellation: tool pairing, not MCP notifications

The MCP spec has a real `notifications/cancelled` mechanism, and the
underlying `mcp`/`fastmcp` libraries implement the *receiving* half
generically. It doesn't work here: `luxai.magpie.adapters.mcp.McpTransport`
(the bridge between `fastmcp.Client` and MAGPIE's RPC transport) silently
drops every JSON-RPC message without an `id` - which is exactly what a
notification is, by definition. The underlying MAGPIE RPC protocol
(`ZMQRpcRequester`/`ZMQRpcResponder`) has no fire-and-forget primitive at all;
every send is a request/ack/reply triple. Getting real MCP cancellation
working would mean extending `luxai.magpie` itself.

Instead, tools are cancelled the same way the robot SDK's own `ActionHandle`
already does it (`robot-sdk-python/.../actions.py`): a **cancel-tool
pairing**. A whitelist entry is `{tool_name: cancel_tool_name | None}` -
e.g. `{"gesture_file_play": "gesture_file_cancel"}` - and `cancel_all()`:

1. Marks every currently in-flight `tool_call_id` cancelled (own tracking,
   independent of MCP).
2. For any call with a pairing, dispatches the cancel tool as an ordinary
   MCP call, fire-and-forget - the original call is expected to resolve on
   its own, faster, once the underlying action honours the stop (same
   assumption `ActionHandle.cancel()` already makes).
3. Calls without a pairing just keep running to completion in the
   background - wasted computation, never force-killed.

`execute_async()`'s callback then checks whether a batch's ids were marked
cancelled; if so, the callback receives the `CANCELLED` sentinel instead of
real results, and the caller (`LLMEngine`) discards the answer rather than
acting on it. This gives **per-call granularity for free** - no MCP bypass,
no notification hack, reuses infrastructure the robot SDK already trusts.

---

## Agent-as-tool pattern (`agents/`)

A sub-agent is a `ToolBase` provider that *also* runs its own isolated
tool-calling loop:

```python
class AgentBase(ToolBase):
    async def run(self, query: str) -> str:
        # fresh ToolEngine (scoped to this agent's own whitelist)
        # fresh RobotMemory (working-memory only, small budget)
        # fresh LLMEngine
        # submit(query), drain output() to one string, shutdown()
```

Each `run()` call builds fresh state end to end - a sub-agent handles one
isolated task per call, so there's nothing to carry over, and fresh state
means concurrent calls (the orchestrator firing two tool calls in one round)
never share mutable memory. The connection/discovery cost is redone every
call too - simpler and correct, and small next to the nested LLM call's own
latency.

`AgentRegistry` is the one place that knows how every agent is built - which
internal tools it needs, which server they live on:

```python
agents = AgentRegistry(client, model)
local_tool_server = LocalToolServer([MemoryTools(long_term), *agents.as_tools()])
```

All agents' *internal* tools (e.g. `WebSearchTools`'s `search_web_api`/
`fetch_url`) share **one** agent-tools MCP server (`AGENT_TOOLS_ENDPOINT`) -
one thread/socket total, not one per agent - reached only through each
agent's own scoped `ToolEngine`, never exposed to the main conversation.

`WebSearchAgent` is the concrete example: a real Tavily search + trafilatura
page-fetch backend, exposed to the main LLM as a single `search_web(query)`
tool. Adding a new agent is: write its `tools.py` provider, write its
`Agent(AgentBase)` subclass, add both lines to `AgentRegistry.__init__`.

Recursion falls out for free: an agent's own cancel-tool pairing (when
built) would forward into its nested engine's `cancel()`, which in turn
cancels *its* nested `ToolEngine`'s in-flight calls - same mechanism, one
level down, no special-casing required anywhere else in the system.

---

## Interruption & cancellation, end to end

Two independent trigger sources, matched to two `LLMEngine` methods:

| Trigger | Method | Stops output | Stops tool/agent calls |
|---|---|---|---|
| Explicit ("cancel that", `/cancel`) | `cancel()` | ✅ | ✅ (`cancel_all()`) |
| Barge-in (user talks over TTS) | `interrupt()` | ✅ | ❌ (deliberately) |

Barge-in doesn't imply the user abandoned whatever's running in the
background - just that they want to be heard now.

### What actually happens on stop

1. **Mid-stream text generation** (`_stream_round`): the cancellation check
   runs once per raw chunk (not once per assembled sentence), so it cuts in
   within roughly one chunk's latency - not after draining the whole
   response. The underlying HTTP stream is explicitly `.close()`d so the
   connection is actually released (most OpenAI-compatible backends stop
   generating/billing on client disconnect; a bare `break` would just
   abandon the iterator).
2. **What's kept**: exactly the sentences that were *already spoken*
   (yielded as complete sentences, and so already sent to `output()`/TTS) -
   any incomplete trailing fragment never actually said is dropped. This
   preserves the model's own memory of what it actually told the user,
   instead of replacing everything with a synthetic-only placeholder.
3. **The closing note**: appended after the real spoken content (or alone,
   if nothing had been spoken yet) - `"I didn't finish that - it was
   cancelled."` for `cancel()`, `"I was interrupted there."` for
   `interrupt()`. Different wording matters: "cancelled" tells the model
   the topic was abandoned; "interrupted" doesn't, since the topic may
   still be relevant.
4. **In-flight tool calls** (`cancel()` only): `ToolEngine.cancel_all()`
   marks every pending id cancelled and fires paired cancel tools where
   available (see above). `_resolve_parked()` then splices the same kind of
   closing note in place of the real answer, so the model never sees an
   unanswered question sitting in history that it might spontaneously retry.
5. **World-state cleanup**: `cancel()` also clears WSM's `[bg action]`
   entries for those same ids **immediately** - not waiting for the
   eventual `CANCELLED` handling above, since a call without a cancel
   pairing can keep running in the background for a while, and its
   "in progress" entry would otherwise stay visible (and misleading) for
   that whole window even though it was already cancelled.

---

## Design principles / what's genuinely different here

- **Non-blocking parking with chronologically-correct splicing** - a slow
  tool call never blocks the conversation, and its eventual answer still
  reads in the right place in the transcript regardless of resolution order.
  Neither of the two agent frameworks researched (OpenAI Agents SDK,
  LlamaIndex Workflows) support this.
- **World State Memory** - a reserved, always-present "what's happening
  right now" slot, decoupled from conversation history (see MEMORY.md).
  Fixes real, observed failure modes: stale answers with no time-awareness,
  self-contradiction after an action was superseded, and hallucinated
  self-unawareness ("I'm just waiting" while actually mid-search).
- **Tool-cancel pairing instead of MCP notifications** - real MCP
  cancellation is architecturally dead in this transport stack; the fix
  reuses the exact idiom the robot SDK's own `ActionHandle` already trusts,
  rather than patching the transport.
- **Two-tier interruption** (`cancel()` vs `interrupt()`) matching the two
  real-world triggers (explicit command vs. voice barge-in) - most simple
  voice stacks conflate these into one blunt "stop" signal.
- **Spoken-content preservation on cut-off** - the model's memory reflects
  what it actually said, not a synthetic "never happened."
- **One local MCP server for arbitrarily many tool providers**, courtesy of
  `luxai.magpie`'s `ServerNode`/`McpSchema`/`ZMQRpcResponder` - and the same
  `ToolEngine` merges that server's tools with the real robot's own MCP
  backend transparently; the LLM never sees the difference.

---

## Known limitations / deferred work

- **Local tool cancel pairing not filled in yet** - `search_documents`,
  `search_web`, etc. have no cancel counterpart (`whitelist` entries are
  `None`); a cancelled call to one just keeps running to completion,
  wasted but harmless.
- **Real robot cancel-tool names not wired in** - `ROBOT_TOOL_WHITELIST` in
  `cli_chat_example.py` has all `None` pairings; fill in the real ones
  (e.g. `gesture_file_play` → `gesture_file_cancel`) as they're confirmed.
- **Chronological-splice-vs-real-time tension for cancellation replies** -
  a cancelled reply is spliced at the *question's* position, which can
  read as "cancelled, then later un-cancelled" if something else was said
  in between in real time. Diagnosed, not yet fixed (would mean appending
  cancellation replies at the current tail instead of the anchor position).
- **No real ASR/VAD wiring yet** - `interrupt()` exists and is tested, but
  nothing currently calls it; that's the barge-in detector's job once
  built.
- **No real perception/sensor component yet** - WSM's `[state]` entries
  (person in view, force detected, etc.) are designed for but not
  implemented; only `[bg action]` entries (tool/agent calls) exist today.
- **`main.py`** (the real-TTS copy of `cli_chat_example.py`) is stale -
  still calls a removed API shape; not yet updated to the current engine.
