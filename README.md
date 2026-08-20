# QTrobot MAGPIE S2S example

Minimal QTrobot application using the reusable `S2SClient` in
[`src/s2s_client.py`](src/s2s_client.py). The client depends only on MAGPIE and has no
knowledge of QTrobot, microphones, speakers, agents, or application policy.

The example application independently:

- feeds QTrobot `AudioFrameRaw` microphone frames to `S2SClient.send_audio()`;
- consumes assistant `S2SAudioFrame` objects from `receive_audio()`;
- consumes `DictFrame` lifecycle and transcript events from `receive_events()`;
- plays and cancels foreground audio through the QTrobot SDK.

It also demonstrates S2S function calling and one background agent-as-tool:

- `LocalToolServer` exposes the application-owned `get_datetime` and
  `get_image` tools over in-process MCP;
- `ToolEngine` discovers those tools together with a small whitelist of
  QTrobot's MCP tools;
- `ToolCallCoordinator` executes calls asynchronously, returns their results
  in response order, and requests one S2S follow-up response;
- camera results are supplied to the multimodal model as `input_image`
  content rather than base64 text;
- `ReminderTools` exposes `set_reminder`, `list_reminders`, and
  `cancel_reminder` through the same in-process MCP server;
- `MemoryTools` exposes `search_memory` for earlier conversations and
  `search_documents` for local reference material;
- `WebSearchAgent` exposes one public `search_web` tool while keeping Tavily
  search and page fetching private to its isolated sub-agent.

Reminders are kept in memory for the lifetime of this application. Scheduling,
listing, and cancellation are ordinary tool calls. When a reminder becomes
due, the application queues one trusted `BACKGROUND_EVENT` and asks S2S for a
response only after current speech, responses, and tool calls are clear.

Web search follows the same generic event path. The public tool returns a
`started` result immediately, then the background agent injects `search.done`
or `search.failed` when its isolated research run finishes.

Long-term memory is intentionally independent from S2S's recent conversation
history. The same final user and assistant transcripts printed by the example
are indexed in the background and appended directly to
`data/long_term_chat_history.jsonl`. `LongTermMemory` reloads that file on the
next start; the application has no shutdown save step. Files under `documents/`
are indexed at startup and are searched only when the model calls
`search_documents`. FastEmbed downloads its embedding and reranking models on
the first run and then reuses its local cache.

The current deployed transport is native ZMQ. Audio is mono PCM16, and the
default Qwen3-TTS CustomVoice speaker is `Ono_Anna`.

## Run

```powershell
uv pip install -r requirements.txt
$env:TAVILY_API_KEY="your-key"
python src/main.py `
  --robot-endpoint tcp://192.168.3.152:50500 `
  --s2s-endpoint tcp://192.168.3.152:50960
```

The web agent uses the local llama.cpp Chat Completions endpoint independently
of S2S. Its defaults match this demo (`http://192.168.3.109:8080/v1` and
`gemma-4-12b-it-Q8_0.gguf`). Override them with `OPENAI_AGENT_BASE_URL`,
`OPENAI_AGENT_MODEL`, and, when required, `OPENAI_AGENT_API_KEY`. If
`TAVILY_API_KEY` is absent, web search is disabled and the rest of the demo
continues normally.

The S2S endpoint is the system-descriptor RPC endpoint. `S2SClient` reads the
descriptor and opens all four advertised audio/event streams internally.

Discovery by node ID is also supported:

```powershell
python src/main.py `
  --robot-endpoint tcp://192.168.3.152:50500 `
  --s2s-node-id luxai-s2s-magpie
```

## Client API

Operations that wait are asynchronous; immediate MAGPIE writes are
synchronous:

```python
import asyncio

async def send_microphone(client, microphone):
    while True:
        client.send_audio(await microphone.read())

async def receive_audio(client):
    async for frame in client.receive_audio():
        ...

async def receive_events(client):
    async for frame in client.receive_events():
        ...

async with S2SClient(node_id="luxai-s2s-magpie") as client:
    await client.update_session(session_config)
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(send_microphone(client, microphone))
        tasks.create_task(receive_audio(client))
        tasks.create_task(receive_events(client))
```

In a real application the two receive iterators run in separate asyncio tasks.
The S2S runtime performs VAD and response creation automatically. The
application remains responsible for microphone capture, audio playback,
handling cancellation frames, tools, agents, and UI behavior.

## Tool lifecycle

Tool schemas are included in `session.update` with `tool_choice="auto"`.
Function calls arrive on the event stream while microphone and speaker tasks
continue normally. Local and robot tools run through the shared MCP
`ToolEngine`. The coordinator returns each result as soon as it is ready while
preserving call order; after the originating response finishes and all results
are delivered, it sends exactly one `response.create`. Barge-in or an explicit
`cancel_all_actions` call also invokes the paired robot stop tools where one is
available.

Additional local tools or future agents can be added as `ToolBase` providers
under `src/tool/providers/` and registered with `LocalToolServer` without
changing `S2SClient`. Providers that emit delayed results pass an event sink to
`ToolBase`; ordinary one-shot providers use its default constructor.
