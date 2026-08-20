# Extending QTrobot AI Agent

This guide explains how to add application tools, background operations, specialized agents, and custom audio integrations to QTrobot AI Agent.

## Add a one-shot application tool

Create a `ToolBase` provider under `src/tool/providers/` and register its methods:

```python
from luxai.magpie.schema import McpSchema
from tool.tool_base import ToolBase


class MyTools(ToolBase):
    def register(self, schema: McpSchema) -> None:
        schema.method()(self.get_room_temperature)

    def get_room_temperature(self) -> dict:
        """Get the current room temperature in degrees Celsius."""
        return {"value": 22.4, "unit": "C"}
```

Add the provider to the `LocalToolServer` provider list in `src/main.py`. Tool schemas are then discovered automatically and made available to the assistant.

## Add a background tool

Background-capable providers receive an `event_sink` through `ToolBase`. Return immediately with:

```json
{"status":"started","task_id":"task_123","summary":"Processing the request"}
```

Publish a correlated event when the work finishes:

```text
[BACKGROUND_EVENT type="task.done" id="task_123" payload="..."]
```

The coordinator safely injects the completed result into the conversation. Reminders and web search demonstrate this pattern.

## Add an isolated agent

Use `src/agents/web_search/` as the reference. The main assistant sees one simple public tool, while the specialized agent has its own instructions, private tools, bounded tool loop, lifecycle, and background completion event. This keeps the main tool surface small and makes each agent independently testable.

## Reuse the S2S client

`src/s2s/client.py` depends on MAGPIE, not QTrobot audio classes. Applications provide their own microphone and speaker integration:

```python
import asyncio

from s2s.client import S2SClient


async def send_microphone(client):
    async for audio_frame in microphone_frames():
        client.send_audio(audio_frame)


async def play_assistant(client):
    async for audio_frame in client.receive_audio():
        await play_audio(audio_frame)


async def handle_events(client):
    async for event_frame in client.receive_events():
        await handle_event(event_frame)


async with S2SClient(node_id="luxai-s2s-magpie") as client:
    await client.update_session(session_config)

    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(send_microphone(client))
        tasks.create_task(play_assistant(client))
        tasks.create_task(handle_events(client))
```

The client accepts either `endpoint=` or MAGPIE `node_id=` discovery. Audio input, audio output, and events are independent streams and should run concurrently.

## Project structure

```text
config/config.yaml                  Paramify schema and application settings
EXTENDING.md                        Developer extension guide
documents/                          Example local knowledge base
src/main.py                         Application composition and lifecycle
src/app_config.py                   Paramify Web live callbacks
src/qtrobot_audio.py                QTrobot microphone and speaker adapters
src/s2s/client.py                   Reusable MAGPIE S2S client
src/s2s/tool_call_coordinator.py    S2S function-call and background-event flow
src/tool/                           MCP discovery, execution, and providers
src/agents/                         Isolated background agent framework
src/memory/                         Document loading and long-term retrieval
src/behaviors/                      Embodied human-attention behavior
tests/                              Focused unit tests
```

[Back to the main README](README.md)
