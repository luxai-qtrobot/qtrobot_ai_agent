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

An isolated agent gives one focused task to a separate model loop with its own instructions and private tools. The main assistant sees only one public tool, while the internal tools stay hidden.

The following example creates a small product-advisor agent.

### 1. Create the private tool

Create `src/agents/product_advisor/tools.py`:

```python
from luxai.magpie.schema import McpSchema

from tool.tool_base import ToolBase


PRODUCTS = {
    "research": "QTrobot Research supports Python, ROS, and C++ development.",
    "education": "QTrobot Education includes visual teaching tools and curricula.",
}


class ProductTools(ToolBase):
    def register(self, schema: McpSchema) -> None:
        schema.method()(self.lookup_product)

    def lookup_product(self, product: str) -> str:
        """Look up verified information about a QTrobot product."""
        return PRODUCTS.get(product.lower(), "No matching product was found.")
```

This tool is available only to the product-advisor agent.

### 2. Write focused instructions

Create `src/agents/product_advisor/instructions.txt`:

```text
You are a QTrobot product advisor.
Use lookup_product before answering product questions.
Give a concise answer based only on the retrieved information.
If no information is found, say so instead of guessing.
```

### 3. Expose one public agent tool

Create `src/agents/product_advisor/agent.py`:

```python
import asyncio
import concurrent.futures
import threading
from pathlib import Path

from luxai.magpie.schema import McpSchema

from ..agent_base import AGENT_TOOLS_ENDPOINT, AgentBase


INSTRUCTIONS = Path(__file__).with_name("instructions.txt")


class ProductAdvisorAgent(AgentBase):
    def __init__(self, client, model, *, owner_loop, endpoint=AGENT_TOOLS_ENDPOINT):
        super().__init__(
            client,
            model,
            {"lookup_product": None},
            INSTRUCTIONS.read_text(encoding="utf-8"),
            endpoint=endpoint,
        )
        self._owner_loop = owner_loop
        self._jobs = set()
        self._jobs_lock = threading.Lock()
        self._closed = False

    def register(self, schema: McpSchema) -> None:
        schema.method()(self.ask_product_advisor)

    def ask_product_advisor(self, question: str) -> str:
        """Answer a question using the QTrobot product advisor."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        with self._jobs_lock:
            if self._closed:
                raise RuntimeError("Product advisor is shutting down")
            future = asyncio.run_coroutine_threadsafe(
                self.run(question), self._owner_loop
            )
            self._jobs.add(future)

        try:
            return future.result(timeout=120)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise RuntimeError("Product advisor timed out") from exc
        finally:
            with self._jobs_lock:
                self._jobs.discard(future)

    async def close(self) -> None:
        with self._jobs_lock:
            self._closed = True
            jobs = tuple(self._jobs)
        for job in jobs:
            job.cancel()
        if jobs:
            await asyncio.gather(
                *(asyncio.wrap_future(job) for job in jobs),
                return_exceptions=True,
            )

    def cleanup(self) -> None:
        with self._jobs_lock:
            self._closed = True
            jobs = tuple(self._jobs)
        for job in jobs:
            job.cancel()
```

The main assistant sees only `ask_product_advisor`. The agent itself can discover and call `lookup_product`, repeat tool calls when needed, and synthesize the final answer.

### 4. Register the agent

In `src/agents/agent_registry.py`:

1. Add `ProductTools()` to the private providers passed to `LocalToolServer`.
2. Create `ProductAdvisorAgent(client, model, owner_loop=owner_loop)` and append it to `_agents`.
3. Keep optional agents independent: a missing Tavily key may disable web search, but it should not return before the product advisor is registered.
4. Export the new classes from `src/agents/product_advisor/__init__.py`.
5. In `src/main.py`, construct `AgentRegistry` whenever the product advisor is enabled, rather than only when `web_search.enabled` is true.

Add the new provider alongside the existing web-search provider; both agents can share the same private MCP server and endpoint. `AgentRegistry.as_tools()` then exposes the product advisor automatically to the main application.

This example is intentionally a normal foreground agent: the conversation waits for its answer. For work that may take a long time, use the background-event pattern and the complete lifecycle in `src/agents/web_search/` as the reference.

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
