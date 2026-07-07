"""
tool_call_simple.py - Tool-calling chat loop using our Agent class (agents/agent.py) for
tool discovery/dispatch, with streaming + sentence-level "[SPEAK]" output exactly as
validated before. Tools are served over a real (local, inproc) MCP server
(agents/user_agents.py), via magpie's McpSchema + ServerNode - the same mechanism a real
robot MCP source would use, just pointed at an in-process server instead of the robot.

No ASR/TTS - terminal in, terminal out. "[SPEAK] ..." stands in for TTS.

Requirements:
    pip install openai luxai-magpie[mcp,video]

Usage (from this file's directory):
    python tool_call_simple.py
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

ROBOT_IP = "192.168.3.111"
ROBOT_ENDPOINT = f"tcp://{ROBOT_IP}:50500"

ROBOT_TOOL_WHITELIST = {
    "gesture_cancel",
    "gesture_file_list",
    "gesture_file_play",
}

LLM_API_BASE = "http://192.168.3.111:8080/v1"
LLM_MODEL = "gemma-4-12B-it-Q8_0.gguf"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools when they let you answer "
    "more accurately. Otherwise just reply in plain text. "
    "When you call a tool, you can include a short spoken reply in the same response "
    "(e.g. 'Sure, one moment!') acknowledging what you're doing where needed. "
    "Strip .xml from gesture names before calling gesture_file_play."
)


# ---------------------------------------------------------------------------
# Streaming, unchanged from the raw-openai validation - still the only place
# that talks to the LLM.
# ---------------------------------------------------------------------------

def stream_round(client, history, tools=None, label=""):
    """Run one streaming completion, printing each delta so we can see the two channels
    (plain content vs. tool_calls) arrive separately. Returns an assistant message dict
    (ready to append to history) plus the parsed {name, args, id} for any tool calls."""
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=history,
        tools=tools,
        tool_choice="auto" if tools else None,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    content = ""
    tool_calls_acc = {}  # index -> {"id": str, "name": str, "arguments": str}

    Logger.debug(f"--- streaming deltas ({label}) ---")
    for chunk in stream:
        delta = chunk.choices[0].delta

        if delta.content:
            content += delta.content
            # Live token-by-token output, kept as a raw print: this simulates the flowing
            # text a real TTS stream would consume - wrapping each fragment in a timestamped
            # Logger line would turn it into log spam instead of readable streaming text.
            print(delta.content, end="", flush=True)

        if delta.tool_calls:
            for tc in delta.tool_calls:
                acc = tool_calls_acc.setdefault(tc.index, {"id": None, "name": None, "arguments": ""})
                if tc.id:
                    acc["id"] = tc.id
                if tc.function and tc.function.name:
                    acc["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    acc["arguments"] += tc.function.arguments
                Logger.debug(f"[tool_call delta] index={tc.index} id={tc.id} "
                             f"name={tc.function.name if tc.function else None} "
                             f"args_fragment={tc.function.arguments if tc.function else None!r}")

    Logger.debug(f"--- end of stream ({label}); full content={content!r} ---")

    if not tool_calls_acc:
        return {"role": "assistant", "content": content}, []

    ordered = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
    assistant_msg = {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": tc["arguments"]}}
            for tc in ordered
        ],
    }
    return assistant_msg, ordered


async def main():
    robot = Robot.connect_zmq(endpoint=ROBOT_ENDPOINT)
    Logger.info(f"Connected to {robot.robot_id} ({robot.robot_type}), SDK version: {robot.sdk_version}")
    robot.enable_plugin_zmq("realsense-driver", endpoint=f"tcp://{ROBOT_IP}:50750")

    user_agents = UserAgents(robot)
    user_requester = ZMQRpcRequester(USER_TOOLS_ENDPOINT)
    robot_requester = ZMQRpcRequester(ROBOT_ENDPOINT)

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

            client = OpenAI(base_url=LLM_API_BASE, api_key="not-needed")
            history = [{"role": "system", "content": SYSTEM_PROMPT}]

            Logger.info("Tool-calling demo ready. Try: 'what time is it?', 'what is 12 plus 30?', "
                        "'what do you see?', or 'play the bye gesture'.")
            while True:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break

                history.append({"role": "user", "content": user_input})

                # No fixed "round 1 / round 2" split - keep looping: each round may carry
                # speakable content, tool_calls, both, or neither. Speak whatever content
                # shows up, execute whatever tools show up, and keep offering tools until
                # a round comes back with no tool_calls (that round ends the turn).
                round_num = 1
                while True:
                    message, tool_calls = stream_round(client, history, tools=agent.schemas(), label=f"round {round_num}")
                    history.append(message)

                    if message.get("content"):
                        Logger.info(f"[SPEAK] {message['content']}")

                    if not tool_calls:
                        break

                    for tc in tool_calls:
                        Logger.info(f"[tool call] {tc['name']}({tc['arguments']})")

                    results = await agent.execute(tool_calls)
                    for r in results:
                        Logger.info(f"[tool result] {r['content']}")
                        history.append({
                            "role": "tool",
                            "tool_call_id": r["tool_call_id"],
                            "content": r["content"],
                        })
                        history.extend(r["extra_messages"])

                    round_num += 1
    finally:
        user_requester.close()
        robot_requester.close()
        user_agents.terminate(timeout=1.0)
        robot.close()


if __name__ == "__main__":
    Logger.set_level("DEBUG")
    asyncio.run(main())
