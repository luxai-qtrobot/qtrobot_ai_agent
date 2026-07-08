"""
token_utils.py - Approximate token counting for memory budget accounting.

Uses tiktoken as a reasonable approximation - it won't exactly match gemma's own
tokenizer, but this is a soft budget (for latency/perf tuning), not a hard
model-imposed ceiling, so exact precision isn't required. Falls back to a
chars/4 heuristic if tiktoken isn't installed.
"""

import json

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENCODING = None

# Vision encoders emit a small, roughly fixed number of tokens per image regardless
# of resolution - counting the raw base64 string length would wildly overestimate
# (a 100k-char data URL is not 100k/4 tokens once it reaches the model).
IMAGE_TOKEN_ESTIMATE = 300


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return max(1, len(text) // 4)


def count_message_tokens(message: dict) -> int:
    """Approximate token cost of one OpenAI-format chat message (content, tool_calls)."""
    content = message.get("content")
    total = 0

    if isinstance(content, str):
        total += count_tokens(content)
    elif isinstance(content, list):
        for block in content:
            if block.get("type") == "image_url":
                total += IMAGE_TOKEN_ESTIMATE
            else:
                total += count_tokens(block.get("text", "") or str(block))

    if message.get("tool_calls"):
        total += count_tokens(json.dumps(message["tool_calls"]))

    return total


def count_messages_tokens(messages: list) -> int:
    return sum(count_message_tokens(m) for m in messages)
