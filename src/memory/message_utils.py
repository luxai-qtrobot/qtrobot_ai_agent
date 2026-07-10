"""
message_utils.py - Shared helpers for rendering chat messages as plain text, used by
any memory tier that needs to hand raw conversation content to an LLM call or an
embedding model (which only take plain text, not the OpenAI message shape).
"""


def raw_excerpt(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content")
        if isinstance(content, str):
            lines.append(f"{role}: {content}")
        elif isinstance(content, list):
            lines.append(f"{role}: <non-text content>")
    return "\n".join(lines)
