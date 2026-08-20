"""LLM-facing tools for conversation-memory and document retrieval."""

from luxai.magpie.schema import McpSchema

from memory.long_term_memory import CHAT_KIND, DOCUMENT_KIND, LongTermMemory
from tool.tool_base import ToolBase

MEMORY_TOOL_INSTRUCTIONS = """
The recent conversation is already available in your context. Use search_memory only
when you need information from an older conversation that is no longer visible. Use
search_documents when the answer may be found in the loaded reference documents.
If no document inventory is listed, do not call search_documents.
Do not claim that you searched memory or documents unless you called the relevant tool.
""".strip()


class MemoryTools(ToolBase):
    """Expose one search tool for chat history and one for reference documents."""

    def __init__(self, long_term: LongTermMemory) -> None:
        super().__init__()
        self.long_term = long_term

    def register(self, schema: McpSchema) -> None:
        schema.method()(self.search_memory)
        schema.method()(self.search_documents)

    def search_memory(self, query: str, time_hint: str | None = None) -> str:
        """Search older conversations with the user.

        Use this only when the needed information is no longer present in the recent
        conversation. ``time_hint`` may be today, yesterday, this_week, last_week,
        this_month, or earlier when the user provides a time reference.
        """
        results = self.long_term.search(
            query,
            kind=CHAT_KIND,
            time_hint=time_hint,
        )
        return _format_results(results)

    def search_documents(self, query: str) -> str:
        """Search the loaded reference documents for information relevant to a query."""
        results = self.long_term.search(query, kind=DOCUMENT_KIND)
        return _format_results(results)


def _format_results(results: list[dict]) -> str:
    if not results:
        return "No matching results found."

    lines = []
    for result in results:
        when = f" ({result['when']})" if result.get("when") else ""
        meta = result.get("meta", {})
        source = f" [{meta['source']}]" if meta.get("source") else ""
        role = f" [{meta['role']}]" if meta.get("role") else ""
        lines.append(f"-{when}{source}{role} {result['text']}")
    return "\n".join(lines)
