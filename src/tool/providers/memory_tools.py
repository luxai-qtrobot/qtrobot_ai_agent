"""LLM-facing tools for conversation-memory and document retrieval."""

from luxai.magpie.schema import McpSchema

from memory.long_term_memory import CHAT_KIND, DOCUMENT_KIND, LongTermMemory
from tool.tool_base import ToolBase

MEMORY_SEARCH_INSTRUCTIONS = """The recent conversation is already available in
your context. Use search_memory only when you need information from an older
conversation that is no longer visible."""

DOCUMENT_SEARCH_INSTRUCTIONS = """Use search_documents when the answer may be
found in the loaded reference documents. If no document inventory is listed,
do not call search_documents."""

MEMORY_TOOL_INSTRUCTIONS = (
    f"{MEMORY_SEARCH_INSTRUCTIONS}\n{DOCUMENT_SEARCH_INSTRUCTIONS}\n"
    "Do not claim that you searched memory or documents unless you called the "
    "relevant tool."
)


def memory_tool_instructions(
    *,
    memory_enabled: bool,
    documents_enabled: bool,
) -> str:
    sections = []
    if memory_enabled:
        sections.append(MEMORY_SEARCH_INSTRUCTIONS)
    if documents_enabled:
        sections.append(DOCUMENT_SEARCH_INSTRUCTIONS)
    if sections:
        sections.append(
            "Do not claim that you searched memory or documents unless you "
            "called the relevant tool."
        )
    return "\n".join(sections)


class MemoryTools(ToolBase):
    """Expose one search tool for chat history and one for reference documents."""

    def __init__(
        self,
        long_term: LongTermMemory,
        *,
        memory_enabled: bool = True,
        documents_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.long_term = long_term
        self.memory_enabled = memory_enabled
        self.documents_enabled = documents_enabled

    def register(self, schema: McpSchema) -> None:
        if self.memory_enabled:
            schema.method()(self.search_memory)
        if self.documents_enabled:
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
