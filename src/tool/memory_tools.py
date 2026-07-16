"""
memory_tools.py - LLM-facing memory retrieval tools: search_memory, search_documents.

Both are backed by the same LongTermMemory instance (memory/long_term_memory.py) -
they only differ in which `kind` they search. Both are always available to the model;
search_documents just returns "No matching results found." if nothing is loaded yet.
"""

from luxai.magpie.schema import McpSchema

from memory.long_term_memory import CHAT_KIND, DOCUMENT_KIND

from .tool_base import ToolBase


class MemoryTools(ToolBase):

    def __init__(self, long_term):
        super().__init__()
        self.long_term = long_term

    def register(self, schema: McpSchema) -> None:
        schema.method()(self.search_memory)
        schema.method()(self.search_documents)

    def search_memory(self, query: str, time_hint: str = None) -> str:
        """Search the robot's own archived conversation history for things the user or
        robot said or did earlier - this session or a previous one. Not for looking up
        reference documents, use search_documents for that.
        time_hint: optional, one of "today", "yesterday", "this_week", "last_week",
        "this_month", "earlier" - set this when the user gives a time reference (e.g.
        "what did I tell you yesterday?") to narrow the search to that window."""
        results = self.long_term.search(query, kind=CHAT_KIND, time_hint=time_hint)
        return _format_results(results)

    def search_documents(self, query: str) -> str:
        """Search the loaded reference documents for content relevant to the query. Use
        this when the user asks about something that might be covered by a loaded
        document - check the document list given in your context for what's available
        before deciding this is the right tool."""
        results = self.long_term.search(query, kind=DOCUMENT_KIND)
        return _format_results(results)


def _format_results(results: list[dict]) -> str:
    if not results:
        return "No matching results found."
    lines = []
    for r in results:
        tag = f" ({r['when']})" if r.get("when") else ""
        source = f" [{r['meta']['source']}]" if r.get("meta", {}).get("source") else ""
        lines.append(f"-{tag}{source} {r['text']}")
    return "\n".join(lines)
