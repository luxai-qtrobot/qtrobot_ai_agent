"""
document_reader.py - DocumentReader / DirectoryReader: minimal text extraction for
feeding LongTermMemory.add_document(). Deliberately separate from long_term_memory.py -
that file only knows text-in/vectors-out, this one only knows filesystem-in/text-out.

Two kinds of "extra info" about a document, kept apart (see LongTermMemory for how each
is used):
- meta - structural facts about the file (source/path/modified date), auto-extracted
  here, handed straight to LongTermMemory.add_document(meta=...) unchanged - it's
  already in the shape that call needs, no reassembly at the call site.
- summary - a human-written one-line description of what a document is *about*, used
  only for the LLM's static "documents available" listing. Can't be reliably
  auto-extracted, so readers never guess it - they just carry whatever the caller
  passed straight through, unchanged.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pypdf

DEFAULT_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    """Fields match LongTermMemory.add_document()'s keyword args exactly, so a caller
    just does add_document(text=doc.text, summary=doc.summary, meta=doc.meta) - no
    positional tuple unpacking, no reassembly."""
    text: str
    meta: dict
    summary: str


class DocumentReader:

    @staticmethod
    def read(path, file_type: str = None, summary: str = "") -> Document:
        """Reads one file. file_type overrides extension-based dispatch ('txt' or
        'pdf'); summary is passed through unchanged, for callers to hand straight to
        LongTermMemory.add_document()."""
        path = Path(path)
        ext = f".{file_type.lstrip('.')}" if file_type else path.suffix.lower()

        text = DocumentReader._read_pdf(path) if ext == ".pdf" else path.read_text(encoding="utf-8")
        meta = {
            "source": path.name,
            "path": str(path),
            "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d"),
        }
        return Document(text=text, meta=meta, summary=summary)

    @staticmethod
    def _read_pdf(path: Path) -> str:
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)


class DirectoryReader:

    @staticmethod
    def read(dir_path, recursive: bool = False, extensions: set = None,
             max_docs: int = None, summary: str = "") -> list[Document]:
        """Walks dir_path (optionally recursive) and reads every matching file (default
        extensions: DEFAULT_EXTENSIONS) via DocumentReader, stopping after max_docs if
        given. summary, if given, is shared across every file in this batch - see
        LongTermMemory.document_index_summary(), which dedups by summary text, so one
        shared summary here collapses to a single line for the LLM regardless of how
        many files it came from."""
        dir_path = Path(dir_path)
        extensions = extensions or DEFAULT_EXTENSIONS

        paths = dir_path.rglob("*") if recursive else dir_path.iterdir()
        files = sorted(p for p in paths if p.is_file() and p.suffix.lower() in extensions)
        if max_docs is not None:
            files = files[:max_docs]

        return [DocumentReader.read(path, summary=summary) for path in files]
