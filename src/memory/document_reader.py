"""Read local reference documents for :class:`LongTermMemory`."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pypdf

DEFAULT_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    """Extracted document text and its retrieval metadata."""

    text: str
    meta: dict
    summary: str


class DocumentReader:
    """Extract text from one supported document."""

    @staticmethod
    def read(path, file_type: str | None = None, summary: str = "") -> Document:
        path = Path(path)
        extension = f".{file_type.lstrip('.')}" if file_type else path.suffix.lower()
        text = (
            DocumentReader._read_pdf(path)
            if extension == ".pdf"
            else path.read_text(encoding="utf-8")
        )
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
    """Read supported documents from a directory."""

    @staticmethod
    def read(
        dir_path,
        recursive: bool = False,
        extensions: set[str] | None = None,
        max_docs: int | None = None,
        summary: str = "",
    ) -> list[Document]:
        directory = Path(dir_path)
        accepted_extensions = extensions or DEFAULT_EXTENSIONS
        paths = directory.rglob("*") if recursive else directory.iterdir()
        files = sorted(
            path
            for path in paths
            if path.is_file() and path.suffix.lower() in accepted_extensions
        )
        if max_docs is not None:
            files = files[:max_docs]

        return [DocumentReader.read(path, summary=summary) for path in files]
