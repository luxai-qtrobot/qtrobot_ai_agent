from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from memory import DirectoryReader, DocumentReader


class DocumentReaderTests(unittest.TestCase):
    def test_directory_reader_loads_supported_files_in_name_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "b.md").write_text("second", encoding="utf-8")
            (directory / "a.txt").write_text("first", encoding="utf-8")
            (directory / "ignored.bin").write_bytes(b"ignored")

            documents = DirectoryReader.read(directory, summary="Test documents")

            self.assertEqual(
                [document.meta["source"] for document in documents],
                ["a.txt", "b.md"],
            )
            self.assertEqual([document.text for document in documents], ["first", "second"])
            self.assertTrue(
                all(document.summary == "Test documents" for document in documents)
            )

    def test_single_document_includes_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.txt"
            path.write_text("hello", encoding="utf-8")

            document = DocumentReader.read(path, summary="Notes")

            self.assertEqual(document.text, "hello")
            self.assertEqual(document.meta["source"], "notes.txt")
            self.assertEqual(document.meta["path"], str(path))
            self.assertEqual(document.summary, "Notes")


if __name__ == "__main__":
    unittest.main()
