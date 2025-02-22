from typing import Dict, List

from llama_index.core import Document, SimpleDirectoryReader
from llama_parse import LlamaParse


class DocumentParser:
    """A class to parse documents from a directory or single file."""

    def __init__(self) -> None:
        """Initialize the DocumentParser with LlamaParse configuration."""
        self.parser: LlamaParse = LlamaParse(
            result_type="markdown",
            auto_mode_trigger_on_table_in_page=True,
            output_tables_as_HTML=True,
            language="es",
            show_progress=True,
        )
        self.file_extractor: Dict[str, LlamaParse] = {".pdf": self.parser}

    def _parse_documents(self, **reader_kwargs) -> List[Document]:
        """
        Internal method to parse documents using SimpleDirectoryReader.

        Args:
            **reader_kwargs: Keyword arguments for SimpleDirectoryReader.

        Returns:
            List of parsed Document objects.
        """
        documents: List[Document] = SimpleDirectoryReader(
            file_extractor=self.file_extractor, **reader_kwargs
        ).load_data()
        return documents

    def parse_dir(self, input_dir: str) -> List[Document]:
        """
        Parse documents from the specified directory.

        Args:
            input_dir: Path to the input directory containing documents.

        Returns:
            List of parsed Document objects.
        """
        documents = self._parse_documents(input_dir=input_dir)
        print(f"Parsed {len(documents)} documents from '{input_dir}'.")
        return documents

    def parse_file(self, input_file_path: str) -> List[Document]:
        """
        Parse a single document from document file path.

        Args:
            input_file_path: Path to the document file.

        Returns:
            List of parsed Document objects.
        """
        documents = self._parse_documents(input_files=[input_file_path])
        print(f"Parsed {len(documents)} documents from '{input_file_path}'.")
        return documents

    # TODO add method to parse file from URL
