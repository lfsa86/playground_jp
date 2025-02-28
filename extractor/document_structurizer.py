import re
from typing import Dict, List, Optional

from pydantic import BaseModel


class Page(BaseModel):
    """
    Represents a page in the document, containing its number and a list of paragraphs.

    Attributes:
        page_number (Optional[int]): The page number. Can be None if not found.
        paragraphs (List[str]): A list of paragraphs in the page.
    """

    page_number: Optional[int]
    paragraphs: List[str]


class Document(BaseModel):
    """
    Represents the entire document, including its pages and metadata.

    Attributes:
        paginas (List[Page]): A list of Page objects representing the document's pages.
        metadata (Dict[str, str]): A dictionary containing document metadata.
    """

    paginas: List[Page]
    metadata: Dict[str, str]


class DocumentStructurizer:
    """
    Handles reading, processing, and structuring a markdown document by segmenting it into pages.
    """

    def __init__(self):
        """
        Initializes the DocumentStructurizer without requiring a file path upfront.
        """
        self.content = None

    def load_file(self, file_path: str):
        """
        Loads the content of a markdown file.

        Args:
            file_path (str): The path to the markdown file.

        Raises:
            FileNotFoundError: If the file is not found.
            Exception: If there is an issue reading the file.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                self.content = file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"El archivo en {file_path} no se encontró.")
        except Exception as e:
            raise Exception(f"Ocurrió un error al leer el archivo: {str(e)}")

    def _split_by_page_number(self, pattern: str = r"Página (\d+)") -> List[Page]:
        """
        Splits the loaded markdown content at each occurrence of 'Página X', where X is any number.

        Args:
            pattern (str): The regular expression pattern to identify page numbers.

        Returns:
            List[Page]: A list of Page objects, each containing the page number and its paragraphs.
        """
        if self.content is None:
            raise ValueError("No file has been loaded. Please call load_file() first.")

        matches = list(re.finditer(pattern, self.content))

        if not matches:
            return [Page(page_number=None, paragraphs=self.content.split("\n\n"))]

        pages = []
        start_idx = 0
        last_page_number = None

        for match in matches:
            page_number = int(match.group(1))
            end_idx = match.start()
            page_content = self.content[start_idx:end_idx].strip()

            if page_content:
                paragraphs = [
                    p.strip() for p in page_content.split("\n\n") if p.strip()
                ]
                pages.append(Page(page_number=last_page_number, paragraphs=paragraphs))

            start_idx = match.end()
            last_page_number = page_number

        last_page_content = self.content[start_idx:].strip()
        if last_page_content:
            paragraphs = [
                p.strip() for p in last_page_content.split("\n\n") if p.strip()
            ]
            pages.append(Page(page_number=last_page_number, paragraphs=paragraphs))

        return pages

    def process_document(self) -> List[Page]:
        """
        Processes the loaded markdown document, segmenting it into pages.

        Returns:
            List[Page]: A list of Page objects containing structured document pages.
        """
        if self.content is None:
            raise ValueError("No file has been loaded. Please call load_file() first.")

        return self._split_by_page_number()
