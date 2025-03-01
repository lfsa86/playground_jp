from typing import Dict, List

from pydantic import BaseModel


class Page(BaseModel):
    """
    Represents a page in the document, containing its number and a list of paragraphs.

    Attributes:
        page_number (Optional[int]): The page number. Can be None if not found.
        paragraphs (List[str]): A list of paragraphs in the page.
    """

    page_summary: str
    page_number: int
    content: List


class Document(BaseModel):
    """
    Represents the entire document, including its pages and metadata.

    Attributes:
        paginas (List[Page]): A list of Page objects representing the document's pages.
        metadata (Dict[str, str]): A dictionary containing document metadata.
    """

    pages: List[Page]
    metadata: Dict
