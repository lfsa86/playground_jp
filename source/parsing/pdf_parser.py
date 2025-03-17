"""Module for parsing PDF files using Mistral OCR."""

import os
from pathlib import Path
from typing import Dict, Union

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()


class PDFParser:
    """A class to handle PDF parsing operations using Mistral OCR."""

    def __init__(self) -> None:
        """Initialize PDFParser instance."""
        pass

    @staticmethod
    def process_pdf(
        input_path: Union[str, Path], output_dir: Union[str, Path] = "data/processed"
    ) -> Dict[str, str]:
        """Process PDF file using Mistral OCR and return the combined content in markdown format.

        Args:
            input_path: Path to the PDF file, can be string or Path object.
            output_dir: Directory path where processed files will be saved (default: 'data/processed').

        Returns:
            Dictionary containing:
                - file_name: Name of the processed file (without extension)
                - md_text: Combined markdown content from all pages

        Raises:
            FileNotFoundError: If the input PDF file doesn't exist.
            MistralError: If OCR processing fails.
            OSError: If output directory creation fails.
        """
        # Convert paths to Path objects
        pdf_path = Path(input_path) if isinstance(input_path, str) else input_path
        output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir

        # Create full output path including file stem
        file_output_path = output_path / pdf_path.stem

        # Verify input file exists
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create output directory if it doesn't exist
        file_output_path.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        output_file = file_output_path / f"{pdf_path.stem}.md"

        # Initialize Mistral client
        client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

        try:
            # Upload PDF file
            with pdf_path.open("rb") as pdf_file:
                uploaded_pdf = client.files.upload(
                    file={
                        "file_name": pdf_path.name,
                        "content": pdf_file,
                    },
                    purpose="ocr",
                )

            # Get signed URL for processing
            signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

            # Process document with OCR
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                },
            )

            # Combine markdown content from all pages
            combined_md_content = ""
            for page in ocr_response.pages:
                combined_md_content += f"{page.markdown}\n-----\n"

            # Save content to file
            output_file.write_text(combined_md_content)

            return combined_md_content

        except Exception as e:
            raise Exception(f"Error processing PDF {pdf_path.name}: {str(e)}")
