"""Module for parsing PDF files using Mistral OCR."""

import os
import re
from pathlib import Path
from typing import Dict, Union

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

def normalize_headings(md_text: str) -> str:
    """
    Detecta encabezados numerados y les agrega formato Markdown manteniendo el numeral,
    o en su defecto promueve a ## o ### encabezados textuales comunes.
    """
    lines = md_text.splitlines()
    new_lines = []
    for line in lines:
        stripped = line.strip()

        # Caso 1: encabezado numerado tipo "4.3.2 Manejo"
        match_num = re.match(r'^(\d+(\.\d+)*\.?)\s+([A-ZÁÉÍÓÚÜÑa-z0-9].+)$', stripped)
        if match_num:
            level = match_num.group(1).count(".") + 1
            new_lines.append(f"{'#' * level} {stripped}")
            continue

        # Caso 2: encabezado sin número, pero en mayúsculas
        if re.match(r'^[A-ZÁÉÍÓÚÜÑ ]{5,}$', stripped):
            new_lines.append(f"## {stripped.title()}")
            continue

        new_lines.append(line)
    return "\n".join(new_lines)

def promote_table_titles(md_text: str) -> str:
    """
    Detecta líneas de tabla con 'Nombre:', 'Carácter:' y 'Fase:', y:
    - Promueve 'Nombre' como encabezado ##.
    - Inserta la fila completa como texto estructurado Markdown debajo del encabezado.
    """
    lines = md_text.splitlines()
    new_lines = []
    for i, line in enumerate(lines):
        # Detectar línea con los tres campos
        match = re.match(
            r'\|\s*Nombre:\s*(.*?)\s*\|\s*Carácter:\s*(.*?)\s*\|\s*Fase:\s*(.*?)\s*\|?',
            line,
            re.IGNORECASE
        )
        if match:
            nombre = match.group(1).replace("<br>", " ").strip()
            caracter = match.group(2).strip()
            fase = match.group(3).strip()

            new_lines.append(f"## {nombre}")
            new_lines.append("")
            new_lines.append(f"**Nombre**: {nombre}")
            new_lines.append(f"**Carácter**: {caracter}")
            new_lines.append(f"**Fase**: {fase}")
            continue

        # Fallback: solo 'Nombre'
        match_simple = re.match(r'\|\s*Nombre:\s*(.*?)\s*\|', line, re.IGNORECASE)
        if match_simple:
            nombre = match_simple.group(1).replace("<br>", " ").strip()
            nombre = re.sub(r'\s+', ' ', nombre)
            new_lines.append(f"## {nombre}")
            new_lines.append("")
            new_lines.append(f"**Nombre**: {nombre}")
            continue

        new_lines.append(line)
    return "\n".join(new_lines)

def extract_all_two_column_subsections(md_text: str) -> str:
    """
    Detecta cualquier fila de tabla con exactamente dos columnas y convierte la primera celda en un encabezado Markdown seguido por el contenido de la segunda celda.

    Ejemplo:
    '| Descripción | Este es el contenido |' -> '### Descripción\n\nEste es el contenido'
    """
    lines = md_text.splitlines()
    new_lines = []
    for line in lines:
        match = re.match(r'^\|\s*(.+?)\s*\|\s*(.+?)\s*\|?$', line)
        if match:
            heading = match.group(1).strip().rstrip('.')
            content = match.group(2).strip()
            new_lines.append(f"### {heading}\n\n{content}")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

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
                page_md = normalize_headings(page.markdown)
                page_md = promote_table_titles(page_md)
                page_md = extract_all_two_column_subsections(page_md)
                combined_md_content += f"\n{page_md.strip()}\n\n-----\n"
            
            # Corrección post-procesamiento para headers mal formateados
            combined_md_content = re.sub(
            r'([^\n])(-{3,}\s*#+)',
            lambda m: f"{m.group(1)}\n{m.group(2).lstrip('-').strip()}",
            combined_md_content
            )
            
            # Save content to file
            output_file.write_text(combined_md_content, encoding='utf-8')

            return combined_md_content

        except Exception as e:
            raise Exception(f"Error processing PDF {pdf_path.name}: {str(e)}")
