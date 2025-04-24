"""Module for parsing PDF files using Mistral OCR."""

import os
import re
from pathlib import Path
from typing import Dict, Union

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

def segmentar_por_nodos_heading(md_text: str) -> Dict[str, Dict[str, str]]:
    """
    Segmenta el markdown en nodos temáticos detectando encabezados tipo 11.1.1, 11.1.1.1, etc.
    
    Retorna un diccionario con claves como '11.1.1.1' y valores que contienen:
    - 'titulo': Título de la sección
    - 'contenido': Markdown correspondiente
    - 'capitulo': Primer número (ej: '11')
    - 'subseccion': ID completo (ej: '11.1.1.1')
    - 'nivel': cantidad de niveles numéricos (3 o más)
    """
    pattern = re.compile(r"###\s+((\d+\.\d+\.\d+(?:\.\d+)*))\s+(.*)")

    nodos = {}
    matches = list(pattern.finditer(md_text))

    for i, match in enumerate(matches):
        subseccion = match.group(1)
        titulo = match.group(3).strip()
        niveles = subseccion.split(".")
        capitulo = niveles[0]
        nivel = len(niveles)

        inicio = match.end()
        fin = matches[i+1].start() if i+1 < len(matches) else len(md_text)

        contenido = md_text[inicio:fin].strip()
        nodos[subseccion] = {
            "titulo": titulo,
            "contenido": contenido,
            "capitulo": capitulo,
            "subseccion": subseccion,
            "nivel": nivel
        }

    return nodos

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

            # Palabras clave contextuales
            componentes_tematicos = [
                "calidad de aire", "ruido", "suelos", "agua", "efluentes", "residuos",
                "biodiversidad", "flora", "fauna", "población", "comunidad", "paisaje"
            ]
            subtemas = ["objetivo", "condiciones", "medidas", "etapa", "actividad"]
            
            # Combine markdown content from all pages
            combined_md_content = ""
            for i, page in enumerate(ocr_response.pages, start=1):
                lines = page.markdown.splitlines()

                # Detectar si esta página parece una tabla de contenido
                toc_like_lines = sum(
                    1 for line in lines
                    if re.match(r'^\s*\d+(\.\d+)+\s+.*\.+\s*\$?\d', line)
                )
                if toc_like_lines >= 5:
                    continue  # ❌ saltar esta página (es muy probable que sea tabla de contenido)

                # Si no es tabla de contenido, seguimos procesando normalmente
                buffer = ""
                inside_table = False

                for line in lines:
                    stripped = line.strip()

                    # Preservar tablas
                    if stripped.startswith("|") and "|" in stripped:
                        if not inside_table:
                            if buffer:
                                combined_md_content += buffer + "\n"
                                buffer = ""
                            inside_table = True
                        combined_md_content += line + "\n"
                        continue
                    elif inside_table and not stripped.startswith("|"):
                        inside_table = False

                    # Encabezado numérico (ej. 11.1.1.1)
                    match = re.match(r"(#+)?\s*(\d+(?:\.\d+)+)\s+(.*)", stripped)
                    # Corregir encabezados sin numeración mal formateados (ej. "# Etapa de cierre")
                    if re.match(r"^#+\s+[A-ZÁÉÍÓÚÜÑ][a-z].*", stripped) and not re.search(r'\d+\.', stripped):
                        buffer += f"**{stripped.lstrip('#').strip()}**\n"
                        continue

                    if match:
                        if buffer:
                            combined_md_content += buffer + "\n"
                            buffer = ""

                        numeracion = match.group(2)
                        titulo = match.group(3).strip()
                        niveles = numeracion.split(".")
                        nivel = len(niveles)
                        titulo_lower = titulo.lower()

                        if nivel == 1:
                            formatted = f"# {numeracion} {titulo}"
                        elif any(p in titulo_lower for p in componentes_tematicos):
                            formatted = f"### {numeracion} {titulo}"
                        elif any(p in titulo_lower for p in subtemas):
                            formatted = f"**{numeracion} {titulo}**"
                        else:
                            formatted = f"#### {numeracion} {titulo}"

                        combined_md_content += formatted + "\n"

                    else:
                        # Si es una línea aislada como "Etapa de cierre", no la conviertas en heading
                        if stripped and any(p in stripped.lower() for p in subtemas):
                            buffer += f"**{stripped}**\n"
                        else:
                            buffer += line + "\n"

                if buffer:
                    combined_md_content += buffer + "\n"

                combined_md_content += f"\n<!-- Página {i} -->\n\n"

            # 🔧 FILTRADO DE ARTEFACTOS COMO 'EPISAMIL'
            filtered_md_content = "\n".join(
                line for line in combined_md_content.splitlines()
                if not re.search(r'\bEPISAMIL\b', line, re.IGNORECASE)
                and not re.match(r'#\s*AtkinsRéalis', line, re.IGNORECASE)
                and not re.match(r'#\s*FOLIO', line, re.IGNORECASE)
            )

            # 🧠 Segmentar por nodos temáticos tipo 11.1.1.1, 11.1.1.2, etc.
            nodos = segmentar_por_nodos_heading(filtered_md_content)

            # Save filtered content to file
            output_file.write_text(filtered_md_content, encoding='utf-8')

            # Return the cleaned markdown
            return {
                "file_name": pdf_path.stem,
                "md_text": filtered_md_content,
                "nodos": nodos
            }

        except Exception as e:
            raise Exception(f"Error processing PDF {pdf_path.name}: {str(e)}")
