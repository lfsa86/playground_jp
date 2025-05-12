"""Module for parsing PDF files using Mistral OCR."""

import os
import re
from pathlib import Path
from typing import Dict, Union

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

class PDFParser:
    """A class to extract raw markdown from PDF using Mistral OCR."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def extract_raw_markdown(
        input_path: Union[str, Path], output_dir: Union[str, Path] = "data/processed"
    ) -> Path:
        pdf_path = Path(input_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_output_path = output_path / pdf_path.stem
        file_output_path.mkdir(parents=True, exist_ok=True)

        raw_output_file = file_output_path / f"{pdf_path.stem}_raw.md"

        client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

        try:
            with pdf_path.open("rb") as pdf_file:
                uploaded_pdf = client.files.upload(
                    file={"file_name": pdf_path.name, "content": pdf_file},
                    purpose="ocr",
                )

            signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "document_url", "document_url": signed_url.url},
            )

            full_text = ""
            for i, page in enumerate(ocr_response.pages):
                print(f"Procesando página {i+1}/{len(ocr_response.pages)}")
                page_text = page.markdown.strip()
                if not page_text:
                    print(f"⚠️ Página {i+1} sin contenido detectable.")
                full_text += f"{page_text}\n\n-----\n\n\n"

            with raw_output_file.open("w", encoding="utf-8", errors="ignore") as f:
                f.write(full_text)

            print(f"📄 Markdown RAW guardado en: {raw_output_file}")
            return raw_output_file  # DEVOLVEMOS LA RUTA

        except Exception as e:
            raise Exception(f"Error procesando {pdf_path.name}: {str(e)}")
        
    process_pdf = extract_raw_markdown

