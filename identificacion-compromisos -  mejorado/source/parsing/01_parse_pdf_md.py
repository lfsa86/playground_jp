from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Union, Tuple

import pymupdf4llm  # pip install pymupdf4llm

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ImprovedPDFParser:
    """
    Parser mejorado para extraer Markdown desde un PDF usando pymupdf4llm.
    Ahora guarda el archivo directamente en output_dir, sin subcarpeta intermedia.
    """

    def __init__(
        self,
        page_separator: str = "-----",
    ) -> None:
        self.page_separator = page_separator

    def _ensure_dirs(self, out_dir: Union[str, Path]) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def extract_raw_markdown(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path] = "data/processed/datamd",  # ← Ruta directa por defecto
        output_filename: Optional[str] = None,  # ← Opcional: nombre personalizado
    ) -> Tuple[Path, str]:
        """
        Extrae Markdown del PDF y lo guarda directamente en output_dir.

        Parameters
        ----------
        output_dir : str o Path
            Carpeta donde guardar el .md (se crea si no existe).
        output_filename : str, optional
            Nombre personalizado para el archivo. Si no se da, usa {stem}_raw.md

        Returns
        -------
        (ruta_md, texto_md)
        """
        pdf_path = Path(input_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"No se encontró el PDF: {pdf_path}")

        base_out = self._ensure_dirs(output_dir)

        # Nombre del archivo de salida
        if output_filename:
            raw_md_path = base_out / output_filename
        else:
            raw_md_path = base_out / f"{pdf_path.stem}_raw.md"

        logger.info(f"Procesando PDF con pymupdf4llm: {pdf_path.name}")
        try:
            md_text = pymupdf4llm.to_markdown(str(pdf_path))

            if not md_text.strip():
                logger.warning("No se detectó texto nativo. El PDF podría ser escaneado (considera OCR).")

        except Exception as e:
            raise RuntimeError(f"Fallo al procesar el PDF: {e}") from e

        raw_text = md_text.strip() + "\n"
        raw_md_path.write_text(raw_text, encoding="utf-8", errors="ignore")
        logger.info(f"Markdown RAW guardado directamente en: {raw_md_path}")

        return raw_md_path, raw_text

    def process_pdf(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path] = "data/processed/datamd",
        output_filename: Optional[str] = None,
    ) -> str:
        """Atajo: devuelve directamente el markdown como string."""
        _, text = self.extract_raw_markdown(
            input_path=input_path,
            output_dir=output_dir,
            output_filename=output_filename
        )
        return text


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python -m source.parsing.pdf_mistral_parser <ruta_pdf> [output_dir] [nombre_archivo_salida.md]")
        print("Ejemplos:")
        print("   python script.py documento.pdf")
        print("   python script.py documento.pdf data/processed/datamd")
        print("   python script.py documento.pdf data/processed/datamd mi_archivo_raw.md")
        raise SystemExit(1)

    inp = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed/datamd")
    out_filename = sys.argv[3] if len(sys.argv) > 3 else None

    parser = ImprovedPDFParser()
    md_path, _ = parser.extract_raw_markdown(inp, out_dir, out_filename)
    print(f"✅ Listo. Archivo generado directamente en: {md_path}")