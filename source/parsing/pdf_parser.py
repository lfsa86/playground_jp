"""Module for parsing PDF files using Mistral OCR (mejorado)."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Union, List

from dotenv import load_dotenv

try:
    from mistralai import Mistral
except ImportError as e:
    raise ImportError("Falta instalar 'mistralai'. Ejecuta: pip install mistralai") from e

load_dotenv()

# =========================================================
# Utilidad: segmentación por encabezados numerados
# =========================================================
def segmentar_por_nodos_heading(md_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Segmenta el markdown en nodos temáticos detectando encabezados tipo:
    ## 11.1, ### 11.1.1, #### 11.1.1.1, etc.

    Retorna un dict con claves '11.1.1.1' y valores:
      - 'titulo', 'contenido', 'capitulo', 'subseccion', 'nivel'
    """
    pattern = re.compile(r"^#{2,6}\s+((\d+(?:\.\d+)+))\s+(.*)$", re.MULTILINE)

    nodos: Dict[str, Dict[str, Any]] = {}
    matches = list(pattern.finditer(md_text))

    for i, match in enumerate(matches):
        subseccion = match.group(1)
        titulo = match.group(3).strip()
        niveles = subseccion.split(".")
        capitulo = niveles[0]
        nivel = len(niveles)
        inicio = match.end()
        fin = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        contenido = md_text[inicio:fin].strip()
        nodos[subseccion] = {
            "titulo": titulo,
            "contenido": contenido,
            "capitulo": capitulo,
            "subseccion": subseccion,
            "nivel": nivel,
        }

    return nodos


# =========================================================
# PDF → Markdown con Mistral OCR
# =========================================================
class PDFParser:
    """Parsea PDF con Mistral OCR y genera Markdown por todo el documento."""

    def __init__(self) -> None:
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise RuntimeError("No se encontró MISTRAL_API_KEY en el entorno (.env).")
        self.client = Mistral(api_key=self.api_key)

        # Heurísticas/regex usados en limpieza
        self.re_toc_line = re.compile(r"^\s*\d+(?:\.\d+)+\s+.+?\.{2,}\s*\$?\d+\s*$")
        self.re_num_heading = re.compile(r"^(?:#+\s*)?(\d+(?:\.\d+)+)\.?\s+(.*)$")
        self.re_alpha_item = re.compile(r"^[A-Z]\.\s+.+$")
        self.re_hash_heading = re.compile(r"^#{1,6}\s+.+$")
        self.re_dup_numeral = re.compile(r"(\d+(?:\.\d+)+\.)\s+\1")
        self.re_inline_heading_in_bold = re.compile(r"^\*+\s*(#{1,6}\s+.+?)\s*\*+\s*$")

        # --- NUEVO: patrones para header/footer por página ---
        # Números de página (Pág. 31, 31, 31/120, Página 31, etc.)
        self.re_page_num = re.compile(
            r"^\s*(?:P(?:á|a)g(?:\.|ina)?\.?\s*)?\d{1,5}(?:\s*/\s*\d+)?\s*$",
            re.IGNORECASE,
        )
        # Códigos/folios/logos breves tipo "00093", "ABC-123"
        self.re_doc_code = re.compile(r"^\s*[A-Z0-9\-_/]{5,}\s*$")
        # Línea casi toda en mayúsculas (suelen ser sellos, razones sociales, etc.)
        self.re_all_caps = re.compile(r"^[^a-záéíóúñ]*[A-ZÁÉÍÓÚÑ]{3,}[^a-záéíóúñ]*$")
        # Frases típicas de pie/encabezado institucionales (ajustable)
        self.re_footer_org = re.compile(
            r"(autopista del norte|fcisa|elaborado por|prepared by|copyright|firma)",
            re.IGNORECASE,
        )

    # ---------------- TOC detection mejorado ----------------
    def _looks_like_toc(self, lines: List[str]) -> bool:
        """
        Detecta páginas de TOC con varias heurísticas:
        - ≥4 líneas con patrón “num ... ..... pág”
        - o presencia de “ÍNDICE” + ≥2 líneas patrón
        - o densidad alta de puntos suspensivos con números al final.
        """
        norm = [self._strip_md_markup(ln) for ln in lines]
        dots_lines = sum(1 for ln in norm if re.search(r"\.{2,}\s*\d+\s*$", ln))
        numdot_lines = sum(1 for ln in norm if self.re_toc_line.match(ln))
        has_indice = any("indice" in ln.lower() or "índice" in ln.lower() for ln in norm)

        if numdot_lines >= 4:
            return True
        if has_indice and numdot_lines >= 2:
            return True
        if dots_lines >= 6 and numdot_lines >= 2:
            return True
        return False

    # ---------------- Artefactos/basura ----------------
    @staticmethod
    def _clean_artifacts(line: str) -> bool:
        if re.search(r"\bEPISAMIL\b", line, re.IGNORECASE):
            return False
        if re.match(r"#\s*Atkins[Ré]alis", line, re.IGNORECASE):
            return False
        if re.match(r"#\s*FOLIO", line, re.IGNORECASE):
            return False
        return True

    @staticmethod
    def _strip_md_markup(s: str) -> str:
        """Quita **, __, #, espacios extras para detección robusta."""
        s = re.sub(r"[*_`]+", "", s)
        s = re.sub(r"^#{1,6}\s*", "", s)
        return s.strip()

    # ---------------- NUEVO: detección de contenido / pie ----------------
    def _is_content_line(self, s: str) -> bool:
        """Heurística: decide si una línea parece contenido real del documento."""
        stripped = s.strip()
        if not stripped:
            return False
        if stripped.startswith("|") and "|" in stripped:  # tablas
            return True
        if self.re_num_heading.match(stripped):           # "3.3.1 Título"
            return True
        if self.re_hash_heading.match(stripped):          # "# Título"
            return True
        if re.match(r"^\s*Tabla\s+\d+(?:[\-–]\d+)?\b", stripped, re.IGNORECASE):
            return True
        if re.match(r"^\s*[-•]\s+\S+", stripped):         # viñetas
            return True
        # Párrafo: minúsculas + longitud razonable
        if re.search(r"[a-záéíóúñ]{3,}", stripped) and len(stripped) >= 40:
            return True
        return False

    def _is_footer_line(self, s: str) -> bool:
        """Heurística: decide si una línea parece pie de página o ruido inferior."""
        stripped = s.strip()
        if not stripped:
            return True
        if self.re_page_num.match(stripped):
            return True
        if self.re_footer_org.search(stripped):
            return True
        if self.re_all_caps.match(stripped) and len(stripped) <= 60:
            return True
        if self.re_doc_code.match(stripped) and len(stripped) <= 12:
            return True
        return False

    def _strip_page_headers_and_footers(self, lines: List[str], page_no: int) -> List[str]:
        """
        Corta:
          - HEADER: desde el tope hasta la primera línea que parezca contenido.
          - FOOTER: desde el final hacia arriba, corta bloques que parezcan pie.
        Conserva tablas y headings. Si no detecta nada, devuelve igual.
        """
        if not lines:
            return lines

        # HEADER: primera línea que luce contenido
        start_idx = 0
        for i, ln in enumerate(lines):
            if self._is_content_line(ln):
                start_idx = i
                break

        # FOOTER: run final de líneas-que-parecen-pie hasta topar contenido
        end_idx = len(lines)
        j = len(lines) - 1
        seen_content = False
        tail_cut = 0
        while j >= start_idx:
            ln = lines[j]
            if not seen_content:
                if self._is_footer_line(ln):
                    tail_cut += 1
                    j -= 1
                    continue
                seen_content = True
                j -= 1
            else:
                break
        if tail_cut > 0:
            end_idx = len(lines) - tail_cut

        sliced = lines[start_idx:end_idx]

        # Limpieza adicional en el borde superior recortado
        while sliced and (
            self.re_all_caps.match(sliced[0].strip()) or self.re_doc_code.match(sliced[0].strip())
        ):
            sliced.pop(0)

        return sliced

    # ---------------- Normalización de líneas ----------------
    def _format_heading_or_text(self, stripped: str) -> str:
        """
        Reglas:
        - Encabezado numérico -> Hx según profundidad: 1->##, 2->###, 3->####, 4->#####, 5->######.
        - “CAPÍTULO X” -> # CAPÍTULO X
        - Segunda línea tipo “DESCRIPCIÓN …” -> ## DESCRIPCIÓN …
        - Items “A. Texto”/“B. Texto” -> bullet: "- A. Texto".
        - Quita negritas falsas que envuelven headings (**### …**).
        """
        m_bold_head = self.re_inline_heading_in_bold.match(stripped)
        if m_bold_head:
            stripped = m_bold_head.group(1).strip()

        if re.match(r"^(cap[ií]tulo)\s+[ivx\d]+", stripped, re.IGNORECASE):
            return f"# {self._strip_md_markup(stripped)}"

        if re.match(r"^(descripci[oó]n)\b", stripped, re.IGNORECASE):
            return f"## {self._strip_md_markup(stripped)}"

        m_num = self.re_num_heading.match(stripped)
        if m_num:
            numeracion = m_num.group(1)
            titulo = m_num.group(2).strip()
            niveles = numeracion.rstrip(".").split(".")
            depth = max(1, min(5, len(niveles)))
            hashes = "#" * (depth + 1)  # 1->##, 2->###, 3->####, 4->#####, 5->######
            return f"{hashes} {numeracion}. {titulo}"

        if self.re_alpha_item.match(stripped):
            return f"- {stripped}"

        if self.re_hash_heading.match(stripped):
            return stripped

        return stripped

    def _normalize_page_lines(self, page_lines: List[str]) -> str:
        """
        Ensambla el contenido de la página preservando tablas,
        normaliza encabezados y arregla duplicaciones de numerales.
        """
        inside_table = False
        buffer: List[str] = []
        out: List[str] = []

        def flush_buffer():
            nonlocal buffer, out
            if buffer:
                out.append("\n".join(buffer))
                buffer = []

        for raw in page_lines:
            if not self._clean_artifacts(raw):
                continue

            line = raw.rstrip("\n")
            stripped = line.strip()

            # Mantener tablas intactas
            if stripped.startswith("|") and "|" in stripped:
                if not inside_table:
                    flush_buffer()
                    inside_table = True
                out.append(line)
                continue
            elif inside_table and not stripped.startswith("|"):
                inside_table = False

            # Normalización de encabezados / texto
            formatted = self._format_heading_or_text(stripped)

            # Arreglo de duplicaciones de numerales en headings
            if formatted.startswith("#"):
                formatted = self.re_dup_numeral.sub(r"\1", formatted)

            if formatted != stripped:
                flush_buffer()
                out.append(formatted)
            else:
                buffer.append(line)

        flush_buffer()
        page_body = "\n".join(out).strip()

        # Limpieza final
        page_body = re.sub(r"\*\s*(#{2,6}\s+)", r"\1", page_body)
        page_body = re.sub(r"(#{2,6}\s+[0-9.]+\.)\s+\1", r"\1", page_body)

        return page_body

    # ---------------- Orquestación ----------------
    def _poll_until_ready(self, file_id: str, timeout_s: int = 120) -> None:
        time.sleep(0.5)

    def process_pdf(
        self, input_path: Union[str, Path], output_dir: Union[str, Path] = "data/processed"
    ) -> Dict[str, Any]:
        """
        Procesa el PDF con Mistral OCR y retorna:
          - file_name
          - md_text (con <!-- Página N -->)
          - nodos (índice por encabezados numerados)
          - output_path (ruta del .md)
        """
        pdf_path = Path(input_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        output_path = Path(output_dir) / pdf_path.stem
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{pdf_path.stem}.md"

        # 1) Subir PDF
        with pdf_path.open("rb") as f:
            uploaded = self.client.files.upload(
                file={"file_name": pdf_path.name, "content": f},
                purpose="ocr",
            )

        # 2) URL firmada
        signed = self.client.files.get_signed_url(file_id=uploaded.id)
        self._poll_until_ready(uploaded.id)

        # 3) Ejecutar OCR
        ocr_response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": signed.url},
        )

        combined_blocks: List[str] = []
        toc_seen = False  # si detectamos que una página es TOC, seguimos saltándola hasta que aparezca contenido real

        for i, page in enumerate(ocr_response.pages, start=1):
            lines = page.markdown.splitlines()

            # --- NUEVO: recortar encabezados/pies por página ---
            lines = self._strip_page_headers_and_footers(lines, i)

            # Detectar TOC
            page_is_toc = self._looks_like_toc(lines)
            if page_is_toc:
                toc_seen = True
                combined_blocks.append(f"<!-- Página {i} -->\n")
                continue

            # Limpiar restos de TOC si venimos de páginas índice
            if toc_seen:
                cleaned_lines = []
                for ln in lines:
                    ln_norm = self._strip_md_markup(ln)
                    if self.re_toc_line.match(ln_norm) or re.search(r"\.{2,}\s*\d+\s*$", ln_norm):
                        continue
                    cleaned_lines.append(ln)
                lines = cleaned_lines

            # Normalizar contenido de la página (encabezados/tablas)
            page_body = self._normalize_page_lines(lines)

            if page_body.strip():
                combined_blocks.append(page_body)
            combined_blocks.append(f"\n<!-- Página {i} -->\n")

        # 4) Ensamble final
        md_text = "\n".join(combined_blocks).strip() + "\n"

        # 5) Segmentar nodos numerados
        nodos = segmentar_por_nodos_heading(md_text)

        # 6) Guardar
        output_file.write_text(md_text, encoding="utf-8")

        return {
            "file_name": pdf_path.stem,
            "md_text": md_text,
            "nodos": nodos,
            "output_path": str(output_file),
        }
