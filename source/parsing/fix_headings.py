"""
fix_headings.py — Normalización minimalista de encabezados Markdown (determinista y robusto).

Mejoras:
- Jerarquiza encabezados según numerales (3., 3.1., 3.3.8.2., etc.).
- Elimina encabezados falsos en negrita (**3.3.8.2. ...**) → convierte a encabezado real.
- Elimina la lógica de "democión" o negritas en TOC (solo encabezados con #).
- Limpieza estricta de tablas: si una tabla tiene celdas muy largas o ruido, se reemplaza por aviso.

Requisitos:
- langchain, tenacity, tqdm (solo si se usa en pipeline con LLM).
"""

from __future__ import annotations
import os, re, logging, threading, multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIG ---
os.environ["GRPC_FORK_SUPPORT_ENABLED"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- REGLAS DE PARSING ---
RE_BOLD_NUMERAL = re.compile(r"^\*\*\s*(\d+(?:\.\d+)+\.?)\s+(.+?)\s*\*\*$")
RE_NUMERAL_ONLY = re.compile(r"^(\d+(?:\.\d+)+\.?)\s+(.+)")
RE_MD_H = re.compile(r"^(#+)\s*(\d+(?:\.\d+)+\.?)\s+(.+)$")
RE_LEADER_DOTS = re.compile(r"\.{2,}\s*\d+\s*$")

RE_TABLE = re.compile(r"^\s*\|.*\|\s*$")
RE_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
RE_LONG_SEQ = re.compile(r"[A-Za-z0-9]{200,}")
RE_HTML_BREAKS = re.compile(r"(?:<br\s*/?>){5,}", flags=re.IGNORECASE)

# --- UTILIDADES ---
def _depth_from_numeral(numeral: str) -> int:
    return len(numeral.rstrip(".").split("."))

def _level_from_depth(depth: int, base_h: int = 2) -> int:
    return max(2, min(6, base_h - 1 + depth))

def _split_table_row(line: str) -> list[str]:
    line = line.strip().strip("|")
    return [c.strip() for c in line.split("|")]

# --- FUNCIONES PRINCIPALES ---
def retag_headings(md_text: str, base_h: int = 2) -> str:
    """
    Detecta numerales (3., 3.3., 3.3.8.2.) y aplica encabezado Markdown (# según profundidad).
    """
    out = []
    for line in md_text.splitlines():
        s = line.strip()
        # ## 3.3.1. Algo
        m1 = RE_MD_H.match(s)
        if m1:
            hashes, num, title = m1.groups()
            depth = _depth_from_numeral(num)
            level = "#" * _level_from_depth(depth, base_h)
            title = RE_LEADER_DOTS.sub("", title)
            out.append(f"{level} {num} {title}")
            continue
        # **3.3.8.2. Algo**
        m2 = RE_BOLD_NUMERAL.match(s)
        if m2:
            num, title = m2.groups()
            depth = _depth_from_numeral(num)
            level = "#" * _level_from_depth(depth, base_h)
            title = RE_LEADER_DOTS.sub("", title)
            out.append(f"{level} {num} {title}")
            continue
        # 3.3.8.2. Algo
        m3 = RE_NUMERAL_ONLY.match(s)
        if m3:
            num, title = m3.groups()
            depth = _depth_from_numeral(num)
            level = "#" * _level_from_depth(depth, base_h)
            title = RE_LEADER_DOTS.sub("", title)
            out.append(f"{level} {num} {title}")
            continue
        out.append(line)
    return "\n".join(out)

def sanitize_tables(md_text: str, max_cell_len: int = 400) -> str:
    """
    Si una tabla tiene filas o celdas sospechosamente largas (OCR corrupto), la sustituye por aviso.
    """
    lines = md_text.splitlines()
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if RE_TABLE.match(line) and i + 1 < len(lines) and RE_TABLE_SEP.match(lines[i + 1]):
            header, sep = line, lines[i + 1]
            rows, j = [], i + 2
            while j < len(lines) and RE_TABLE.match(lines[j]):
                rows.append(lines[j]); j += 1
            joined = " ".join(rows)
            if RE_HTML_BREAKS.search(joined) or RE_LONG_SEQ.search(joined) or len(joined) > 8000:
                out.append("> ⚠️ **Tabla omitida por longitud o ruido excesivo**")
                i = j
                continue
            too_long = any(len(c) > max_cell_len for r in rows for c in _split_table_row(r))
            if too_long:
                out.append(header)
                out.append(sep)
                out.append("> ⚠️ **Celdas omitidas por longitud excesiva**")
            else:
                out.append(header); out.append(sep); out.extend(rows)
            i = j; continue
        out.append(line); i += 1
    return "\n".join(out)

# --- LLM THREAD-SAFE ---
class ThreadSafeLLM:
    def __init__(self):
        self._lock = threading.Lock()
        self._llm_instances = {}
    def get_llm(self):
        tid = threading.get_ident()
        with self._lock:
            if tid not in self._llm_instances:
                self._llm_instances[tid] = init_chat_model(
                    model="gemini-2.0-flash", model_provider="google_genai", temperature=0
                )
            return self._llm_instances[tid]
    def cleanup(self):
        with self._lock:
            self._llm_instances.clear()

# --- PARSER PRINCIPAL ---
class HeadingParser:
    def __init__(self):
        self.llm_handler = ThreadSafeLLM()
        self.results = {}
        self.page_split_regex = re.compile(r"(?=<!--\s*Página\s*\d+\s*-->)")
        self.page_joiner = "\n"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def process_page(self, content: str, idx: int) -> str:
        if not content.strip():
            return content
        return content  # Sin LLM: flujo directo

    def normalize_headings(self, file_name: str, md_content: str, output_dir="data/processed/") -> str:
        pages = [p for p in self.page_split_regex.split(md_content) if p.strip()]
        os.makedirs(os.path.join(output_dir, file_name), exist_ok=True)
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(self.process_page, p, i): i for i, p in enumerate(pages)}
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Procesando páginas"):
                pass
        merged = "\n".join(self.results.get(i, p) for i, p in enumerate(pages))
        # --- POSPROCESADO ---
        merged = retag_headings(merged)
        merged = sanitize_tables(merged)
        out_path = os.path.join(output_dir, file_name, f"{file_name}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(merged)
        logger.info(f"✅ Normalización completada: {out_path}")
        return merged
