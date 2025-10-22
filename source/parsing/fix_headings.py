"""
fix_headings.py — Normalización de encabezados Markdown con LLM (paralelo, robusto).

Mejoras:
- Split por páginas usando el separador real del OCR: <!-- Página N --> (conserva el marcador).
- Prompt reforzado: NO renombrar encabezados numerados ni inventar secciones.
- Posprocesado determinista: retag H2..H6 según profundidad del numeral.
- Unwrap de negritas alrededor de headings (**### ...** -> ### ...).
- Deduplicación de numerales repetidos en el mismo heading.

Requisitos:
- langchain, tenacity, tqdm; credenciales del provider LLM (google_genai).
- GRPC_FORK_SUPPORT_ENABLED=1
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

# Set multiprocessing start method to "spawn" before any other imports
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError as e:
        logging.error(f"Failed to set multiprocessing start method: {e}")

# Enable gRPC fork support
os.environ["GRPC_FORK_SUPPORT_ENABLED"] = "1"

# Configure logging early
try:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
except Exception as e:
    print(f"Failed to configure logging: {e}")
    # Fallback logger
    logger = logging.getLogger(__name__)

try:
    from langchain.chat_models import init_chat_model
    from langchain_core.prompts import ChatPromptTemplate
    from tenacity import retry, stop_after_attempt, wait_exponential
    from tqdm import tqdm
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

# ----------------------------
# Prompt reforzado (no inventes ni renombres)
# ----------------------------
system_prompt = """
Eres un asistente experto en dar FORMATO a documentos técnicos en Markdown.
NO debes alterar el CONTENIDO FACTUAL ni RENOMBRAR encabezados existentes.
NO inventes encabezados ni secciones nuevas.

REGLAS:
1) Jerarquía: Usa encabezados Markdown desde ##. Mantén EXACTOS los encabezados que ya tengan numeral (ej. "3.3.5.2. Título").
   - Si un párrafo pertenece a ese encabezado, colócalo debajo sin cambiar el texto del encabezado.
2) Listas/tablas: Convierte numeraciones o bullets del texto a listas Markdown; las tablas deben tener cabecera.
3) No crees títulos nuevos; no cambies el texto de títulos existentes, sobre todo si comienzan con numeral.
4) Si hay subitems "A. ...", "B. ...", respétalos como listas y NO los transformes en encabezados.
5) MANTÉN el contenido literal, solo reestructura el formato.
6) Respuesta: solo el Markdown; sin backticks.
"""

NORMALIZE_TEMPLATE = """
{system_prompt}

Contenido a normalizar:

{content}
"""

# ----------------------------
# Utilidades
# ----------------------------
def get_optimal_thread_count():
    """Calculate optimal thread count based on CPU cores."""
    try:
        return min(32, (multiprocessing.cpu_count() * 2))
    except Exception as e:
        logger.warning(f"Error determining CPU count: {e}. Using default value of 4.")
        return 4


# ----------------------------
# Retag determinista por numeral
# ----------------------------
RE_MD_H = re.compile(r"^(?P<Hash>#{1,6})\s*(?P<num>(?:\d+\.)+\d+|\d+\.)\s+(?P<title>.+?)\s*$")
RE_NUM_TITLE = re.compile(r"^(?P<num>(?:\d+\.)+\d+|\d+\.)\s+(?P<title>.+?)$")

def _depth_from_numeral(numeral: str) -> int:
    return len(numeral.rstrip(".").split("."))

def _level_from_depth(depth: int, base_h: int = 2) -> int:
    # base_h=2 => "3." -> ##, "3.3." -> ###, ...
    return max(2, min(6, base_h - 1 + depth))

def retag_md_headings_by_numeral(md_text: str, base_h: int = 2) -> str:
    """Ajusta niveles Hx coherentes con la profundidad del numeral (sin cambiar títulos)."""
    out = []
    for ln in md_text.splitlines():
        s = ln.strip()

        # Heading con hashes y numeral
        m = RE_MD_H.match(s)
        if m:
            num = m.group("num")
            title = m.group("title")
            depth = _depth_from_numeral(num)
            hx = "#" * _level_from_depth(depth, base_h)
            # dedupe numeral duplicado (e.g., "3.3.8.7. 3.3.8.7.")
            num_dedup = re.sub(rf"^({re.escape(num)})(\s+{re.escape(num)})\b", r"\1", f"{num}")
            out.append(f"{hx} {num_dedup} {title}")
            continue

        # Línea que empieza con numeral + título (sin hashes)
        m2 = RE_NUM_TITLE.match(s)
        if m2:
            num = m2.group("num")
            title = m2.group("title")
            depth = _depth_from_numeral(num)
            hx = "#" * _level_from_depth(depth, base_h)
            out.append(f"{hx} {num} {title}")
            continue

        # Unwrap de ** alrededor de headings (caso OCR: **### 3.1 ...**)
        if s.startswith("**#"):
            s2 = s.strip("* ")
            if RE_MD_H.match(s2):
                out.append(s2)
                continue

        out.append(ln)
    text = "\n".join(out)
    # Arreglo extra de numerales duplicados dentro de la misma línea
    text = re.sub(r"((?:^|\s)(#{2,6}\s+)(\d+(?:\.\d+)+\.))\s+\3", r"\1", text)
    return text


# ----------------------------
# LLM thread-safe
# ----------------------------
class ThreadSafeLLM:
    """Thread-safe LLM handler."""

    def __init__(self):
        self._lock = threading.Lock()
        self._llm_instances = {}

    def get_llm(self):
        """Get or create LLM instance for current thread."""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id not in self._llm_instances:
                try:
                    self._llm_instances[thread_id] = init_chat_model(
                        model="gemini-2.0-flash",
                        model_provider="google_genai",
                        temperature=0,
                    )
                except Exception as e:
                    logger.error(
                        f"Error initializing LLM for thread {thread_id}: {str(e)}"
                    )
                    raise
            return self._llm_instances[thread_id]

    def cleanup(self):
        """Cleanup LLM instances."""
        try:
            with self._lock:
                self._llm_instances.clear()
        except Exception as e:
            logger.error(f"Error during LLM cleanup: {e}")


# ----------------------------
# Parser
# ----------------------------
class HeadingParser:
    """Class for parsing and normalizing headings in markdown documents using LLM."""

    def __init__(self):
        """Initialize the parser with LLM model and prompt template."""
        try:
            self.prompt_template = ChatPromptTemplate.from_template(
                template=NORMALIZE_TEMPLATE
            )
            self.llm_handler = ThreadSafeLLM()
            self.results = {}
            self._lock = threading.Lock()

            # Nuevo: separador real de páginas proveniente del OCR
            # split por regex que CONSERVA el marcador <!-- Página N -->
            self.page_split_regex = re.compile(r"(?=<!--\s*Página\s*\d+\s*-->)")
            self.page_joiner = "\n"
        except Exception as e:
            logger.error(f"Error initializing HeadingParser: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def process_page(self, page_content: str, page_index: int) -> str:
        """
        Process a single page of markdown content and normalize headings.

        Args:
            page_content: Markdown content to normalize.
            page_index: Index of the page in the document.

        Returns:
            Normalized page content.
        """
        try:
            if not page_content.strip():
                return page_content

            prompt = self.prompt_template.format_messages(
                system_prompt=system_prompt,
                content=page_content,
            )
            llm = self.llm_handler.get_llm()
            response = llm.invoke(prompt)

            normalized_content = response.content

            with self._lock:
                self.results[page_index] = normalized_content

            return normalized_content

        except Exception as e:
            logger.error(f"Error processing page {page_index}: {str(e)}")
            return page_content  # Return original content on error

    def normalize_headings(
        self, file_name: str, md_content: str, output_dir="data/processed/"
    ) -> str:
        """
        Normalize headings in markdown content using parallel processing.

        Args:
            file_name: Name of the file being processed.
            md_content: Markdown content to normalize.
            output_dir: Directory to save processed files.

        Returns:
            Normalized markdown content as a string.
        """
        try:
            # Split content into pages (conservando el marcador <!-- Página N -->)
            pages = self.page_split_regex.split(md_content)
            pages = [p for p in pages if p.strip()]  # descarta vacíos
            logger.info(f"Starting heading normalization for {len(pages)} pages")

            # Create output directory if it doesn't exist
            try:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, file_name)
                os.makedirs(output_path, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create output directory: {e}")
                output_path = "."  # Fallback to current directory

            # Reset results
            self.results = {}

            # Process pages in parallel with progress bar
            with ThreadPoolExecutor(max_workers=get_optimal_thread_count()) as executor:
                futures = {
                    executor.submit(
                        self.process_page, page_content=page, page_index=idx
                    ): idx
                    for idx, page in enumerate(pages)
                }

                with tqdm(total=len(futures), desc="Processing pages") as pbar:
                    for future in as_completed(futures):
                        try:
                            future.result(timeout=180)  # 3-minute timeout per page
                        except TimeoutError:
                            logger.error(f"Timeout processing page {futures[future]}")
                        except Exception as e:
                            logger.error(
                                f"Error processing page {futures[future]}: {str(e)}"
                            )
                        finally:
                            pbar.update(1)  # Always update progress

            # Merge en orden
            normalized_pages = []
            for i in range(len(pages)):
                normalized_pages.append(self.results.get(i, pages[i]))

            normalized_content = self.page_joiner.join(normalized_pages)

            # 🔧 Posprocesado determinista:
            # - retag H2..H6 según profundidad del numeral
            # - unwrap ** alrededor de headings
            # - dedupe numerales duplicados
            normalized_content = retag_md_headings_by_numeral(normalized_content, base_h=2)

            # Persistir
            try:
                output_file = os.path.join(output_path, f"{file_name}.md")
                with open(output_file, "w", encoding="utf-8") as file:
                    file.write(normalized_content)
                logger.info(f"Successfully wrote normalized content to {output_file}")
            except IOError as e:
                logger.error(f"Failed to write output file: {e}")

            logger.info("Completed heading normalization")
            return normalized_content

        except Exception as e:
            logger.error(f"Error during normalization process: {str(e)}")
            return md_content  # Return original content on error

        finally:
            try:
                self.llm_handler.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
