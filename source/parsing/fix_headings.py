"""
fix_headings.py — Normalización de encabezados Markdown con LLM (paralelo, robusto).

Objetivo:
- Reestructurar y normalizar encabezados (a partir de ##), listas y tablas en contenido Markdown,
  manteniendo el contenido factual intacto y mejorando la legibilidad.

Componentes:
- get_optimal_thread_count(): determina hilos óptimos (min(32, 2*CPU)).
- ThreadSafeLLM: gestor thread-safe de instancias de chat model (una por hilo).
- HeadingParser:
    - process_page(page_content, page_index): normaliza una “página” con LLM (reintentos y logs).
    - normalize_headings(file_name, md_content, output_dir): divide por separador, procesa en paralelo,
      reensambla en orden y persiste en data/processed/<file_name>/<file_name>.md. Retorna el markdown final.

Flujo:
1) split por separador de páginas (self.page_separator).
2) parallel map con ThreadPoolExecutor → process_page().
3) merge ordenado por índice; fallback a original en errores.
4) write .md en carpeta de salida; cleanup de instancias LLM.

Requisitos:
- LangChain, Tenacity, TQDM; credenciales del provider LLM (google_genai).
- GRPC_FORK_SUPPORT_ENABLED=1 (evita issues con gRPC y forking).

Notas:
- Asegura alinear el separador de páginas con el parser que genera el markdown (p. ej., <!-- Página N -->).
- Controla max_workers si hay límites de cuota del LLM.
"""

import logging
import multiprocessing
import os
import threading
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
    from langchain.prompts import ChatPromptTemplate
    from tenacity import retry, stop_after_attempt, wait_exponential
    from tqdm import tqdm
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

# System prompt for heading normalization
system_prompt = """
Eres un asistente experto en reformatear documentos legales y técnicos, con un enfoque en la creación de estructuras jerárquicas claras y concisas. Tu tarea principal es transformar un texto plano en un documento estructurado utilizando Markdown, específicamente enfatizando la organización multinivel a partir del nivel de encabezado 2 (##).

**Instrucciones específicas:**

1.  **Jerarquía Markdown:**  Organiza el texto utilizando encabezados Markdown (##, ###, ####, etc.) para representar la jerarquía lógica del documento. Asegúrate de que cada sección tenga un encabezado apropiado y que las subsecciones estén anidadas correctamente. Comienza la jerarquía en el nivel de encabezado 2.
2.  **Formato de listas y tablas:** Cuando corresponda, utiliza listas (ordenadas y no ordenadas) y tablas para presentar información de manera clara y organizada. Formatea las tablas con encabezados y alineación apropiados.
3.  **Claridad y concisión:** Reestructura las oraciones y párrafos para mejorar la legibilidad y la fluidez del texto, sin alterar su significado original.  Evita la redundancia y la ambigüedad.
4.  **Respetar el contenido original:** No modifiques el contenido factual del documento. Tu objetivo es mejorar la estructura y la presentación, no alterar la información proporcionada.
5.  **Consistencia:** Aplica un estilo consistente en todo el documento, incluyendo el uso de mayúsculas, minúsculas, puntuación y formato de listas y tablas.
6. **énfasis a listas:** Si hay instrucciones, regulaciones o ítems enumerados, asegúrate de representarlos en un formato de lista con viñetas o numerados para que sean fáciles de identificar y entender.
7. **énfasis a tablas:** Las tablas siempre deben incluir títulos claros y descriptivos.
8. En tu respuesta solamente da el contenido del MD sin triple backtick (```)
9. MANTEN EL CONTENIDO ORIGINAL SIN MODIFICARLO.
10. EVITA REDUNDANCIAS AL MOMENTO DE TRANSCRIBIR.

**Actúa como un asistente legal/técnico profesional.** Tu objetivo es producir documentos bien estructurados, fáciles de leer y que mantengan la integridad del contenido original.
"""

NORMALIZE_TEMPLATE = """
{system_prompt}

Contenido a normalizar:

{content}
"""


def get_optimal_thread_count():
    """Calculate optimal thread count based on CPU cores."""
    try:
        return min(32, (multiprocessing.cpu_count() * 2))
    except Exception as e:
        logger.warning(f"Error determining CPU count: {e}. Using default value of 4.")
        return 4


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
            self.page_separator = "-----"
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
            # Split content into pages
            pages = md_content.split(self.page_separator)
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
                            future.result(timeout=120)  # 2-minute timeout
                        except TimeoutError:
                            logger.error(f"Timeout processing page {futures[future]}")
                        except Exception as e:
                            logger.error(
                                f"Error processing page {futures[future]}: {str(e)}"
                            )
                        finally:
                            pbar.update(1)  # Always update progress

            # Create final normalized content
            normalized_pages = []
            for i in range(len(pages)):
                normalized_pages.append(self.results.get(i, pages[i]))

            normalized_content = self.page_separator.join(normalized_pages)

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