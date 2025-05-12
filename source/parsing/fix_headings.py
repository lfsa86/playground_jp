"""Module for normalizing headings in markdown documents using LLM."""

import logging
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import re

def demote_table_headers(md_text: str) -> str:
    """
    Convierte encabezados Markdown que comienzan con 'Tabla' en texto plano destacado con negrita,
    y agrega saltos de línea arriba y abajo para claridad visual.
    """
    lines = md_text.splitlines()
    new_lines = []
    pattern = re.compile(r"^(#{1,6})\s+(Tabla(?:\s+(Original|Rectificada))?\s[\d\.\s\(\)A-Za-zÁÉÍÓÚÑñ:]+)", re.IGNORECASE)

    for i, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            titulo = match.group(2).strip()
            new_lines.append("")  # línea vacía arriba
            new_lines.append(f"**{titulo}**")  # título en negrita
            new_lines.append("")  # línea vacía abajo
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

def protect_sensitive_blocks(text: str) -> str:
    """
    Añade marcas de protección alrededor de tablas, listas enumeradas o texto con estilo técnico
    para que no sea modificado por el modelo.
    """
    lines = text.splitlines()
    protected = []
    in_table = False
    for line in lines:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            if not in_table:
                protected.append("<PROTECTED_BLOCK>")
                in_table = True
            protected.append(line)
        elif in_table and not line.strip().startswith("|"):
            protected.append("</PROTECTED_BLOCK>")
            protected.append(line)
            in_table = False
        else:
            protected.append(line)
    if in_table:
        protected.append("</PROTECTED_BLOCK>")
    return "\n".join(protected)

def restore_protected_blocks(text: str, original: str) -> str:
    """
    Reemplaza los bloques marcados como <PROTECTED_BLOCK>...</PROTECTED_BLOCK>
    con el contenido original correspondiente (línea por línea).
    """
    original_lines = original.splitlines()
    result_lines = []
    i = 0
    while i < len(original_lines):
        if "<PROTECTED_BLOCK>" in original_lines[i]:
            protected_block = []
            i += 1
            while i < len(original_lines) and "</PROTECTED_BLOCK>" not in original_lines[i]:
                protected_block.append(original_lines[i])
                i += 1
            result_lines.extend(protected_block)
            i += 1  # Saltar </PROTECTED_BLOCK>
        else:
            result_lines.append(original_lines[i])
            i += 1
    return "\n".join(result_lines)


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
3.  **Respetar el contenido original:** No modifiques el contenido factual del documento. Tu objetivo es mejorar la estructura y la presentación, no alterar la información proporcionada.
4.  **Consistencia:** Aplica un estilo consistente en todo el documento, incluyendo el uso de mayúsculas, minúsculas, puntuación y formato de listas y tablas.
5. **énfasis a listas:** Si hay instrucciones, regulaciones o ítems enumerados, asegúrate de representarlos en un formato de lista con viñetas o numerados para que sean fáciles de identificar y entender.
6. **énfasis a tablas:** Las tablas siempre deben incluir títulos claros y descriptivos.
7. En tu respuesta solamente da el contenido del MD sin triple backtick (```)
8. MANTEN EL CONTENIDO ORIGINAL SIN MODIFICARLO.
9. EVITA REDUNDANCIAS AL MOMENTO DE TRANSCRIBIR.
10. NO resumas, interpretes o completes el contenido. No omitas ninguna línea, celda o tabla, incluso si parecen repetidas, vacías o poco informativas.

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
        Adds protection around sensitive blocks like tables before passing to LLM.
        """
        try:
            if not page_content.strip():
                return page_content

            # 👉 Paso 1: proteger contenido sensible
            protected_content = protect_sensitive_blocks(page_content)

            # 👉 Paso 2: preparar el prompt
            prompt = self.prompt_template.format_messages(
                system_prompt=system_prompt,
                content=protected_content,
            )

            # 👉 Paso 3: ejecutar el modelo
            llm = self.llm_handler.get_llm()
            response = llm.invoke(prompt)

            # 👉 Paso 4: restaurar el contenido sensible original
            if hasattr(response, "content") and response.content:
                normalized_content = restore_protected_blocks(response.content, protected_content)
            else:
                logger.warning(f"Empty response at page {page_index}")
                normalized_content = page_content

            # 👉 Guardar resultado
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

            normalized_content = demote_table_headers(normalized_content)

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