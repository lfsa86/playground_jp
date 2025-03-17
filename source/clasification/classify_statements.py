"""Module for extracting environmental components from commitments using LLM."""

import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Dict, List

# Set environment variables before any other imports
os.environ["GRPC_FORK_SUPPORT_ENABLED"] = "1"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"

# Configure logging early
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set multiprocessing start method to "spawn" before any other imports
# This is critical for avoiding fork-related issues with gRPC
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

# Now import other dependencies
import pandas as pd
from dotenv import load_dotenv
from gradio.components import file
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from ..models import ElementIterator, ElementParser
from ..schemas import Statement

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """Eres un asistente especializado en analizar documentos regulatorios ambientales chilenos. Tu tarea es extraer y categorizar información en las siguientes secciones:

1. Generalidades: Extrae información contextual sobre el proyecto incluyendo ubicación, propósito, escala, cronograma y antecedentes que ayuden a comprender el alcance general del proyecto.

2. Componentes: Identifica todos los elementos físicos, infraestructura o sistemas que serán construidos, instalados o modificados como parte del proyecto. Incluye especificaciones técnicas cuando estén disponibles.

3. Implicaciones: Extrae información sobre los elementos implicados para el desarrollo del proyecto (acciones,actividades del proyecto, uso de recursos, etc.), incluyendo su relación con el entorno, la comunidad.

4. Riesgos e Impactos: Identifica los potenciales riesgos e impactos ambientales descritos en el documento, tanto durante las fases de construcción como de operación.

5. Compromisos: Extrae todos los compromisos realizados por el titular del proyecto, incluyendo medidas voluntarias que se vuelven obligatorias al ser incluidas en la Resolución de Calificación Ambiental (RCA).

6. Permisos: Enumera todos los permisos, autorizaciones y aprobaciones regulatorias mencionadas en el documento, incluyendo aquellos ya obtenidos y los que aún se requieren.

7. Otros: Ninguna de las anteriores.

Presenta la información en un formato estructurado con encabezados claros para cada categoría. Si una categoría no tiene información relevante en el documento, indícalo explícitamente."""

TEMPLATE = """
{system_prompt}

Clasifica el siguiente enunciado dentro de obligación, compromiso, acuerdo u 'otro':
```{statement}```

NO ALUCINES
"""


class ThreadSafeLLM:
    """Thread-safe LLM handler with proper initialization."""

    def __init__(self):
        self._lock = threading.Lock()
        self._llm_instances = {}
        self._initialized = False

    def initialize(self):
        """Initialize the LLM handler once before threading."""
        if not self._initialized:
            # Pre-initialize one instance to avoid concurrent initialization issues
            try:
                llm = init_chat_model(
                    model="gemini-2.0-flash",
                    model_provider="google_genai",
                    temperature=0.1,
                    max_tokens=8192,
                )
                # Just initialize but don't store
                llm.with_structured_output(Statement)
                self._initialized = True
                logger.info("LLM pre-initialized successfully")
            except Exception as e:
                logger.error(f"Error pre-initializing LLM: {str(e)}")
                raise

    def get_llm(self):
        """Get or create LLM instance for current thread."""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id not in self._llm_instances:
                try:
                    llm = init_chat_model(
                        model="gemini-2.0-flash",
                        model_provider="google_genai",
                        temperature=0.1,
                        max_tokens=8192,
                    )
                    self._llm_instances[thread_id] = llm.with_structured_output(
                        Statement
                    )
                except Exception as e:
                    logger.error(
                        f"Error initializing LLM for thread {thread_id}: {str(e)}"
                    )
                    raise
            return self._llm_instances[thread_id]

    def cleanup(self):
        """Cleanup LLM instances."""
        with self._lock:
            self._llm_instances.clear()


class ThreadSafeList:
    """Thread-safe list implementation for storing statements."""

    def __init__(self):
        self._lock = threading.Lock()
        self._list = []

    def append(self, item):
        with self._lock:
            self._list.append(item)

    def extend(self, items):
        with self._lock:
            self._list.extend(items)

    def get_all(self) -> List:
        with self._lock:
            return self._list.copy()


class StatementProcessor:
    """Processor for handling LLM initialization and statement storage."""

    def __init__(self):
        self.statements = ThreadSafeList()
        self.llm_handler = ThreadSafeLLM()
        self.prompt_template = ChatPromptTemplate.from_template(template=TEMPLATE)

        # Pre-initialize LLM before threading
        self.llm_handler.initialize()

    def get_llm(self):
        """Thread-safe LLM initialization."""
        return self.llm_handler.get_llm()

    def cleanup(self):
        """Cleanup resources."""
        self.llm_handler.cleanup()


def get_optimal_thread_count():
    """Calculate optimal thread count based on CPU cores."""
    return min(32, (multiprocessing.cpu_count()))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_statements(title: str, content: str, processor: StatementProcessor) -> Dict:
    """
    Extract and classify a statement from title and content.

    Args:
        title: Node title.
        content: Node content.
        processor: StatementProcessor instance for LLM handling.

    Returns:
        Dictionary containing statement classification.
    """
    try:
        prompt = processor.prompt_template.format_messages(
            system_prompt=SYSTEM_PROMPT, statement=f"{title}\n{content}"
        )

        llm = processor.get_llm()
        response: Statement = llm.invoke(prompt)

        return {
            "category": response.category,
            "justification": response.justification,
            "synthesis": response.synthesis,
            "related_items": response.elements,
            "heading_path": title,
            "text_content": str(title + "\n" + content),
        }

    except Exception as e:
        logger.error(f"Error processing '{title}': {str(e)}")
        return {
            "category": "error",
            "justification": f"Error in processing: {str(e)}",
            "synthesis": "",
            "related_items": [],
            "heading_path": title,
            "text_content": content.strip(),
        }


def process_node(node_tuple, processor: StatementProcessor):
    """Process a single node."""
    try:
        node_id, node, depth = node_tuple
        result = extract_statements(node.title, node.content, processor)
        return result
    except Exception as e:
        logger.error(f"Error processing node: {str(e)}")
        return None


def process_nodes(executor, nodes_to_process, processor):
    """Process nodes using a thread pool."""
    results_dict = {}  # Use dictionary to preserve order by index

    with tqdm(total=len(nodes_to_process), desc="Processing nodes") as pbar:
        # Submit each node with a small delay to avoid gRPC initialization conflicts
        futures = {}  # Map futures to their original indices
        for i, node_tuple in enumerate(nodes_to_process):
            future = executor.submit(process_node, node_tuple, processor)
            futures[future] = i  # Store the original index
            # Small delay to avoid concurrent gRPC initialization
            if i % 5 == 0:  # Add delay every 5 submissions
                time.sleep(0.5)

        for future in as_completed(futures):
            try:
                result = future.result(timeout=300)
                if result:
                    results_dict[futures[future]] = result  # Store with original index
                pbar.update(1)
            except TimeoutError:
                logger.error(f"Timeout processing node at index {futures[future]}")
                pbar.update(1)
            except Exception as e:
                logger.error(
                    f"Error processing node at index {futures[future]}: {str(e)}"
                )
                pbar.update(1)

    # Collect results in order
    ordered_results = []
    for i in sorted(results_dict.keys()):
        ordered_results.append(results_dict[i])

    processor.statements.extend(ordered_results)


def save_results(statements: List[Dict], output_path: str):
    """Save results to multiple file formats."""
    output_files = {
        "statements.parquet": lambda df: df.to_parquet(
            f"{output_path}/statements.parquet"
        ),
        "statements.pkl": lambda df: df.to_pickle(f"{output_path}/statements.pkl"),
        "statements.csv": lambda df: df.to_csv(f"{output_path}/statements.csv"),
        "statements.xlsx": lambda df: df.explode("related_items").to_excel(
            f"{output_path}/statements.xlsx"
        ),
    }

    df = pd.DataFrame(statements)
    for output_file_name, save_func in output_files.items():
        try:
            save_func(df)
            logger.info(f"Successfully saved {output_file_name}")
        except Exception as e:
            logger.error(f"Error saving {output_file_name}: {str(e)}")

    logger.info(f"Created files: statements.parquet, statements.pkl in {output_path}")


def classify_statements(
    file_name: str, md_content: str, output_dir: str = "data/processed/"
):
    """
    Process and classify statements from a file.

    Args:
        file_dict: Dictionary containing file information.
        output_dir: Directory path for output files.
    """
    logger.info(f"Starting processing of {file_name}")

    output_path = os.path.join(output_dir, file_name, "statements")
    os.makedirs(output_path, exist_ok=True)

    parser = ElementParser()
    tree = parser.parse_text(md_content)
    iterator = ElementIterator(tree)
    nodes_to_process = list(iterator.iterate_depth_first(skip_root=False))

    # Print sample node structure for debugging
    if nodes_to_process:
        logger.info(f"Node structure: {nodes_to_process}")

    processor = StatementProcessor()

    try:
        # Use a smaller thread pool to reduce contention
        with ThreadPoolExecutor(
            max_workers=min(8, get_optimal_thread_count())
        ) as executor:
            try:
                process_nodes(executor, nodes_to_process, processor)
            except Exception as e:
                logger.error(f"Error during processing: {str(e)}")
            finally:
                # Ensure results are saved even if processing fails
                save_results(processor.statements.get_all(), output_path)
    finally:
        processor.cleanup()

    return pd.DataFrame(processor.statements.get_all())
