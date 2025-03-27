"""Module for extracting commitments from environmental documents using LLM."""

import logging
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Dict

# Set multiprocessing start method to "spawn" before any other imports
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

# Enable gRPC fork support
os.environ["GRPC_FORK_SUPPORT_ENABLED"] = "1"

# Configure logging early
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import pandas as pd
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from pandas import DataFrame
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from ..schemas import Commitment

# System prompt for commitment extraction
SYSTEM_PROMPT = """
Eres un asistente experto en análisis de documentos ambientales. Tu tarea es identificar y extraer compromisos ambientales de textos, clasificarlos y determinar sus características principales.
"""

ANALYSIS_TEMPLATE = """
{system_prompt}

Texto a analizar:
{texto}
NO ALUCINES
"""


def get_optimal_thread_count():
    """Calculate optimal thread count based on CPU cores."""
    return min(32, ((multiprocessing.cpu_count() * 2) - 10))


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
                    llm = init_chat_model(
                        model="gemini-2.0-flash",
                        model_provider="google_genai",
                        temperature=0,
                    )
                    self._llm_instances[thread_id] = llm.with_structured_output(
                        Commitment
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


class CommitmentExtractor:
    """Class for extracting commitments from environmental documents"""

    def __init__(self):
        """Initialize the extractor with LLM model and prompt template."""
        self.prompt_template = ChatPromptTemplate.from_template(
            template=ANALYSIS_TEMPLATE
        )
        self.llm_handler = ThreadSafeLLM()
        self.results = []
        self._lock = threading.Lock()

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def process_row(self, row: DataFrame) -> Dict:
        """
        Process a single row of text and extract commitments.

        Args:
            row: Text content to analyze.

        Returns:
            Dictionary containing extracted commitment information.
        """
        try:
            prompt = self.prompt_template.format_messages(
                system_prompt=SYSTEM_PROMPT,
                texto=row["text_content"].values[0],
            )
            llm = self.llm_handler.get_llm()
            response: Commitment = llm.invoke(prompt)

            result = {
                "statement_index": row.index[0],
                "titulo": row["heading_path"].values[0],
                "summary": response.summary,
                "componente_operativo": response.coa,
                "componente_ambiental": response.caa,
                "fase_aplicacion_del_compromiso": response.fase_aplicacion,
                "frecuencia_de_reporte": response.frecuencia_reporte,
                "text_content": row["text_content"].values[0],
            }

            with self._lock:
                self.results.append(result)

            return result

        except Exception as e:
            logger.error(f"Error processing row: {str(e)}")
            return {
                "error": str(e),
                "row_content": row["text_content"],
            }

    def save_results(self, df: DataFrame, output_path: str):
        """Save results to multiple file formats."""
        output_files = {
            "extracted_commitments.parquet": lambda df: df.to_parquet(
                f"{output_path}/extracted_commitments.parquet"
            ),
            "extracted_commitments.pkl": lambda df: df.to_pickle(
                f"{output_path}/extracted_commitments.pkl"
            ),
            "extracted_commitments.csv": lambda df: df.to_csv(
                f"{output_path}/extracted_commitments.csv"
            ),
            "extracted_commitments.xlsx": lambda df: df.to_excel(
                f"{output_path}/extracted_commitments.xlsx"
            ),
        }

        for file_name, save_func in output_files.items():
            try:
                save_func(df)
                logger.info(f"Successfully saved {file_name}")
            except Exception as e:
                logger.error(f"Error saving {file_name}: {str(e)}")

    def extract(
        self, file_name: str, df: DataFrame, output_dir: str = "data/processed"
    ) -> DataFrame:
        """
        Extract and analyze commitments from a DataFrame.

        Args:
            file_name: Name of the file being processed.
            df: DataFrame containing the data to analyze.
            output_dir: Directory path for output files.

        Returns:
            DataFrame containing analyzed commitments.
        """
        commitment_df = df[df["category"].str.lower() == "compromisos"]
        logger.info(f"Starting commitment extraction for file: {file_name}")

        # Create output directory
        output_path = os.path.join(output_dir, file_name, "commitments")
        os.makedirs(output_path, exist_ok=True)

        try:
            # Process rows in parallel with progress bar
            with ThreadPoolExecutor(max_workers=get_optimal_thread_count()) as executor:
                futures = {
                    executor.submit(self.process_row, row=DataFrame([row])): idx
                    for idx, row in commitment_df.iterrows()
                }

                with tqdm(total=len(futures), desc="Processing rows") as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=60)  # Add timeout
                            pbar.update(1)
                        except TimeoutError:
                            logger.error(f"Timeout processing row {futures[future]}")
                        except Exception as e:
                            logger.error(
                                f"Error processing row {futures[future]}: {str(e)}"
                            )
                        finally:
                            pbar.update(
                                0
                            )  # Ensure progress bar updates even on failure

            # Create DataFrame from results
            responses_df = pd.DataFrame(self.results)
            responses_df = responses_df.sort_values(by="statement_index")

            # Save results
            self.save_results(responses_df, output_path)

            logger.info(f"Completed commitment extraction for file: {file_name}")
            return responses_df

        except Exception as e:
            logger.error(f"Error during extraction process: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

        finally:
            self.llm_handler.cleanup()
