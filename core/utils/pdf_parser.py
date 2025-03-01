import io
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI as genai
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


load_dotenv()


class PDFParser:
    """
    A class for transcribing PDF documents to Markdown with HTML tables using Google Gemini.

    This class handles the entire transcription pipeline:
    1. Converting PDF pages to high-quality images
    2. Generating custom prompts based on document samples
    3. Transcribing content with parallel processing
    4. Cleaning and formatting the output

    Attributes:
        api_key (str): Google Gemini API key
        dpi (int): Image quality for PDF conversion
        max_workers (int): Maximum number of parallel workers
        remove_code_markers (bool): Whether to remove code markers from response
        model: Google Gemini model instance
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        dpi: int = 150,
        max_workers: int = 5,
        remove_code_markers: bool = True,
    ):
        """
        Initialize PDFParser.

        Args:
            api_key: Google Gemini API key. If None, will try to get from GOOGLE_API_KEY env var.
            dpi: Image quality for PDF conversion. Higher values produce better quality but larger files.
            max_workers: Maximum number of parallel workers for transcription.
            remove_code_markers: Whether to remove markdown code block markers from response.

        Raises:
            ValueError: If API key is not provided and not found in environment variables.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not provided and GOOGLE_API_KEY not found in environment"
            )

        self.dpi = dpi
        self.max_workers = max_workers
        self.remove_code_markers = remove_code_markers

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Generation configuration
        generation_config = {
            "temperature": 0.1,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 4096,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-001",
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

    def _convert_pdf_to_images(self, pdf_path: Union[str, Path]) -> List[Image.Image]:
        """
        Convert PDF pages to high-quality images with progress tracking.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of PIL Image objects representing each page

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            fitz.FileDataError: If the file is not a valid PDF
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        images = []
        total_pages = len(doc)

        with tqdm(
            total=total_pages, desc="Converting PDF to images", unit="page"
        ) as pbar:
            try:
                for page in doc:
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(self.dpi / 72, self.dpi / 72)
                    )
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
                    pbar.update(1)
            finally:
                doc.close()

        return images

    def _prepare_image(self, image: Image.Image) -> bytes:
        """
        Convert and compress PIL Image to bytes for API transmission.

        Args:
            image: The PIL Image to prepare

        Returns:
            Compressed image bytes in JPEG format
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG", quality=85)
        return img_byte_arr.getvalue()

    def _clean_response_text(self, text: str) -> str:
        """
        Clean response text by removing unnecessary code block markers.

        Args:
            text: Original response text from the model

        Returns:
            Cleaned text with code markers removed
        """
        # Split text into lines
        lines = text.split("\n")
        cleaned_lines = []
        skip_line = False

        for line in lines:
            # Skip lines that only contain markdown or html code markers
            if line.strip() in ["```markdown", "```html", "```", "```md"]:
                skip_line = False
                continue

            if not skip_line:
                cleaned_lines.append(line)

        # Join lines back together
        cleaned_text = "\n".join(cleaned_lines)

        # Remove any remaining single backticks around html tags
        html_tags = ["table", "tr", "td", "th"]
        for tag in html_tags:
            cleaned_text = cleaned_text.replace(f"`<{tag}>`", f"<{tag}>")
            cleaned_text = cleaned_text.replace(f"`</{tag}>`", f"</{tag}>")

        return cleaned_text.strip()

    def _generate_prompt(self, sample_images: List[Image.Image]) -> str:
        """
        Generate a transcription prompt based on sample images from the document.

        This method analyzes sample pages to create a custom prompt that will
        guide the transcription of the entire document.

        Args:
            sample_images: List of sample pages as PIL Images

        Returns:
            Generated prompt for transcription

        Raises:
            ValueError: If the model returns an empty response
            Exception: For other API errors, with retry logic
        """
        system_message = """Analiza estas imágenes del documento y genera un prompt detallado
        para transcribir el documento completo. El documento final debe:
        1. Usar Markdown para todo el contenido excepto las tablas
        2. Usar HTML solo para las tablas (<table>, <tr>, <td>), IMPORTANTE: Es importante adaptarse a la estructura de las tablas que pueda tener el documento, puede ser que tengan layouts muy extraños o no sean tablas perfectas, tu trabajo es organizarlas lo mejor posible para que queden lo mejor posible y se puedan leer facilmente, sin alterar el contenido que tienen dentro
        3. Mantener la estructura jerárquica exacta del documento
        4. Preservar el formato (negritas, cursivas, etc.)
        5. Mantener enlaces y referencias
        6. Conservar listas y enumeraciones
        7. Respetar el espaciado original
        8. Mantener el formato de código si existe
        9. Preservar citas y bloques de texto
        10. Mantener notas al pie y referencias cruzadas
        11. Eliminar pie de paginas y links de verificacion de autenticidad o firmas digitales
        12. Ignora todas las imagenes del documento
        13. Asegurate que las tablas esten estructuradas bajo los mejores estandares posibles"""

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content([system_message, *sample_images])
                if response.text:
                    return response.text
                else:
                    raise ValueError("Empty response from Gemini")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    logger.error(
                        f"Failed to generate prompt after {max_retries} attempts"
                    )
                    raise

    def _transcribe_page_parallel(
        self, image: Image.Image, custom_prompt: str, page_index: int
    ) -> str:
        """
        Transcribe a single page in parallel.

        Args:
            image: The image of the page to transcribe
            custom_prompt: The custom prompt for transcription
            page_index: The index of the page (0-based)

        Returns:
            Transcribed content in Markdown/HTML format with page number at the beginning

        Raises:
            Exception: If transcription fails, with error details
        """
        try:
            prepared_image = self._prepare_image(image)
            transcription_prompt = f"""
            {custom_prompt}

            Instrucciones específicas para esta página:
            1. Transcribe el contenido manteniendo el formato exacto
            2. Usa Markdown para todo excepto tablas
            3. Las tablas deben estar en HTML con <table>, <tr>, y <td>
            4. Preserva todos los elementos de formato (negritas, cursivas, etc.)
            5. Mantén la estructura jerárquica del documento
            6. No agregues contenido adicional
            7. No omitas ningún detalle del documento original
            8. No incluyas marcadores de bloques de código (```) en la respuesta
            9. Esta es la página {page_index + 1} del documento
            """

            response = self.model.generate_content(
                [
                    transcription_prompt,
                    {"mime_type": "image/jpeg", "data": prepared_image},
                ]
            )

            content = response.text
            if self.remove_code_markers:
                content = self._clean_response_text(content)

            # Ensure page number appears at the beginning
            page_header = f"Página {page_index + 1}\n\n"
            return page_header + content

        except Exception as e:
            logger.error(f"Error transcribing page: {e}")
            return f"Error transcribing page: {e}"

    def transcribe_pdf(
        self,
        pdf_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        sample_pages: int = 3,
    ) -> Path:
        """
        Transcribe PDF document to Markdown with tables in HTML using parallel processing.

        This is the main method that orchestrates the entire transcription process.

        Args:
            pdf_path: Path to PDF file
            output_path: Path for output markdown file. If None, uses the PDF filename with .md extension.
            sample_pages: Number of sample pages to use for prompt generation

        Returns:
            Path to the generated markdown file

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            ValueError: If sample_pages is less than 1
            Exception: For other errors during transcription
        """
        if sample_pages < 1:
            raise ValueError("sample_pages must be at least 1")

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            logger.info(f"Starting parallel transcription of {pdf_path}")

            # Convert PDF to images
            images = self._convert_pdf_to_images(pdf_path)
            logger.info(f"Converted PDF to {len(images)} images")

            # Select sample pages for prompt generation
            sample_count = min(sample_pages, len(images))
            sample_images = random.sample(images, sample_count)
            logger.info(
                f"Selected {len(sample_images)} sample pages for prompt generation"
            )

            # Generate custom prompt
            with tqdm(total=1, desc="Generating custom prompt") as pbar:
                custom_prompt = self._generate_prompt(sample_images)
                pbar.update(1)
            logger.info("Generated custom prompt for transcription")

            # Process pages in parallel
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for i, image in enumerate(images):
                    future = executor.submit(
                        self._transcribe_page_parallel, image, custom_prompt, i
                    )
                    futures.append(future)

                # Show progress bar for parallel processing
                with tqdm(
                    total=len(futures), desc="Transcribing pages in parallel"
                ) as pbar:
                    for future in futures:
                        results.append(future.result())
                        pbar.update(1)

            # Save results
            if output_path is None:
                output_path = pdf_path.with_suffix(".md")
            else:
                output_path = Path(output_path)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(results))

            logger.info(f"Transcription completed and saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise
