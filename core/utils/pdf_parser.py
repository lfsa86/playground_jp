import io
import os
import random
import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class PDFParser:
    """
    Class to transcribe PDF documents into Markdown with HTML tables.

    This class handles the entire transcription pipeline:
    1. Converts PDF pages into high-quality images
    2. Generates custom prompts based on sample pages
    3. Transcribes the content with intelligent structure analysis
    4. Provides document analysis and table of contents extraction
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            dpi: int = 150,
            max_workers: int = 50,
            remove_code_markers: bool = True,
            max_requests_per_minute: int = 2000,
    ):
        """
        Initialize the PDFParser.

        Args:
            api_key: Google Gemini API key. If not provided, uses GOOGLE_API_KEY
                environment variable.
            dpi: Image quality for PDF conversion. Higher values produce better
                quality but larger files.
            max_workers: Maximum number of parallel workers for transcription.
            remove_code_markers: Whether to remove code block markers from results.
            max_requests_per_minute: Maximum API requests allowed per minute.

        Raises:
            ValueError: If API key is not provided and not in environment variables.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not provided and GOOGLE_API_KEY not found in environment"
            )

        self.dpi = dpi
        self.max_workers = max_workers
        self.remove_code_markers = remove_code_markers
        self.max_requests_per_minute = max_requests_per_minute

        # Cache for storing converted images
        self._image_cache = {}

        # Initialize the model using LangChain
        self.model = init_chat_model(
            model="gemini-2.0-flash",
            model_provider="google-genai",
            temperature=0.1,
            max_tokens=8192
        )

    def _convert_pdf_to_images(
            self, pdf_path: Union[str, Path], force_reload: bool = False
    ) -> List[Image.Image]:
        """
        Convert PDF pages into high-quality images.
        Uses caching to avoid redundant conversions.

        Args:
            pdf_path: Path to the PDF file.
            force_reload: If True, ignore cache and reload images.

        Returns:
            List of PIL Image objects representing each page.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            fitz.FileDataError: If the file is not a valid PDF.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Check if images are already in cache
        cache_key = str(pdf_path.absolute())
        if not force_reload and cache_key in self._image_cache:
            logger.info(f"Using cached images for {pdf_path}")
            return self._image_cache[cache_key]

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
                    img = Image.frombytes(
                        "RGB", [pix.width, pix.height], pix.samples
                    )
                    images.append(img)
                    pbar.update(1)
            finally:
                doc.close()

        # Store images in cache
        self._image_cache[cache_key] = images
        return images

    def _prepare_image(self, image: Image.Image) -> str:
        """
        Convert a PIL image into a base64-encoded data URI string.

        Args:
            image: The PIL image to prepare.

        Returns:
            A base64-encoded string in data URI format.
        """
        # Convert the image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_bytes = buffered.getvalue()

        # Encode the image in base64
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        # Format as a data URI
        data_uri = f"data:image/jpeg;base64,{img_base64}"
        return data_uri

    def _clean_response_text(self, text: str) -> str:
        """
        Clean the response text by removing code block markers.

        Args:
            text: The text to clean.

        Returns:
            Cleaned text.
        """
        # Remove code block markers
        text = text.replace("```markdown", "")
        text = text.replace("```html", "")
        text = text.replace("```", "")

        # Remove leading and trailing whitespace
        text = text.strip()

        return text

    def _generate_custom_prompt(self, sample_images: List[Image.Image]) -> str:
        """
        Generate a custom prompt based on sample pages.

        Args:
            sample_images: List of sample images.

        Returns:
            Custom prompt for transcription.
        """
        logger.info("Analyzing sample pages to generate custom prompt...")

        # Prepare the first sample image for analysis
        if not sample_images:
            return self._get_default_prompt()

        sample_image = sample_images[0]
        image_data_uri = self._prepare_image(sample_image)

        analysis_prompt = """
        Analyze this sample page from a PDF document.

        I need you to identify:
        1. The type of document (academic, technical, legal, etc.)
        2. The main structural elements (headings, subheadings, paragraphs, lists,
           tables, etc.)
        3. ESPECIALLY IMPORTANT: Identify if there are elements that LOOK like tables
           but are actually:
           - Paragraphs formatted in columns
           - Narrative text inside cells
           - Structures that do not contain genuine tabular data
        4. Any special or complex formatting that requires particular attention

        Provide a brief analysis to help me understand the document's format and
        structure. Do not transcribe the content, just analyze its structure and format.
        """

        # Create a message for analysis
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": analysis_prompt,
                },
                {"type": "image_url", "image_url": image_data_uri},
            ]
        )

        try:
            # Get analysis from the model
            response = self.model.invoke([message])
            analysis = response.content

            # Generate a custom prompt based on the analysis
            custom_prompt = f"""
            Transcribe this PDF document into Markdown with an intelligent focus on
            structure.

            Based on the analysis of the sample pages, this appears to be a document
            with the following characteristics:

            {analysis}

            CRITICAL INSTRUCTIONS ON STRUCTURE:
            1. FIRST ANALYZE each element to determine if it is really:
               - A normal text paragraph
               - A genuine table with tabular data
               - A list
               - A heading
               - Another structural element

            2. DECISION ON TABLES:
               - ONLY use HTML tables (<table>, <tr>, <td>) when the content is
                 GENUINELY TABULAR
               - If you find text formatted as a table but conceptually it is a
                 paragraph, CONVERT IT TO NORMAL TEXT
               - If a table contains mostly narrative text or long paragraphs,
                 REMOVE the table structure
               - Preserve tables ONLY when the relationship between rows and columns
                 is meaningful

            3. GENERAL FORMAT:
               - Use Markdown for everything except genuine tables
               - Preserve formatting elements (bold, italics, etc.)
               - Maintain the hierarchical structure of the document
               - Ensure the text flow is natural and readable

            4. FOR COMPLEX ELEMENTS:
               - Diagrams: briefly describe them in brackets [Diagram: description]
               - Mathematical formulas: use LaTeX syntax
               - Code: use code blocks with the appropriate language
               - Complex visual elements: restructure them while maintaining their
                 meaning and hierarchy

            REMEMBER: The main goal is to produce a readable and well-structured
            document, NOT to exactly reproduce the visual format of the PDF.
            """

            logger.info("Custom prompt successfully generated")
            return custom_prompt

        except Exception as e:
            logger.warning(
                f"Error generating custom prompt: {e}. Using default prompt."
            )
            return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """
        Return a default prompt for transcription.

        Returns:
            Default prompt.
        """
        return """
        Transcribe this PDF document into Markdown with an intelligent focus on
        structure.

        CRITICAL INSTRUCTIONS ON STRUCTURE:
        1. FIRST ANALYZE each element to determine if it is really:
           - A normal text paragraph
           - A genuine table with tabular data
           - A list
           - A heading
           - Another structural element

        2. DECISION ON TABLES:
           - ONLY use HTML tables (<table>, <tr>, <td>) when the content is
             GENUINELY TABULAR
           - If you find text formatted as a table but conceptually it is a paragraph,
             CONVERT IT TO NORMAL TEXT
           - If a table contains mostly narrative text or long paragraphs,
             REMOVE the table structure
           - Preserve tables ONLY when the relationship between rows and columns
             is meaningful

        3. GENERAL FORMAT:
           - Use Markdown for everything except genuine tables
           - Preserve formatting elements (bold, italics, etc.)
           - Maintain the hierarchical structure of the document
           - Ensure the text flow is natural and readable

        4. FOR COMPLEX ELEMENTS:
           - Diagrams: briefly describe them in brackets [Diagram: description]
           - Mathematical formulas: use LaTeX syntax
           - Code: use code blocks with the appropriate language
           - Complex visual elements: restructure them while maintaining their meaning
             and hierarchy

        REMEMBER: The main goal is to produce a readable and well-structured document,
        NOT to exactly reproduce the visual format of the PDF.
        """

    def _transcribe_page(
            self, image: Image.Image, custom_prompt: str, page_index: int
    ) -> str:
        """
        Transcribe a single page using a base64-encoded image.

        Args:
            image: The image of the page to transcribe.
            custom_prompt: The custom prompt for transcription.
            page_index: The index of the page (0-based).

        Returns:
            Transcribed content in Markdown/HTML format with page number.

        Raises:
            Exception: If transcription fails, with error details.
        """
        try:
            # Prepare the image as a base64-encoded string
            image_data_uri = self._prepare_image(image)

            transcription_prompt = f"""
            {custom_prompt}

            SPECIFIC INSTRUCTIONS FOR THIS PAGE (Page {page_index + 1}):

            TWO-STEP TRANSCRIPTION PROCESS:

            STEP 1 - STRUCTURE ANALYSIS:
            - Carefully examine each element on the page
            - Identify which elements are genuinely tabular and which are narrative text
            - Determine if there are "false tables" that contain paragraphs or
              narrative text

            STEP 2 - INTELLIGENT TRANSCRIPTION:
            - For normal text: use paragraphs in Markdown
            - For GENUINE tables (tabular data): use HTML (<table>, <tr>, <td>)
            - For "false tables" or tables with paragraphs: CONVERT to normal text
              in Markdown
            - For lists: use Markdown list format
            - For headings: use # with the appropriate level

            CRITICAL RULES:
            1. DO NOT preserve tables that mainly contain narrative text
            2. DO NOT create HTML tables for content that should be normal text
            3. DO NOT omit any content from the original document
            4. DO NOT add additional content
            5. DO NOT include code block markers (```) in the response

            If you encounter complex formatting on this page:
            - Restructure the content according to its type and purpose
            - If you encounter a large table containing a long paragraph and think it
              would be useful to restructure the page by removing that unnecessary
              table, DO IT
              
            CRITICAL INSTRUCTIONS ON STRUCTURE:
            1. FIRST ANALYZE each element to determine if it is really:
               - A normal text paragraph
               - A genuine table with tabular data
               - A list
               - A heading
               - Another structural element

            2. DECISION ON TABLES:
               - ONLY use HTML tables (<table>, <tr>, <td>) when the content is
                 GENUINELY TABULAR
               - If you find text formatted as a table but conceptually it is a
                 paragraph, CONVERT IT TO NORMAL TEXT
               - If a table contains mostly narrative text or long paragraphs,
                 REMOVE the table structure
               - Preserve tables ONLY when the relationship between rows and columns
                 is meaningful

            3. GENERAL FORMAT:
               - Use Markdown for everything except genuine tables
               - Preserve formatting elements (bold, italics, etc.)
               - Maintain the hierarchical structure of the document
               - Ensure the text flow is natural and readable

            4. FOR COMPLEX ELEMENTS:
               - Diagrams: briefly describe them in brackets [Diagram: description]
               - Mathematical formulas: use LaTeX syntax
               - Code: use code blocks with the appropriate language
               - Complex visual elements: restructure them while maintaining their
                 meaning and hierarchy

            REMEMBER: The main goal is to produce a readable and well-structured
            document, NOT to exactly reproduce the visual format of the PDF.
            """

            # Create a message for the model using HumanMessage
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": transcription_prompt,
                    },
                    {"type": "image_url", "image_url": image_data_uri},
                ]
            )

            # Get response from the model
            response = self.model.invoke([message])

            content = response.content
            if self.remove_code_markers:
                content = self._clean_response_text(content)

            # Ensure the page number appears at the beginning
            page_header = f"Page {page_index + 1}\n\n"
            return page_header + content

        except Exception as e:
            logger.error(f"Error transcribing page {page_index + 1}: {e}")
            return f"Error transcribing page {page_index + 1}: {e}"

    def transcribe_pdf(
            self,
            pdf_path: Union[str, Path],
            output_path: Optional[Union[str, Path]] = None,
            sample_pages: int = 3,
            images: Optional[List[Image.Image]] = None,
    ) -> Path:
        """
        Transcribe a PDF document into Markdown with HTML tables.

        Args:
            pdf_path: Path to the PDF file.
            output_path: Path for the output Markdown file. If None, uses the PDF
                name with a .md extension.
            sample_pages: Number of sample pages to generate the prompt.
            images: Optional pre-loaded images to use instead of loading from PDF.

        Returns:
            Path to the generated Markdown file.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If sample_pages is less than 1.
            Exception: For other errors during transcription.
        """
        if sample_pages < 1:
            raise ValueError("sample_pages must be at least 1")

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            logger.info(f"Starting transcription of {pdf_path}")

            # Use provided images or convert PDF to images
            if images is None:
                images = self._convert_pdf_to_images(pdf_path)

            logger.info(f"Working with {len(images)} images")

            # Select sample pages to generate the prompt
            sample_count = min(sample_pages, len(images))
            sample_indices = random.sample(range(len(images)), sample_count)
            sample_images = [images[i] for i in sample_indices]
            logger.info(
                f"Selected {len(sample_images)} sample pages for prompt generation"
            )

            # Generate custom prompt based on the sample pages
            custom_prompt = self._generate_custom_prompt(sample_images)

            # Process pages in parallel
            results = [None] * len(images)  # Pre-allocate results to maintain order

            with tqdm(
                    total=len(images), desc="Transcribing pages", unit="page"
            ) as pbar:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._transcribe_page, img, custom_prompt, idx
                        ): idx
                        for idx, img in enumerate(images)
                    }

                    for future in as_completed(futures):
                        original_idx = futures[future]
                        results[original_idx] = future.result()
                        pbar.update(1)

            # Save results
            if output_path is None:
                output_path = pdf_path.with_suffix(".md")
            else:
                output_path = Path(output_path)

            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(results))

            logger.info(f"Transcription completed and saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise

    def batch_transcribe_pdfs(
            self,
            pdf_dir: Union[str, Path],
            output_dir: Optional[Union[str, Path]] = None,
            sample_pages: int = 3,
            file_pattern: str = "*.pdf",
    ) -> List[Path]:
        """
        Batch transcribe multiple PDF documents in a directory.

        Args:
            pdf_dir: Directory containing PDF files.
            output_dir: Directory for output Markdown files. If None, uses the same
                directory as the PDFs.
            sample_pages: Number of sample pages to generate the prompt for each PDF.
            file_pattern: Pattern to match PDF files.

        Returns:
            List of paths to the generated Markdown files.

        Raises:
            FileNotFoundError: If the directory does not exist.
            Exception: For other errors during transcription.
        """
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists() or not pdf_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {pdf_dir}")

        if output_dir is None:
            output_dir = pdf_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = list(pdf_dir.glob(file_pattern))
        if not pdf_files:
            logger.warning(
                f"No PDF files found in {pdf_dir} matching pattern {file_pattern}"
            )
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to transcribe")
        output_paths = []

        for pdf_file in tqdm(pdf_files, desc="Transcribing PDFs", unit="file"):
            try:
                output_path = output_dir / f"{pdf_file.stem}.md"
                result_path = self.transcribe_pdf(
                    pdf_file, output_path, sample_pages=sample_pages
                )
                output_paths.append(result_path)
            except Exception as e:
                logger.error(f"Error transcribing {pdf_file}: {e}")

        return output_paths

    def analyze_document_structure(
            self, pdf_path: Union[str, Path], images: Optional[List[Image.Image]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the structure of a PDF document without transcribing it.

        Args:
            pdf_path: Path to the PDF file.
            images: Optional pre-loaded images to use instead of loading from PDF.

        Returns:
            Dictionary containing analysis results.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            Exception: For other errors during analysis.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            # Use provided images or convert PDF to images
            if images is None:
                images = self._convert_pdf_to_images(pdf_path)

            # Use first 3 pages for analysis
            sample_images = images[:3]

            if not sample_images:
                return {"error": "Could not extract images from PDF"}

            # Prepare the first image
            image_data_uri = self._prepare_image(sample_images[0])

            analysis_prompt = """
            Analyze this PDF document and provide a detailed structural assessment.

            Please identify and quantify:
            1. The type of document (academic, technical, legal, etc.)
            2. The approximate number and types of structural elements:
               - Headings and subheadings
               - Paragraphs
               - Lists (bulleted, numbered)
               - Tables (genuine tabular data vs. text formatted as tables)
               - Figures, charts, or diagrams
               - Special elements (equations, code blocks, etc.)
            3. Any potential challenges in transcribing this document:
               - Complex layouts
               - Multi-column text
               - Text embedded in images
               - "False tables" that should be converted to text
               - Other structural issues

            Provide your analysis in a structured format that can be easily parsed.
            """

            # Create a message for analysis
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": analysis_prompt,
                    },
                    {"type": "image_url", "image_url": image_data_uri},
                ]
            )

            # Get analysis from the model
            response = self.model.invoke([message])
            analysis = response.content

            # Return the analysis as a dictionary
            return {
                "pdf_path": str(pdf_path),
                "analysis": analysis,
                "page_count": len(images),
            }

        except Exception as e:
            logger.error(f"Error analyzing document structure: {e}")
            return {"error": str(e)}

    def extract_table_of_contents(
            self, pdf_path: Union[str, Path], images: Optional[List[Image.Image]] = None, block_size: int = 5
    ) -> str:
        """
        Extract the table of contents from all pages of a PDF document in smaller blocks.

        Args:
            pdf_path: Path to the PDF file.
            images: Optional pre-loaded images to use instead of loading from PDF.
            block_size: Number of pages to process in each block.

        Returns:
            Markdown-formatted table of contents.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            Exception: For other errors during extraction.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            # Use provided images or convert PDF to images
            if images is None:
                images = self._convert_pdf_to_images(pdf_path)

            if not images:
                return "Could not extract images from PDF"

            # Split images into blocks
            toc_parts = []
            total_blocks = (len(images) + block_size - 1) // block_size  # Ceiling division

            with tqdm(total=total_blocks, desc="Extracting TOC in blocks", unit="block") as pbar:
                for i in range(0, len(images), block_size):
                    block_images = images[i:i + block_size]
                    block_page_range = f"(Pages {i+1}-{min(i+block_size, len(images))})"

                    # Prepare images in this block
                    image_data_uris = [self._prepare_image(img) for img in block_images]

                    toc_prompt = f"""
                    Extract the Table of Contents from this block of pages {block_page_range}.

                    Instructions:
                    1. Analyze the pages to identify the structure.
                    2. Look for section titles, chapter headings, or any hierarchical structure
                       that resembles a Table of Contents.
                    3. Extract all section titles and their corresponding page numbers.
                    4. Format the result as a Markdown list with proper indentation for
                       subsections.
                    5. If no explicit Table of Contents exists, create a structural outline
                       based on the headings and sections found throughout the document.
                    6. Include page numbers for each section or heading.

                    Format the Table of Contents in clean, hierarchical Markdown.
                    """

                    # Create content with the block of images
                    content = [{"type": "text", "text": toc_prompt}]
                    for uri in image_data_uris:
                        content.append({"type": "image_url", "image_url": uri})

                    # Create a message for extraction
                    message = HumanMessage(content=content)

                    # Get TOC from the model
                    try:
                        response = self.model.invoke([message])
                        toc_parts.append(response.content)
                    except Exception as e:
                        error_msg = f"Error processing block {i // block_size + 1}: {e}"
                        logger.error(error_msg)
                        toc_parts.append(f"### {error_msg}")

                    pbar.update(1)

            # Combine TOC from all blocks
            combined_toc = "\n\n".join(toc_parts)

            # If we processed multiple blocks, add a note
            if len(toc_parts) > 1:
                combined_toc = "# Combined Table of Contents\n\n" + \
                               "_Note: This TOC was extracted in multiple blocks and combined._\n\n" + \
                               combined_toc

            return combined_toc

        except Exception as e:
            logger.error(f"Error extracting table of contents: {e}")
            return f"Error extracting table of contents: {e}"

    def process_document(
            self, pdf_path: Union[str, Path], output_dir: str = "results"
    ) -> Dict[str, Any]:
        """
        Process a PDF document with a complete workflow:
        1. Analyze structure
        2. Extract table of contents
        3. Transcribe with optimized settings
        4. Save supplementary files

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save results

        Returns:
            Dictionary with paths to all generated files
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Get filename without extension for output files
        pdf_path = Path(pdf_path)
        filename = pdf_path.stem
        base_output_path = Path(output_dir) / filename

        # Setup progress tracking
        steps = ["PDF Conversion", "Analysis", "TOC Extraction",
                 "Transcription", "Saving Files"]
        results = {}

        with tqdm(
                total=len(steps),
                desc=f"Processing {pdf_path.name}",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) as pbar:
            # Step 1: Convert PDF to images (only once)
            pbar.set_description("Converting PDF to images")
            images = self._convert_pdf_to_images(pdf_path)
            results['images'] = len(images)
            pbar.update(1)

            # Step 2: Analyze document structure
            pbar.set_description("Analyzing document structure")
            analysis = self.analyze_document_structure(pdf_path, images=images)
            results['analysis'] = analysis
            pbar.update(1)

            # Check if document has complex structure
            analysis_text = analysis.get('analysis', '')
            has_complex_tables = ("false tables" in analysis_text.lower() or
                                  "complex tables" in analysis_text.lower())

            # Step 3: Extract table of contents
            pbar.set_description("Extracting table of contents")
            # Use smaller block size for larger documents
            block_size = 3 if len(images) > 30 else 5
            toc = self.extract_table_of_contents(pdf_path, images=images, block_size=block_size)
            results['toc'] = toc
            pbar.update(1)

            # Step 4: Transcribe with optimized settings
            sample_pages = 5 if has_complex_tables else 3
            pbar.set_description(
                f"Transcribing document (using {sample_pages} sample pages)"
            )

            output_path = self.transcribe_pdf(
                pdf_path=pdf_path,
                output_path=f"{base_output_path}_transcribed.md",
                sample_pages=sample_pages,
                images=images
            )
            results['transcription'] = output_path
            pbar.update(1)

            # Step 5: Save supplementary files
            pbar.set_description("Saving supplementary files")

            # Save TOC
            toc_path = f"{base_output_path}_toc.md"
            with open(toc_path, "w", encoding="utf-8") as f:
                f.write("# Table of Contents\n\n")
                f.write(toc)
            results['toc_file'] = toc_path

            # Save analysis
            analysis_path = f"{base_output_path}_analysis.md"
            with open(analysis_path, "w", encoding="utf-8") as f:
                f.write("# Document Structure Analysis\n\n")
                f.write(analysis.get('analysis', 'Analysis not available'))
            results['analysis_file'] = analysis_path

            pbar.update(1)

        # Print summary of results
        print("\n✅ Document processing completed successfully!")
        print("📄 Files generated:")
        print(f"  • Transcription: {results['transcription']}")
        print(f"  • Table of Contents: {results['toc_file']}")
        print(f"  • Structure Analysis: {results['analysis_file']}")
        print(f"  • Total pages processed: {results['images']}")

        if has_complex_tables:
            print("\n⚠️ Note: Complex or 'false' tables detected in document.")
            print(f"   Used {sample_pages} sample pages for improved analysis.")

        return results