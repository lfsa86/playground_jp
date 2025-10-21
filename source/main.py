from ast import List
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.globals import set_verbose
import pandas as pd

from source.clasification import classify_statements
from source.extraction import CommitmentExtractor
from source.parsing import HeadingParser, PDFParser

set_verbose(True)
load_dotenv()
file_path = "data/raw/ToromochoMEIA.pdf"
file_path = Path(file_path)
file_name = file_path.stem

ocr_pdf_parser = PDFParser()
file_md_content = ocr_pdf_parser.process_pdf(input_path=file_path)
# heading_parser = HeadingParser()
# fixed_md_content: str = heading_parser.normalize_headings(
#    file_name=file_name, md_content=file_md_content
# )
df = classify_statements(file_name=file_name, md_content=file_md_content) #file_md_content or #fixed_md_content
commitment_extractor = CommitmentExtractor()
df = commitment_extractor.extract(file_name=file_name, df=df)
