import os
import time
from pathlib import Path
from core.utils.pdf_parser import PDFParser

pdf_path = "../../data/raw/rca.pdf"
output_dir = Path("../../data/processed/output")
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize parser with basic configuration
parser = PDFParser(max_workers=100)

print(f"Processing: {pdf_path}")

# Convert PDF to images
images = parser._convert_pdf_to_images(pdf_path)
# Transcribe document
output_path = parser.transcribe_pdf(
    pdf_path=pdf_path,
    output_path=output_dir / "rca.md",
    sample_pages=10,
    images=images
)

print(f"\nProcessing completed:")
print(f"- Transcription: {output_path}")
