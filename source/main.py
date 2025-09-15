from pathlib import Path
from dotenv import load_dotenv
from langchain.globals import set_verbose
import pandas as pd

from source.clasification import classify_statements
from source.extraction import CommitmentExtractor
from source.parsing import PDFParser

# Configuración
set_verbose(True)
load_dotenv()

# Ruta del archivo
file_path = Path("data/raw/Quellaveco_ITS_PMA_Cap11Texto.pdf")
file_name = file_path.stem

# Procesar PDF y obtener markdown completo
ocr_pdf_parser = PDFParser()
file_md_content = ocr_pdf_parser.process_pdf(input_path=file_path)["md_text"]

# Clasificar compromisos
df = classify_statements(file_name=file_name, md_content=file_md_content)

# Extraer compromisos detallados
commitment_extractor = CommitmentExtractor()
df = commitment_extractor.extract(file_name=file_name, df=df)

# (Opcional) Exportar resultados
df.to_excel(f"data/processed/{file_name}_compromisos.xlsx", index=False)