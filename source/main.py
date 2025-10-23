from pathlib import Path
from dotenv import load_dotenv
from langchain_core.globals import set_verbose
import pandas as pd
import os

from source.clasification import classify_statements
from source.extraction import CommitmentExtractor
from source.parsing import PDFParser
try:
    from source.parsing import HeadingParser  # opcional
except ImportError:
    HeadingParser = None

# -------------------- CONFIG --------------------
set_verbose(True)
load_dotenv()

file_path = Path("data/raw/Aunor/2023-81-T-ITS-00349-2022/139611-2022-57807-03_Tortuga_Capítulo_III_Descripción_del_ITSFolio063-0311.pdf")
file_name = file_path.stem

BASE_OUT = Path("data/processed") / file_name
BASE_OUT.mkdir(parents=True, exist_ok=True)

# -------------------- MENÚ --------------------
print("\n🛠 OPCIONES DE EJECUCIÓN:")
print("1. Hasta markdown con headings normalizados")
print("2. Hasta CSV/XLSX/Parquet de statements")
print("3. Pipeline completo (commitments)\n")
choice = input("Selecciona una opción (1, 2 o 3): ").strip()

# -------------------- PASO 1: PARSER --------------------
print("\n▶ Paso 1: Procesando PDF con OCR...")
ocr_pdf_parser = PDFParser()
pdf_result = ocr_pdf_parser.process_pdf(input_path=file_path)
file_md_content = pdf_result["md_text"] if isinstance(pdf_result, dict) and "md_text" in pdf_result else str(pdf_result)

# Normalizar headings
if HeadingParser:
    print("▶ Normalizando encabezados...")
    heading_parser = HeadingParser()
    fixed_md_content = heading_parser.normalize_headings(
        file_name=file_name,
        md_content=file_md_content,
        # Esta función YA guarda en data/processed/<file_name>/<file_name>.md
    )
else:
    fixed_md_content = file_md_content
    # Guardamos SOLO si no existe salida estándar:
    md_std_path = BASE_OUT / f"{file_name}.md"
    md_std_path.write_text(fixed_md_content, encoding="utf-8")
    print(f"✅ Markdown guardado en: {md_std_path}")

if choice == "1":
    print("\n🎉 Pipeline detenido tras generar el markdown normalizado.")
    raise SystemExit(0)

# -------------------- PASO 2: CLASSIFY --------------------
print("\n▶ Paso 2: Clasificando statements...")
# classify_statements YA guarda en data/processed/<file_name>/statements/{parquet,csv,xlsx}
df_statements = classify_statements(file_name=file_name, md_content=fixed_md_content, output_dir=str(BASE_OUT.parent))

if choice == "2":
    print("\n🎉 Pipeline detenido tras generar los statements (archivos en carpeta 'statements').")
    raise SystemExit(0)

# -------------------- PASO 3: EXTRACT --------------------
print("\n▶ Paso 3: Extrayendo compromisos detallados...")
commitment_extractor = CommitmentExtractor()
df_commitments = commitment_extractor.extract(file_name=file_name, df=df_statements)

# ⚠️ Ya no guardamos aquí: el método extract() lo hace internamente
print(f"✅ Archivos de commitments generados automáticamente en: data/processed/{file_name}/commitments/")
print("\n🎉 Pipeline completo ejecutado exitosamente.")
