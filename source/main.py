from pathlib import Path
from dotenv import load_dotenv
from langchain_core.globals import set_verbose
import pandas as pd

from source.clasification import classify_statements
from source.extraction import CommitmentExtractor
from source.parsing import PDFParser
try:
    from source.parsing import HeadingParser  # opcional, si lo tienes implementado
except ImportError:
    HeadingParser = None

# -------------------- CONFIGURACIÓN --------------------
set_verbose(True)
load_dotenv()

# Ruta del archivo PDF
file_path = Path("data/raw/Aunor/2023-81-T-ITS-00349-2022/139611-2022-57807-03_Tortuga_Capítulo_III_Descripción_del_ITSFolio063-0311.pdf")
file_name = file_path.stem

# Directorio de salida
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# -------------------- SELECCIÓN DE PASO --------------------
print("\n🛠 OPCIONES DE EJECUCIÓN:")
print("1. Hasta markdown con headings normalizados")
print("2. Hasta CSV de statements")
print("3. Hasta CSV/Excel de commitments (pipeline completo)\n")

choice = input("Selecciona una opción (1, 2 o 3): ").strip()

# -------------------- PASO 1: PARSER --------------------
print("\n▶ Paso 1: Procesando PDF con OCR...")
ocr_pdf_parser = PDFParser()
pdf_result = ocr_pdf_parser.process_pdf(input_path=file_path)

# El resultado puede ser un dict o un string
if isinstance(pdf_result, dict) and "md_text" in pdf_result:
    file_md_content = pdf_result["md_text"]
else:
    file_md_content = str(pdf_result)

# Normalización de headings si existe HeadingParser
if HeadingParser:
    print("▶ Normalizando encabezados...")
    heading_parser = HeadingParser()
    fixed_md_content = heading_parser.normalize_headings(
        file_name=file_name,
        md_content=file_md_content
    )
else:
    fixed_md_content = file_md_content

# Guardar el markdown con headings normalizados
md_output_path = output_dir / f"{file_name}.md"
md_output_path.write_text(fixed_md_content, encoding="utf-8")
print(f"✅ Markdown guardado en: {md_output_path}")

if choice == "1":
    print("\n🎉 Pipeline detenido tras generar el markdown normalizado.")
    exit(0)

# -------------------- PASO 2: CLASSIFY --------------------
print("\n▶ Paso 2: Clasificando statements...")
df_statements = classify_statements(file_name=file_name, md_content=fixed_md_content)

# Guardar resultados en CSV
statements_path = output_dir / f"{file_name}_statements.csv"
df_statements.to_csv(statements_path, index=False)
print(f"✅ Statements guardados en: {statements_path}")

if choice == "2":
    print("\n🎉 Pipeline detenido tras generar el CSV de statements.")
    exit(0)

# -------------------- PASO 3: EXTRACT --------------------
print("\n▶ Paso 3: Extrayendo compromisos detallados...")
commitment_extractor = CommitmentExtractor()
df_commitments = commitment_extractor.extract(file_name=file_name, df=df_statements)

# Guardar resultados finales
csv_output_path = output_dir / f"{file_name}_compromisos.csv"
xlsx_output_path = output_dir / f"{file_name}_compromisos.xlsx"

df_commitments.to_csv(csv_output_path, index=False)
try:
    df_commitments.to_excel(xlsx_output_path, index=False)
except Exception as e:
    print(f"ℹ️ No se pudo exportar a Excel (instala xlsxwriter si lo necesitas): {e}")

print(f"✅ CSV guardado en: {csv_output_path}")
print(f"✅ Excel guardado en: {xlsx_output_path}")
print("\n🎉 Pipeline completo ejecutado exitosamente.")
