from ast import List
from pathlib import Path
from dotenv import load_dotenv
from langchain.globals import set_verbose

from source.clasification import classify_statements
from source.extraction import CommitmentExtractor
from source.parsing import HeadingParser, PDFParser

import pandas as pd

set_verbose(True)
load_dotenv()


def run_parser_only(file_path: Path):
    print("▶ Ejecutando solo el PARSER...")
    ocr_pdf_parser = PDFParser()

    # Ahora devuelve una ruta, no el contenido
    raw_md_path = ocr_pdf_parser.process_pdf(input_path=file_path)

    heading_parser = HeadingParser()
    # Leemos el contenido desde el archivo
    file_md_content = raw_md_path.read_text(encoding="utf-8")

    # Normalizamos los encabezados a partir del contenido leído
    fixed_md_content = heading_parser.normalize_headings(
        file_name=file_path.stem, md_content=file_md_content
    )

    # Guardamos el nuevo markdown con headings normalizados
    md_output_path = raw_md_path.parent / f"{file_path.stem}.md"
    md_output_path.write_text(fixed_md_content, encoding="utf-8")
    print(f"✅ Archivo markdown generado en: {md_output_path}")


def run_parser_and_classifier(file_path: Path):
    print("▶ Ejecutando PARSER + CLASSIFIER...")
    ocr_pdf_parser = PDFParser()
    file_md_content = ocr_pdf_parser.process_pdf(input_path=file_path)

    heading_parser = HeadingParser()
    fixed_md_content = heading_parser.normalize_headings(
        file_name=file_path.stem, md_content=file_md_content
    )

    df = classify_statements(file_name=file_path.stem, md_content=fixed_md_content)

    # Guardar ambos resultados
    md_output_path = Path(f"data/processed/{file_path.stem}.md")
    md_output_path.write_text(fixed_md_content, encoding="utf-8")

    df_output_path = Path(f"data/processed/{file_path.stem}_statements.csv")
    df.to_csv(df_output_path, index=False)

    print(f"✅ Markdown guardado en: {md_output_path}")
    print(f"✅ Statements guardados en: {df_output_path}")


def run_full_pipeline(file_path: Path):
    print("▶ Ejecutando PIPELINE COMPLETO...")
    ocr_pdf_parser = PDFParser()
    file_md_content = ocr_pdf_parser.process_pdf(input_path=file_path)

    heading_parser = HeadingParser()
    fixed_md_content = heading_parser.normalize_headings(
        file_name=file_path.stem, md_content=file_md_content
    )

    df = classify_statements(file_name=file_path.stem, md_content=fixed_md_content)

    commitment_extractor = CommitmentExtractor()
    df = commitment_extractor.extract(file_name=file_path.stem, df=df)

    # Guardar todos los resultados
    md_output_path = Path(f"data/processed/{file_path.stem}.md")
    md_output_path.write_text(fixed_md_content, encoding="utf-8")

    df_output_path = Path(f"data/processed/{file_path.stem}_commitments.csv")
    df.to_csv(df_output_path, index=False)

    print(f"✅ Markdown guardado en: {md_output_path}")
    print(f"✅ Commitments guardados en: {df_output_path}")


if __name__ == "__main__":
    file_path = Path("data/raw/rca2022.pdf") #cambiar archivo aqui

    print("\n🛠 OPCIONES DE EJECUCIÓN:")
    print("1. Solo parser (markdown)")
    print("2. Parser + classifier (markdown + statements)")
    print("3. Full pipeline (markdown + statements + commitments)\n")

    choice = input("Selecciona una opción (1, 2 o 3): ").strip()

    try:
        if choice == "1":
            run_parser_only(file_path)
        elif choice == "2":
            run_parser_and_classifier(file_path)
        elif choice == "3":
            run_full_pipeline(file_path)
        else:
            print("❌ Opción inválida. Por favor selecciona 1, 2 o 3.")
    except Exception as e:
        print(f"❌ Error al ejecutar el flujo: {e}")
