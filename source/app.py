from pathlib import Path

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from langchain.globals import set_verbose
from pandas import DataFrame

from source.clasification import classify_statements
from source.extraction import CommitmentExtractor, ComponentExtractor
from source.parsing import PDFParser

# Estilo CSS personalizado para las tablas
custom_css = """
.table-wrap table td {
    white-space: normal !important;
    word-wrap: break-word !important;
    max-width: 300px !important;
    padding: 8px !important;
}
.table-wrap table {
    width: 100% !important;
    table-layout: fixed !important;
}
"""


def process_pdf_file(pdf_file):
    set_verbose(True)
    load_dotenv()

    results = []

    # Step 1: Parse PDF
    parser = PDFParser()
    file_name = Path(pdf_file).stem
    file_content = parser.process_pdf(pdf_file)
    results.append("1. PDF parseado exitosamente")

    # Step 2: Classify statements
    file_df = classify_statements(file_name=file_name, md_content=file_content)
    results.append("2. Statements clasificados")
    results.append(f"Archivo: {file_name}")
    results.append(f"Statements encontrados: {len(file_df)}")

    # Step 3: Extract commitments
    commitment_extractor = CommitmentExtractor()
    final_df = commitment_extractor.extract(file_name=file_name, df=file_df)
    results.append("4. Compromisos extraídos")
    results.append(f"Compromisos totales: {len(final_df)}")

    # Estadísticas generales
    stats = f"""
    Resumen del proceso:
    - Statements totales: {len(file_df)}
    - Compromisos finales: {len(final_df)}
    """

    # Ajustar el ancho de las columnas para mejor visualización
    pd.set_option("display.max_colwidth", None)

    return ("\n".join(results), stats, file_df, final_df)


with gr.Blocks(title="Herramienta de Análisis de PDF", css=custom_css) as demo:
    gr.Markdown("# Herramienta de Análisis de PDF")
    gr.Markdown("""
    Esta herramienta procesa documentos PDF en cuatro pasos:
    1. Parseo del PDF
    2. Clasificación de statements
    3. Extracción de componentes
    4. Extracción de compromisos
    """)

    with gr.Row():
        file_input = gr.File(label="Subir PDF", file_types=[".pdf"])

    with gr.Row():
        with gr.Column():
            process_output = gr.Textbox(label="Proceso", lines=10)
            stats_output = gr.Textbox(label="Estadísticas", lines=5)

    with gr.Tabs() as tabs:
        with gr.Tab("Statements"):
            with gr.Column(elem_classes="table-wrap"):
                statements_df = gr.Dataframe(
                    label="Statements Clasificados", wrap=True, interactive=False
                )

        with gr.Tab("Componentes"):
            with gr.Column(elem_classes="table-wrap"):
                components_df = gr.Dataframe(
                    label="Componentes Extraídos", wrap=True, interactive=False
                )

        with gr.Tab("Compromisos"):
            with gr.Column(elem_classes="table-wrap"):
                commitments_df = gr.Dataframe(
                    label="Compromisos Finales", wrap=True, interactive=False
                )

    file_input.change(
        fn=process_pdf_file,
        inputs=[file_input],
        outputs=[
            process_output,
            stats_output,
            statements_df,
            components_df,
            commitments_df,
        ],
    )

    gr.Examples(
        examples=[["data/raw/rca.pdf"]],
        inputs=file_input,
        outputs=[
            process_output,
            stats_output,
            statements_df,
            components_df,
            commitments_df,
        ],
    )

if __name__ == "__main__":
    demo.launch(share=True)
