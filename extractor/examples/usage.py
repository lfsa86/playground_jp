import concurrent.futures
import json
import time

from tqdm import tqdm

from extractor.base import extract_commitments, load_md


def process_paragraph(args):
    paragraph, page_number = args
    commitment = extract_commitments(paragraph)
    commitment["text"] = paragraph
    commitment["page_number"] = page_number
    time.sleep(45)  # Simulación de procesamiento
    return commitment


# Load all pages
pages = load_md("extractor/data/rca.md")
total_pages = len(pages)
commitments = []

# Preparar todos los párrafos con sus números de página
all_paragraphs = []
for page in pages:
    for p in page.paragraphs:
        all_paragraphs.append((p, page.page_number))

total_paragraphs = len(all_paragraphs)

# Procesar párrafos en paralelo con una barra de progreso adecuada
with tqdm(total=total_paragraphs, desc="Processing paragraphs") as pbar:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Mapear la función a todos los párrafos
        for commitment in executor.map(process_paragraph, all_paragraphs):
            commitments.append(commitment)

print(f"Total statements categorized: {len(commitments)}")

with open("commitments.json", "w", encoding="utf-8") as f:
    json.dump(commitments, f, ensure_ascii=False, indent=4)

compromisos = [c for c in commitments if c["category"] == "compromiso"]

with open("compromisos.json", "w", encoding="utf-8") as f:
    json.dump(compromisos, f, ensure_ascii=False, indent=4)
