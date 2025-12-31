from __future__ import annotations
import os
import time
import logging
from pathlib import Path
from typing import Union, List
import re
from mistralai import Mistral
from dotenv import load_dotenv
from tqdm import tqdm
import datetime

# ----------------------------------------------------------------------------- #
# Logging
# ----------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- #
# Cargar API key desde .env
# ----------------------------------------------------------------------------- #
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("Falta MISTRAL_API_KEY. Define la variable de entorno o pásala al constructor.")


class MarkdownCleaner:
    """
    Limpia Markdown extraído de PDFs usando Mistral.
    Divide por bloques de filas COMPLETAS de tabla (nunca corta una fila a la mitad).
    """
    def __init__(self, api_key: str = None, model: str = "mistral-large-latest"):
        self.api_key = api_key or MISTRAL_API_KEY
        self.model = model
        self.client = Mistral(api_key=self.api_key)

    # ------------------------------------------------------------------------- #
    # Dividir texto en fragmentos
    # ------------------------------------------------------------------------- #
    def _split_text(self, text: str, max_chars: int = 40000) -> List[str]:
        lines = text.splitlines()
        chunks, current_chunk, current_len = [], [], 0
        for line in lines:
            line_len = len(line) + 1
            if current_len + line_len > max_chars:
                chunks.append("\n".join(current_chunk))
                current_chunk, current_len = [line], line_len
            else:
                current_chunk.append(line)
                current_len += line_len
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        logger.info(f"📄 Segmento dividido en {len(chunks)} fragmentos (~{max_chars} caracteres cada uno).")
        return chunks

    # ------------------------------------------------------------------------- #
    # Generación con reintentos
    # ------------------------------------------------------------------------- #
    def _generate_with_retry(self, prompt: str, max_retries: int = 5, wait_seconds: int = 20) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout_ms=600000
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = wait_seconds * (2 ** attempt)
                    logger.warning(f"⚠️ Error en intento {attempt + 1}: {e}. Reintentando en {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"❌ Falló después de {max_retries} intentos: {e}")
                    raise RuntimeError(f"Falló después de {max_retries} intentos: {e}") from e

    # ------------------------------------------------------------------------- #
    # Asegurar directorio
    # ------------------------------------------------------------------------- #
    def _ensure_dirs(self, out_dir: Union[str, Path]) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # ------------------------------------------------------------------------- #
    # Limpieza de un bloque (tu prompt original, intacto)
    # ------------------------------------------------------------------------- #
    def _clean_block(self, segment: str, block_idx: int) -> str:
        chunks = self._split_text(segment, max_chars=40000)
        cleaned_chunks = []

        with tqdm(total=len(chunks), desc=f"🔄 Bloque {block_idx}", unit="parte", colour="green", leave=False) as pbar:
            for idx, chunk in enumerate(chunks, 1):
                prompt = f"""
                                Eres un experto en limpiar y normalizar Markdown generado a partir de PDFs.
                    El texto contiene tablas con celdas partidas o mal alineadas.
                    Tareas:
                    - Une contenido de celdas divididas erróneamente.
                    - Asegura que todas las filas tengan el mismo número de columnas.
                    - Mantén encabezados y separadores originales.
                    - Elimina artefactos como “Página X de Y” o notas al pie redundantes.
                    - NO parafrasees, NO reformules ni cambies ninguna palabra del texto.
                    - Mantén exactamente la redacción original del contenido.
                    - NO alteres el orden de las frases ni su puntuación.
                    - Respeta el formato del texto fuente, solo corrige la estructura y los saltos de línea.
                    - Devuelve solo el Markdown limpio, sin explicaciones.
                    ────────────────────────────────────────
                    REGLAS ADICIONALES PARA TABLAS (NO ELIMINAR TEXTO)
                    ────────────────────────────────────────
                    - Si detectas encabezados de tabla fusionados en una sola celda, SEPÁRALOS en columnas distintas.
                    - NO permitas encabezados genéricos como Col2, Col3, Col4, etc.; reemplázalos por los encabezados reales detectados en el texto.
                    - Elimina etiquetas HTML como <br> y usa saltos de línea normales SOLO dentro de la celda correspondiente.
                    - Cada fila de tabla DEBE tener el mismo número de columnas que el encabezado.
                    - Si una celda no tiene contenido en el texto original, déjala vacía, pero CONSERVA la columna.
                    ────────────────────────────────────────
                    ESTRUCTURA DE TABLA OBLIGATORIA (CUANDO APLIQUE)
                    ────────────────────────────────────────
                    Cuando el contenido corresponda a una tabla de observaciones, la tabla DEBE normalizarse
                    con EXACTAMENTE las siguientes columnas, en este orden y con este texto exacto:
                    | N° | ITEM | FUNDAMENTOS / SUSTENTOS | OBSERVACIONES | SUBSANACION¹ | INFORMACIÓN COMPLEMENTARIA | ANÁLISIS DE LA SUBSANACIÓN | ABSUELTO (SI/NO) |
                    Y el separador Markdown debe ser EXACTAMENTE:
                    |---|---|---|---|---|---|---|---|
                    - NO agregues columnas nuevas.
                    - NO elimines columnas.
                    - NO cambies el nombre de estas columnas.
                    - Si el texto de una columna no existe, deja la celda vacía.
                    ────────────────────────────────────────
                    REGLA OBLIGATORIA SOBRE <br>
                    ────────────────────────────────────────
                    - Está PROHIBIDO devolver etiquetas HTML como <br>, <br/>, <br />.
                    - Todo <br> DEBE eliminarse.
                    - Si <br> separa elementos relacionados dentro de una misma celda,
                    reemplázalo por:
                    • un espacio simple, o
                    • un salto de línea normal (texto plano),
                    pero NUNCA por <br>.
                    - El resultado final NO DEBE contener ninguna etiqueta HTML.
                Fragmento {idx}/{len(chunks)}:
                {chunk}
                """
                try:
                    cleaned_part = self._generate_with_retry(prompt)
                    cleaned_chunks.append(cleaned_part)
                except Exception as e:
                    logger.error(f"❌ Error al limpiar fragmento {idx} del bloque {block_idx}: {e}")
                    cleaned_chunks.append(chunk)
                finally:
                    pbar.update(1)

        return "\n\n".join(cleaned_chunks)

    # ------------------------------------------------------------------------- #
    # Limpieza principal: BLOQUES POR FILAS COMPLETAS (nunca corta una fila)
    # ------------------------------------------------------------------------- #
    def clean_markdown(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path] = None,
    ) -> List[Path]:
        start_time = time.time()
        md_path = Path(input_path)
        if not md_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo Markdown: {md_path}")

        base_out = self._ensure_dirs(output_dir or md_path.parent)
        parts_dir = base_out / "cleaned_parts"
        parts_dir.mkdir(exist_ok=True)

        raw_text = md_path.read_text(encoding="utf-8", errors="ignore")
        lines = raw_text.splitlines()
        total_lines = len(lines)

        # ===================================================================== #
        # CONFIGURACIÓN: filas aproximadas por bloque
        # ===================================================================== #
        ROWS_PER_BLOCK = 3  # ← Cambia a 10, 20, 50 para producción

        # Patrón para inicio de fila: |1.| |2.| |20.| etc.
        row_start_pattern = re.compile(r'^\|\s*\d+\.\s*\|', re.IGNORECASE)

        logger.info(f"🧹 Iniciando limpieza RECURSIVA por FILAS COMPLETAS de {md_path.name}")
        logger.info(f"🟦 ~{ROWS_PER_BLOCK} filas por bloque (siempre completas) → carpeta: {parts_dir}")

        generated_files = []
        block_idx = 1
        current_block_lines = []
        row_count = 0

        i = 0
        while i < total_lines:
            current_block_lines.append(lines[i])

            # Contar inicio de fila
            if row_start_pattern.match(lines[i].strip()):
                row_count += 1

            # Cuando llegamos al límite, esperamos a que termine la última fila
            if row_count >= ROWS_PER_BLOCK:
                # Miramos hacia adelante hasta encontrar el inicio de la siguiente fila
                j = i + 1
                while j < total_lines and not row_start_pattern.match(lines[j].strip()):
                    current_block_lines.append(lines[j])
                    j += 1

                # Ahora el bloque tiene filas completas
                segment = "\n".join(current_block_lines)
                logger.info(f"📦 Procesando bloque {block_idx}: {row_count} filas completas")

                cleaned_content = self._clean_block(segment, block_idx)

                cleaned_md_path = parts_dir / f"{md_path.stem}_cleaned_{block_idx}.md"
                cleaned_md_path.write_text(cleaned_content, encoding="utf-8", errors="ignore")
                generated_files.append(cleaned_md_path)
                logger.info(f"✅ Bloque {block_idx} guardado: {cleaned_md_path}")

                # Reiniciar
                block_idx += 1
                current_block_lines = []
                row_count = 0
                i = j - 1  # retrocedemos porque el bucle incrementará i

            i += 1

        # Último bloque (resto)
        if current_block_lines:
            segment = "\n".join(current_block_lines)
            logger.info(f"📦 Procesando bloque final {block_idx}: {row_count} filas restantes")

            cleaned_content = self._clean_block(segment, block_idx)

            cleaned_md_path = parts_dir / f"{md_path.stem}_cleaned_{block_idx}.md"
            cleaned_md_path.write_text(cleaned_content, encoding="utf-8", errors="ignore")
            generated_files.append(cleaned_md_path)
            logger.info(f"✅ Bloque final {block_idx} guardado: {cleaned_md_path}")

        total_time = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=int(total_time)))

        print("\n──────────────────────────────────────────────")
        print(f"✅ LIMPIEZA COMPLETADA (filas SIEMPRE completas)")
        print(f"📁 Carpeta: {parts_dir}")
        print(f"📊 Bloques generados: {block_idx}")
        print(f"⏱️ Tiempo total: {elapsed}")
        print("──────────────────────────────────────────────")

        return generated_files

    def process_md(self, input_path: Union[str, Path], output_dir: Union[str, Path] = None) -> List[Path]:
        return self.clean_markdown(input_path=input_path, output_dir=output_dir)


# ----------------------------------------------------------------------------- #
# Ejecución directa
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python -m source.parsing.md_cleaner <ruta_raw_md> [output_dir]")
        raise SystemExit(1)
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    cleaner = MarkdownCleaner()
    files = cleaner.clean_markdown(inp, out)
    print(f"\n✅ Listo. Generados {len(files)} bloques (filas completas garantizadas) en 'cleaned_parts'")