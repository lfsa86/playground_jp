"""Module for normalizing headings in markdown documents using LLM."""

import logging
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import re

def es_tabla_atributos(linea: str) -> bool:
    clean_line = re.sub(r"<br\s*/?>", " ", linea)
    columnas = [c.strip() for c in clean_line.strip().split("|") if c.strip()]
    return all(":" in c for c in columnas)

def demote_table_headers(md_text: str) -> str:
    """
    Convierte encabezados Markdown que comienzan con 'Tabla' en texto plano destacado con negrita,
    y agrega saltos de línea arriba y abajo para claridad visual.
    """
    lines = md_text.splitlines()
    new_lines = []
    pattern = re.compile(r"^(#{1,6})\s+(Tabla(?:\s+(Original|Rectificada))?\s[\d\.\s\(\)A-Za-zÁÉÍÓÚÑñ:]+)", re.IGNORECASE)

    for i, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            titulo = match.group(2).strip()
            new_lines.append("")  # línea vacía arriba
            new_lines.append(f"**{titulo}**")  # título en negrita
            new_lines.append("")  # línea vacía abajo
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

def protect_sensitive_blocks(text: str) -> str:
    """
    Añade marcas de protección alrededor de tablas, listas enumeradas o texto con estilo técnico
    para que no sea modificado por el modelo.
    """
    lines = text.splitlines()
    protected = []
    in_table = False
    for line in lines:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            if not in_table:
                protected.append("<PROTECTED_BLOCK>")
                in_table = True
            protected.append(line)
        elif in_table and not line.strip().startswith("|"):
            protected.append("</PROTECTED_BLOCK>")
            protected.append(line)
            in_table = False
        else:
            protected.append(line)
    if in_table:
        protected.append("</PROTECTED_BLOCK>")
    return "\n".join(protected)

def restore_protected_blocks(text: str, original: str) -> str:
    """
    Reemplaza los bloques marcados como <PROTECTED_BLOCK>...</PROTECTED_BLOCK>
    con el contenido original correspondiente (línea por línea).
    """
    original_lines = original.splitlines()
    result_lines = []
    i = 0
    while i < len(original_lines):
        if "<PROTECTED_BLOCK>" in original_lines[i]:
            protected_block = []
            i += 1
            while i < len(original_lines) and "</PROTECTED_BLOCK>" not in original_lines[i]:
                protected_block.append(original_lines[i])
                i += 1
            result_lines.extend(protected_block)
            i += 1  # Saltar </PROTECTED_BLOCK>
        else:
            result_lines.append(original_lines[i])
            i += 1
    return "\n".join(result_lines)

def estructurar_bloques_generales_con_descripcion(md_text: str, claves_titulo=None) -> str:
    """
    Detecta bloques de atributos con encabezado y conserva su descripción posterior como parte del componente.
    Además, corrige encabezados como '## DESCRIPCIÓN' transformándolos en atributos tipo '- **Descripción:**'.
    """
    import re

    if claves_titulo is None:
        claves_titulo = [
            "Nombre", "Nombre del componente", "Acción", "Acciones",
            "Insumo básico", "Recurso natural renovable", "Componente ambiental"
        ]

    lines = md_text.splitlines()

    # Preprocesar líneas tipo tabla con atributos
    nueva_lista = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and ":" in stripped and not stripped.endswith("|:--"):
            stripped = re.sub(r"<br\s*/?>", " ", stripped)  # reemplaza <br> o <br/> por espacio
            columnas = [c.strip(" |") for c in stripped.split("|") if ":" in c]
            atributos = [c.split(":", 1) for c in columnas]
            atributos = [(k.strip(), v.strip().rstrip(".")) for k, v in atributos]

            clave_kv = next(((k, v) for k, v in atributos if k.rstrip(":") in claves_titulo), None)
            clave_titulo = clave_kv[0].rstrip(":") if clave_kv else None

            if clave_kv:
                k, v = clave_kv
                nueva_lista.append(f"### {k.rstrip(':')}: {v}")
                for k2, v2 in atributos:
                    if k2 != k:
                        nueva_lista.append(f"- **{k2.rstrip(':')}:** {v2}")
                nueva_lista.append("")
            else:
                for k, v in atributos:
                    nueva_lista.append(f"- **{k.rstrip(':')}:** {v}")
                nueva_lista.append("")
            
            if clave_titulo:
                for k, v in atributos:
                    if k == clave_titulo:
                        nueva_lista.append(f"### {k}: {v}")
                    else:
                        nueva_lista.append(f"- **{k}:** {v}")
                nueva_lista.append("")
            else:
                for k, v in atributos:
                    nueva_lista.append(f"- **{k}:** {v}")
                nueva_lista.append("")
        else:
            nueva_lista.append(line)
    lines = nueva_lista

    output = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Detecta grupo de atributos en líneas independientes
        if line.lower().startswith("nombre:"):
                line_content = line.split(":", 1)[1].strip()
                caracter = fase = ""

                # Extraer posibles campos incrustados
                caracter_match = re.search(r"Carácter:\s*([^\.]+)", line_content, re.IGNORECASE)
                fase_match = re.search(r"Fase[s]?:\s*([^\.]+)", line_content, re.IGNORECASE)

                if caracter_match:
                    caracter = caracter_match.group(1).strip()
                    line_content = re.sub(r"Carácter:\s*[^\.]+\.?", "", line_content, flags=re.IGNORECASE)

                if fase_match:
                    fase = fase_match.group(1).strip()
                    line_content = re.sub(r"Fase[s]?:\s*[^\.]+\.?", "", line_content, flags=re.IGNORECASE)

                # Lo que queda puede ser parte de la descripción embebida
                nombre = re.sub(r"<br\s*/?>", " ", line_content)  # eliminar <br> o <br/>
                nombre = re.sub(r"\s+", " ", nombre).strip(" .")  # normalizar espacios
                i += 1

                # Buscar si las siguientes líneas son Carácter y Fase
                for _ in range(3):  # máximo 3 líneas
                    if i < len(lines):
                        next_line = lines[i].strip()
                        if next_line.lower().startswith("carácter:") and not caracter:
                            caracter = next_line.split(":", 1)[1].strip().rstrip(".")
                            i += 1
                        elif next_line.lower().startswith("fase") or next_line.lower().startswith("fases"):
                            if not fase:
                                fase = next_line.split(":", 1)[1].strip().rstrip(".")
                            i += 1
                        else:
                            break

                output.append(f"### Nombre: {nombre}")
                if caracter:
                    output.append(f"- **Carácter:** {caracter}")
                if fase:
                    output.append(f"- **Fase:** {fase}")
                output.append("")

                # Buscar descripción como sigue
                descripcion_lines = []
                while i < len(lines):
                    siguiente = lines[i].strip()
                    if siguiente.startswith("##") or siguiente.startswith("|") or any(siguiente.startswith(h) for h in ["###", "---"]):
                        break
                    if siguiente:
                        descripcion_lines.append(siguiente)
                    i += 1

                if descripcion_lines:
                    descripcion_text = " ".join(descripcion_lines).strip()

                    # Extraer atributos embebidos en la descripción
                    caracter_match = re.search(r"Carácter:\s*([^\.]+)", descripcion_text, re.IGNORECASE)
                    fase_match = re.search(r"Fase[s]?:\s*([^\.]+)", descripcion_text, re.IGNORECASE)

                    if caracter_match:
                        caracter = caracter_match.group(1).strip()
                        output.append(f"- **Carácter:** {caracter}")
                        descripcion_text = re.sub(r"Carácter:\s*[^\.]+\.?", "", descripcion_text, flags=re.IGNORECASE)

                    if fase_match:
                        fase = fase_match.group(1).strip()
                        output.append(f"- **Fase:** {fase}")
                        descripcion_text = re.sub(r"Fase[s]?:\s*[^\.]+\.?", "", descripcion_text, flags=re.IGNORECASE)

                    # Limpiar doble espacio y puntos sobrantes
                    descripcion_text = re.sub(r"\s{2,}", " ", descripcion_text).strip(" .")

                    if descripcion_text:
                        output.append(f"- **Descripción:** {descripcion_text}")
                output.append("")
                continue

        # Si no es bloque de atributos, agregar línea tal cual
        output.append(lines[i])
        i += 1

    # Segunda pasada para interceptar encabezados de "Descripción"
    final_lines = []
    i = 0
    while i < len(output):
        line = output[i]
        if re.match(r"^#{2,6}\s*descripción\.?\s*$", line.strip(), re.IGNORECASE):
            descripcion_lines = []
            i += 1
            while i < len(output):
                siguiente = output[i]
                if re.match(r"^#{1,6}\s*\w+", siguiente) or siguiente.strip().startswith("### "):
                    break
                descripcion_lines.append(siguiente.strip())
                i += 1
            if descripcion_lines:
                descripcion_text = " ".join(descripcion_lines).strip()
                final_lines.append(f"- **Descripción:** {descripcion_text}")
                final_lines.append("")
        else:
            final_lines.append(line)
            i += 1

    return "\n".join(final_lines)

def formatear_listado_en_descripcion(md_text: str) -> str:
    """
    Detecta listados tipo a., b., c. dentro de descripciones y los convierte en listas ordenadas Markdown.
    """
    def reemplazar(match):
        letra = match.group(1)
        contenido = match.group(2)
        return f"\n{letra}. {contenido.strip()}"

    # Aplicar solo dentro de bloques de descripción
    pattern = r"\b([a-k])\.\s*(.*?)\s*(?=\b[a-k]\.|\Z)"
    return re.sub(pattern, reemplazar, md_text, flags=re.DOTALL)

def dividir_por_bloques_tematicos(md_text: str) -> list[str]:
    """
    Divide el markdown en bloques temáticos comenzando por encabezados '##'.
    """
    bloques = re.split(r'(?=^##\s+)', md_text, flags=re.MULTILINE)
    return [b.strip() for b in bloques if b.strip()]

def limpiar_notacion_latex(md_text: str) -> str:
    r"""
    Limpia expresiones LaTeX comunes y fragmentadas y las convierte en texto plano legible.

    Reemplazos:
    - $1^{\\circ}$ → 1º
    - $1^{a}$ → 1ª
    - $\mathrm{N}^{\\circ}$ → N°
    - kcal / kg en notación LaTeX → kcal/kg
    - $\mathrm{m}^{3}$ → m³
    - N ${ }^{\\circ}$ → N°
    - Eliminación de ${ }$ u otros residuos
    """
    # 1. Ordinales como 1º, 1ª
    md_text = re.sub(r"\{\s*\}\s*\^\s*\{\\circ\}", "°", md_text)  # casos como: { }^{\circ}
    md_text = re.sub(r"\$\s*(\d+)\s*\^\s*\{\\circ\}\s*\$", r"\1º", md_text)
    md_text = re.sub(r"\$\s*(\d+)\s*\^\s*\{a\}\s*\$", r"\1ª", md_text)

    # 2. Variantes de 'N°'
    md_text = re.sub(r"N\s*\$\{\s*\}\^\{\\circ\}\$", "N°", md_text)  # N ${ }^{\circ}$
    md_text = re.sub(r"N\s*\$\s*\^\{\\circ\}\s*\$", "N°", md_text)    # N $^{\circ}$
    md_text = re.sub(r"\$\\mathrm\{N\}\^\{\\circ\}\$", "N°", md_text)  # $\mathrm{N}^{\circ}$
    md_text = re.sub(r"\\mathrm\{N\}\^\{\\circ\}", "N°", md_text)      # \mathrm{N}^{\circ}
    md_text = re.sub(r"N\s*\^\{\\circ\}", "N°", md_text)               # N^{\circ}

    # 3. Unidades tipo kcal/kg
    md_text = re.sub(r"\$\\mathrm\{([^}]+)\}\s*/\s*\\mathrm\{([^}]+)\}\$", r"\1/\2", md_text)

    # 4. Potencias como m³ y km²
    md_text = re.sub(r"\$\\mathrm\{([^}]+)\}\^\{3\}\$", r"\1³", md_text)
    md_text = re.sub(r"\$\\mathrm\{([^}]+)\}\^\{2\}\$", r"\1²", md_text)

    # 5. Eliminar \mathrm{} simple
    md_text = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", md_text)

    # 6. Eliminar residuos tipo ${ }$
    md_text = re.sub(r"\$\{\s*\}\$", "", md_text)

    # 7. Eliminar delimitadores $...$ residuales
    md_text = re.sub(r"\$([^\$]+)\$", r"\1", md_text)

    return md_text

# Set multiprocessing start method to "spawn" before any other imports
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError as e:
        logging.error(f"Failed to set multiprocessing start method: {e}")

# Enable gRPC fork support
os.environ["GRPC_FORK_SUPPORT_ENABLED"] = "1"

# Configure logging early
try:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
except Exception as e:
    print(f"Failed to configure logging: {e}")
    # Fallback logger
    logger = logging.getLogger(__name__)

try:
    from langchain.chat_models import init_chat_model
    from langchain.prompts import ChatPromptTemplate
    from tenacity import retry, stop_after_attempt, wait_exponential
    from tqdm import tqdm
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

# System prompt for heading normalization
system_prompt = """
Eres un asistente experto en reformatear documentos legales y técnicos, con un enfoque en la creación de estructuras jerárquicas claras y concisas. Tu tarea principal es transformar un texto plano en un documento estructurado utilizando Markdown, específicamente enfatizando la organización multinivel a partir del nivel de encabezado 2 (##).

**Instrucciones específicas:**

1.  **Jerarquía Markdown:**  Organiza el texto utilizando encabezados Markdown (##, ###, ####, etc.) para representar la jerarquía lógica del documento. Asegúrate de que cada sección tenga un encabezado apropiado y que las subsecciones estén anidadas correctamente. Comienza la jerarquía en el nivel de encabezado 2.
2.  **Formato de listas y tablas:** Cuando corresponda, utiliza listas (ordenadas y no ordenadas) y tablas para presentar información de manera clara y organizada. Formatea las tablas con encabezados y alineación apropiados.
3.  **Respetar el contenido original:** No modifiques el contenido factual del documento. Tu objetivo es mejorar la estructura y la presentación, no alterar la información proporcionada.
4.  **Consistencia:** Aplica un estilo consistente en todo el documento, incluyendo el uso de mayúsculas, minúsculas, puntuación y formato de listas y tablas.
5. **énfasis a listas:** Si hay instrucciones, regulaciones o ítems enumerados, asegúrate de representarlos en un formato de lista con viñetas o numerados para que sean fáciles de identificar y entender.
6. **énfasis a tablas:** Las tablas siempre deben incluir títulos claros y descriptivos.
7. En tu respuesta solamente da el contenido del MD sin triple backtick (```)
8. MANTEN EL CONTENIDO ORIGINAL SIN MODIFICARLO.
9. EVITA REDUNDANCIAS AL MOMENTO DE TRANSCRIBIR.
10. NO resumas, interpretes o completes el contenido. No omitas ninguna línea, celda o tabla, incluso si parecen repetidas, vacías o poco informativas.

**Actúa como un asistente legal/técnico profesional.** Tu objetivo es producir documentos bien estructurados, fáciles de leer y que mantengan la integridad del contenido original.
"""

NORMALIZE_TEMPLATE = """
{system_prompt}

Contenido a normalizar:

{content}
"""

system_prompt_tematica = """
Eres un asistente experto en organización documental en formato Markdown. Tu tarea es segmentar y jerarquizar temáticamente el siguiente contenido **sin modificar el texto original**.

**Reglas estrictas:**
1. NO edites, resumas, interpretes ni reescribas el contenido.
2. NO cambies el orden ni la redacción original del texto.
3. NO completes oraciones faltantes ni corrijas errores.
4. NO introduzcas contenido nuevo.

**Lo que sí debes hacer:**

- Identifica bloques temáticos principales como: `VISTOS`, `MARCO LEGAL`, `CONSIDERANDO`, `ANTECEDENTES`, `DESCRIPCIÓN`, `MEDIDAS`, `COMPONENTES AMBIENTALES`, entre otros.
- Inicia cada uno de estos bloques con un encabezado de nivel `##` con su nombre literal.

- Dentro de los bloques donde se describen componentes técnicos, partes del proyecto u otros elementos estructurados:
  - Si el bloque contiene un campo **"Nombre"**, conviértelo en un encabezado de nivel `##`.
  - Si existen atributos como **“Fase”, “Carácter”, “Ubicación”**, **colócalos como lista debajo del Nombre**.
  - Si el contenido incluye una sección llamada “Descripción”, **no la trates como un encabezado**. En su lugar, incorpora su contenido como un campo adicional debajo del resto de atributos, con el nombre `**Descripción:**`.
  - Asegúrate de que **la sección nunca empiece con el encabezado "Descripción"**. Comienza por el atributo `Nombre`, si existe.

- Mantén cualquier tabla, párrafo, subtítulo o contenido complementario en el orden original en el que aparece.

- No modifiques el contenido ni sustituyas texto, incluso si tiene errores de redacción o formato.

Contenido a segmentar:
{content}
"""

def segmentar_tematicamente_con_llm(md_text: str) -> str:
    bloques = dividir_por_bloques_tematicos(md_text)
    llm = init_chat_model(
        model="gemini-2.0-flash",
        model_provider="google_genai",
        temperature=0,
    )
    resultados = []

    for idx, bloque in enumerate(tqdm(bloques, desc="Segmentando bloques temáticos")):
        prompt = system_prompt_tematica.format(content=bloque)
        try:
            response = llm.invoke(prompt)
            if hasattr(response, "content") and response.content:
                resultados.append(response.content.strip())
            else:
                resultados.append(bloque)  # fallback
        except Exception as e:
            logger.error(f"Error segmentando bloque {idx}: {e}")
            resultados.append(bloque)  # fallback

    return "\n\n".join(resultados)


def get_optimal_thread_count():
    """Calculate optimal thread count based on CPU cores."""
    try:
        return min(32, (multiprocessing.cpu_count() * 2))
    except Exception as e:
        logger.warning(f"Error determining CPU count: {e}. Using default value of 4.")
        return 4


class ThreadSafeLLM:
    """Thread-safe LLM handler."""

    def __init__(self):
        self._lock = threading.Lock()
        self._llm_instances = {}

    def get_llm(self):
        """Get or create LLM instance for current thread."""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id not in self._llm_instances:
                try:
                    self._llm_instances[thread_id] = init_chat_model(
                        model="gemini-2.0-flash",
                        model_provider="google_genai",
                        temperature=0,
                    )
                except Exception as e:
                    logger.error(
                        f"Error initializing LLM for thread {thread_id}: {str(e)}"
                    )
                    raise
            return self._llm_instances[thread_id]

    def cleanup(self):
        """Cleanup LLM instances."""
        try:
            with self._lock:
                self._llm_instances.clear()
        except Exception as e:
            logger.error(f"Error during LLM cleanup: {e}")


class HeadingParser:
    """Class for parsing and normalizing headings in markdown documents using LLM."""

    def __init__(self):
        """Initialize the parser with LLM model and prompt template."""
        try:
            self.prompt_template = ChatPromptTemplate.from_template(
                template=NORMALIZE_TEMPLATE
            )
            self.llm_handler = ThreadSafeLLM()
            self.results = {}
            self._lock = threading.Lock()
            self.page_separator = "-----"
        except Exception as e:
            logger.error(f"Error initializing HeadingParser: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def process_page(self, page_content: str, page_index: int) -> str:
        """
        Process a single page of markdown content and normalize headings.
        Adds protection around sensitive blocks like tables before passing to LLM.
        """
        try:
            if not page_content.strip():
                return page_content

            # 👉 Paso 1: proteger contenido sensible
            protected_content = protect_sensitive_blocks(page_content)

            # 👉 Paso 2: preparar el prompt
            prompt = self.prompt_template.format_messages(
                system_prompt=system_prompt,
                content=protected_content,
            )

            # 👉 Paso 3: ejecutar el modelo
            llm = self.llm_handler.get_llm()
            response = llm.invoke(prompt)

            # 👉 Paso 4: restaurar el contenido sensible original
            if hasattr(response, "content") and response.content:
                normalized_content = restore_protected_blocks(response.content, protected_content)

                # ✅ Transformar tablas tipo ficha en encabezado y atributos
                output_lines = []
                for line in normalized_content.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("|") and es_tabla_atributos(stripped):
                        clean_line = re.sub(r"<br\s*/?>", " ", stripped)
                        columnas = [c.strip() for c in clean_line.strip().split("|") if c.strip()]
                        for col in columnas:
                            if ":" in col:
                                clave, valor = col.split(":", 1)
                                clave = clave.strip().rstrip(".")
                                valor = valor.strip().rstrip(".")
                                if clave.lower() == "nombre":
                                    output_lines.append(f"### Nombre: {valor}")
                                else:
                                    output_lines.append(f"- **{clave}:** {valor}")
                        continue  # omitir línea original de la tabla
                    output_lines.append(line)

                normalized_content = "\n".join(output_lines)

            else:
                logger.warning(f"Empty response at page {page_index}")
                normalized_content = page_content

        except Exception as e:
            logger.error(f"Error processing page {page_index}: {str(e)}")
            return page_content  # Return original content on error
        
        return normalized_content

    def normalize_headings(
            self, file_name: str, md_content: str, output_dir="data/processed/"
        ) -> str:
        try:
            pages = md_content.split(self.page_separator)
            logger.info(f"Starting heading normalization for {len(pages)} pages")

            try:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, file_name)
                os.makedirs(output_path, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create output directory: {e}")
                output_path = "."

            self.results = {}

            with ThreadPoolExecutor(max_workers=get_optimal_thread_count()) as executor:
                futures = {
                    executor.submit(self.process_page, page_content=page, page_index=idx): idx
                    for idx, page in enumerate(pages)
                }

                with tqdm(total=len(futures), desc="Processing pages") as pbar:
                    for future in as_completed(futures):
                        page_index = futures[future]
                        try:
                            result = future.result(timeout=120)
                            self.results[page_index] = result
                        except TimeoutError:
                            logger.error(f"Timeout processing page {page_index}")
                            self.results[page_index] = pages[page_index]
                        except Exception as e:
                            logger.error(f"Error processing page {page_index}: {str(e)}")
                            self.results[page_index] = pages[page_index]
                        finally:
                            pbar.update(1)

            # ✅ Unir resultados en orden
            normalized_pages = [self.results.get(i, pages[i]) for i in range(len(pages))]
            normalized_content = self.page_separator.join(normalized_pages)

            # 🔁 Procesamiento adicional (tabla ficha)
            lines = normalized_content.splitlines()
            output = []

            for line in lines:
                stripped_line = line.strip()
                if not stripped_line.startswith("|") or not es_tabla_atributos(stripped_line):
                    output.append(line)
                    continue
                clean_line = re.sub(r"<br\s*/?>", " ", line)
                columnas = [c.strip() for c in clean_line.strip().split("|") if c.strip()]
                for col in columnas:
                    if ":" in col:
                        clave, valor = col.split(":", 1)
                        clave = clave.strip().rstrip(".")
                        valor = valor.strip().rstrip(".")
                        if clave.lower() == "nombre":
                            output.append(f"### Nombre: {valor}")
                        else:
                            output.append(f"- **{clave}:** {valor}")

            normalized_content = "\n".join(output)

            # 🧠 Estructuración y limpieza
            normalized_content = estructurar_bloques_generales_con_descripcion(normalized_content)

            if "4.3" in normalized_content or "componentes ambientales" in normalized_content.lower():
                logger.info("Aplicando segmentación temática con LLM...")
                normalized_content = segmentar_tematicamente_con_llm(normalized_content)

            normalized_content = limpiar_notacion_latex(normalized_content)
            normalized_content = re.sub(r"<br\s*/?>", "\n", normalized_content)
            normalized_content = formatear_listado_en_descripcion(normalized_content)

            try:
                output_file = os.path.join(output_path, f"{file_name}.md")
                with open(output_file, "w", encoding="utf-8") as file:
                    file.write(normalized_content)
                logger.info(f"Successfully wrote normalized content to {output_file}")
            except IOError as e:
                logger.error(f"Failed to write output file: {e}")

            logger.info("Completed heading normalization")
            return normalized_content

        except Exception as e:
            logger.exception("Error during normalization process")
            return md_content

        finally:
            try:
                self.llm_handler.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")