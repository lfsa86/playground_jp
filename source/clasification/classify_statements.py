"""Module for extracting environmental components from commitments using LLM."""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Dict, List, Tuple, Optional

# ---- Early env & logging -----------------------------------------------------
os.environ["GRPC_FORK_SUPPORT_ENABLED"] = "1"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

# ---- Deps tardías ------------------------------------------------------------
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from ..models import ElementIterator, ElementParser
from ..schemas import Statement

# ------------------------------------------------------------------------------
# Config LLM / Prompts
# ------------------------------------------------------------------------------
load_dotenv()

SYSTEM_PROMPT = """Eres un asistente especializado en analizar documentos regulatorios ambientales peruanos. Tu tarea es extraer y categorizar información en las siguientes secciones:

1. Generalidades: Extrae información contextual sobre el proyecto incluyendo ubicación, propósito, escala, cronograma y antecedentes que ayuden a comprender el alcance general del proyecto.

2. Componentes: Identifica todos los elementos físicos, infraestructura o sistemas que serán construidos, instalados o modificados como parte del proyecto. Incluye especificaciones técnicas cuando estén disponibles.

3. Riesgos e Impactos: Identifica los potenciales riesgos e impactos ambientales descritos en el documento, en todas las fases de desarrollo del proyecto.

5. Compromisos: Extrae todos los compromisos realizados por el titular del proyecto, incluyendo medidas voluntarias que se vuelven obligatorias al ser incluidas en el expediente analizado. 

Presenta la información en un formato estructurado con encabezados claros para cada categoría. Si una categoría no tiene información relevante en el documento, indícalo explícitamente."""

TEMPLATE = """
{system_prompt}

Clasifica el siguiente enunciado dentro de obligación, compromiso, acuerdo u 'otro':
```{statement}```

NO ALUCINES
"""

# ------------------------------------------------------------------------------
# Forzar agrupación del bloque bajo "# ÍNDICE" hasta la siguiente línea con "#"
# ------------------------------------------------------------------------------

def force_group_index_block(md: str) -> str:
    if not md:
        return md

    lines = md.splitlines()
    n = len(lines)

    idx_start = None
    pat_start = re.compile(r'^\s*#\s*Í?NDICE\s*$', flags=re.IGNORECASE)
    for i, ln in enumerate(lines):
        if pat_start.match(ln):
            idx_start = i
            break
    if idx_start is None:
        return md

    idx_end = n
    pat_head = re.compile(r'^\s*#')
    for j in range(idx_start + 1, n):
        if pat_head.match(lines[j]):
            idx_end = j
            break

    toc_body_lines = lines[idx_start + 1: idx_end]
    if not toc_body_lines:
        return md

    toc_body_text = "\n".join(toc_body_lines).strip("\n")

    if "```" not in toc_body_text:
        replacement_block = [lines[idx_start], "", "```", toc_body_text, "```", ""]
    else:
        escaped = [re.sub(r'^(\s*)#', r'\1\#', raw) for raw in toc_body_lines]
        replacement_block = [lines[idx_start], ""] + escaped + [""]

    new_lines = lines[:idx_start] + replacement_block + lines[idx_end:]
    return "\n".join(new_lines)

# ------------------------------------------------------------------------------
# Utils de heading: construir ruta empezando en numeral más cercano
# ------------------------------------------------------------------------------

# Numerales tipo 3., 3.3., 3.3.5.2 (acepta con/sin espacio y cualquier char después)
_NUMBERED_RE = re.compile(r'^\s*\d+(?:\.\d+)*\.?\s*')

# Títulos macro que NO deben iniciar la ruta
_MACRO_RE = re.compile(r'^\s*(CAP[IÍ]TULO\b|Í?NDICE\b)', re.IGNORECASE)

# Bullets / guiones comunes a limpiar al inicio
_BULLET_PREFIX_RE = re.compile(r'^\s*([•·\-–—]+|\*\s+)+')

# --- NUEVO: detectar subtítulos con letra (A., B., a., b., etc.)
LETTER_RE = re.compile(r'^\s*[\(\[]?([A-Za-z])[\)\]]?[\.\-]?\s+')

def _clean_title(title: str) -> str:
    if not isinstance(title, str):
        return ""
    t = title.strip()

    # Quitar negritas/itálicas markdown envolventes
    if (t.startswith("**") and t.endswith("**")) or (t.startswith("__") and t.endswith("__")):
        t = t[2:-2].strip()
    if (t.startswith("*") and t.endswith("*")) or (t.startswith("_") and t.endswith("_")):
        t = t[1:-1].strip()

    # Quitar bullets/guiones iniciales
    t = _BULLET_PREFIX_RE.sub("", t)

    # Normalizar espacios y signos
    t = re.sub(r'\s+', ' ', t).strip(" \t:;.-·•\u200b")
    return t

def _is_numbered(title: str) -> bool:
    """
    Devuelve True solo si el título empieza con un numeral jerárquico válido tipo:
      1. , 1.2 , 3.4.1.3 , etc.
    Excluye años (>=1900), códigos numéricos largos (más de 3 dígitos sin punto)
    y cadenas que no contengan al menos un punto.
    """
    if not isinstance(title, str):
        return False
    t = title.strip()

    # Rechazar años o códigos (≥4 dígitos seguidos sin punto)
    if re.match(r"^\s*\d{4,}\b(?!\.)", t):
        return False

    # Detectar numerales jerárquicos válidos (al menos un punto)
    if re.match(r"^\s*\d{1,2}(\.\d{1,3})+\.?\s*", t):
        return True

    # Casos simples: "1." o "2." o "3"
    if re.match(r"^\s*\d{1,2}\.\s*", t):
        return True

    return False

def _is_macro(title: str) -> bool:
    return bool(_MACRO_RE.match(title or ""))

def _is_lettered(title: str) -> bool:
    return bool(LETTER_RE.match(title or ""))

def _get_parent(tree, node):
    try:
        return tree.get_node(node.parent_id) if getattr(node, "parent_id", None) else None
    except Exception:
        return None

# ---------- Índice de hijos/hermanos (cacheado en tree) -----------------------

def _ensure_children_index(tree) -> None:
    """Construye un índice {parent_id: [child_nodes_en_orden]} y {parent_id: [child_ids]} en cache."""
    if hasattr(tree, "__children_index") and hasattr(tree, "__children_index_ids"):
        return
    children_index: Dict[Optional[str], List[object]] = {}
    children_index_ids: Dict[Optional[str], List[str]] = {}

    try:
        iterator = ElementIterator(tree)
        # Recorremos todos los nodos respetando el orden DFS del iterador
        for node_id, node, depth in iterator.iterate_depth_first(skip_root=False):
            pid = getattr(node, "parent_id", None)
            children_index.setdefault(pid, []).append(node)
            try:
                nid = getattr(node, "id", node_id)
            except Exception:
                nid = node_id
            children_index_ids.setdefault(pid, []).append(nid)
    except Exception as e:
        logger.warning(f"Could not build children index from iterator: {e}")
        # fallback vacío

    setattr(tree, "__children_index", children_index)
    setattr(tree, "__children_index_ids", children_index_ids)

def _get_siblings(tree, node) -> List[object]:
    """Devuelve la lista de hijos del padre (hermanos, incluyendo el propio) en orden."""
    _ensure_children_index(tree)
    pid = getattr(node, "parent_id", None)
    index = getattr(tree, "__children_index", {})
    return index.get(pid, [])

def _get_sibling_index(tree, node) -> int:
    """Índice del nodo dentro de la lista de hermanos."""
    _ensure_children_index(tree)
    pid = getattr(node, "parent_id", None)
    index_ids = getattr(tree, "__children_index_ids", {})
    siblings_ids = index_ids.get(pid, [])
    nid = getattr(node, "id", None)
    try:
        return siblings_ids.index(nid)
    except Exception:
        return -1

def _find_prev_numbered_among_siblings(tree, node) -> Optional[object]:
    """Busca el hermano numerado más cercano hacia atrás (misma profundidad)."""
    siblings = _get_siblings(tree, node)
    pos = _get_sibling_index(tree, node)
    if pos <= 0:
        return None
    for i in range(pos - 1, -1, -1):
        cand = siblings[i]
        t = _clean_title(getattr(cand, "title", "") or "")
        if t and not _is_macro(t) and _is_numbered(t):
            return cand
    return None

def _find_closest_numbered_anchor(tree, node) -> Optional[object]:
    """
    Devuelve el mejor 'ancla numerada':
      1) Ancestro numerado (el más cercano).
      2) Si no hay, hermano anterior numerado (en el mismo nivel).
      3) Si no hay, subir un nivel y repetir (hermano anterior del padre, etc.).
    """
    # 1) Ancestro numerado
    cur = node
    safety = 0
    while cur is not None and safety < 200:
        t = _clean_title(getattr(cur, "title", "") or "")
        if t and not _is_macro(t) and _is_numbered(t):
            return cur
        cur = _get_parent(tree, cur)
        safety += 1

    # 2-3) Hermanos anteriores (en este nivel) y al subir niveles
    cur_node = node
    safety = 0
    while cur_node is not None and safety < 200:
        prev_num = _find_prev_numbered_among_siblings(tree, cur_node)
        if prev_num is not None:
            return prev_num
        # subir un nivel y buscar el hermano anterior del padre
        cur_node = _get_parent(tree, cur_node)
        safety += 1

    return None  # si no se encuentra nada

def _linear_dfs_order(tree):
    """Devuelve lista [(node_id, node)] en orden DFS (incluye root)."""
    try:
        iterator = ElementIterator(tree)
        return [(nid, n) for nid, n, _d in iterator.iterate_depth_first(skip_root=False)]
    except Exception:
        # Fallback sin garantía de orden
        return list(getattr(tree, "nodes", {}).items())

def _find_prev_numbered_in_dfs(tree, node) -> Optional[object]:
    """
    Busca hacia atrás en el recorrido DFS el último nodo numerado válido.
    Útil cuando un subtítulo 'B.' no tiene ancestro/hermano numerado detectable.
    """
    ordered = _linear_dfs_order(tree)
    idx = -1
    nid = getattr(node, "id", None)
    for i, (cand_id, _cand_node) in enumerate(ordered):
        if cand_id == nid:
            idx = i
            break
    if idx <= 0:
        return None

    for j in range(idx - 1, -1, -1):
        _idj, cand = ordered[j]
        t = _clean_title(getattr(cand, "title", "") or "")
        if t and not _is_macro(t) and _is_numbered(t):
            return cand
    return None

def _join_anchor_with_leaf(anchor_title: str, leaf_title: str) -> str:
    left = _clean_title(anchor_title or "")
    right = _clean_title(leaf_title or "")
    if left and right:
        return f"{left} > {right}"
    return left or right or "Sin numeral"

def build_numeric_first_heading_path(tree, node) -> str:
    """
    Construye un heading_path que SIEMPRE inicia en el numeral más cercano:
      - Si hay ancestro numerado, se usa como inicio y se continúa hasta el nodo actual,
        preservando intermedios (saltando macros).
      - Si NO hay ancestro numerado, se busca el hermano numerado más cercano (o del padre al subir).
        En ese caso la ruta será: "<ancla_numerada> > <título_actual>".
      - Si aún no hay ancla y el título es tipo "B. ...", se busca hacia atrás en DFS
        el último encabezado numerado válido y se usa como ancla.
    """
    # Cadena node->root
    chain: List[Tuple[object, str]] = []
    cur = node
    safety = 0
    while cur is not None and safety < 200:
        title = _clean_title(getattr(cur, "title", "") or "")
        chain.append((cur, title))
        cur = _get_parent(tree, cur)
        safety += 1

    if not chain:
        return ""

    # 1) Intentar con ancestro numerado
    idx_num = None
    for i, (_n, t) in enumerate(chain):
        if t and not _is_macro(t) and _is_numbered(t):
            idx_num = i
            break

    if idx_num is not None:
        # Construir ruta desde ese numeral hasta el nodo (root->node del tramo)
        segment = list(reversed(chain[: idx_num + 1]))  # [numeral, ..., nodo]
        parts: List[str] = []
        for _n, t in segment:
            if not t or _is_macro(t):
                continue
            parts.append(t)

        # Asegurar que la primera parte sea numerada (sanity)
        if parts and not _is_numbered(parts[0]):
            for k, pt in enumerate(parts):
                if _is_numbered(pt):
                    parts = parts[k:]
                    break

        if not parts:
            # Fallback suave al título de la hoja si no macro
            leaf_title = chain[0][1]
            return leaf_title if (leaf_title and not _is_macro(leaf_title)) else "Sin numeral"

        return " > ".join(parts).strip(" >")

    # 2) No hay ancestro numerado: buscar ancla numerada más cercana (hermano anterior, etc.)
    _ensure_children_index(tree)
    anchor = _find_closest_numbered_anchor(tree, node)
    leaf_title = chain[0][1]  # título del nodo actual (limpio)

    if anchor is not None:
        anchor_title = _clean_title(getattr(anchor, "title", "") or "")
        # Filtrar macro y empty
        left = anchor_title if (anchor_title and not _is_macro(anchor_title)) else ""
        right = leaf_title if (leaf_title and not _is_macro(leaf_title)) else ""
        if left and right:
            return f"{left} > {right}"
        if left:
            return left
        if right:
            return right
        return "Sin numeral"

    # 2-bis) Si el nodo es tipo "B. ..." intenta el ancla por DFS hacia atrás
    if _is_lettered(leaf_title):
        cand = _find_prev_numbered_in_dfs(tree, node)
        if cand is not None:
            return _join_anchor_with_leaf(getattr(cand, "title", ""), leaf_title)

    # 3) Último fallback
    return leaf_title if (leaf_title and not _is_macro(leaf_title)) else "Sin numeral"

# ------------------------------------------------------------------------------
# Utilidades de fusión (vista)
# ------------------------------------------------------------------------------

def merge_subnodes_view(nodes: List[Tuple[str, object, int]], tree) -> List[Tuple[str, object, int]]:
    def _clone_with_content(node, content):
        clone = type(node)(
            title=node.title, level=node.level, content=content, parent_id=node.parent_id
        )
        try:
            clone.id = node.id
        except Exception:
            pass
        return clone

    merged: List[Tuple[str, object, int]] = []
    i = 0
    n = len(nodes)
    while i < n:
        node_id, node, depth = nodes[i]
        if depth == 3:
            content = (node.content or "").strip()
            j = i + 1
            while j < n:
                nid_j, node_j, depth_j = nodes[j]
                if depth_j == 4 and getattr(node_j, "parent_id", None) == getattr(node, "id", None):
                    if node_j.content:
                        content = (content + "\n\n" + node_j.content).strip() if content else node_j.content.strip()
                    j += 1
                else:
                    break
            merged.append((node_id, _clone_with_content(node, content), depth))
            i = j
        else:
            merged.append((node_id, node, depth))
            i += 1
    return merged

# ------------------------------------------------------------------------------
# LLM thread-safe
# ------------------------------------------------------------------------------

class ThreadSafeLLM:
    def __init__(self):
        self._lock = threading.Lock()
        self._llm_instances = {}
        self._initialized = False

    def initialize(self):
        if not self._initialized:
            try:
                llm = init_chat_model(
                    model="gemini-2.0-flash",
                    model_provider="google_genai",
                    temperature=0.1,
                    max_tokens=8192,
                )
                llm.with_structured_output(Statement)
                self._initialized = True
                logger.info("LLM pre-initialized successfully")
            except Exception as e:
                logger.error(f"Error pre-initializing LLM: {str(e)}")
                raise

    def get_llm(self):
        tid = threading.get_ident()
        with self._lock:
            if tid not in self._llm_instances:
                try:
                    llm = init_chat_model(
                        model="gemini-2.0-flash",
                        model_provider="google_genai",
                        temperature=0.1,
                        max_tokens=8192,
                    )
                    self._llm_instances[tid] = llm.with_structured_output(Statement)
                except Exception as e:
                    logger.error(f"Error initializing LLM for thread {tid}: {str(e)}")
                    raise
            return self._llm_instances[tid]

    def cleanup(self):
        with self._lock:
            self._llm_instances.clear()

class ThreadSafeList:
    def __init__(self):
        self._lock = threading.Lock()
        self._list: List[Dict] = []

    def append(self, item):
        with self._lock:
            self._list.append(item)

    def extend(self, items: List[Dict]):
        with self._lock:
            self._list.extend(items)

    def get_all(self) -> List[Dict]:
        with self._lock:
            return list(self._list)

class StatementProcessor:
    def __init__(self):
        self.statements = ThreadSafeList()
        self.llm_handler = ThreadSafeLLM()
        self.prompt_template = ChatPromptTemplate.from_template(template=TEMPLATE)
        self.llm_handler.initialize()

    def get_llm(self):
        return self.llm_handler.get_llm()

    def cleanup(self):
        self.llm_handler.cleanup()

# ------------------------------------------------------------------------------
# Concurrencia
# ------------------------------------------------------------------------------

def get_optimal_thread_count():
    return min(8, max(2, multiprocessing.cpu_count()))

def _truncate_for_prompt(title: str, content: str, max_chars: int = 18000) -> str:
    title = title or ""
    content = content or ""
    joined = f"{title}\n{content}".strip()
    if len(joined) <= max_chars:
        return joined
    head = (title + "\n").encode("utf-8", "ignore")[:2000].decode("utf-8", "ignore")
    body_budget = max_chars - len(head)
    tail = content.encode("utf-8", "ignore")[:max(0, body_budget)].decode("utf-8", "ignore")
    return (head + tail).strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_statements(node, tree, processor: StatementProcessor) -> Dict:
    try:
        statement_text = _truncate_for_prompt(node.title, node.content, max_chars=18000)
        prompt = processor.prompt_template.format_messages(
            system_prompt=SYSTEM_PROMPT, statement=statement_text
        )
        llm = processor.get_llm()
        response: Statement = llm.invoke(prompt)
        return {
            "category": response.category,
            "justification": response.justification,
            "synthesis": response.synthesis,
            "heading_path": build_numeric_first_heading_path(tree, node),
            "text_content": statement_text,
        }
    except Exception as e:
        logger.error(f"Error processing '{getattr(node, 'title', '')}': {str(e)}")
        return {
            "category": "error",
            "justification": f"Error in processing: {str(e)}",
            "synthesis": "",
            "heading_path": build_numeric_first_heading_path(tree, node),
            "text_content": (node.content or "").strip(),
        }

def process_node(node_tuple, tree, processor: StatementProcessor):
    _node_id, node, _depth = node_tuple
    return extract_statements(node, tree, processor)

def process_nodes(executor, nodes_to_process, processor, tree):
    results_dict: Dict[int, Dict] = {}
    with tqdm(total=len(nodes_to_process), desc="Processing nodes") as pbar:
        futures = {}
        for i, node_tuple in enumerate(nodes_to_process):
            fut = executor.submit(process_node, node_tuple, tree, processor)
            futures[fut] = i
            if i % 5 == 0:
                time.sleep(0.3)

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result(timeout=300)
                if result:
                    results_dict[idx] = result
            except TimeoutError:
                logger.error(f"Timeout processing node at index {idx}")
            except Exception as e:
                logger.error(f"Error processing node at index {idx}: {str(e)}")
            finally:
                pbar.update(1)

    ordered_results = [results_dict[i] for i in sorted(results_dict.keys())]
    processor.statements.extend(ordered_results)

# ------------------------------------------------------------------------------
# Persistencia
# ------------------------------------------------------------------------------

def save_results(statements: List[Dict], output_path: str, formats=("parquet","csv"), write_xlsx=True):
    os.makedirs(output_path, exist_ok=True)

    cols = ["category","justification","synthesis","heading_path","text_content"]
    if not statements:
        logger.warning("No statements to save.")
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(statements)
        df = df.applymap(lambda x: x.encode("utf-8","replace").decode("utf-8") if isinstance(x,str) else x)

    if "parquet" in formats:
        try:
            df.to_parquet(f"{output_path}/statements.parquet")
            logger.info("Saved statements.parquet")
        except Exception as e:
            logger.error(f"Parquet save error: {e}")

    if "csv" in formats:
        try:
            df.to_csv(f"{output_path}/statements.csv", index=False, encoding="utf-8-sig")
            logger.info("Saved statements.csv")
        except Exception as e:
            logger.error(f"CSV save error: {e}")

    if write_xlsx:
        try:
            try:
                df.to_excel(f"{output_path}/statements.xlsx", index=False, engine="xlsxwriter")
            except Exception:
                df.to_excel(f"{output_path}/statements.xlsx", index=False)
            logger.info("Saved statements.xlsx")
        except Exception as e:
            logger.warning(f"Skipping XLSX: {e}")

    logger.info(f"Artifacts stored in: {output_path}")

# ------------------------------------------------------------------------------
# API principal
# ------------------------------------------------------------------------------

def classify_statements(
    file_name: str, md_content: str, output_dir: str = "data/processed/"
):
    logger.info(f"Starting processing of {file_name}")

    output_path = os.path.join(output_dir, file_name, "statements")
    os.makedirs(output_path, exist_ok=True)

    md_content = force_group_index_block(md_content)

    parser = ElementParser()
    tree = parser.parse_text(md_content)

    iterator = ElementIterator(tree)
    nodes = list(iterator.iterate_depth_first(skip_root=True))

    nodes_to_process = merge_subnodes_view(nodes, tree)

    if nodes_to_process:
        logger.info(f"Nodes to process: {len(nodes_to_process)}")

    processor = StatementProcessor()
    try:
        with ThreadPoolExecutor(max_workers=get_optimal_thread_count()) as executor:
            try:
                process_nodes(executor, nodes_to_process, processor, tree)
            finally:
                save_results(processor.statements.get_all(), output_path)
    finally:
        processor.cleanup()

    return pd.DataFrame(processor.statements.get_all())
