import json
import re
import uuid
from collections import deque
from typing import Dict, List

from pydantic import BaseModel


class Page(BaseModel):
    """
    Represents a page in the document, containing its number and a list of paragraphs.

    Attributes:
        page_number (Optional[int]): The page number. Can be None if not found.
        paragraphs (List[str]): A list of paragraphs in the page.
    """

    page_summary: str
    page_number: int
    content: List


class Document(BaseModel):
    """
    Represents the entire document, including its pages and metadata.

    Attributes:
        pages (List[Page]): A list of Page objects representing the document's pages.
        metadata (Dict[str, str]): A dictionary containing document metadata.
    """

    pages: List[Page]
    metadata: Dict

class ElementNode:
    def __init__(self, title="", level=0,content="", parent_id=None):
        self.id = str(uuid.uuid4())
        self.title = title
        self.level = level
        self.content = content
        self.parent_id = parent_id
        self.children_ids = []


    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "content": self.content,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids.copy()
        }

    @classmethod
    def from_dicts(cls, node_dict):
        node = cls(
            title=node_dict["title"],
            level=node_dict["level"],
            content=node_dict["content"],
            parent_id=node_dict["parent_id"],
        )

        node.id = node_dict["id"]
        node.children_ids = node_dict["children_ids"]

        return node
class DocumentTree:
    def __init__(self):
        self.nodes = {}
        self.root_node = ElementNode(title="root", level=0)
        self.nodes[self.root_node.id] = self.root_node

    def add_node(self, title, level, content="", parent_id=None):
        node = ElementNode(title=title, level=level, content=content, parent_id=parent_id)
        self.nodes[node.id] = node
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(node.id)
        return node.id

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def find_node_by_title(self, title):
        for node in self.nodes.values():
            if node.title == title:
                return node
        return None

    def get_path_to_node(self, node_id):
        path: List = []
        current_id = node_id
        while current_id is not None:
            node = self.get_node(current_id)
            if node and node.title != "root":
                path.append((current_id, node.title))
            current_id = node.parent_id if node else None
            path.reverse()
            return path
    def to_dict(self):
        return {
            "root_id": self.root_node.id,
            "nodes": {node.id: node.to_dict() for node_id, node in self.nodes.items()}
        }

    @classmethod
    def from_dict(cls, tree_dict):
        tree: DocumentTree = cls()
        tree.nodes= {
        }
        for node_id, node_data in tree_dict["nodes"].items():
            tree.nodes[node_id] = ElementNode.from_dicts(node_data)
            tree.root_node = tree.nodes[tree_dict["root_id"]]
            return tree
    def save_to_json(self, filename):
        try:
            tree_dict = self.to_dict()
            with open(filename, 'w', encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)
        except TypeError as e:
            print("Serialization Error")

    @classmethod
    def load_from_json(cls, filename):
        with open(filename, 'r', encoding="utf-8") as f:
            tree_dict = json.load(f)
        return cls.from_dict(tree_dict)

    def generate_toc(self):
        lines = ["Tabla de Contenido\n"]

        def build_toc(node_id, level=0):
            node = self.get_node(node_id)
            toc_lines = []
            if node.title != "root":
                indent = '  ' * (level -1) if level > 0 else ''
                toc_lines.append(f"{indent}- [{node.title}](#{node_id})")
            for child_id in node.children_ids:
                toc_lines.extend(build_toc(child_id, level + 1 if node.title != "root" else level))
            return toc_lines

        lines.extend(build_toc(self.root_node.id))
        return '\n'.join(lines)

class ElementParser():
    def __init__(self):
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')

    def parse_file(self, file):
        with open(file, 'r', encoding="utf-8") as f:
            content = f.read()
        return self.parse_text(content)

    def parse_text(self, text):
        tree = DocumentTree()
        lines = text.split("\n")
        current_node_id = tree.root_node.id
        level_stack = {0: tree.root_node.id}
        section_content = []

        for line in lines:
            match = self.heading_pattern.match(line)
            if match:
                if section_content:
                    node = tree.get_node(current_node_id)
                    if node:
                        node.content = "\n".join(section_content)
                        section_content = []
                level = len(match.group(1))
                title = match.group(2).strip()
                parent_level = max([l for l in level_stack.keys() if l < level], default=0)
                parent_id = level_stack[parent_level]
                current_node_id = tree.add_node(title=title, level=level, parent_id=parent_id)
                level_stack[level] = current_node_id
                keys_to_remove = [l for l in level_stack if l > level]
                for key in keys_to_remove:
                    del level_stack[key]
            else:
                section_content.append(line)

        if section_content:
            node=tree.get_node(current_node_id)
            if node:
                node.content = "\n".join(section_content)
        return tree

class ElementIterator:
    def __init__(self, tree: DocumentTree):
        self.tree = tree

    def iterate_all(self):
        for node_id, node in self.tree.nodes.items():
            yield node_id, node

    def iterate_depth_first(self, start_node_id=None, skip_root=True):
        """
        Recorre los nodos en orden jerárquico (profundidad primero).

        Args:
            start_node_id (str, optional): ID del nodo desde donde comenzar
            skip_root (bool): Si se debe saltar el nodo raíz

        Yields:
            tuple: (node_id, node, depth) para cada nodo
        """
        if start_node_id is None:
            start_node_id = self.tree.root_node.id

        def _dfs(current_node_id, depth=0):
            """Función recursiva interna para DFS."""
            # Obtener el nodo actual
            node = self.tree.get_node(current_node_id)
            if node is None:
                return

            # Emitir el nodo actual (a menos que sea el nodo raíz y skip_root=True)
            if not (skip_root and node.title == "Root"):
                yield current_node_id, node, depth

            # Recorrer recursivamente cada hijo
            for child_id in node.children_ids:
                # Usar yield from para delegar a la llamada recursiva
                yield from _dfs(child_id, depth + 1)

        # Iniciar la recursión desde el nodo de inicio
        yield from _dfs(start_node_id)

    def iterate_breadth_first(self, start_node_id=None, skip_root=True):
        if start_node_id is None:
            start_node_id = self.tree.root_node.id
            queue = deque([(start_node_id, 0)])
            visited = set()
            while queue:
                node_id, depth = queue.popleft()
                if node_id in visited:
                    continue
                visited.add(node_id)

                node = self.tree.get_node(node_id)
                if not (skip_root and node.title =="root"):
                    yield node_id, node, depth
                for child_id in node.children_ids:
                    queue.append((child_id, depth + 1))

    def filter_nodes(self, criteria_func):
        return [(node_id, node) for node_id, node in self.tree.nodes.items() if criteria_func(node)]

class DocumentProcessor:
    def __init__(self):
        self.parser = ElementParser()

    def process_file(self, file):
        return self.parser.parse_file(file)

    def save_tree(self, tree: DocumentTree, output_file):
        tree.save_to_json(output_file)

    def generate_toc(self, tree: DocumentTree, output_file):
        toc = tree.generate_toc()
        with open(output_file, 'w', encoding="utf-8") as f:
            f.write(toc)

    def export_to_html(self, tree, output_file):
        html = ["<!DOCTYPE html>", "<html>", "<head>",
                "<title>Documento Exportado</title>",
                "<meta charset='utf-8'>",
                "<style>",
                "body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }",
                "h1, h2, h3, h4, h5, h6 { color: #333; }",
                "</style>",
                "</head>", "<body>"]
        iterator = ElementIterator(tree)
        for node_id, node, depth in iterator.iterate_depth_first():
            html.append(f"<h{node.level}>{node.title}</h{node.level}>")
            if node.content:
                content_html = node.content.replace('\n', '<br>\n')
                html.append(f"<p>{content_html}</p>")
        html.extend(["</body>", "</html>"])
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(html))
                


