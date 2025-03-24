import concurrent.futures
import json
import re
import uuid
from collections import deque
from typing import Any, Callable, List, Tuple


class ElementNode:
    """
    Represents a node in a document tree structure.

    Each node contains a title, content, hierarchical level, and references to parent and children nodes.

    Attributes:
        id (str): Unique identifier for the node, automatically generated using UUID.
        title (str): The title or heading text of the node.
        level (int): The hierarchical level of the node (e.g., 1 for h1, 2 for h2).
        content (str): The text content associated with this node.
        parent_id (str): The ID of the parent node, or None if this is a root node.
        children_ids (list): List of IDs of child nodes.
    """

    def __init__(self, title="", level=0, content="", parent_id=None):
        """
        Initialize a new ElementNode.

            Args:
                title (str, optional): The title of the node. Defaults to empty string.
                level (int, optional): The hierarchical level. Defaults to 0.
                content (str, optional): The text content. Defaults to empty string.
                parent_id (str, optional): The ID of the parent node. Defaults to None.
        """
        self.id = str(uuid.uuid4())
        self.title = title
        self.level = level
        self.content = content
        self.parent_id = parent_id
        self.children_ids = []

    def to_dict(self):
        """
        Convert the node to a dictionary representation.

        Returns:
            dict: A dictionary containing all node attributes.
        """
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "content": self.content,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids.copy(),
        }

    @classmethod
    def from_dicts(cls, node_dict):
        """
        Create an ElementNode instance from a dictionary.

        Args:
            node_dict (dict): Dictionary containing node attributes.

        Returns:
            ElementNode: A new instance with attributes set from the dictionary.
        """
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
    """
    Represents a hierarchical document structure as a tree of ElementNodes.

    This class provides methods to build, manipulate, and serialize a document tree.

    Attributes:
        nodes (dict): Dictionary mapping node IDs to ElementNode objects.
        root_node (ElementNode): The root node of the tree.
    """

    def __init__(self):
        """
        Initialize a new DocumentTree with a root node.
        """
        self.nodes = {}
        self.root_node = ElementNode(title="root", level=0)
        self.nodes[self.root_node.id] = self.root_node

    def add_node(self, title, level, content="", parent_id=None):
        """
        Add a new node to the document tree.

        Args:
            title (str): The title of the node.
            level (int): The hierarchical level of the node.
            content (str, optional): The text content. Defaults to empty string.
            parent_id (str, optional): The ID of the parent node. Defaults to None.

        Returns:
            str: The ID of the newly created node.
        """
        node = ElementNode(
            title=title, level=level, content=content, parent_id=parent_id
        )
        self.nodes[node.id] = node
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(node.id)
        return node.id

    def get_node(self, node_id):
        """
        Retrieve a node by its ID.

        Args:
            node_id (str): The ID of the node to retrieve.

        Returns:
            ElementNode: The node with the specified ID, or None if not found.
        """
        return self.nodes.get(node_id)

    def find_node_by_title(self, title):
        """
        Find a node by its title.

        Args:
            title (str): The title to search for.

        Returns:
            ElementNode: The first node with the matching title, or None if not found.
        """
        for node in self.nodes.values():
            if node.title == title:
                return node
        return None

    def get_path_to_node(self, node_id):
        """
        Get the path from the root to a specified node.

        Args:
            node_id (str): The ID of the target node.

        Returns:
            list: A list of tuples (node_id, title) representing the path.
        """
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
        """
        Convert the entire document tree to a dictionary representation.

        Returns:
            dict: A dictionary containing the root ID and all nodes.
        """
        return {
            "root_id": self.root_node.id,
            "nodes": {node.id: node.to_dict() for node_id, node in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, tree_dict):
        """
        Create a DocumentTree instance from a dictionary.

        Args:
            tree_dict (dict): Dictionary containing tree structure.

        Returns:
            DocumentTree: A new instance with nodes reconstructed from the dictionary.
        """
        tree: DocumentTree = cls()
        tree.nodes = {}
        for node_id, node_data in tree_dict["nodes"].items():
            tree.nodes[node_id] = ElementNode.from_dicts(node_data)
            tree.root_node = tree.nodes[tree_dict["root_id"]]
            return tree

    def save_to_json(self, filename):
        """
        Save the document tree to a JSON file.

        Args:
            filename (str): The path to the output JSON file.
        """
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)
        except TypeError as e:
            print("Serialization Error")

    @classmethod
    def load_from_json(cls, filename):
        """
        Load a document tree from a JSON file.

        Args:
            filename (str): The path to the JSON file.

        Returns:
            DocumentTree: A new instance loaded from the file.
        """
        with open(filename, "r", encoding="utf-8") as f:
            tree_dict = json.load(f)
        return cls.from_dict(tree_dict)

    def generate_toc(self):
        """
        Generate a table of contents for the document.

        Returns:
            str: A markdown-formatted table of contents.
        """
        lines = ["Tabla de Contenido\n"]

        def build_toc(node_id, level=0):
            node = self.get_node(node_id)
            toc_lines = []
            if node.title != "root":
                indent = "  " * (level - 1) if level > 0 else ""
                toc_lines.append(f"{indent}- [{node.title}](#{node_id})")
            for child_id in node.children_ids:
                toc_lines.extend(
                    build_toc(child_id, level + 1 if node.title != "root" else level)
                )
            return toc_lines

        lines.extend(build_toc(self.root_node.id))
        return "\n".join(lines)


class ElementParser:
    """
    Parser for extracting document structure from markdown text.

    This class parses markdown headings and content to build a DocumentTree.
    """

    def __init__(self):
        """
        Initialize the parser with a regex pattern for markdown headings.
        """
        self.heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

    def parse_file(self, file):
        """
        Parse a markdown file into a DocumentTree.

        Args:
            file (str): Path to the markdown file.

        Returns:
            DocumentTree: A tree representing the document structure.
        """
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        return self.parse_text(content)

    def parse_text(self, text):
        """
        Parse markdown text into a DocumentTree.

        Args:
            text (str): Markdown text to parse.

        Returns:
            DocumentTree: A tree representing the document structure.
        """
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
                parent_level = max(
                    [l for l in level_stack.keys() if l < level], default=0
                )
                parent_id = level_stack[parent_level]
                current_node_id = tree.add_node(
                    title=title, level=level, parent_id=parent_id
                )
                level_stack[level] = current_node_id
                keys_to_remove = [l for l in level_stack if l > level]
                for key in keys_to_remove:
                    del level_stack[key]
            else:
                section_content.append(line)

        if section_content:
            node = tree.get_node(current_node_id)
            if node:
                node.content = "\n".join(section_content)
        return tree


class ElementIterator:
    """
    Iterator for traversing nodes in a DocumentTree.

    Provides methods for different traversal strategies (depth-first, breadth-first)
    and filtering nodes based on criteria.
    """

    def __init__(self, tree: DocumentTree):
        """
        Initialize the iterator with a document tree.

        Args:
            tree (DocumentTree): The document tree to iterate over.
        """
        self.tree = tree

    def iterate_all(self):
        """
        Iterate through all nodes in the tree without specific order.

        Yields:
            tuple: (node_id, node) for each node in the tree.
        """
        for node_id, node in self.tree.nodes.items():
            yield node_id, node

    def iterate_depth_first(self, start_node_id=None, skip_root=True):
        """
        Traverse nodes in depth-first order (hierarchical order).

        Args:
            start_node_id (str, optional): ID of the node to start from. Defaults to root.
            skip_root (bool): Whether to skip the root node. Defaults to True.

        Yields:
            tuple: (node_id, node, depth) for each node in the traversal.
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
        """
        Traverse nodes in breadth-first order (level by level).

        Args:
            start_node_id (str, optional): ID of the node to start from. Defaults to root.
            skip_root (bool): Whether to skip the root node. Defaults to True.

        Yields:
            tuple: (node_id, node, depth) for each node in the traversal.
        """
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
                if not (skip_root and node.title == "root"):
                    yield node_id, node, depth
                for child_id in node.children_ids:
                    queue.append((child_id, depth + 1))

    def filter_nodes(self, criteria_func):
        """
        Filter nodes based on a criteria function.

        Args:
            criteria_func (callable): A function that takes a node and returns a boolean.

        Returns:
            list: List of (node_id, node) tuples for nodes that match the criteria.
        """
        return [
            (node_id, node)
            for node_id, node in self.tree.nodes.items()
            if criteria_func(node)
        ]


class DocumentProcessor:
    """
    High-level processor for document operations.

    Provides methods for parsing, saving, and exporting document trees.
    """

    def __init__(self):
        """
        Initialize the document processor with an ElementParser.
        """
        self.parser = ElementParser()

    def process_file(self, file):
        """
        Process a markdown file into a DocumentTree.

        Args:
            file (str): Path to the markdown file.

        Returns:
            DocumentTree: A tree representing the document structure.
        """
        return self.parser.parse_file(file)

    def save_tree(self, tree: DocumentTree, output_file):
        """
        Save a document tree to a JSON file.

        Args:
            tree (DocumentTree): The document tree to save.
            output_file (str): Path to the output JSON file.
        """
        tree.save_to_json(output_file)

    def generate_toc(self, tree: DocumentTree, output_file):
        """
        Generate and save a table of contents for a document tree.

        Args:
            tree (DocumentTree): The document tree.
            output_file (str): Path to the output markdown file.
        """
        toc = tree.generate_toc()
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(toc)

    def export_to_html(self, tree, output_file):
        """
        Export a document tree to an HTML file.

        Args:
            tree (DocumentTree): The document tree to export.
            output_file (str): Path to the output HTML file.
        """
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Documento Exportado</title>",
            "<meta charset='utf-8'>",
            "<style>",
            "body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }",
            "h1, h2, h3, h4, h5, h6 { color: #333; }",
            "</style>",
            "</head>",
            "<body>",
        ]
        iterator = ElementIterator(tree)
        for node_id, node, depth in iterator.iterate_depth_first():
            html.append(f"<h{node.level}>{node.title}</h{node.level}>")
            if node.content:
                content_html = node.content.replace("\n", "<br>\n")
                html.append(f"<p>{content_html}</p>")
        html.extend(["</body>", "</html>"])
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(html))


class NodeProcessor:
    """
    Clase para procesar el contenido y título de nodos de manera eficiente.
    """

    def __init__(self, tree: DocumentTree, max_workers=None):
        """
        Inicializa el procesador de nodos.

        Args:
            tree (DocumentTree): El árbol de documento a procesar.
            max_workers (int, optional): Número máximo de trabajadores en el pool.
        """
        self.tree = tree
        self.max_workers = max_workers
        self.iterator = ElementIterator(tree)

    def _process_node(
        self, node_data: Tuple[str, ElementNode, int], processor_func: Callable
    ):
        """
        Procesa un nodo individual (título y contenido).

        Args:
            node_data (tuple): Tupla (node_id, node, depth) del nodo a procesar.
            processor_func (callable): Función que procesa el título y contenido del nodo.

        Returns:
            tuple: (node_id, resultado_procesamiento)
        """
        node_id, node, depth = node_data
        result = processor_func(node.title, node.content)
        return (node_id, result)

    def process_nodes_parallel(
        self,
        processor_func: Callable,
        traversal_method="depth_first",
        start_node_id=None,
        skip_root=True,
        update_nodes=False,
        use_threads=True,
    ) -> List[Tuple[str, Any]]:
        """
        Procesa el título y contenido de cada nodo en paralelo.

        Args:
            processor_func (callable): Función que recibe el título y contenido del nodo (str, str)
                                      y devuelve un resultado procesado.
            traversal_method (str): Metodo de recorrido: "depth_first" o "breadth_first".
            start_node_id (str, optional): ID del nodo desde donde comenzar.
            skip_root (bool): Si se debe saltar el nodo raíz.
            update_nodes (bool): Si se debe actualizar el contenido de los nodos con el resultado.
            use_threads (bool): Si se deben usar hilos en lugar de procesos.

        Returns:
            list: Lista de tuplas (node_id, resultado_procesamiento) para cada nodo.
        """
        # Obtener los nodos a procesar
        if traversal_method == "depth_first":
            nodes_to_process = list(
                self.iterator.iterate_depth_first(start_node_id, skip_root)
            )
        elif traversal_method == "breadth_first":
            nodes_to_process = list(
                self.iterator.iterate_breadth_first(start_node_id, skip_root)
            )
        else:
            raise ValueError(f"Método de recorrido no válido: {traversal_method}")

        results = []

        # Usar ThreadPoolExecutor por defecto para evitar problemas de inicialización
        executor_class = (
            concurrent.futures.ThreadPoolExecutor
            if use_threads
            else concurrent.futures.ProcessPoolExecutor
        )

        # Procesar en paralelo
        with executor_class(max_workers=self.max_workers) as executor:
            # Enviar tareas al pool
            future_to_node = {
                executor.submit(
                    self._process_node, node_data, processor_func
                ): node_data[0]
                for node_data in nodes_to_process
            }

            # Recoger resultados a medida que se completan
            for future in concurrent.futures.as_completed(future_to_node):
                node_id = future_to_node[future]
                try:
                    node_id, result = future.result()
                    results.append((node_id, result))

                    # Actualizar el contenido del nodo si se solicita
                    if update_nodes:
                        node = self.tree.get_node(node_id)
                        if node:
                            node.content = result

                except Exception as exc:
                    print(
                        f"El procesamiento del nodo {node_id} generó una excepción: {exc}"
                    )

        return results
