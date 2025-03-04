from core.models import DocumentProcessor, ElementIterator


def main():
    processor = DocumentProcessor()
    markdown_file = "../../data/processed/output/rca.md"
    tree = processor.process_file(markdown_file)

    nodes_output = markdown_file.rsplit('.', 1)[0] + '_nodes.json'
    toc_output = markdown_file.rsplit('.', 1)[0] + '_toc.md'
    html_output = markdown_file.rsplit('.', 1)[0] + '.html'

    processor.save_tree(tree, nodes_output)
    processor.generate_toc(tree, toc_output)
    processor.export_to_html(tree, html_output)

    print(f"Estructura de nodos guardada en: {nodes_output}")
    print(f"Tabla de contenido guardada en: {toc_output}")
    print(f"Documento HTML generado en: {html_output}")

    iterator = ElementIterator(tree)
    for i, (node_id, node, depth) in enumerate(iterator.iterate_depth_first()):
        indent = "  " * depth
        print(f"{indent}- {node.title} (Nivel: {node.level})")

if __name__ == "__main__":
    main()