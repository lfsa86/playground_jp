def read_md_file(file_path):
    """
    Reads the contents of a markdown file.

    Args:
        file_path (str): Path to the markdown file.

    Returns:
        str: Content of the markdown file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {str(e)}")


doc = read_md_file("rca.md")

doc = doc.split("Página `")
