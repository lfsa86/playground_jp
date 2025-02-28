from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate

from extractor.chat_model.gemini import ChatGemini
from extractor.document_structurizer import DocumentStructurizer, Page

load_dotenv()


def load_md(md_path: Path | str) -> List[Page]:
    """
    Load a markdown file and return a list of pages.

    Args:
        md_path (Path | str): The path to the markdown file.

    Returns:
        List[Page]: A list of Page objects.
    """
    # Initialize the structurizer
    structurizer = DocumentStructurizer()
    # Load the markdown file
    structurizer.load_file(md_path)
    # Process the document
    return structurizer.process_document()


def extract_commitments(paragraph: str):
    """
    Extract commitments from a paragraph.

    Args:
        paragraph (str): The paragraph to extract commitments from.

    Returns:
        dict: A dictionary containing the commitment category and justification.
    """
    # Define SYSTEM_PROMPT
    SYSTEM_PROMPT = """
    Eres un asistente especializado en normativa ambiental y regulación en Chile. Tu tarea es clasificar un enunciado en una de las siguientes categorías:
    1. **Obligación**: Requisito normativo o regulatorio que debe cumplirse para evitar sanciones.
    2. **Compromiso**: Acción voluntaria propuesta por el titular del proyecto que, al ser incorporada en la Resolución de Calificación Ambiental (RCA), se vuelve obligatoria.
    3. **Acuerdo**: Pacto negociado entre el titular del proyecto y otros actores (comunidades, gobierno, ONGs) para mejorar la gestión ambiental del proyecto.
    4. **Otro**: Si el enunciado no corresponde a ninguna de las categorías anteriores.

    **Reglas para la clasificación:**

    - **Obligación**: Si el enunciado menciona el cumplimiento de normas legales, requisitos regulatorios o hace referencia a sanciones por incumplimiento, clasifícalo como **Obligación**.
    - *Justificación*: Estos enunciados se refieren a mandatos establecidos por la legislación ambiental chilena que son de cumplimiento obligatorio para todos los proyectos, independientemente de su naturaleza.

    - **Compromiso**: Si el enunciado describe una medida voluntaria propuesta por el titular del proyecto para mejorar el desempeño ambiental, que al ser incluida en la RCA adquiere carácter obligatorio, clasifícalo como **Compromiso**.
    - *Justificación*: Aunque inicialmente voluntarias, estas acciones se formalizan y se tornan exigibles al ser incorporadas en la RCA, según lo establecido en la normativa ambiental chilena.

    - **Acuerdo**: Si el enunciado menciona negociaciones, pactos o convenios entre el titular del proyecto y otras partes interesadas (como comunidades locales, organismos gubernamentales u organizaciones no gubernamentales) orientados a mejorar la gestión ambiental, clasifícalo como **Acuerdo**.
    - *Justificación*: Estos enunciados reflejan entendimientos alcanzados mediante diálogo y consenso, que pueden ir más allá de las exigencias legales y compromisos voluntarios, buscando beneficios ambientales adicionales.

    - **Otro**: Si el enunciado no encaja en ninguna de las categorías anteriores, clasifícalo como **Otro**.
    - *Justificación*: Esta categoría abarca enunciados que no se relacionan directamente con obligaciones legales, compromisos voluntarios formalizados en la RCA o acuerdos negociados, y por lo tanto requieren una clasificación distinta.

    Ten presente que, en el contexto de la legislación ambiental chilena, la RCA es un documento administrativo emitido por el Servicio de Evaluación Ambiental (SEA) que certifica la viabilidad ambiental de un proyecto o actividad. Este documento establece las condiciones y exigencias que el titular del proyecto debe cumplir para mitigar o compensar los posibles efectos en el medio ambiente. Además, las medidas voluntarias propuestas por el titular, una vez incorporadas en la RCA, adquieren carácter obligatorio. 

    Por lo tanto, al clasificar los enunciados, es esencial considerar si se trata de:

    - Requisitos legales preexistentes (**Obligación**).
    - Compromisos voluntarios asumidos por el titular que se vuelven obligatorios al ser incluidos en la RCA (**Compromiso**).
    - Acuerdos resultantes de negociaciones con otros actores (**Acuerdo**).
    - Enunciados que no se ajustan a ninguna de las categorías anteriores (**Otro**).
    """

    # Define output schema
    category_schema = ResponseSchema(
        name="category", description="obligacion, compromiso, acuerdo u otro"
    )
    justification_schema = ResponseSchema(
        name="justification",
        description="Explicación breve del motivo de la clasificación.",
    )

    # Initialize OutputParser
    output_parser = StructuredOutputParser.from_response_schemas(
        [category_schema, justification_schema]
    )
    format_instructions = output_parser.get_format_instructions()

    # Create Template
    template = """
    {system_prompt}

    Clasifica el siguiente enunciado dentro de obligación, compromiso, acuerdo u 'otro':
    ```{statement}```

    {format_instructions}
    """

    # Initialize ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_template(template)

    # Format messages
    prompt = prompt_template.format_messages(
        statement=paragraph,
        system_prompt=SYSTEM_PROMPT,
        format_instructions=format_instructions,
    )

    # Initialize Chat
    llm = ChatGemini(model_name="gemini-2.0-flash-001", temperature=0.0)

    # Get Response
    response = llm.invoke(prompt)

    # Parse commitment
    commitment = output_parser.parse(response.content)

    return commitment
