El proceso debe seguir este orden:

Paso 1 para convertir el PDF a Markdown: 
ejecutar:  
python -m source.parsing.parse_pdf_md data/datapdf/“nombre_de_tu_archivo”.pdf
el archivo se creará en data/processed/datamd con el nombre de “nombre_de_tu_archivo_raw”.

Paso 2 para limpiar y estructurar el archivo generado: 
ejecutar: 
python -m source.parsing.md_cleaner_mistral data/processed/datamd/“nombre_de_tu_archivo_generado”.md
aquí se ingresa el número de filas de la tabla, en la parte del código donde se indicó. Se recomienda ingresar de 20 a 30 filas de tabla.
se creará una nueva carpeta con los archivos md creados limpios en data/processed/datamd/cleaned_parts (una vez terminado el proceso eliminar o cambiar el nombre de la carpeta, para que se pueda procesar un nuevo archivo).

Paso 3 para transformar el Markdown limpio en un archivo Excel: 
ejecutar:  
python -m source.parsing.to_excel1.py
aquí se ingresa el número de columnas y el nombre de cada columna de la tabla, también el nombre de la carpeta generada con las partes de los archivos md limpios para convertir a excel, en la parte del código donde se indicó.
Se creará un nuevo Excel con todos los compromisos generados y convertarchivo en la raíz del proyecto con el nombre “Compromisos_Unificados” cámbialo luego de culminar el proceso a otro nombre, para que puedas generar un nuevo proceso.
El archivo PDF a procesar debe ubicarse en data/datapdf y al correr los comandos se debe usar siempre el mismo nombre del archivo presente en esa carpeta.

En el archivo
.env : MISTRAL_API_KEY=wxdV2M1hgJt9hp0Cwwp7PwWBH1ryG46M
setx MISTRAL_API_KEY "wxdV2M1hgJt9hp0Cwwp7PwWBH1ryG46M"

