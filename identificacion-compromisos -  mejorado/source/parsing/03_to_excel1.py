import pandas as pd
from pathlib import Path
import re

def _procesar_bloque_tabla(bloque_lineas, num_cols):
    filas = []
    for linea in bloque_lineas:
        if linea.strip().startswith('|'):
            celdas = [c.strip() for c in linea.split('|')[1:-1]]
            if len(celdas) > 0:
                filas.append(celdas)

    if len(filas) < 2:
        return []

    # Quitar fila de separadores
    filas_limpias = []
    for fila in filas:
        if all(c.startswith('-') or c == '' for c in fila):
            continue
        filas_limpias.append(fila)

    if len(filas_limpias) <= 1:
        return []

    # Saltar encabezado (primera fila después del separador)
    datos = filas_limpias[1:]

    # Normalizar columnas
    filas_normalizadas = []
    for fila in datos:
        if len(fila) < num_cols:
            fila += [''] * (num_cols - len(fila))
        elif len(fila) > num_cols:
            exceso = ' | '.join(fila[num_cols-1:])
            fila = fila[:num_cols-1] + [exceso]
        filas_normalizadas.append(fila[:num_cols])

    return filas_normalizadas


def markdown_a_excel_desde_carpeta(
    carpeta_entrada: Path,
    nombres_columnas,
    num_columnas_esperadas=8,
    archivo_salida="Compromisos_Unificados.xlsx"
):
    """
    Procesa TODOS los archivos .md en una carpeta (en orden numérico)
    y genera un solo Excel con todas las filas unidas.
    """
    if not carpeta_entrada.exists():
        raise FileNotFoundError(f"Carpeta no encontrada: {carpeta_entrada}")

    # Buscar todos los archivos .md que terminen con _cleaned_X.md
    archivos_md = sorted(
        carpeta_entrada.glob("*_cleaned_*.md"),
        key=lambda p: int(re.search(r'_cleaned_(\d+)\.md', p.name).group(1))
    )

    if not archivos_md:
        raise FileNotFoundError(f"No se encontraron archivos *_cleaned_*.md en: {carpeta_entrada}")

    print(f"📂 Encontrados {len(archivos_md)} archivos en {carpeta_entrada.name}")
    print("   → Procesando en orden...")

    todas_las_filas_datos = []

    for idx, archivo in enumerate(archivos_md, 1):
        print(f"   [{idx}/{len(archivos_md)}] Leyendo: {archivo.name}")
        texto_md = archivo.read_text(encoding="utf-8", errors="ignore")

        lineas_originales = texto_md.split('\n')
        lineas = []
        buffer = ""
        for linea in lineas_originales:
            linea_strip = linea.strip()
            if linea_strip.startswith('|'):
                if buffer:
                    lineas.append(buffer.rstrip())
                    buffer = ""
                buffer = linea
            else:
                if buffer and linea_strip:
                    buffer += " " + linea_strip
                elif buffer:
                    buffer += "\n"
        
        if buffer:
            lineas.append(buffer.rstrip())

        # Procesar todos los bloques de tabla en este archivo
        i = 0
        while i < len(lineas):
            linea = lineas[i]
            if linea.strip().startswith('|'):
                inicio_tabla = i
                fin_tabla = i + 1
                while fin_tabla < len(lineas) and lineas[fin_tabla].strip().startswith('|'):
                    fin_tabla += 1
                bloque = lineas[inicio_tabla:fin_tabla]
                filas_datos = _procesar_bloque_tabla(bloque, num_columnas_esperadas)
                todas_las_filas_datos.extend(filas_datos)
                i = fin_tabla
            else:
                i += 1

    if not todas_las_filas_datos:
        print("⚠️ No se encontraron datos de tabla en ningún archivo.")
        return None

    # === FILTRO FINAL: eliminar filas basura ===
    filas_buenas = []
    for fila in todas_las_filas_datos:
        texto_fila = ' '.join([c.strip() for c in fila]).strip()
        
        if not texto_fila:
            continue
        
        if texto_fila.upper() in [
            "SUBSANACION¹", "SUBSANACIÓN¹", "INFORMACIÓN COMPLEMENTARIA",
            "SUBSANACION¹ INFORMACIÓN COMPLEMENTARIA", "SUBSANACIÓN¹ INFORMACIÓN COMPLEMENTARIA"
        ]:
            continue
        
        if not fila[0].strip() or not fila[0].strip().replace('.', '').isdigit():
            continue
        
        filas_buenas.append(fila)

    df = pd.DataFrame(filas_buenas, columns=nombres_columnas[:num_columnas_esperadas])

    # Guardar Excel
    with pd.ExcelWriter(archivo_salida, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Observaciones", index=False)
        
        worksheet = writer.sheets["Observaciones"]
        for idx, columna in enumerate(df.columns):
            longitud_max = max(
                df[columna].astype(str).map(len).max(),
                len(columna)
            ) + 6
            worksheet.column_dimensions[chr(65 + idx)].width = min(longitud_max, 80)

    print("\n¡ÉXITO TOTAL!")
    print(f"   → {len(archivos_md)} archivos procesados")
    print(f"   → {len(df)} filas de observaciones unificadas")
    print(f"   → Archivo final guardado: {archivo_salida}")
    return archivo_salida


# ==================================================================
# TUS COLUMNAS
# ==================================================================
MIS_COLUMNAS = [
    "N°",
    "ITEM",
    "FUNDAMENTOS / SUSTENTOS",
    "OBSERVACIONES",
    "SUBSANACIÓN",
    "INFORMACIÓN COMPLEMENTARIA",
    "ANÁLISIS DE LA SUBSANACIÓN",
    "ABSUELTO (SI/NO)"
]

# ==================================================================
# CONFIGURACIÓN
# ==================================================================
carpeta_con_partes = Path("data/processed/datamd/cleaned_parts")  # ← Ajusta si es necesario
archivo_excel_final = "compromisos_1bloque.xlsx"

# ==================================================================
# EJECUTAR
# ==================================================================
if not carpeta_con_partes.exists():
    raise FileNotFoundError(f"Carpeta no encontrada: {carpeta_con_partes}")

markdown_a_excel_desde_carpeta(
    carpeta_entrada=carpeta_con_partes,
    nombres_columnas=MIS_COLUMNAS,
    num_columnas_esperadas=8,
    archivo_salida=archivo_excel_final
)