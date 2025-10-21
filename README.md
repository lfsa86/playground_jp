
# Configuración del Entorno

Sigue estos pasos para configurar el entorno de desarrollo:

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/jdqniel/Identificacion-de-Compromisos.git
   cd Identificacion-de-Compromisos
   ```

2. **Instala uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Crea y activa el entorno virtual**:
   ```bash
   uv sync
   uv venv --python 3.12.4
   source .venv/bin/activate  # Unix/macOS
   # .venv\Scripts\activate  # Windows
   ```
4. **Corre el codigo
   ```bash
   Paso 1: Define la ruta del documento que en main.py
   Paso 2: d "C:\Github\NaturAI Modulo 1\Identificacion-de-Compromisos"
   uv run python -m source.main
   ```  
¡Tu entorno está listo para usar!
