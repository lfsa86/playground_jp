
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
   uv venv --python 3.12.4
   source .venv/bin/activate  # Unix/macOS
   # .venv\Scripts\activate  # Windows
   ```

4. **Instala las dependencias**:
   ```bash
   uv pip install -r requirements.txt
   ```

5. **Configura Ruff** (opcional):
   - Asegúrate de que el archivo `pyproject.toml` esté configurado correctamente.
   - Ejecuta Ruff para verificar el código:
     ```bash
     ruff check .
     ```
   - Aplica correcciones automáticas:
     ```bash
     ruff check --fix .
     ```

¡Tu entorno está listo para usar!