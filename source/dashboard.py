import pandas as pd
import plotly.graph_objects as go

# Cargar y limpiar datos
df = pd.read_csv("data/processed/df.csv", encoding="utf-8")

# Estandarizar nombres de columnas
df.columns = df.columns.str.strip()

# Limpiar columna de estado de cumplimiento
df["Cumple (SI/NO)"] = df["Cumple (SI/NO)"].str.strip().str.upper()
df["Cumple (SI/NO)"] = df["Cumple (SI/NO)"].fillna("POR DETERMINAR")
df["Cumple (SI/NO)"] = df["Cumple (SI/NO)"].replace({"SI": "SÍ"})

# Verificar valores únicos para diagnóstico
print("Valores únicos en 'Cumple (SI/NO)':", df["Cumple (SI/NO)"].unique())

# Limpiar y estandarizar otras columnas
for col in ["Área Responsable de Kolpa", "Componente minero", "Etapa", "Código IGA"]:
    if col in df.columns:
        df[col] = df[col].fillna("No especificado")
        df[col] = df[col].str.strip().str.upper()

# Calcular estadísticas clave
total_commitments = len(df)
compliant = df[df["Cumple (SI/NO)"] == "SÍ"].shape[0]
non_compliant = df[df["Cumple (SI/NO)"] == "NO"].shape[0]
undetermined = df[df["Cumple (SI/NO)"] == "POR DETERMINAR"].shape[0]
compliance_percentage = (compliant / total_commitments) * 100

# Imprimir estadísticas para verificación
print(f"Total de compromisos: {total_commitments}")
print(f"Cumplidos (SÍ): {compliant}")
print(f"No cumplidos (NO): {non_compliant}")
print(f"No determinados: {undetermined}")
print(f"Porcentaje de cumplimiento: {compliance_percentage:.1f}%")

# Crear una paleta de colores profesional
colors = {
    "SÍ": "#2E8B57",  # Verde bosque
    "NO": "#E63946",  # Rojo
    "NO DETERMINADO": "#FFB703",  # Amarillo/dorado
    "background": "#FFFFFF",  # Blanco
    "card_bg": "#F8F9FA",  # Gris muy claro
    "text": "#1D3557",  # Azul oscuro
    "subtitle": "#457B9D",  # Azul medio
    "grid": "#E9ECEF",  # Gris claro
    "accent": "#457B9D",  # Azul medio
}

# Preparar datos para el gráfico de áreas
area_compliance = (
    df.groupby(["Área Responsable de Kolpa", "Cumple (SI/NO)"])
    .size()
    .reset_index(name="Count")
)
area_pivot = area_compliance.pivot_table(
    index="Área Responsable de Kolpa",
    columns="Cumple (SI/NO)",
    values="Count",
    fill_value=0,
).reset_index()

# Asegurar que todas las columnas de estado existan
for status in ["SÍ", "NO", "NO DETERMINADO"]:
    if status not in area_pivot.columns:
        area_pivot[status] = 0

# Calcular total y ordenar
area_pivot["Total"] = area_pivot["SÍ"] + area_pivot["NO"] + area_pivot["NO DETERMINADO"]
area_pivot = area_pivot.sort_values("Total", ascending=True)
area_pivot = area_pivot.tail(8)  # Mostrar solo las 8 áreas principales

# Preparar datos para el gráfico de componentes
component_compliance = (
    df.groupby(["Componente minero", "Cumple (SI/NO)"]).size().reset_index(name="Count")
)
component_pivot = component_compliance.pivot_table(
    index="Componente minero", columns="Cumple (SI/NO)", values="Count", fill_value=0
).reset_index()

# Asegurar que todas las columnas de estado existan
for status in ["SÍ", "NO", "NO DETERMINADO"]:
    if status not in component_pivot.columns:
        component_pivot[status] = 0

# Calcular total y ordenar
component_pivot["Total"] = (
    component_pivot["SÍ"] + component_pivot["NO"] + component_pivot["NO DETERMINADO"]
)
component_pivot = component_pivot.sort_values("Total", ascending=False)
component_pivot = component_pivot.head(6)  # Mostrar solo los 6 componentes principales

# Preparar datos para el gráfico de etapas
phase_compliance = (
    df.groupby(["Etapa", "Cumple (SI/NO)"]).size().reset_index(name="Count")
)
phase_pivot = phase_compliance.pivot_table(
    index="Etapa", columns="Cumple (SI/NO)", values="Count", fill_value=0
).reset_index()

# Asegurar que todas las columnas de estado existan
for status in ["SÍ", "NO", "NO DETERMINADO"]:
    if status not in phase_pivot.columns:
        phase_pivot[status] = 0

# Calcular total y ordenar
phase_pivot["Total"] = (
    phase_pivot["SÍ"] + phase_pivot["NO"] + phase_pivot["NO DETERMINADO"]
)
phase_pivot = phase_pivot.sort_values("Total", ascending=False)

# Crear gráficos individuales y guardarlos como archivos HTML separados

# 1. Gráfico de pastel
fig_pie = go.Figure(
    data=[
        go.Pie(
            labels=["SÍ", "NO", "NO DETERMINADO"],
            values=[compliant, non_compliant, undetermined],
            hole=0.6,
            marker=dict(
                colors=[colors["SÍ"], colors["NO"], colors["NO DETERMINADO"]],
                line=dict(color="white", width=2),
            ),
            textinfo="percent",
            textposition="inside",
            hoverinfo="label+value+percent",
            hovertemplate="<b>%{label}</b><br>Compromisos: %{value}<br>Porcentaje: %{percent}<extra></extra>",
            pull=[0, 0.05, 0],
        )
    ]
)

fig_pie.update_layout(
    showlegend=True,
    legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    margin=dict(t=0, b=0, l=0, r=0),
    paper_bgcolor=colors["card_bg"],
    plot_bgcolor=colors["card_bg"],
    font=dict(family="Roboto, sans-serif", size=12, color=colors["text"]),
)

fig_pie.write_html("pie_chart.html")

# 2. Gráfico de gauge
fig_gauge = go.Figure(
    data=[
        go.Indicator(
            mode="gauge+number+delta",
            value=compliance_percentage,
            delta=dict(reference=100, increasing=dict(color=colors["SÍ"])),
            number=dict(suffix="%", font=dict(size=26, color=colors["text"])),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor=colors["text"]),
                bar=dict(color=colors["SÍ"]),
                bgcolor="white",
                borderwidth=2,
                bordercolor=colors["text"],
                steps=[
                    dict(range=[0, 50], color="rgba(230, 57, 70, 0.2)"),
                    dict(range=[50, 80], color="rgba(255, 183, 3, 0.2)"),
                    dict(range=[80, 100], color="rgba(46, 139, 87, 0.2)"),
                ],
                threshold=dict(
                    line=dict(color=colors["text"], width=4),
                    thickness=0.75,
                    value=compliance_percentage,
                ),
            ),
        )
    ]
)

fig_gauge.update_layout(
    margin=dict(t=0, b=0, l=0, r=0),
    paper_bgcolor=colors["card_bg"],
    font=dict(family="Roboto, sans-serif", size=12, color=colors["text"]),
)

fig_gauge.write_html("gauge_chart.html")

# 3. Gráfico de áreas
fig_area = go.Figure()

for status in ["SÍ", "NO", "NO DETERMINADO"]:
    fig_area.add_trace(
        go.Bar(
            y=area_pivot["Área Responsable de Kolpa"],
            x=area_pivot[status],
            name=status,
            orientation="h",
            marker=dict(color=colors[status]),
            hovertemplate="<b>%{y}</b><br>%{x} compromisos "
            + status.lower()
            + "<extra></extra>",
        )
    )

fig_area.update_layout(
    barmode="stack",
    showlegend=True,
    legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    margin=dict(t=0, b=50, l=150, r=20),
    paper_bgcolor=colors["card_bg"],
    plot_bgcolor=colors["card_bg"],
    font=dict(family="Roboto, sans-serif", size=12, color=colors["text"]),
    xaxis=dict(
        title="Número de Compromisos",
        gridcolor=colors["grid"],
        zerolinecolor=colors["grid"],
    ),
    yaxis=dict(gridcolor=colors["grid"], zerolinecolor=colors["grid"]),
)

fig_area.write_html("area_chart.html")

# 4. Gráfico de componentes
fig_component = go.Figure()

for status in ["SÍ", "NO", "NO DETERMINADO"]:
    fig_component.add_trace(
        go.Bar(
            x=component_pivot["Componente minero"],
            y=component_pivot[status],
            name=status,
            marker=dict(color=colors[status]),
            hovertemplate="<b>%{x}</b><br>%{y} compromisos "
            + status.lower()
            + "<extra></extra>",
        )
    )

fig_component.update_layout(
    barmode="stack",
    showlegend=False,
    margin=dict(t=0, b=100, l=50, r=20),
    paper_bgcolor=colors["card_bg"],
    plot_bgcolor=colors["card_bg"],
    font=dict(family="Roboto, sans-serif", size=12, color=colors["text"]),
    xaxis=dict(tickangle=45, gridcolor=colors["grid"], zerolinecolor=colors["grid"]),
    yaxis=dict(
        title="Número de Compromisos",
        gridcolor=colors["grid"],
        zerolinecolor=colors["grid"],
    ),
)

fig_component.write_html("component_chart.html")

# 5. Gráfico de etapas
fig_phase = go.Figure()

for status in ["SÍ", "NO", "NO DETERMINADO"]:
    fig_phase.add_trace(
        go.Bar(
            x=phase_pivot["Etapa"],
            y=phase_pivot[status],
            name=status,
            marker=dict(color=colors[status]),
            hovertemplate="<b>%{x}</b><br>%{y} compromisos "
            + status.lower()
            + "<extra></extra>",
        )
    )

fig_phase.update_layout(
    barmode="stack",
    showlegend=False,
    margin=dict(t=0, b=50, l=50, r=20),
    paper_bgcolor=colors["card_bg"],
    plot_bgcolor=colors["card_bg"],
    font=dict(family="Roboto, sans-serif", size=12, color=colors["text"]),
    xaxis=dict(gridcolor=colors["grid"], zerolinecolor=colors["grid"]),
    yaxis=dict(
        title="Número de Compromisos",
        gridcolor=colors["grid"],
        zerolinecolor=colors["grid"],
    ),
)

fig_phase.write_html("phase_chart.html")

# Crear HTML para el dashboard
html_content = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard de Compromisos Ambientales</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Roboto', sans-serif;
        }
        body {
            background-color: #F1F3F5;
            color: #1D3557;
            padding: 20px;
        }
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: #FFFFFF;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            padding: 30px;
            overflow: hidden;
        }
        .dashboard-header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #E9ECEF;
        }
        .dashboard-title {
            font-size: 28px;
            font-weight: 700;
            color: #1D3557;
            margin-bottom: 5px;
        }
        .dashboard-subtitle {
            font-size: 14px;
            color: #457B9D;
            margin-bottom: 15px;
        }
        .stats-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .stat-card {
            background-color: #F8F9FA;
            border-radius: 8px;
            padding: 15px 20px;
            min-width: 180px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
            text-align: center;
            border-top: 3px solid #457B9D;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        }
        .stat-value {
            font-size: 24px;
            font-weight: 700;
            margin: 5px 0;
        }
        .stat-label {
            font-size: 12px;
            color: #457B9D;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-compliant .stat-value { color: #2E8B57; }
        .stat-non-compliant .stat-value { color: #E63946; }
        .stat-undetermined .stat-value { color: #FFB703; }

        .row {
            display: flex;
            flex-wrap: wrap;
            margin: -10px;
            margin-bottom: 20px;
        }
        .col {
            padding: 10px;
        }
        .col-6 {
            width: 50%;
        }
        .col-12 {
            width: 100%;
        }
        .chart-card {
            background-color: #F8F9FA;
            border-radius: 8px;
            padding: 20px;
            height: 100%;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .chart-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        }
        .chart-title {
            font-size: 16px;
            font-weight: 500;
            margin-bottom: 15px;
            color: #1D3557;
            border-bottom: 1px solid #E9ECEF;
            padding-bottom: 8px;
        }
        .chart-container {
            width: 100%;
            height: 300px;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #E9ECEF;
            color: #457B9D;
            font-size: 12px;
        }
        @media (max-width: 768px) {
            .col-6 {
                width: 100%;
            }
            .stat-card {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="dashboard-header">
            <h1 class="dashboard-title">Dashboard de Compromisos Ambientales</h1>
            <p class="dashboard-subtitle">Monitoreo y seguimiento de compromisos ambientales | Actualizado: DATE_PLACEHOLDER</p>

            <div class="stats-container">
                <div class="stat-card">
                    <div class="stat-label">Total de Compromisos</div>
                    <div class="stat-value">TOTAL_PLACEHOLDER</div>
                </div>
                <div class="stat-card stat-compliant">
                    <div class="stat-label">Cumplidos</div>
                    <div class="stat-value">COMPLIANT_PLACEHOLDER</div>
                </div>
                <div class="stat-card stat-non-compliant">
                    <div class="stat-label">No Cumplidos</div>
                    <div class="stat-value">NON_COMPLIANT</div>
                </div>
                <div class="stat-card stat-undetermined">
                    <div class="stat-label">No Determinados</div>
                    <div class="stat-value">UNDETERMINED_PLACEHOLDER</div>
                </div>
                <div class="stat-card stat-compliant">
                    <div class="stat-label">% Cumplimiento</div>
                    <div class="stat-value">PERCENTAGE_PLACEHOLDER</div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col col-6">
                <div class="chart-card">
                    <h3 class="chart-title">Estado de Cumplimiento</h3>
                    <div class="chart-container" id="pie-chart"></div>
                </div>
            </div>
            <div class="col col-6">
                <div class="chart-card">
                    <h3 class="chart-title">Porcentaje de Cumplimiento</h3>
                    <div class="chart-container" id="gauge-chart"></div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col col-12">
                <div class="chart-card">
                    <h3 class="chart-title">Cumplimiento por Área Responsable</h3>
                    <div class="chart-container" id="area-chart" style="height: 350px;"></div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col col-6">
                <div class="chart-card">
                    <h3 class="chart-title">Cumplimiento por Componente Minero</h3>
                    <div class="chart-container" id="component-chart"></div>
                </div>
            </div>
            <div class="col col-6">
                <div class="chart-card">
                    <h3 class="chart-title">Cumplimiento por Etapa</h3>
                    <div class="chart-container" id="phase-chart"></div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>© YEAR_PLACEHOLDER Dashboard de Compromisos Ambientales | Desarrollado para el seguimiento de compromisos ambientales mineros</p>
        </div>
    </div>

    <script>
        // Cargar gráficos desde archivos HTML individuales usando iframes
        document.addEventListener('DOMContentLoaded', function() {
            const charts = {
                'pie-chart': 'pie_chart.html',
                'gauge-chart': 'gauge_chart.html',
                'area-chart': 'area_chart.html',
                'component-chart': 'component_chart.html',
                'phase-chart': 'phase_chart.html'
            };

            for (const [divId, htmlFile] of Object.entries(charts)) {
                const iframe = document.createElement('iframe');
                iframe.src = htmlFile;
                iframe.style.width = '100%';
                iframe.style.height = '100%';
                iframe.style.border = 'none';
                iframe.style.overflow = 'hidden';

                const container = document.getElementById(divId);
                container.appendChild(iframe);
            }
        });
    </script>
</body>
</html>
"""

# Reemplazar placeholders
current_date = pd.Timestamp.now().strftime("%d/%m/%Y")
current_year = pd.Timestamp.now().year
percentage_formatted = f"{compliance_percentage:.1f}%"

html_content = html_content.replace("DATE_PLACEHOLDER", current_date)
html_content = html_content.replace("TOTAL_PLACEHOLDER", str(total_commitments))
html_content = html_content.replace("COMPLIANT_PLACEHOLDER", str(compliant))
html_content = html_content.replace("NON_COMPLIANT", str(non_compliant))
html_content = html_content.replace("UNDETERMINED_PLACEHOLDER", str(undetermined))
html_content = html_content.replace("PERCENTAGE_PLACEHOLDER", percentage_formatted)
html_content = html_content.replace("YEAR_PLACEHOLDER", str(current_year))

# Guardar el dashboard como un archivo HTML
with open("dashboard_compromisos_ordenado.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Dashboard ordenado creado y guardado como 'dashboard_compromisos_ordenado.html'")
print("También se han creado archivos HTML individuales para cada gráfico.")
