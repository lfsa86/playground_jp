import pandas as pd
import plotly.express as px
import vizro.models as vm
import vizro.plotly.express as vx
from vizro import Vizro

# Load the CSV file
df = pd.read_csv("data/processed/rca/commitments/extracted_commitments.csv")

# Clean up the data
categorical_cols = [
    "category",
    "location",
    "use_case",
    "tipology",
    "zonification",
    "reportability",
    "fqreportability",
]
for col in categorical_cols:
    df[col] = df[col].fillna("No especificado")


# Create a new column for execution phase
def categorize_phase(time_str):
    if pd.isna(time_str):
        return "No especificado"
    time_str = str(time_str).lower()
    if "construcción" in time_str:
        if "operación" in time_str:
            if "cierre" in time_str:
                return "Construcción, Operación y Cierre"
            return "Construcción y Operación"
        return "Construcción"
    elif "operación" in time_str:
        if "cierre" in time_str:
            return "Operación y Cierre"
        return "Operación"
    elif "cierre" in time_str:
        return "Cierre"
    else:
        return "Otro"


df["execution_phase"] = df["time_of_execution"].apply(categorize_phase)

# Prepare DataFrames for visualizations
category_df = df["category"].value_counts().reset_index()
category_df.columns = ["category_name", "count"]

phase_df = df["execution_phase"].value_counts().reset_index()
phase_df.columns = ["phase", "count"]

location_df = df["location"].value_counts().reset_index().head(10)
location_df.columns = ["location", "count"]

usecase_df = df["use_case"].value_counts().reset_index().head(10)
usecase_df.columns = ["use_case", "count"]

# Create the dashboard
dashboard = vm.Dashboard(
    title="Dashboard de Compromisos Ambientales",
    pages=[
        vm.Page(
            id="vision_general",
            title="Visión General",
            components=[
                vm.Card(
                    id="intro_card",
                    text="""
                    # Dashboard de Compromisos Ambientales

                    Este dashboard presenta un análisis de los compromisos ambientales extraídos del archivo CSV.
                    Explore las diferentes visualizaciones para entender mejor la distribución y características de los compromisos.
                    """,
                ),
                vm.Graph(
                    id="category_graph",
                    figure=vx.bar(
                        data_frame=category_df,
                        x="category_name",
                        y="count",
                        title="Compromisos por Categoría",
                        labels={"category_name": "Categoría", "count": "Cantidad"},
                        color="category_name",
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        text="count",
                    ),
                ),
                vm.Graph(
                    id="phase_graph",
                    figure=vx.pie(
                        data_frame=phase_df,
                        names="phase",
                        values="count",
                        title="Compromisos por Fase de Ejecución",
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        hole=0.3,
                    ),
                ),
            ],
        ),
        vm.Page(
            id="analisis_ubicacion",
            title="Análisis por Ubicación",
            components=[
                vm.Graph(
                    id="location_graph",
                    figure=vx.bar(
                        data_frame=location_df,
                        x="count",
                        y="location",
                        title="Top 10 Ubicaciones con más Compromisos",
                        labels={"count": "Cantidad", "location": "Ubicación"},
                        orientation="h",
                        color="count",
                        color_continuous_scale="Viridis",
                        text="count",
                    ),
                ),
                vm.Graph(
                    id="location_scatter",
                    figure=vx.scatter(
                        data_frame=df,
                        x="location",
                        y="compromiso",
                        color="execution_phase",
                        title="Detalles de Compromisos por Ubicación",
                        labels={
                            "location": "Ubicación",
                            "compromiso": "Compromiso",
                            "execution_phase": "Fase de Ejecución",
                        },
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                    ),
                ),
            ],
        ),
        vm.Page(
            id="analisis_caso_uso",
            title="Análisis por Caso de Uso",
            components=[
                vm.Graph(
                    id="usecase_graph",
                    figure=vx.bar(
                        data_frame=usecase_df,
                        x="count",
                        y="use_case",
                        title="Top 10 Casos de Uso con más Compromisos",
                        labels={"count": "Cantidad", "use_case": "Caso de Uso"},
                        orientation="h",
                        color="count",
                        color_continuous_scale="Viridis",
                        text="count",
                    ),
                ),
                vm.Graph(
                    id="usecase_scatter",
                    figure=vx.scatter(
                        data_frame=df,
                        x="use_case",
                        y="compromiso",
                        color="execution_phase",
                        title="Detalles de Compromisos por Caso de Uso",
                        labels={
                            "use_case": "Caso de Uso",
                            "compromiso": "Compromiso",
                            "execution_phase": "Fase de Ejecución",
                        },
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                    ),
                ),
            ],
        ),
    ],
)

# Crear un HTML simple que muestre todos los gráficos
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard de Compromisos Ambientales</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        h1 { color: #333; text-align: center; }
        .dashboard { display: flex; flex-wrap: wrap; justify-content: center; }
        .chart { width: 48%; margin: 1%; border: 1px solid #ddd; padding: 10px; box-sizing: border-box; background-color: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        iframe { width: 100%; height: 400px; border: none; }
        .intro { width: 98%; margin: 1%; padding: 15px; background-color: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        @media (max-width: 768px) {
            .chart { width: 98%; }
        }
    </style>
</head>
<body>
    <h1>Dashboard de Compromisos Ambientales</h1>
    
    <div class="intro">
        <h2>Visión General</h2>
        <p>Este dashboard presenta un análisis de los compromisos ambientales extraídos del archivo CSV. 
        Explore las diferentes visualizaciones para entender mejor la distribución y características de los compromisos.</p>
    </div>

    <div class="dashboard">
        <div class="chart">
            <h2>Compromisos por Categoría</h2>
            <iframe src="category_chart.html"></iframe>
        </div>
        <div class="chart">
            <h2>Compromisos por Fase de Ejecución</h2>
            <iframe src="phase_chart.html"></iframe>
        </div>
        <div class="chart">
            <h2>Top 10 Ubicaciones con más Compromisos</h2>
            <iframe src="location_chart.html"></iframe>
        </div>
        <div class="chart">
            <h2>Top 10 Casos de Uso con más Compromisos</h2>
            <iframe src="usecase_chart.html"></iframe>
        </div>
    </div>
</body>
</html>
"""

# Create regular Plotly figures for individual HTML exports with improved styling
fig_category = px.bar(
    category_df,
    x="category_name",
    y="count",
    title="Compromisos por Categoría",
    labels={"category_name": "Categoría", "count": "Cantidad"},
    color="category_name",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    text="count",
)
fig_category.update_layout(
    xaxis_tickangle=-45,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
    margin=dict(l=50, r=20, t=50, b=100),
)
fig_category.update_traces(textposition="outside")
fig_category.write_html("category_chart.html")

fig_phase = px.pie(
    phase_df,
    names="phase",
    values="count",
    title="Compromisos por Fase de Ejecución",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    hole=0.3,
)
fig_phase.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
)
fig_phase.update_traces(textinfo="percent+label+value")
fig_phase.write_html("phase_chart.html")

fig_location = px.bar(
    location_df,
    x="count",
    y="location",
    title="Top 10 Ubicaciones con más Compromisos",
    labels={"count": "Cantidad", "location": "Ubicación"},
    orientation="h",
    color="count",
    color_continuous_scale="Viridis",
    text="count",
)
fig_location.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
    margin=dict(l=200, r=20, t=50, b=20),
)
fig_location.update_traces(textposition="outside")
fig_location.write_html("location_chart.html")

fig_usecase = px.bar(
    usecase_df,
    x="count",
    y="use_case",
    title="Top 10 Casos de Uso con más Compromisos",
    labels={"count": "Cantidad", "use_case": "Caso de Uso"},
    orientation="h",
    color="count",
    color_continuous_scale="Viridis",
    text="count",
)
fig_usecase.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
    margin=dict(l=200, r=20, t=50, b=20),
)
fig_usecase.update_traces(textposition="outside")
fig_usecase.write_html("usecase_chart.html")

# Guardar el HTML
with open("dashboard_manual.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Dashboard manual guardado como dashboard_manual.html")

# Intentar ejecutar el dashboard de Vizro

app = Vizro()
app.build(dashboard=dashboard).run()
