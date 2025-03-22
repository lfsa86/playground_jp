from typing import List, Optional

from pydantic import BaseModel, Field


class ComponenteAsociado(BaseModel):
    componente: str


class Reportabilidad(BaseModel):
    valor: str


class FrecuenciaDeReporte(BaseModel):
    valor: str


class Componentes(BaseModel):
    componente_operativo_asociado: List[ComponenteAsociado] = Field(
        description="Componente operativo asociado"
    )
    componente_ambiental_asociado: List[ComponenteAsociado] = Field(
        description="Componente ambiental asociado"
    )
    reportabilidad: Reportabilidad
    frecuencia_de_reporte: Optional[FrecuenciaDeReporte] = None


class Statement(BaseModel):
    """Model for environmental statement classification."""

    category: str = Field(
        description="Generalidades, Implicaciones del Proyecto, Actividades, Riesgos e Impactos, Compromisos, Permisos"
    )
    justification: str = (
        Field(description="Explicación breve del motivo de la clasificación."),
    )
    synthesis: str = Field(
        description="Sintesis extraida del texto acerca de la categorizacion del enunciado"
    )
    elements: List[str] = Field(
        description="Elementos relacionados con el desarrollo del proyecto, impacto o compromiso ambiental"
    )


class Compromiso(BaseModel):
    category: str = Field(..., description="Categoría del compromiso")
    location: Optional[str] = Field(
        None,
        description="Ubicacion geografica del compromiso (Puede ser None si no se especifica)",
    )
    use_case: str = Field(..., description="Explicacion del caso de uso del item")
    description: str = Field(
        ...,
        description="Descripcion y relaciones que tiene el componente con el medio ambiente y el proyecto por desarrollar",
    )
    commitment: str = Field(
        ...,
        description="Sintesis del compromiso ambiental identificado en relacion al componente",
    )
    tipology: Optional[str] = Field(
        None,
        description="Tipo de análisis o muestreo realizado (ej. Puede ser None si no se especifica.",
    )
    zonification: Optional[str] = Field(
        None,
        description="Área geográfica donde se aplica el muestreo. Puede ser None si no se especifica.",
    )
    related_metrics: Optional[str] = Field(
        None,
        description="Métrica utilizada para medir o evaluar el suelo. Puede ser None si no se especifica.",
    )
    reportability: Optional[str] = Field(
        None,
        description="Información sobre cómo y dónde se reportan los resultados. Puede ser None si no se especifica.",
    )
    fqreportability: Optional[str] = Field(
        None,
        description="Frecuencia con la que se reportan los resultados. Puede ser None si no se especifica.",
    )
    time_of_execution: Optional[str] = Field(
        None,
        description="Momento o fase en el que se ejecuta el compromiso (ej., antes de la construcción)",
    )
    general_objective: str = Field(
        ...,
        description="El objetivo General del compromiso, explicado de manera precisa y directa",
    )
