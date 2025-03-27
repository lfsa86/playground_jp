from typing import List, Optional

from pydantic import BaseModel, Field


class Statement(BaseModel):
    """Model for environmental statement classification."""

    category: str = Field(
        description="Generalidades, Implicaciones del Proyecto, Actividades, Riesgos e Impactos, Compromisos, Permisos"
    )
    justification: str = Field(
        description="Explicación breve del motivo de la clasificación."
    )
    synthesis: str = Field(
        description="Sintesis extraida del texto acerca de la categorizacion del enunciado"
    )


class Commitment(BaseModel):
    """Model for environmental commitment classification."""

    summary: str = Field(
        description="Resumen del explicativo del compromiso"
    )
    coa: Optional[str] = Field(None, description="Componente operativo asociado")
    caa: Optional[str] = Field(
        None, description="Componente ambiental asociado (aire, agua, ruido, etc)"
    )
    fase_aplicacion: Optional[str] = Field(
        None,
        description="Fase de aplicacion del compromiso (construccion, operacion, cierre, todas, etc)",
    )
    frecuencia_reporte: Optional[str] = Field(None, description="Frecuencia de reporte")
