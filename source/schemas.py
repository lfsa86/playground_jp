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
            description="Resumen breve, claro y específico del compromiso ambiental declarado por el titular del proyecto. Evita frases genéricas."
        )
    coa: Optional[str] = Field(
            None,
            description="Componente operativo minero al que se refiere el compromiso. Ejemplos: taller de camiones, deposito de desmonte, planta de cal, planta de lixiviación, toma de agua, PTARD  ."
        )
    caa: Optional[str] = Field(
            None,
            description="Componente ambiental afectado o protegido por el compromiso. Evita terminos genericos ya que hay diferencia entre agua subterranea, agua superfial, lagunas o similares. Ejemplos: aire, agua, suelo, biodiversidad, ruido, paisaje, etc."
        )
    fase_aplicacion: Optional[str] = Field(
            None,
            description="Fase del proyecto donde se aplica el compromiso. Puede ser: construcción, operación, cierre, post cierre o todas las fases."
        )
    frecuencia_reporte: Optional[str] = Field(
            None,
            description="Frecuencia con la que se debe reportar el cumplimiento del compromiso. Ejemplos: mensual, trimestral, anual, antes del inicio de obras, etc."
        )
    ubicacion: Optional[str] = Field(None, description="Lugar geográfico o infraestructura asociada al compromiso (por ejemplo: Mina, Línea de transmisión, Puerto, Todas las ubicaciones, etc.)")
    tematica: Optional[str] = Field(None, description="Temática asociada al compromiso. Puede incluir temas como diseño de componentes operativos, eficiencia energética, monitoreo, etc.")
    dificultad_cumplimiento: Optional[str] = Field(None, description="Nivel de dificultad del cumplimiento: puede ser rutina operativa o requerir coordinación y servicios especializados. Catalogalo como Sencillo, Moderado o Complejo")
        

class MultipleCommitments(BaseModel):
    """Modelo para múltiples compromisos extraídos desde un mismo enunciado."""
    compromisos: List[Commitment] = Field(
        default_factory=list,
        description="Lista de compromisos identificados en un bloque de texto"
    )