"""
Schemas Pydantic para el endpoint POST /predict.
"""
from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request desde el Spring Boot — alineado con German Credit renombrado."""
    model_config = {"extra": "allow"}

    # --- Numericas ---
    duration: int = Field(..., ge=6, le=120, description="Plazo en meses")
    credit_amount: float = Field(..., gt=0, description="Monto solicitado")
    installment_rate: int = Field(3, ge=1, le=4)
    residence_since: int = Field(2, ge=1, le=4)
    age: int = Field(..., ge=18, le=75)
    existing_credits: int = Field(1, ge=1, le=4)
    dependents: int = Field(1, ge=1, le=2)

    # --- Categoricas (codigos A** del dataset) ---
    checking_status: str = Field("A14")
    credit_history:  str = Field("A32")
    purpose:         str = Field("A43")
    savings:         str = Field("A63")
    employment:      str = Field("A73")
    personal_status: str = Field("A93")
    other_debtors:   str = Field("A101")
    property:        str = Field("A123")
    other_installment_plans: str = Field("A143")
    housing:         str = Field("A152")
    job:             str = Field("A173")
    telephone:       str = Field("A192")
    foreign_worker:  str = Field("A201")


class FactorClave(BaseModel):
    factor: str
    impacto: float
    descripcion: str
    direction: Literal["AUMENTA_RIESGO", "REDUCE_RIESGO"]


class PredictResponse(BaseModel):
    score: int = Field(..., ge=0, le=1000)
    probabilidad_default: float = Field(..., ge=0.0, le=1.0)
    decision: Literal["APROBADO", "RECHAZADO"]
    factores_clave: list[FactorClave]
    explicacion: str
    model_version: str = "logreg-german-v1.0"
    latencia_ms: int = 0