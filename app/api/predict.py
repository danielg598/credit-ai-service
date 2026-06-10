"""
Endpoint POST /predict — orquesta el pipeline de evaluacion crediticia.

Flujo:
  1. Recibe features del Spring Boot (PredictRequest)
  2. Inferencia del modelo sklearn       -> probabilidad de default
  3. SHAP local                          -> top 5 factores influyentes
  4. Decision                             -> APROBADO si prob < umbral
  5. Groq Llama 3.3                       -> explicacion en español
  6. Ensambla PredictResponse

Latencia esperada: ~400-800 ms (Groq es el dominante, ~300-600 ms).
"""
import logging
import time

from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.schemas.predict import FactorClave, PredictRequest, PredictResponse
from app.services.groq_explainer import explain_decision
from app.services.model_service import predict_probability
from app.services.shap_service import top_contributing_factors
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

log = logging.getLogger("credit-ai.api")
router = APIRouter(tags=["Predict"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Evalua solicitud con modelo ML + SHAP + LLM",
)
def predict(request: PredictRequest) -> PredictResponse:
    """Pipeline completo: ML -> SHAP -> LLM."""
    t0 = time.time()
    settings = get_settings()

    features = request.model_dump()
    log.info(
        f"POST /predict duration={features.get('duration')}, "
        f"amount={features.get('credit_amount')}, age={features.get('age')}"
    )

    try:
        # --- 1. Inferencia del modelo ---
        with tracer.start_as_current_span("model.infer") as span:
            probability, X_row = predict_probability(features)
            span.set_attribute("credit.probability_default", probability)

        # --- 2. Decision segun umbral ---
        decision = "APROBADO" if probability < settings.DEFAULT_THRESHOLD else "RECHAZADO"
        score = int((1.0 - probability) * 1000)

        # --- 3. SHAP: top 5 factores ---
        with tracer.start_as_current_span("shap.explain"):
            shap_factors = top_contributing_factors(X_row, top_n=5)

        # --- 4. Groq: explicacion en español ---
        with tracer.start_as_current_span("groq.explain") as span:
            span.set_attribute("credit.decision", decision)
            span.set_attribute("credit.score", score)
            explanation = explain_decision(
                applicant=features,
                probability=probability,
                threshold=settings.DEFAULT_THRESHOLD,
                decision=decision,
                shap_factors=shap_factors,
            )

        # --- 5. Ensamblar response ---
        factores_dto = [
            FactorClave(
                factor=f["feature"],
                impacto=f["shap_value"],
                descripcion=f["descripcion"],
                direction=f["direction"],
            )
            for f in shap_factors
        ]

        latencia = int((time.time() - t0) * 1000)
        log.info(
            f"OK decision={decision} score={score} pd={probability:.3f} "
            f"latencia={latencia}ms"
        )

        return PredictResponse(
            score=score,
            probabilidad_default=probability,
            decision=decision,  # type: ignore  (literal match)
            factores_clave=factores_dto,
            explicacion=explanation["explicacion"],
            latencia_ms=latencia,
        )

    except Exception as e:
        log.exception("Error en /predict")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )