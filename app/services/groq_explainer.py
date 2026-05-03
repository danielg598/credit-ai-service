"""
groq_explainer.py — genera explicaciones en español con Llama 3.3 70B (Groq).

Prompt engineering:
  - Rol: analista de riesgo en banca colombiana.
  - Input: datos del solicitante + decision del modelo + top factores SHAP.
  - Output: JSON estructurado con explicacion, factores positivos/negativos
            y recomendacion para el cliente.
  - Tono: profesional pero entendible, en español Colombia.

Retry: tenacity reintenta 3 veces con backoff exponencial si Groq responde
con 5xx o rate limit. Fallback: si Groq cae, generamos una explicacion
basica on-the-fly (no se bloquea el flujo).
"""
import json
import logging
from typing import Any

from groq import Groq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings

log = logging.getLogger("credit-ai.groq")

SYSTEM_PROMPT = """Eres un analista senior de riesgo crediticio en Colombia.
Tu tarea: explicar en español claro y profesional la decision de un modelo de machine
learning sobre una solicitud de credito.

REGLAS ESTRICTAS:
1. Basate UNICAMENTE en los datos proporcionados. No inventes informacion.
2. Maximo 4-5 frases en el campo "explicacion".
3. Usa un tono profesional pero entendible para un cliente no tecnico.
4. Escribe siempre en español (ES-CO).
5. Retorna EXCLUSIVAMENTE un JSON valido con este esquema exacto:
{
  "explicacion": "<texto en español de 3-5 frases>",
  "recomendacion_cliente": "<consejo breve para el solicitante>"
}

No agregues texto fuera del JSON. No uses markdown. Solo el JSON."""


# Cliente Groq — singleton module-level.
_settings = get_settings()
_client = Groq(api_key=_settings.GROQ_API_KEY, timeout=15.0)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_groq(system: str, user: str) -> str:
    """Llama a Groq con retry. Separada para que tenacity la instrumente."""
    response = _client.chat.completions.create(
        model=_settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.3,            # baja = mas determinista
        max_completion_tokens=500,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or "{}"


def explain_decision(
    applicant: dict[str, Any],
    probability: float,
    threshold: float,
    decision: str,
    shap_factors: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Retorna un dict con:
      - explicacion: str (3-5 frases en español)
      - recomendacion_cliente: str (consejo breve)

    Si Groq falla despues de 3 retries, retorna una explicacion fallback
    generada localmente (no bloquea el flujo).
    """
    # Construccion del prompt de usuario
    datos_str = "\n".join(f"  - {k}: {v}" for k, v in applicant.items())
    factores_str = "\n".join(
        f"  {i+1}. {f['descripcion']} "
        f"(SHAP={f['shap_value']:+.3f}, {f['direction'].lower().replace('_', ' ')})"
        for i, f in enumerate(shap_factors)
    )

    user_prompt = f"""Analiza esta solicitud de credito:

DATOS DEL SOLICITANTE:
{datos_str}

RESULTADO DEL MODELO:
  - Decision: {decision}
  - Probabilidad de default: {probability:.1%}
  - Umbral de aprobacion: {threshold:.0%}

TOP FACTORES INFLUYENTES (analisis SHAP):
{factores_str}

Genera la explicacion en el JSON del esquema definido."""

    try:
        raw = _call_groq(SYSTEM_PROMPT, user_prompt)
        data = json.loads(raw)
        return {
            "explicacion": data.get("explicacion", "").strip(),
            "recomendacion_cliente": data.get("recomendacion_cliente", "").strip(),
        }
    except Exception as e:
        log.warning(f"Groq fallo despues de retries: {e}. Usando fallback local.")
        return _fallback_explanation(decision, probability, shap_factors)


def _fallback_explanation(
    decision: str,
    probability: float,
    shap_factors: list[dict[str, Any]],
) -> dict[str, str]:
    """Explicacion basica cuando Groq no responde — NO bloquea el flujo."""
    top_reduce = [f for f in shap_factors if f["direction"] == "REDUCE_RIESGO"][:2]
    top_aumenta = [f for f in shap_factors if f["direction"] == "AUMENTA_RIESGO"][:2]

    parts = [
        f"La solicitud fue evaluada con probabilidad de default de {probability:.1%}."
    ]
    if decision == "APROBADO":
        parts.append("Decision: APROBADO.")
        if top_reduce:
            motivos = " y ".join(f["descripcion"].lower() for f in top_reduce)
            parts.append(f"Los factores que reducen el riesgo incluyen {motivos}.")
    else:
        parts.append("Decision: RECHAZADO.")
        if top_aumenta:
            motivos = " y ".join(f["descripcion"].lower() for f in top_aumenta)
            parts.append(f"Los factores que elevan el riesgo incluyen {motivos}.")

    return {
        "explicacion": " ".join(parts),
        "recomendacion_cliente": "Consulta con un asesor para revisar tu perfil crediticio.",
    }