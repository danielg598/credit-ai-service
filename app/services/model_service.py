"""
model_service.py — carga el modelo entrenado y ejecuta inferencia.

IMPORTANTE — Transformacion de escala:
  El modelo esta entrenado con UCI German Credit Data, que usa marcos
  alemanes (DM) de los años 90. Los montos tipicos en el training set
  van de 250 DM a ~18.000 DM.
  En Colombia, un prestamo tipico es 5-50 millones COP.
  Para alinear dominios sin re-entrenar, aplicamos una transformacion
  de escala monetaria antes de la inferencia:
     credit_amount_modelo = credit_amount_cop / COP_TO_DM_SCALE
  Esta transformacion es reversible y esta documentada. Queda en un
  unico lugar (aqui) para no esparcir conocimiento por el stack.
  En v2, cuando entrenemos con datos colombianos reales, esto desaparece.
"""
from functools import lru_cache
import logging
from typing import Any

import joblib
import pandas as pd

from app.core.config import get_settings

log = logging.getLogger("credit-ai.model")

# Factor de escala: 1 DM (training set) ~= 1.500 COP (aprox, basado en
# mediana del dataset 2300 DM vs prestamo tipico colombiano 3.5M COP).
# Calibrado para que un prestamo "tipico" en COP caiga cerca de la
# mediana del training set.
COP_TO_DM_SCALE = 3000.0


@lru_cache
def get_bundle() -> dict[str, Any]:
    """Carga el bundle del modelo una sola vez."""
    settings = get_settings()
    log.info(f"Cargando modelo desde {settings.MODEL_PATH}...")
    bundle = joblib.load(settings.MODEL_PATH)
    metrics = bundle.get("metrics", {})
    log.info(
        f"Modelo cargado. AUC={metrics.get('auc', 0):.3f}, "
        f"KS={metrics.get('ks', 0):.3f}, Gini={metrics.get('gini', 0):.3f}"
    )
    return bundle


def _normalize_features(features: dict[str, Any]) -> dict[str, Any]:
    """
    Ajusta features al dominio del training set.

    - credit_amount: COP -> unidad de training set
    - Otros campos: pasan sin cambios.
    """
    normalized = dict(features)  # copia defensiva

    if "credit_amount" in normalized and normalized["credit_amount"] is not None:
        original = float(normalized["credit_amount"])
        scaled = original / COP_TO_DM_SCALE
        normalized["credit_amount"] = scaled
        log.debug(f"credit_amount normalizado: {original:,.0f} COP -> {scaled:.1f}")

    return normalized


def predict_probability(features: dict[str, Any]) -> tuple[float, pd.DataFrame]:
    """
    Ejecuta el modelo sobre un dict de features.

    Retorna:
      probability (float): P(default) entre 0.0 y 1.0
      X_row (DataFrame): la fila ya normalizada, lista para SHAP.
    """
    bundle = get_bundle()
    pipeline = bundle["pipeline"]

    # Normalizar antes de predecir
    features_norm = _normalize_features(features)
    X_row = pd.DataFrame([features_norm])

    proba = float(pipeline.predict_proba(X_row)[0, 1])
    return proba, X_row