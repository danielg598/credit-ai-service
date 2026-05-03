"""
shap_service.py — explicabilidad local con SHAP.

Para modelos lineales como Logistic Regression, SHAP.LinearExplainer es
el metodo optimo (O(1) por prediccion vs O(n) del KernelExplainer).

Se inicializa con un "background" set de 200 muestras aleatorias del
dataset original. Esto es estandar en SHAP: representa la distribucion
"promedio" contra la cual comparar una solicitud individual.

Descripciones legibles: mapeamos los nombres tecnicos del dataset
(Attribute1 = "checking_status") a textos entendibles por un cliente.
"""
import logging
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import shap
from ucimlrepo import fetch_ucirepo

from app.services.model_service import get_bundle

log = logging.getLogger("credit-ai.shap")


# Descripciones legibles de cada feature (para el cliente final).
# Los codigos (A11, A14, ...) son del dataset German Credit UCI — los
# convertimos a texto claro en ES-CO.
FEATURE_DESCRIPTIONS = {
    # --- Numericas ---
    "num__duration":          "Plazo del prestamo en meses",
    "num__credit_amount":     "Monto del prestamo solicitado",
    "num__installment_rate":  "Peso de la cuota sobre el ingreso disponible",
    "num__age":               "Edad del solicitante",
    "num__residence_since":   "Antigüedad en el domicilio actual",
    "num__existing_credits":  "Numero de creditos vigentes",
    "num__dependents":        "Numero de personas a cargo",

    # --- Categoricas: checking_status ---
    "cat__checking_status_A11": "Sin cuenta corriente o con saldo negativo",
    "cat__checking_status_A12": "Cuenta corriente con saldo bajo",
    "cat__checking_status_A13": "Cuenta corriente con saldo positivo",
    "cat__checking_status_A14": "No tiene cuenta corriente",

    # --- credit_history ---
    "cat__credit_history_A30": "Sin creditos previos",
    "cat__credit_history_A31": "Todos los creditos pagados",
    "cat__credit_history_A32": "Historial crediticio al dia",
    "cat__credit_history_A33": "Retraso en pago previo",
    "cat__credit_history_A34": "Historial crediticio critico",

    # --- purpose ---
    "cat__purpose_A40": "Proposito: vehiculo nuevo",
    "cat__purpose_A41": "Proposito: vehiculo usado",
    "cat__purpose_A42": "Proposito: muebles/electrodomesticos",
    "cat__purpose_A43": "Proposito: electrodomesticos/electronica",
    "cat__purpose_A44": "Proposito: electrodomesticos",
    "cat__purpose_A45": "Proposito: reparaciones",
    "cat__purpose_A46": "Proposito: educacion",
    "cat__purpose_A48": "Proposito: capacitacion profesional",
    "cat__purpose_A49": "Proposito: negocio propio",
    "cat__purpose_A410": "Proposito: otros",

    # --- savings ---
    "cat__savings_A61": "Ahorros < 100 EUR",
    "cat__savings_A62": "Ahorros 100-500 EUR",
    "cat__savings_A63": "Ahorros 500-1000 EUR",
    "cat__savings_A64": "Ahorros > 1000 EUR",
    "cat__savings_A65": "Sin ahorros registrados",

    # --- employment ---
    "cat__employment_A71": "Desempleado",
    "cat__employment_A72": "Empleado menos de 1 año",
    "cat__employment_A73": "Empleado entre 1-4 años",
    "cat__employment_A74": "Empleado entre 4-7 años",
    "cat__employment_A75": "Empleado mas de 7 años",

    # --- personal_status ---
    "cat__personal_status_A91": "Hombre divorciado/separado",
    "cat__personal_status_A92": "Mujer divorciada/separada/casada",
    "cat__personal_status_A93": "Hombre soltero",
    "cat__personal_status_A94": "Hombre casado/viudo",
    "cat__personal_status_A95": "Mujer soltera",

    # --- housing ---
    "cat__housing_A151": "Alquila vivienda",
    "cat__housing_A152": "Vivienda propia",
    "cat__housing_A153": "Vivienda sin costo (familiar)",

    # --- job ---
    "cat__job_A171": "Desempleado o sin cualificacion (no residente)",
    "cat__job_A172": "Sin cualificacion (residente)",
    "cat__job_A173": "Empleado cualificado",
    "cat__job_A174": "Directivo/profesional independiente",

    # --- foreign_worker ---
    "cat__foreign_worker_A201": "Trabajador extranjero",
    "cat__foreign_worker_A202": "Residente local",
}


def _describe_feature(feature_name: str) -> str:
    """Retorna descripcion legible o, si no existe, limpia el nombre tecnico."""
    if feature_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature_name]
    # Fallback: quita prefijos tecnicos y devuelve algo al menos legible
    clean = feature_name.replace("num__", "").replace("cat__", "").replace("_", " ")
    return f"Factor: {clean}"


@lru_cache
def get_explainer() -> tuple[shap.LinearExplainer, list[str]]:
    """
    Construye el SHAP explainer una sola vez (cacheado).

    El background son 200 muestras del dataset original transformadas
    con el mismo preprocessor del modelo — representa "la solicitud
    promedio" contra la cual se compara cada solicitud individual.
    """
    log.info("Inicializando SHAP LinearExplainer...")

    bundle = get_bundle()
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]

    # Descargar un subset del dataset para el background
    german = fetch_ucirepo(id=144)
    X_full = german.data.features.copy()

    # Mismo renombrado que en train.py — mantener sincronizado.
    X_full = X_full.rename(columns={
        "Attribute1":  "checking_status",
        "Attribute2":  "duration",
        "Attribute3":  "credit_history",
        "Attribute4":  "purpose",
        "Attribute5":  "credit_amount",
        "Attribute6":  "savings",
        "Attribute7":  "employment",
        "Attribute8":  "installment_rate",
        "Attribute9":  "personal_status",
        "Attribute10": "other_debtors",
        "Attribute11": "residence_since",
        "Attribute12": "property",
        "Attribute13": "age",
        "Attribute14": "other_installment_plans",
        "Attribute15": "housing",
        "Attribute16": "existing_credits",
        "Attribute17": "job",
        "Attribute18": "dependents",
        "Attribute19": "telephone",
        "Attribute20": "foreign_worker",
    })
    X_sample = X_full.sample(200, random_state=42)

    # Transformar con el mismo preprocessor del modelo
    preprocessor = pipeline.named_steps["preprocessor"]
    X_bg_transformed = preprocessor.transform(X_sample)

    # LinearExplainer: optimo para logistic regression
    classifier = pipeline.named_steps["classifier"]
    explainer = shap.LinearExplainer(classifier, X_bg_transformed)

    log.info(f"SHAP listo. Background: {X_bg_transformed.shape[0]} muestras.")
    return explainer, feature_names


def top_contributing_factors(X_row: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    """
    Para una solicitud, retorna los top N factores que mas movieron la prediccion.

    Cada factor incluye:
      - feature: nombre tecnico (ej: "cat__Attribute3_A32")
      - descripcion: texto legible (ej: "Historial crediticio al dia")
      - shap_value: valor SHAP (positivo=aumenta_riesgo, negativo=reduce_riesgo)
      - direction: "AUMENTA_RIESGO" | "REDUCE_RIESGO"
    """
    explainer, feature_names = get_explainer()
    bundle = get_bundle()
    preprocessor = bundle["pipeline"].named_steps["preprocessor"]

    # Transformar la fila
    X_transformed = preprocessor.transform(X_row)

    # Calcular SHAP values
    shap_values = explainer.shap_values(X_transformed)[0]  # [0] porque es una sola fila

    # Ordenar por magnitud absoluta y tomar top_n
    top_indices = np.argsort(np.abs(shap_values))[::-1][:top_n]

    factors = []
    for idx in top_indices:
        shap_val = float(shap_values[idx])
        feat_name = feature_names[idx]
        factors.append({
            "feature": feat_name,
            "descripcion": _describe_feature(feat_name),
            "shap_value": shap_val,
            "direction": "AUMENTA_RIESGO" if shap_val > 0 else "REDUCE_RIESGO",
        })

    return factors