"""
train.py — entrena el modelo de credit scoring sobre German Credit Data.

Uso:
    python scripts/train.py

Resultado:
    models/credit_model.joblib   (pipeline completo listo para inferencia)
    Imprime metricas: AUC, KS, Gini, confusion matrix, classification report

El bundle guardado contiene:
  - pipeline:       Pipeline sklearn (preprocesamiento + clasificador)
  - feature_names:  nombres de las features POST transformacion (para SHAP)
  - num_cols:       lista de columnas numericas
  - cat_cols:       lista de columnas categoricas
  - metrics:        dict con AUC, KS, Gini

Dataset: UCI Statlog German Credit Data (id=144)
  1000 registros, 20 features (7 numericas + 13 categoricas), clase binaria.
  Licencia: CC BY 4.0.
"""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

# Configuracion
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_PATH = Path("models/credit_model.joblib")


def main() -> None:
    print("=" * 70)
    print("  CREDIT SCORING — ENTRENAMIENTO")
    print("  Dataset: UCI Statlog German Credit Data")
    print("=" * 70)

    # 1. Descargar dataset
    print("\n[1/6] Descargando dataset desde UCI...")
    german = fetch_ucirepo(id=144)
    X = german.data.features.copy()

    # Renombrado a columnas descriptivas
    # El dataset UCI usa 'Attribute1', 'Attribute2', ... como nombres.
    # Los mapeamos a nombres descriptivos segun el README oficial del dataset
    # (https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)).
    # Asi el modelo, SHAP y el schema de la API hablan el mismo lenguaje.
    column_mapping = {
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
    }
    X = X.rename(columns=column_mapping)
    # El target original es 1=Good, 2=Bad. Convertimos a 1=Bad (default) para
    # que roc_auc_score interprete "probabilidad de la clase positiva" como
    # "probabilidad de default" que es lo que queremos modelar.
    y = (german.data.targets.iloc[:, 0] == 2).astype(int)

    print(f"    Muestras: {len(X)}  |  Features: {X.shape[1]}  |  Defaults: {y.sum()} ({y.mean()*100:.1f}%)")

    # 2. Identificar tipos de columnas
    print("\n[2/6] Identificando tipos de columnas...")
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    print(f"    Numericas   ({len(num_cols):2d}): {num_cols}")
    print(f"    Categoricas ({len(cat_cols):2d}): {cat_cols}")

    # 3. Pipeline de preprocesamiento
    # Numericas: imputar mediana + estandarizar (media 0, desv 1)
    # Categoricas: imputar moda + one-hot encoding
    print("\n[3/6] Construyendo pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc",  StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        # class_weight="balanced" compensa el desbalance 70/30 Good/Bad:
        # le da mas peso a la clase minoritaria (Bad) al entrenar.
        # Sin esto, el modelo aprende a decir "Good" siempre y parece
        # tener 70% accuracy engañoso.
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    # 4. Train/test split estratificado
    print("\n[4/6] Dividiendo train/test (80/20 estratificado)...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"    Train: {len(X_tr)}  |  Test: {len(X_te)}")

    # 5. Cross-validation + entrenamiento final
    print("\n[5/6] Evaluando con 5-fold CV...")
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    auc_cv = cross_val_score(pipeline, X_tr, y_tr, cv=cv, scoring="roc_auc")
    print(f"    CV AUC: {auc_cv.mean():.3f} (+/- {auc_cv.std():.3f})")

    print("\n    Entrenando modelo final con todo el train set...")
    pipeline.fit(X_tr, y_tr)

    # 6. Metricas sobre test set
    print("\n[6/6] Evaluando sobre test set...")
    proba = pipeline.predict_proba(X_te)[:, 1]
    pred = pipeline.predict(X_te)

    auc = roc_auc_score(y_te, proba)
    # KS (Kolmogorov-Smirnov): maxima separacion entre distribuciones de
    # prob. de los que defaultaron vs los que pagaron. >0.30 es decente,
    # >0.40 es bueno en credit scoring.
    ks = ks_2samp(proba[y_te == 1], proba[y_te == 0]).statistic
    # Gini = 2*AUC - 1. Un modelo aleatorio tiene Gini=0, perfecto Gini=1.
    gini = 2 * auc - 1

    print(f"    AUC:  {auc:.3f}")
    print(f"    KS:   {ks:.3f}")
    print(f"    Gini: {gini:.3f}")
    print(f"\n    Matriz de confusion:")
    cm = confusion_matrix(y_te, pred)
    print(f"                    Pred Good  Pred Bad")
    print(f"    Real Good       {cm[0,0]:4d}       {cm[0,1]:4d}")
    print(f"    Real Bad        {cm[1,0]:4d}       {cm[1,1]:4d}")
    print("\n    Classification report:")
    print(classification_report(y_te, pred, target_names=["Good", "Bad"], digits=3))

    # 7. Serializar bundle completo
    print(f"\nGuardando modelo en {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # feature_names: nombres DESPUES de la transformacion (con OHE expandido).
    # Necesario para SHAP, para saber que columna corresponde a que feature.
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()

    bundle = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "metrics": {
            "auc": float(auc),
            "ks": float(ks),
            "gini": float(gini),
            "cv_auc_mean": float(auc_cv.mean()),
            "cv_auc_std":  float(auc_cv.std()),
        },
    }
    joblib.dump(bundle, OUTPUT_PATH)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"    OK  ({size_kb:.1f} KB)")
    print("\n" + "=" * 70)
    print("  ENTRENAMIENTO COMPLETO")
    print("=" * 70)
    print(f"\nFeatures transformadas: {len(feature_names)}")
    print(f"Clasificador: Logistic Regression (balanced)")
    print(f"\nProximos pasos: implementar POST /predict usando models/credit_model.joblib")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise