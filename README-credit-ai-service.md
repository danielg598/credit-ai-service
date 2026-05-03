# credit-ai-service

> Microservicio de **scoring crediticio** con IA explicable.
> Combina un modelo de machine learning (scikit-learn) con
> explicaciones locales (SHAP) y explicaciones en lenguaje natural
> generadas por un LLM (Groq Llama 3.3 70B).

Puerto: **8000** · Python 3.12 · FastAPI

---

## Responsabilidades

Este servicio tiene UNA responsabilidad: dado un perfil de solicitante,
devolver una decisión crediticia explicable.

```
Input:  8-20 features del solicitante (edad, monto, plazo, historial, ...)
Output: score 0-1000, probabilidad default, top-5 factores SHAP,
        explicación en español generada por LLM
```

**No** se comunica con el core bancario ni con el frontend. Su cliente es
exclusivamente `credit-orchestrator` (Spring Boot).

---

## Arquitectura interna

```
┌──────────────────────────────────────────────────────────┐
│  POST /predict                                            │
│                                                           │
│  ┌─────────────────┐    ┌─────────────────┐              │
│  │ model_service   │    │ shap_service    │              │
│  │                 │───▶│                 │              │
│  │ sklearn Pipeline│    │ LinearExplainer │              │
│  │ (LogReg)        │    │ + top-5 factors │              │
│  └─────────────────┘    └────────┬────────┘              │
│                                  │                        │
│                                  ▼                        │
│                         ┌─────────────────┐              │
│                         │ groq_explainer  │              │
│                         │                 │              │
│                         │ Llama 3.3 70B   │              │
│                         │ (prompt ES-CO)  │              │
│                         └─────────────────┘              │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

Las 3 capas se ejecutan secuencialmente en cada request pero son
**independientes**: si Groq cae, el fallback local sigue devolviendo
explicación. Si SHAP tarda, tiene cache. Si el modelo falla, FastAPI
devuelve 500 con detalle.

---

## Stack técnico

| Componente | Librería | Rol |
|---|---|---|
| API framework | FastAPI 0.115+ | ASGI async, OpenAPI auto-generado |
| Validación | Pydantic 2.9+ | Tipado estricto de request/response |
| Config | pydantic-settings | Lectura tipada de `.env` |
| ML | scikit-learn 1.5+ | Pipeline de clasificación |
| Explicabilidad | SHAP 0.46+ | LinearExplainer para LogReg |
| LLM | groq 0.11+ | SDK oficial de Groq |
| Resiliencia | tenacity | Retry exponencial a Groq |
| Dataset | ucimlrepo | Descarga UCI German Credit |

---

## Estructura del proyecto

```
credit-ai-service/
├── app/
│   ├── main.py                     FastAPI app + lifespan + routers
│   ├── core/
│   │   └── config.py               Settings tipadas desde .env
│   ├── api/
│   │   └── predict.py              Endpoint POST /predict
│   ├── schemas/
│   │   └── predict.py              Pydantic: Request / Response
│   └── services/
│       ├── model_service.py        Carga joblib + infer + normalización
│       ├── shap_service.py         LinearExplainer + descripciones ES
│       └── groq_explainer.py       Prompt + retry + fallback local
├── scripts/
│   └── train.py                    Entrenamiento (corrida única)
├── models/
│   └── credit_model.joblib         Bundle serializado (post entrenamiento)
├── .env                            GROQ_API_KEY, GROQ_MODEL, ...
├── .gitignore
├── requirements.txt
└── README.md                       (este archivo)
```

---

## Setup desde cero

```bash
cd credit-ai-service

# 1. Crear y activar venv
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate         # Linux/Mac

# 2. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 3. Crear .env a partir del ejemplo (ver abajo)

# 4. Entrenar el modelo (descarga dataset ~50KB, tarda ~20s)
python scripts/train.py

# 5. Arrancar el servidor
uvicorn app.main:app --reload --port 8000
```

### Variables de entorno (`.env`)

```ini
# Groq — API key en https://console.groq.com/keys
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama-3.3-70b-versatile

# Modelo ML
MODEL_PATH=models/credit_model.joblib

# Reglas de negocio
DEFAULT_THRESHOLD=0.5

# Server
HOST=0.0.0.0
PORT=8000
```

**El `.env` NO se versiona** (está en `.gitignore`). Para producción, las
variables se inyectan vía Kubernetes Secrets o AWS Parameter Store.

---

## El modelo ML en detalle

### Dataset

**UCI Statlog German Credit Data** (id=144 en el UCI ML Repository).

- 1000 registros, 20 features (7 numéricas, 13 categóricas)
- Clase binaria: 1 = "Good credit", 2 = "Bad credit" (defaulter)
- Desbalance 70/30 — compensado con `class_weight="balanced"`
- Licencia: CC BY 4.0

El dataset viene de Statlog (1994), un estudio estándar en banca europea.
Sus variables son conceptualmente aplicables a Colombia (historial,
empleo, propósito, garantías) aunque los nombres son códigos alemanes
(`A11`, `A32`, `A73`) que mapeamos a descripciones legibles.

### Pipeline

```python
Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh",  OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # compensa 70/30
        random_state=42,
    )),
])
```

**Por qué Logistic Regression y no XGBoost** (la pregunta obvia):

1. **Interpretabilidad regulatoria**: la SFC puede leer coeficientes
   lineales directamente. No necesita herramientas específicas de ML.
2. **SHAP lineal es O(1)** por predicción vs O(árboles × profundidad)
   del TreeExplainer. Latencia predecible < 50ms.
3. **1000 muestras no justifican un modelo complejo**. Con XGBoost el
   AUC subiría de 0.80 a 0.83 — ganancia marginal a costa de overfitting.
4. **Determinismo**: mismos inputs → mismo output. XGBoost puede variar
   con el orden de entrenamiento. Importante para auditoría.

Cuando el dataset crezca a 50k+ muestras (v2 con datos del banco), se
re-evaluará XGBoost + TreeExplainer.

### Métricas del modelo (test set)

| Métrica | Valor | Interpretación |
|---|---|---|
| **AUC** | 0.806 | Área bajo la curva ROC. >0.75 es bueno en credit scoring. |
| **KS** | 0.540 | Kolmogorov-Smirnov. Separación entre defaulters/no. >0.40 es bueno. |
| **Gini** | 0.612 | 2·AUC−1. Coeficiente estándar en banca. >0.50 es bueno. |
| **CV AUC** | 0.780 ± 0.035 | 5-fold, confirma que no hay overfitting. |

Cuando pidan ver la prueba:
```bash
python scripts/train.py
```
imprime las métricas completas + matriz de confusión.

---

## Explicabilidad con SHAP

SHAP (SHapley Additive exPlanations) descompone cada predicción en
contribuciones por feature. Usamos `LinearExplainer` porque el modelo
es lineal — óptimo para este caso.

### Flujo

1. Al primer request, SHAP se inicializa:
   - Descarga 200 muestras aleatorias del dataset como "background"
   - Construye el explainer (esto toma ~2-4s la primera vez)
   - Se cachea con `@lru_cache` — siguientes calls son <50ms

2. En cada predicción:
   - Calcula SHAP values por cada una de las 61 features transformadas
   - Ordena por magnitud absoluta, toma top 5
   - Traduce el nombre técnico (`cat__credit_history_A32`) a descripción
     legible (`"Historial crediticio al día"`)

### Mapeo de códigos a descripciones

`shap_service.py` contiene un diccionario `FEATURE_DESCRIPTIONS` que
traduce los códigos German Credit a español Colombia. Ejemplo:

```python
"cat__employment_A73":         "Empleado entre 1-4 años",
"cat__credit_history_A34":     "Historial crediticio critico",
"cat__savings_A65":            "Sin ahorros registrados",
```

Si llega un código no mapeado, hay fallback a una descripción genérica
(`"Factor: <clean_name>"`) — el servicio nunca revienta por un código
desconocido.

---

## Explicación en lenguaje natural

La salida del LLM es lo que **el cliente final lee** en el dashboard.
Debe ser coherente con los factores SHAP, profesional, en español,
y sin inventar datos.

### Prompt engineering

System prompt (en `groq_explainer.py`):

```
Eres un analista senior de riesgo crediticio en Colombia.

REGLAS ESTRICTAS:
1. Básate ÚNICAMENTE en los datos proporcionados.
2. Máximo 4-5 frases en "explicacion".
3. Tono profesional pero entendible.
4. Escribe siempre en español ES-CO.
5. Retorna EXCLUSIVAMENTE JSON válido con el esquema exacto.
```

User prompt (se construye dinámicamente con los valores de la solicitud
+ decisión del modelo + top-5 factores SHAP).

Temperatura **0.3** — baja para respuestas consistentes pero no
totalmente determinísticas. Cada request da una redacción ligeramente
distinta con el mismo mensaje de fondo.

### Resiliencia

**Retry**: `tenacity` reintenta 3 veces con backoff exponencial
(2s, 4s, 8s) ante fallos 5xx o rate limit.

**Fallback local**: si después de 3 intentos Groq sigue caído, el
servicio genera una explicación básica sin LLM a partir de los factores
SHAP:

> *"La solicitud fue evaluada con probabilidad de default de 24.4%.
> Decisión: APROBADO. Los factores que reducen el riesgo incluyen
> historial crediticio al día y empleado entre 4-7 años."*

El flujo **nunca se bloquea** por problemas con Groq. El usuario final
puede no notar la diferencia.

---

## La transformación de escala (deuda técnica documentada)

**Problema**: El modelo se entrenó con montos en marcos alemanes (DM)
de los años 90 — rango típico 250-18.000 DM. En Colombia pedimos
créditos de 500.000 a 100.000.000 COP.

**Solución MVP**: en `model_service.py` aplicamos una transformación
documentada antes de la inferencia:

```python
COP_TO_DM_SCALE = 3000.0
credit_amount_modelo = credit_amount_cop / COP_TO_DM_SCALE
```

Así un préstamo de 20M COP se ve como ~6.666 DM — dentro de la
distribución del training set, cerca de la mediana.

**Por qué es deuda técnica y no bug**: es un hack intencional para
MVP. Lo correcto sería re-entrenar con datos colombianos reales, y
entonces la transformación desaparece (se vuelve identidad).

**Cuándo desaparece**: en v2, cuando tengamos el book real del banco.

---

## API

### `GET /health`

```json
{
  "status": "ok",
  "service": "credit-ai",
  "version": "1.0.0",
  "llm_model": "llama-3.3-70b-versatile"
}
```

### `POST /predict`

Request:

```json
{
  "duration": 36,
  "credit_amount": 20000000,
  "age": 32,
  "installment_rate": 2,
  "residence_since": 2,
  "existing_credits": 1,
  "dependents": 1,
  "checking_status": "A13",
  "credit_history": "A32",
  "purpose": "A41",
  "savings": "A63",
  "employment": "A74",
  "personal_status": "A92",
  "other_debtors": "A101",
  "property": "A123",
  "other_installment_plans": "A143",
  "housing": "A152",
  "job": "A173",
  "telephone": "A192",
  "foreign_worker": "A201"
}
```

Response (200 OK):

```json
{
  "score": 755,
  "probabilidad_default": 0.2443,
  "decision": "APROBADO",
  "factores_clave": [
    {
      "factor": "cat__purpose_A41",
      "impacto": -0.8161,
      "descripcion": "Proposito: vehiculo usado",
      "direction": "REDUCE_RIESGO"
    },
    ...
  ],
  "explicacion": "La decisión de aprobar su solicitud...",
  "model_version": "logreg-german-v1.0",
  "latencia_ms": 826
}
```

Latencia típica:
- **Primera llamada** tras arrancar uvicorn: 3-5 seg (warm-up SHAP)
- **Siguientes**: 600-900 ms (Groq domina ~400-600 ms)

### `GET /docs`

Swagger UI auto-generado con todos los endpoints. Útil para testing
manual y para compartir el contrato con el equipo de Spring Boot.

---

## Observabilidad

Cada request genera un log estructurado:

```
[INFO] credit-ai.api: POST /predict duration=36, amount=20000000.0, age=32
[INFO] credit-ai.model: Cargando modelo desde models/credit_model.joblib...
[INFO] credit-ai.model: Modelo cargado. AUC=0.806, KS=0.540, Gini=0.612
[INFO] credit-ai.api: OK decision=APROBADO score=755 pd=0.244 latencia=826ms
```

Para producción se sustituye `logging.basicConfig` por JSON logs
estructurados (`python-json-logger`) y se conecta a Datadog / Loki /
CloudWatch.

---

## Re-entrenar el modelo

```bash
python scripts/train.py
```

Regenera `models/credit_model.joblib`. El servicio debe reiniciarse para
cargar el modelo nuevo (el singleton `@lru_cache` cachea el bundle). En
producción, un blue/green deploy resuelve esto sin downtime.

---

## Testing manual

1. Swagger UI: http://localhost:8000/docs → `POST /predict` → **Try it
   out** → **Execute** con el payload default.
2. Verificar que `decision` sea coherente con `probabilidad_default` y
   `DEFAULT_THRESHOLD`.
3. Verificar que `explicacion` esté en español y sea coherente con los
   factores SHAP.

---

## Próximos pasos

- [ ] Unit tests con pytest (cobertura >80%)
- [ ] Integration test con mock de Groq
- [ ] Bias testing (disparate impact por género, edad)
- [ ] Migrar a `logreg-colombia-v2` entrenado con datos reales
- [ ] Prompt versioning y A/B testing con Langfuse
- [ ] Auto-hosting del LLM con vLLM si hay política de no-salida de datos
