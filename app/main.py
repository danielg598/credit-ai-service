"""
Credit AI Service — microservicio de scoring crediticio con IA explicable.

Arquitectura:
  1. Modelo de ML scikit-learn (Logistic Regression sobre German Credit)
     → calcula probabilidad de default.
  2. SHAP local → identifica los factores que más influyeron.
  3. Groq Llama 3.3 70B → genera una explicacion en lenguaje natural
     en español para cliente colombiano.

Endpoints:
  - GET  /health         Health check
  - POST /predict        Evalua una solicitud (recibe features, devuelve
                         score + probabilidad + factores SHAP + explicacion LLM)
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import predict as predict_router
from contextlib import asynccontextmanager

from app.core.config import get_settings

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("credit-ai")

# --- Settings (valida .env al arrancar) ---
settings = get_settings()

# --- Lifespan (reemplaza el deprecated @app.on_event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    log.info("Credit AI Service arrancando...")
    log.info(f"Modelo: {settings.MODEL_PATH}")
    log.info(f"Groq model: {settings.GROQ_MODEL}")
    log.info(f"Umbral default: {settings.DEFAULT_THRESHOLD}")
    yield
    # Shutdown (si en el futuro necesitas cerrar conexiones, etc.)
    log.info("Credit AI Service detenido.")


# --- App ---
app = FastAPI(
    title="Credit AI Service",
    description="Scoring crediticio con ML + explicabilidad LLM en español",
    version="1.0.0",
    lifespan=lifespan,    # ← registramos el lifespan aqui
)

# --- CORS (por si Angular pega directo para debug, igual que el vault-mock) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(predict_router.router)


@app.get("/health")
def health_check():
    """Health check. Spring Boot lo usara para verificar disponibilidad."""
    return {
        "status": "ok",
        "service": "credit-ai",
        "version": "1.0.0",
        "llm_model": settings.GROQ_MODEL,
    }


@app.get("/")
def root():
    return {
        "service": "Credit AI Service",
        "docs": "/docs",
        "endpoints": [
            "GET  /health       Health check",
            "POST /predict      Evalua solicitud con ML + SHAP + LLM  ✅",
        ],
    }