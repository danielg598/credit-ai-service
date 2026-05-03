"""
Configuración del servicio — se lee desde .env con pydantic-settings.

Pydantic valida los tipos y lanza error al arrancar si falta alguna
variable requerida (ej: GROQ_API_KEY). Mejor fallar al inicio que
descubrir el problema en la tercera petición.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # --- Groq ---
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # --- Modelo ML ---
    MODEL_PATH: str = "models/credit_model.joblib"

    # --- Reglas de negocio ---
    DEFAULT_THRESHOLD: float = 0.5

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Singleton — se cachea para no leer .env en cada request."""
    return Settings()