"""
Configuration management for the application.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """Application settings."""

    # --- LLM ---
    groq_api_key: str
    llm_model_name: str = "llama-3.1-8b-instant"

    # Database/Storage paths
    chroma_db_path: str = "./chroma_db"

    # --- Embeddings (Google AI Studio) ---
    google_api_key: str
    embedding_model_name: str = "models/embedding-001"

    # Processing limits
    max_chunk_size: int = 1000
    chunk_overlap: int = 200

    # Observability
    langchain_api_key: Optional[str] = None
    langchain_tracing_v2: Optional[str] = None
    langchain_project: Optional[str] = None

    # Deliberation Loop Limits
    max_deliberation_loops: int = 3
    convergence_threshold: float = 0.3

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Global settings instance
settings = Settings()