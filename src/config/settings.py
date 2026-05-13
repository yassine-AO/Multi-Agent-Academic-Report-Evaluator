"""
Configuration management for the application.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """Application settings."""

    # API Keys
    openai_api_key: Optional[str] = None
    # Add other API keys as needed

    # Database/Storage paths
    chroma_db_path: str = "./chroma_db"

    # Model configurations
    embedding_model_name: str = "text-embedding-3-small"
    llm_model_name: str = "gpt-4o-mini"

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