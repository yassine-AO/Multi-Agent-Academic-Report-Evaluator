"""
ChromaDB client initialization.
No default embedding function attached — LlamaIndex passes pre-computed vectors.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)

_client = None


def get_chroma_client():
    """Get or create the persistent ChromaDB client (singleton)."""
    global _client
    if _client is None:
        logger.info(
            "Initializing ChromaDB persistent client",
            extra={"path": settings.chroma_db_path}
        )
        _client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
    return _client


def get_or_create_collection(name: str):
    """
    Get an existing collection or create it.
    No embedding_function passed — LlamaIndex handles all embedding.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(
        "Collection ready",
        extra={"collection": name, "document_count": collection.count()}
    )
    return collection


def health_check() -> dict:
    """Quick connectivity check for FastAPI /health."""
    try:
        client = get_chroma_client()
        cols = client.list_collections()
        return {"status": "healthy", "collections": [c.name for c in cols]}
    except Exception as e:
        logger.error("ChromaDB health check failed", exc_info=True)
        return {"status": "unhealthy", "error": str(e)}