"""
Embedding model initialization using Google AI Studio (Gemini).
Implements LlamaIndex's BaseEmbedding interface so it works
seamlessly with the RAG pipeline.
"""

from google import genai
from google.genai import types
from llama_index.core.base.embeddings.base import BaseEmbedding

from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)

# Singleton instance
_embedding_model = None


class GoogleStudioEmbedding(BaseEmbedding):
    """
    Thin wrapper around Google AI Studio's embed_content API.
    LlamaIndex calls this when it needs to turn text into vectors.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        model_name = model_name or settings.embedding_model_name
        
        # Pass model_name up to LlamaIndex's base class
        super().__init__(model_name=model_name, **kwargs)
        
        self._model_name = model_name
        self._api_key = api_key or settings.google_api_key
        
        # Configure Google's SDK once
        self._client = genai.Client(api_key=self._api_key)

    def _get_text_embedding(self, text: str) -> list[float]:
        """
        Embed a document (used during ingestion).
        task_type="RETRIEVAL_DOCUMENT" tells Google this is a stored document.
        """
        result = self._client.models.embed_content(
            model=self._model_name,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768
            )
        )
        return result.embeddings[0].values

    def _get_query_embedding(self, query: str) -> list[float]:
        """
        Embed a search query (used during retrieval).
        task_type="RETRIEVAL_QUERY" tells Google this is a question.
        """
        result = self._client.models.embed_content(
            model=self._model_name,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768
            )
        )
        return result.embeddings[0].values

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Async version — Google SDK is sync, so we delegate."""
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async version — Google SDK is sync, so we delegate."""
        return self._get_query_embedding(query)


def get_embedding_model() -> GoogleStudioEmbedding:
    """Get or create the Google embedding model singleton."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(
            "Initializing Google AI Studio embedding model",
            extra={
                "model": settings.embedding_model_name,
                "dimensions": 768,  # Google embedding-001 produces 768-dim vectors
            }
        )
        _embedding_model = GoogleStudioEmbedding(
            model_name=settings.embedding_model_name,
            api_key=settings.google_api_key,
        )
    return _embedding_model


def embed_text(text: str) -> list[float]:
    """
    Convenience function: embed a single string and return the vector.
    """
    model = get_embedding_model()
    return model.get_text_embedding(text)