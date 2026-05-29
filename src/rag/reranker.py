"""
Cross-encoder reranker.
Local model — no API calls, runs on CPU.
Takes candidate chunks from hybrid search and reorders them by true relevance.
"""

from sentence_transformers import CrossEncoder

from src.utils import get_logger

logger = get_logger(__name__)

# Singleton
_reranker = None


def get_reranker() -> CrossEncoder:
    """Load or return the cross-encoder model."""
    global _reranker
    if _reranker is None:
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info(f"Loading cross-encoder reranker: {model_name}")
        _reranker = CrossEncoder(model_name)
    return _reranker


def rerank(query: str, documents: list[str], top_k: int = 5) -> list[tuple[str, float]]:
    """
    Re-rank documents by relevance to the query.
    
    Args:
        query: The search query.
        documents: Candidate document texts.
        top_k: How many to keep after reranking.
    
    Returns:
        List of (document_text, score) tuples, sorted by score descending.
    """
    if not documents:
        return []
    
    reranker = get_reranker()
    
    # Cross-encoder scores each (query, doc) pair
    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)
    
    # Sort by score, keep top_k
    scored = list(zip(documents, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(
        "Reranking complete",
        extra={
            "input_candidates": len(documents),
            "output_top_k": min(top_k, len(scored)),
            "best_score": round(float(scored[0][1]), 4) if scored else None,
        }
    )
    
    return scored[:top_k]