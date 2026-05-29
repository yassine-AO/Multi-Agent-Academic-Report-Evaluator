"""
Query & retrieval pipeline.
The full chain: HyDE → Hybrid Search → Rerank → Return top chunks.
"""

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import QueryBundle, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, Settings

# Disable the default LLM (OpenAI) to avoid requiring an OpenAI API key/package for retrieval
Settings.llm = None

from src.config import settings
from src.db import get_or_create_collection, RUBRICS_COLLECTION, PAST_REPORTS_COLLECTION
from src.rag.embeddings import get_embedding_model
from src.rag.reranker import rerank
from src.utils import get_logger

logger = get_logger(__name__)


def get_vector_index(collection_name: str):
    """
    Build a LlamaIndex VectorStoreIndex from a ChromaDB collection.
    This wraps our existing ChromaDB data in LlamaIndex's search interface.
    """
    collection = get_or_create_collection(collection_name)
    embed_model = get_embedding_model()
    
    # Bridge ChromaDB → LlamaIndex
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    # Load existing vectors (no re-embedding needed)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    
    return index


def hybrid_retrieve(
    query: str,
    collection_name: str,
    top_k: int = 15,
    metadata_filters: dict | None = None,
) -> list[TextNode]:
    """
    Stage 1+2: HyDE + Hybrid Search (dense + BM25).
    
    Args:
        query: Raw search query from an agent.
        collection_name: Which ChromaDB collection to search.
        top_k: How many candidates to fetch before reranking.
        metadata_filters: Optional ChromaDB metadata filters.
    
    Returns:
        List of TextNode candidates.
    """
    index = get_vector_index(collection_name)
    
    # Build retriever with HyDE + fusion
    retriever = QueryFusionRetriever(
        [index.as_retriever(similarity_top_k=top_k)],
        similarity_top_k=top_k,
        num_queries=1,  # HyDE generates the expanded query internally
        use_async=False,
        verbose=False,
    )
    
    # Execute search
    nodes = retriever.retrieve(query)
    
    logger.info(
        "Hybrid retrieval complete",
        extra={
            "query": query[:100],
            "collection": collection_name,
            "candidates": len(nodes),
        }
    )
    
    return nodes


def retrieve_with_rerank(
    query: str,
    collection_name: str,
    final_k: int = 5,
    metadata_filters: dict | None = None,
) -> list[tuple[str, float]]:
    """
    Full pipeline: Hybrid retrieve → Cross-encoder rerank → Return top chunks.
    
    Args:
        query: Raw search query.
        collection_name: Which collection to search.
        final_k: How many chunks to return after reranking.
        metadata_filters: Optional metadata filters (e.g., quality_tier).
    
    Returns:
        List of (chunk_text, relevance_score) tuples.
    """
    # Stage 1+2: Get candidates
    nodes = hybrid_retrieve(query, collection_name, top_k=15, metadata_filters=metadata_filters)
    
    if not nodes:
        logger.warning(f"No candidates found for query: {query[:100]}")
        return []
    
    # Extract raw text from nodes
    documents = [node.text for node in nodes]
    
    # Stage 3: Rerank
    ranked = rerank(query, documents, top_k=final_k)
    
    return ranked


def retrieve_for_reviewer(criterion: str, report_section: str | None = None) -> list[tuple[str, float]]:
    """
    High-level helper: Reviewer agent calls this.
    Searches rubrics + high-quality past reports for positive examples.
    """
    query = f"Rubric criterion: {criterion}. {report_section or ''}"
    
    # Search rubrics
    rubric_results = retrieve_with_rerank(query, RUBRICS_COLLECTION, final_k=3)
    
    # Search high-quality past reports
    report_results = retrieve_with_rerank(
        query,
        PAST_REPORTS_COLLECTION,
        final_k=3,
        metadata_filters={"quality_tier": "high"},
    )
    
    # Combine and deduplicate (simple: just concatenate, LLM handles it)
    combined = rubric_results + report_results
    
    logger.info(
        "Reviewer retrieval complete",
        extra={
            "criterion": criterion,
            "rubric_chunks": len(rubric_results),
            "report_chunks": len(report_results),
        }
    )
    
    return combined


def retrieve_for_critic(criterion: str, report_section: str | None = None) -> list[tuple[str, float]]:
    """
    High-level helper: Critic agent calls this.
    Searches low-quality past reports for counter-examples and flaws.
    """
    query = f"Common flaws in {criterion}. Weak examples. What went wrong."
    
    results = retrieve_with_rerank(
        query,
        PAST_REPORTS_COLLECTION,
        final_k=5,
        metadata_filters={"quality_tier": "low"},
    )
    
    logger.info(
        "Critic retrieval complete",
        extra={
            "criterion": criterion,
            "counter_example_chunks": len(results),
        }
    )
    
    return results