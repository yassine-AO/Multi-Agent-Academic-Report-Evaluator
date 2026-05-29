"""
Semantic chunking configuration.
Uses LlamaIndex's SemanticSplitterNodeParser to split text
at topic boundaries, not arbitrary token counts.
"""

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document

from src.rag.embeddings import get_embedding_model
from src.utils import get_logger

logger = get_logger(__name__)


def get_semantic_splitter() -> SemanticSplitterNodeParser:
    """
    Create and return the semantic splitter.
    
    Uses the same embedding model as the rest of the system
    to ensure consistent semantic understanding.
    """
    embed_model = get_embedding_model()
    
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,                      # sentences to buffer around split point
        breakpoint_percentile_threshold=70,  # similarity drop threshold (0-100)
        embed_model=embed_model,
    )
    
    logger.info(
        "Semantic splitter initialized",
        extra={
            "buffer_size": 1,
            "breakpoint_percentile": 80,
            "embed_model": embed_model.model_name,
        }
    )
    return splitter


def split_documents(documents: list[Document]) -> list:
    """
    Split a list of LlamaIndex Documents into semantic nodes.
    
    Args:
        documents: Raw LlamaIndex Document objects (usually one per PDF).
    
    Returns:
        List of TextNode objects, each containing a semantically coherent chunk.
    """
    splitter = get_semantic_splitter()
    nodes = splitter.get_nodes_from_documents(documents)
    
    logger.info(
        "Documents split into semantic chunks",
        extra={
            "input_documents": len(documents),
            "output_nodes": len(nodes),
            "avg_chunk_size": sum(len(n.get_content()) for n in nodes) // max(len(nodes), 1),
        }
    )
    return nodes