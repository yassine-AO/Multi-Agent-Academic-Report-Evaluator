"""
Data ingestion pipeline.
Offline, one-time process that reads PDFs, chunks them, embeds them,
and stores them in ChromaDB with metadata tags.
"""

import os
from pathlib import Path

from llama_index.core.schema import Document

from src.config import settings
from src.db import (
    get_or_create_collection,
    RUBRICS_COLLECTION,
    PAST_REPORTS_COLLECTION,
    METADATA_SOURCE_TYPE,
    METADATA_QUALITY_TIER,
    METADATA_FILENAME,
    METADATA_PAGE_NUMBER,
)
from src.rag.chunking import split_documents
from src.rag.embeddings import get_embedding_model
from src.services.pdf_parser import extract_text_from_pdf
from src.utils import get_logger

logger = get_logger(__name__)


def ingest_directory(directory: Path, collection_name: str, source_type: str, quality_tier: str | None = None):
    """
    Ingest all PDFs from a directory into a ChromaDB collection.
    
    Args:
        directory: Path to folder containing PDFs.
        collection_name: ChromaDB collection to write into.
        source_type: "rubric" or "past_report".
        quality_tier: "high" or "low" (only for past reports).
    """
    collection = get_or_create_collection(collection_name)
    embed_model = get_embedding_model()
    
    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return
    
    logger.info(
        "Starting ingestion",
        extra={
            "directory": str(directory),
            "collection": collection_name,
            "file_count": len(pdf_files),
        }
    )
    
    total_chunks = 0
    
    for pdf_path in pdf_files:
        logger.info(f"Processing {pdf_path.name}")
        
        # Step 1: Extract text with page boundaries
        pages = extract_text_from_pdf(str(pdf_path))
        
        # Step 2: Combine pages into one document for semantic splitting
        full_text = "\n\n".join(pages)
        doc = Document(
            text=full_text,
            metadata={
                METADATA_FILENAME: pdf_path.name,
                METADATA_SOURCE_TYPE: source_type,
            }
        )
        
        # Step 3: Semantic chunking
        nodes = split_documents([doc])
        
        # Step 4: Embed and store each chunk
        for node in nodes:
            # Determine which page this chunk came from (best-effort)
            page_num = _estimate_page_number(node.text, pages)
            
            metadata = {
                METADATA_SOURCE_TYPE: source_type,
                METADATA_FILENAME: pdf_path.name,
                METADATA_PAGE_NUMBER: page_num,
            }
            if quality_tier:
                metadata[METADATA_QUALITY_TIER] = quality_tier
            
            # Generate embedding
            embedding = embed_model.get_text_embedding(node.text)
            
            # Store in ChromaDB
            collection.add(
                ids=[_chunk_id(pdf_path.name, total_chunks)],
                documents=[node.text],
                embeddings=[embedding],
                metadatas=[metadata],
            )
            
            total_chunks += 1
    
    logger.info(
        "Ingestion complete",
        extra={
            "collection": collection_name,
            "files_processed": len(pdf_files),
            "total_chunks": total_chunks,
        }
    )


def _estimate_page_number(chunk_text: str, pages: list[str]) -> int:
    """
    Best-effort page attribution. Finds which raw page contains
    the most overlap with this chunk.
    """
    best_page = 1
    best_overlap = 0
    
    for i, page_text in enumerate(pages, start=1):
        # Simple heuristic: count shared words
        chunk_words = set(chunk_text.lower().split())
        page_words = set(page_text.lower().split())
        overlap = len(chunk_words & page_words)
        
        if overlap > best_overlap:
            best_overlap = overlap
            best_page = i
    
    return best_page


def _chunk_id(filename: str, index: int) -> str:
    """Generate a unique ID for a chunk."""
    safe_name = filename.replace(" ", "_").replace(".pdf", "")
    return f"{safe_name}_chunk_{index}"


def run_full_ingestion():
    """
    Run the complete ingestion for both rubrics and past reports.
    Call this from scripts/ingest.py.
    """
    # Ingest rubrics
    rubrics_dir = Path("data/rubrics")
    if rubrics_dir.exists():
        ingest_directory(
            directory=rubrics_dir,
            collection_name=RUBRICS_COLLECTION,
            source_type="rubric",
        )
    else:
        logger.warning(f"Rubrics directory not found: {rubrics_dir}")
    
    # Ingest past reports (high quality)
    reports_dir = Path("data/past_reports")
    if reports_dir.exists():
        # For now, we assume all reports in the folder are high-quality
        # You can organize into subfolders (high/, low/) later
        ingest_directory(
            directory=reports_dir,
            collection_name=PAST_REPORTS_COLLECTION,
            source_type="past_report",
            quality_tier="high",
        )
    else:
        logger.warning(f"Past reports directory not found: {reports_dir}")