#!/usr/bin/env python
"""
CLI script to run the full ingestion pipeline.
Usage: python scripts/ingest.py
"""

import sys
from pathlib import Path

# Add project root to path so imports work when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.ingestion import run_full_ingestion
from src.utils import get_logger

logger = get_logger("ingest_cli")

if __name__ == "__main__":
    logger.info("Starting ingestion pipeline")
    run_full_ingestion()
    logger.info("Ingestion pipeline finished")