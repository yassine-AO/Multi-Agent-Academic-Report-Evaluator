#!/usr/bin/env python
"""
Inspect ChromaDB contents — view chunks, metadata, and collection stats.
"""

import sys
from pathlib import Path

# Add project root to path so imports work when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import (
    get_or_create_collection,
    RUBRICS_COLLECTION,
    PAST_REPORTS_COLLECTION,
)


def inspect_collection(name: str, limit: int = 10):
    print(f"\n{'=' * 60}")
    print(f"Collection: {name}")
    print(f"{'=' * 60}")
    
    collection = get_or_create_collection(name)
    count = collection.count()
    print(f"Total documents: {count}")
    
    if count == 0:
        print("Collection is empty.")
        return
    
    results = collection.get(limit=limit)
    
    for i, (doc, meta, id_) in enumerate(zip(results['documents'], results['metadatas'], results['ids'])):
        print(f"\n--- Chunk {i+1} | ID: {id_} ---")
        print(f"  Source type: {meta.get('source_type', 'unknown')}")
        print(f"  Filename:    {meta.get('filename', 'unknown')}")
        print(f"  Page:        {meta.get('page_number', '?')}")
        print(f"  Quality:     {meta.get('quality_tier', 'N/A')}")
        print(f"  Text ({len(doc)} chars):")
        preview = doc[:400] + "..." if len(doc) > 400 else doc
        for line in preview.split('\n'):
            print(f"    {line}")
        print("-" * 40)


if __name__ == "__main__":
    inspect_collection(RUBRICS_COLLECTION)
    inspect_collection(PAST_REPORTS_COLLECTION)