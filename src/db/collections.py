"""
Collection name constants and metadata schema conventions.
Centralized here to avoid typos across the entire codebase.
"""

# Collection names
RUBRICS_COLLECTION = "rubrics"
PAST_REPORTS_COLLECTION = "past_reports"

# Metadata keys used when storing documents in ChromaDB
METADATA_SOURCE_TYPE = "source_type"       # Values: "rubric" | "past_report"
METADATA_QUALITY_TIER = "quality_tier"     # Values: "high" | "low" (only for past reports)
METADATA_FILENAME = "filename"
METADATA_PAGE_NUMBER = "page_number"