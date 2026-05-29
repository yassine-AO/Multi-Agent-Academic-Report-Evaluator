from src.db import (
    get_or_create_collection,
    health_check,
    RUBRICS_COLLECTION,
    PAST_REPORTS_COLLECTION,
)

print("=== ChromaDB Health Check ===")
status = health_check()
print(status)

print(f"\n=== Getting collection: {RUBRICS_COLLECTION} ===")
rubrics = get_or_create_collection(RUBRICS_COLLECTION)
print(f"Documents in rubrics: {rubrics.count()}")

print(f"\n=== Getting collection: {PAST_REPORTS_COLLECTION} ===")
reports = get_or_create_collection(PAST_REPORTS_COLLECTION)
print(f"Documents in past_reports: {reports.count()}")

print("\n✅ ChromaDB client is alive and collections are ready!")