import glob
from pathlib import Path
from src.services import extract_text_from_pdf, parse_pdf_structure

# Dynamically find the first PDF in data/rubrics/
pdfs = glob.glob("data/rubrics/*.pdf")
test_pdf = pdfs[0] if pdfs else "data/rubrics/test_rubric.pdf"

if not Path(test_pdf).exists():
    print("No test PDF found. Place any PDF in data/rubrics/ and retry.")
    exit()

print("=== Test 1: Extract text ===")
pages, has_ocr = extract_text_from_pdf(test_pdf)
print(f"Pages: {len(pages)}")
print(f"OCR fallback used: {has_ocr}")
print(f"\nPage 1 preview:\n{pages[0][:300]}...")

print("\n=== Test 2: Parse structure ===")
structure = parse_pdf_structure(test_pdf)
print(f"Page count: {structure['page_count']}")
print(f"Has OCR: {structure['has_ocr_fallback']}")
print(f"Total text: {len(structure['raw_text'])} chars")

print("\n[OK] PDF parser working!")