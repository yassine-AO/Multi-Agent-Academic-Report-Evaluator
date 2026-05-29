"""
Minimal PDF text extraction (placeholder).
Full implementation with OCR fallback comes in Step 10.
"""

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> list[str]:
    """
    Extract text from each page of a PDF.
    
    Returns:
        List of strings, one per page.
    """
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        pages.append(text)
    
    doc.close()
    return pages