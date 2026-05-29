"""
PDF text extraction with EasyOCR fallback.
Logic: Check PyMuPDF first. If page has text, use it.
If no text found, fall back to EasyOCR (scanned/image PDFs).
"""

from pathlib import Path

import fitz  # PyMuPDF
import easyocr
import numpy as np
from PIL import Image

from src.utils import get_logger, clean_text

logger = get_logger(__name__)

# Singleton: EasyOCR reader (expensive to create, reuse across pages)
_ocr_reader = None


def _get_ocr_reader():
    """Lazy-load EasyOCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Initializing EasyOCR (English)")
        _ocr_reader = easyocr.Reader(["en"], gpu=False)  # gpu=True if CUDA available
    return _ocr_reader


def extract_text_from_pdf(pdf_path: str) -> tuple[list[str], bool]:
    """
    Extract text from PDF.
    
    Per-page logic:
    1. Try PyMuPDF.get_text()
    2. If result is empty/whitespace → EasyOCR on page image
    3. Return cleaned text + flag if OCR was used
    
    Args:
        pdf_path: Path to PDF file.
    
    Returns:
        (pages_text, has_ocr_fallback)
    """
    doc = fitz.open(pdf_path)
    pages = []
    has_ocr_fallback = False
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        raw_text = page.get_text()
        
        # Check 1: Did PyMuPDF find actual text?
        stripped = raw_text.strip()
        
        if stripped:
            # Yes: use PyMuPDF result
            cleaned = clean_text(raw_text)
            pages.append(cleaned)
            logger.debug(f"Page {page_num + 1}: PyMuPDF text found ({len(cleaned)} chars)")
        else:
            # No: PDF page is image-only, use EasyOCR
            logger.info(f"Page {page_num + 1}: No text found, falling back to EasyOCR")
            ocr_text = _ocr_page(page)
            pages.append(ocr_text)
            has_ocr_fallback = True
    
    doc.close()
    
    logger.info(
        "PDF extraction complete",
        extra={
            "file": Path(pdf_path).name,
            "pages": len(pages),
            "ocr_used": has_ocr_fallback,
        }
    )
    
    return pages, has_ocr_fallback


def _ocr_page(page: fitz.Page) -> str:
    """
    Extract text from a single page using EasyOCR.
    
    Converts page to image, runs OCR, returns cleaned text.
    """
    # Render at 300 DPI for good OCR accuracy
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    
    # Convert to numpy array (EasyOCR expects this)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8)
    img_array = img_array.reshape(pix.height, pix.width, pix.n)
    
    # EasyOCR reads RGB, PyMuPDF gives RGBA → drop alpha channel if present
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Run OCR
    reader = _get_ocr_reader()
    results = reader.readtext(img_array, detail=0, paragraph=True)
    
    # Join lines and clean
    raw_text = "\n".join(results)
    cleaned = clean_text(raw_text)
    
    logger.debug(f"EasyOCR extracted {len(cleaned)} chars")
    return cleaned


def parse_pdf_structure(pdf_path: str) -> dict:
    """
    High-level parser output.
    
    Returns skeleton dict. The Parser agent (next step) fills sections
    via LLM-based identification.
    """
    pages, has_ocr = extract_text_from_pdf(pdf_path)
    full_text = "\n\n".join(pages)
    
    return {
        "title": "",
        "abstract": "",
        "introduction": "",
        "methodology": "",
        "results": "",
        "conclusion": "",
        "bibliography": "",
        "raw_text": full_text,
        "page_count": len(pages),
        "has_ocr_fallback": has_ocr,
    }