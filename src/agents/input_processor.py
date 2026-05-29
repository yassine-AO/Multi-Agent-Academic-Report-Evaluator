"""
Parser Agent Node.
Extracts structured sections from raw PDF text.
"""

from src.agents.base_agent import run_agent
from src.agents.prompts import PARSER_SYSTEM_PROMPT
from src.models.schemas import ParsedReport
from src.services.pdf_parser import parse_pdf_structure
from src.utils import get_logger

logger = get_logger(__name__)


def parser_node(state: dict) -> dict:
    """
    LangGraph node: Parse PDF structure.
    
    Args:
        state: EvaluationState dict with 'pdf_path'.
    
    Returns:
        Updated state with 'parsed_report' field.
    """
    pdf_path = state["pdf_path"]
    logger.info(f"Parser node starting: {pdf_path}")
    
    # Extract raw structure (text + metadata)
    raw_structure = parse_pdf_structure(pdf_path)
    
    # Use LLM to identify sections if needed, or just validate structure
    # For now, we pass raw text to LLM for section identification
    user_content = f"""
Extract the academic sections from this report.

RAW TEXT:
{raw_structure['raw_text'][:8000]}  # Truncate if very long

METADATA:
- Pages: {raw_structure['page_count']}
- OCR used: {raw_structure['has_ocr_fallback']}
"""
    
    parsed = run_agent(
        system_prompt=PARSER_SYSTEM_PROMPT,
        user_content=user_content,
        output_schema=ParsedReport,
        temperature=0.0,
    )
    
    # Preserve metadata from extraction layer
    parsed.page_count = raw_structure["page_count"]
    parsed.has_ocr_fallback = raw_structure["has_ocr_fallback"]
    
    # Update state
    state["parsed_report"] = parsed.model_dump()
    logger.info("Parser node complete")
    
    return state