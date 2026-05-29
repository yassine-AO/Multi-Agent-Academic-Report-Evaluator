"""
Reviewer Agent Node.
Scores the report against rubric criteria using RAG-grounded evidence.
"""

from src.agents.base_agent import run_agent
from src.agents.prompts import REVIEWER_SYSTEM_PROMPT
from src.models.schemas import ReviewScores
from src.rag import retrieve_for_reviewer
from src.utils import get_logger

logger = get_logger(__name__)


def reviewer_node(state: dict) -> dict:
    """
    LangGraph node: Review and score the parsed report.
    
    Args:
        state: EvaluationState with 'parsed_report'.
    
    Returns:
        Updated state with 'review_scores' field.
    """
    parsed = state["parsed_report"]
    logger.info("Reviewer node starting")
    
    # Build context from RAG
    rag_context = []
    
    # Query for each major section
    sections = {
        "methodology": parsed.get("methodology", ""),
        "results": parsed.get("results", ""),
        "introduction": parsed.get("introduction", ""),
    }
    
    for section_name, section_text in sections.items():
        if section_text:
            chunks = retrieve_for_reviewer(
                criterion=section_name,
                report_section=section_text[:500],  # Truncate for query
            )
            rag_context.append(f"\n--- RAG for {section_name} ---")
            for text, score in chunks:
                rag_context.append(f"[score: {score:.3f}] {text[:300]}")
    
    rag_text = "\n".join(rag_context)
    
    user_content = f"""
PARSED REPORT:
Title: {parsed.get('title', 'N/A')}
Abstract: {parsed.get('abstract', 'N/A')[:1000]}

Methodology: {parsed.get('methodology', 'N/A')[:2000]}

Results: {parsed.get('results', 'N/A')[:2000]}

RAG CONTEXT (Rubric criteria + high-quality past reports):
{rag_text[:6000]}

Score each criterion 0.0-5.0. Cite RAG evidence in your justifications.
"""
    
    critic_report = state.get("critic_report", {})
    if critic_report:
        user_content += f"\n\nPREVIOUS CRITIC FEEDBACK (address these):\n{critic_report.get('overall_assessment', 'None')}"
    
    review_scores = run_agent(
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        user_content=user_content,
        output_schema=ReviewScores,
        temperature=0.1,
    )
    
    state["review_scores"] = review_scores.model_dump()
    logger.info("Reviewer node complete")
    
    return state