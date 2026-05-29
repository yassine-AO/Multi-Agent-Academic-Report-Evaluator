"""
Rapporteur Agent Node.
Synthesizes final evaluation from converged deliberation data.
"""
from src.config import settings
from src.agents.base_agent import run_agent
from src.agents.prompts import RAPPORTEUR_SYSTEM_PROMPT
from src.models.schemas import FinalEvaluationReport
from src.utils import get_logger
from datetime import datetime, timezone

logger = get_logger(__name__)


def rapporteur_node(state: dict) -> dict:
    """
    LangGraph node: Generate final evaluation report.
    
    Args:
        state: EvaluationState with converged 'review_scores'.
    
    Returns:
        Updated state with 'final_report' field.
    """
    parsed = state["parsed_report"]
    review_scores = state["review_scores"]
    critic_report = state.get("critic_report", {})
    deliberation_count = state.get("deliberation_count", 0)
    
    logger.info("Rapporteur node starting")
    
    user_content = f"""
PROJECT TITLE: {parsed.get('title', 'N/A')}

FINAL CONVERGED SCORES:
{review_scores}

CRITIC FEEDBACK (incorporated):
{ critic_report.get('overall_assessment', 'No major challenges') }

DELIBERATION ROUNDS: {deliberation_count}

Synthesize a fair, balanced final evaluation.
"""
    
    final_report = run_agent(
        system_prompt=RAPPORTEUR_SYSTEM_PROMPT,
        user_content=user_content,
        output_schema=FinalEvaluationReport,
        temperature=0.2,
    )
    
    # Fill metadata
    final_report.project_title = parsed.get("title", "Unknown")
    final_report.deliberation_rounds = deliberation_count
    final_report.metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_used": settings.llm_model_name,
        "tokens_consumed": 0,  # Could track via LangSmith
    }
    
    state["final_report"] = final_report.model_dump()
    logger.info("Rapporteur node complete")
    
    return state