"""
Deliberation / Convergence Check.
Decides whether to loop back to Reviewer or proceed to Rapporteur.
"""

from src.config import settings
from src.utils import get_logger, calculate_score_delta

logger = get_logger(__name__)


def deliberation_node(state: dict) -> str:
    """
    LangGraph conditional edge: Check if scores have converged.
    
    Args:
        state: EvaluationState with 'critic_report' and 'deliberation_count'.
    
    Returns:
        "reviewer" to loop back, or "rapporteur" to finalize.
    """
    critic_report = state.get("critic_report", {})
    deliberation_count = state.get("deliberation_count", 0)
    
    max_delta = critic_report.get("max_score_delta", 0.0)
    
    logger.info(
        "Deliberation check",
        extra={
            "max_delta": max_delta,
            "threshold": settings.convergence_threshold,
            "deliberation_count": deliberation_count,
            "max_loops": settings.max_deliberation_loops,
        }
    )
    
    import time
    
    # Convergence condition: delta small OR max loops reached
    if max_delta > settings.convergence_threshold and deliberation_count < settings.max_deliberation_loops:
        logger.info("Scores not converged -> loop back to reviewer")
        time.sleep(5)  # Brief pause before loop-back
        return "reviewer"
    else:
        logger.info("Scores converged -> proceed to rapporteur")
        state["is_converged"] = True
        return "rapporteur"