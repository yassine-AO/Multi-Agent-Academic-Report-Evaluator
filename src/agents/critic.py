"""
Critic Agent Node.
Challenges Reviewer scores with counter-evidence from low-quality reports.
"""

from src.agents.base_agent import run_agent
from src.agents.prompts import CRITIC_SYSTEM_PROMPT
from src.models.schemas import CriticReport
from src.rag import retrieve_for_critic
from src.utils import get_logger

logger = get_logger(__name__)


def critic_node(state: dict) -> dict:
    """
    LangGraph node: Critically evaluate Reviewer's scores.
    
    Args:
        state: EvaluationState with 'parsed_report' and 'review_scores'.
    
    Returns:
        Updated state with 'critic_report' and incremented 'deliberation_count'.
    """
    parsed = state["parsed_report"]
    review_scores = state["review_scores"]
    logger.info("Critic node starting")
    
    # Get counter-evidence from low-quality reports
    rag_context = []
    
    for score_item in review_scores.get("scores", []):
        criterion = score_item["criterion_name"]
        chunks = retrieve_for_critic(criterion=criterion)
        
        rag_context.append(f"\n--- Counter-evidence for {criterion} ---")
        for text, score in chunks:
            rag_context.append(f"[score: {score:.3f}] {text[:300]}")
    
    rag_text = "\n".join(rag_context)
    
    user_content = f"""
ORIGINAL REPORT:
Title: {parsed.get('title', 'N/A')}
Methodology: {parsed.get('methodology', 'N/A')[:2000]}

REVIEWER SCORES:
{review_scores}

RAG CONTEXT (Low-quality past reports — common flaws):
{rag_text[:6000]}

Challenge scores where the Reviewer was too lenient. Provide specific evidence.
"""
    
    critic_report = run_agent(
        system_prompt=CRITIC_SYSTEM_PROMPT,
        user_content=user_content,
        output_schema=CriticReport,
        temperature=0.1,
    )
    
    # Update state
    state["critic_report"] = critic_report.model_dump()
    state["deliberation_count"] = state.get("deliberation_count", 0) + 1
    
    logger.info(f"Critic node complete. Deliberation count: {state['deliberation_count']}")
    
    return state