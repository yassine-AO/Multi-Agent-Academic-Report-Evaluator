"""
LangGraph workflow definition.
Wires all agent nodes into a stateful, cyclic graph.
"""

from langgraph.graph import StateGraph, END

from src.models.schemas import EvaluationState
from src.agents import (
    parser_node,
    reviewer_node,
    critic_node,
    deliberation_node,
    rapporteur_node,
)
from src.utils import get_logger

logger = get_logger(__name__)


def build_evaluation_graph():
    """
    Build and compile the LangGraph evaluation workflow.
    
    Graph structure:
        START → parser → reviewer → critic → deliberation
                            ↑___________|
                            (loop if not converged)
        deliberation → rapporteur → END
                            (if converged)
    """
    # Initialize graph with shared state type
    workflow = StateGraph(EvaluationState)
    
    # Add nodes
    workflow.add_node("parser", parser_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("rapporteur", rapporteur_node)
    
    # Deliberation is not a node — it's a conditional edge function
    # We don't add it as a node; we use it in add_conditional_edges
    
    # Define edges
    workflow.set_entry_point("parser")
    workflow.add_edge("parser", "reviewer")
    workflow.add_edge("reviewer", "critic")
    
    # Conditional edge: critic → either reviewer (loop) or rapporteur (done)
    workflow.add_conditional_edges(
        "critic",
        deliberation_node,  # Returns "reviewer" or "rapporteur"
        {
            "reviewer": "reviewer",      # Loop back
            "rapporteur": "rapporteur",  # Proceed to final
        }
    )
    
    # Final edge
    workflow.add_edge("rapporteur", END)
    
    # Compile
    graph = workflow.compile()
    logger.info("Evaluation graph compiled successfully")
    
    return graph


# Singleton — compile once, reuse
_compiled_graph = None


def get_graph():
    """Get or create the compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_evaluation_graph()
    return _compiled_graph