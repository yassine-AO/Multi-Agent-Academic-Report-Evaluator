"""
Agent responsible for retrieving relevant rubric context via RAG.
"""

from .base_agent import BaseAgent
from ..rag.retriever import Retriever
from ..models.schemas import ProcessingState

class ContextRetrieverAgent(BaseAgent):
    """Agent that retrieves relevant rubric context using RAG."""

    def __init__(self, retriever: Retriever):
        super().__init__("ContextRetriever")
        self.retriever = retriever

    def process(self, state: ProcessingState) -> ProcessingState:
        """
        Retrieve relevant context for the current report.

        Args:
            state: Current processing state

        Returns:
            Updated processing state with retrieved context
        """
        # Placeholder: Implementation would generate queries from report and retrieve relevant rubric sections
        query = f"Evaluation criteria for: {getattr(state.report, 'title', 'unknown report')}"
        context = self.retriever.retrieve_relevant_context(query, k=5)

        return self.update_state(
            state,
            retrieved_context=context
        )