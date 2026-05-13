"""
Output generator that formats the final evaluation for presentation.
"""

from .base_agent import BaseAgent
from ..models.schemas import ProcessingState

class OutputGeneratorAgent(BaseAgent):
    """Agent that generates the final formatted output."""

    def __init__(self):
        super().__init__("OutputGenerator")

    def process(self, state: ProcessingState) -> ProcessingState:
        """
        Generate the final formatted output from the evaluation.

        Args:
            state: Current processing state

        Returns:
            Updated processing state with final output format
        """
        # Placeholder: Implementation would format the final evaluation
        # into JSON, PDF, HTML, or other required format
        return state