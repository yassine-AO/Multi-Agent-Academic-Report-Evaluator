"""
Critic agent that reviews and challenges the reviewer's evaluation.
"""

from .base_agent import BaseAgent
from ..models.schemas import ProcessingState

class CriticAgent(BaseAgent):
    """Agent that reviews and challenges the reviewer's evaluation."""

    def __init__(self):
        super().__init__("Critic")

    def process(self, state: ProcessingState) -> ProcessingState:
        """
        Review the reviewer's evaluation and provide critical feedback.

        Args:
            state: Current processing state

        Returns:
            Updated processing state with critic feedback
        """
        # Placeholder: Implementation would analyze the reviewer's assessment
        # for biases, missed criteria, or inconsistencies
        if state.current_evaluation:
            # Add critical feedback to critic notes
            critic_feedback = "Critical review of evaluation pending."
        else:
            critic_feedback = "No evaluation to critique."

        return self.update_state(
            state,
            critic_notes=state.critic_notes + [critic_feedback]
        )