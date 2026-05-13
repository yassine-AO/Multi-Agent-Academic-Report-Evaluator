"""
Base agent class for all specialized agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from ..models.schemas import ProcessingState

class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, state: ProcessingState) -> ProcessingState:
        """
        Process the current state and return updated state.

        Args:
            state: Current processing state

        Returns:
            Updated processing state
        """
        ...

    def update_state(self, state: ProcessingState, **kwargs) -> ProcessingState:
        """Helper method to update state fields."""
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state