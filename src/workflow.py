"""
Workflow orchestrator using LangGraph to manage the agent pipeline.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from .models.schemas import ProcessingState
from .agents.input_processor import InputProcessorAgent
from .agents.context_retriever import ContextRetrieverAgent
from .agents.reviewer import ReviewerAgent
from .agents.critic import CriticAgent
from .agents.deliberation import DeliberationAgent
from .agents.output_generator import OutputGeneratorAgent
from .rag.retriever import Retriever

class AgentWorkflow:
    """Manages the workflow of agents using LangGraph."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow with all agents.

        Returns:
            Configured StateGraph workflow
        """
        # Initialize agents
        input_processor = InputProcessorAgent()
        context_retriever = ContextRetrieverAgent(self.retriever)
        reviewer = ReviewerAgent()
        critic = CriticAgent()
        deliberator = DeliberationAgent()
        output_generator = OutputGeneratorAgent()

        # Create workflow graph
        workflow = StateGraph(ProcessingState)

        # Add nodes for each agent
        workflow.add_node("input_processor", input_processor.process)
        workflow.add_node("context_retriever", context_retriever.process)
        workflow.add_node("reviewer", reviewer.process)
        workflow.add_node("critic", critic.process)
        workflow.add_node("deliberator", deliberator.process)
        workflow.add_node("output_generator", output_generator.process)

        # Define the flow
        workflow.set_entry_point("input_processor")
        workflow.add_edge("input_processor", "context_retriever")
        workflow.add_edge("context_retriever", "reviewer")
        workflow.add_edge("reviewer", "critic")
        workflow.add_edge("critic", "deliberator")
        workflow.add_edge("deliberator", "output_generator")
        workflow.add_edge("output_generator", END)

        return workflow.compile()

    def run(self, initial_state: ProcessingState) -> ProcessingState:
        """
        Run the workflow from initial state to completion.

        Args:
            initial_state: Initial processing state

        Returns:
            Final processing state
        """
        # Placeholder: Actual implementation would invoke the workflow
        return self.workflow.invoke(initial_state)