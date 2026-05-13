"""
Data models and schemas for the application.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ReportStatus(str, Enum):
    """Status of report processing."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    EVALUATED = "evaluated"
    FAILED = "failed"

class EvaluationCriteria(BaseModel):
    """Individual evaluation criterion from rubric."""
    id: str
    name: str
    description: str
    weight: float = Field(..., ge=0.0, le=1.0)
    score: Optional[float] = None
    feedback: Optional[str] = None

class ReportInput(BaseModel):
    """Input academic report data."""
    id: str
    title: str
    author: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EvaluationOutput(BaseModel):
    """Final evaluation output."""
    report_id: str
    overall_score: float
    criteria_scores: List[EvaluationCriteria]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    summary: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ProcessingState(BaseModel):
    """State for the LangGraph workflow."""
    report: ReportInput
    status: ReportStatus
    retrieved_context: List[str] = Field(default_factory=list)
    current_evaluation: Optional[EvaluationOutput] = None
    reviewer_notes: List[str] = Field(default_factory=list)
    critic_notes: List[str] = Field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 3