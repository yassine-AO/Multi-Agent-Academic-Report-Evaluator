"""
Data models and schemas for the application.
"""
from typing import TypedDict, Any
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel
from typing import Any

class ReportStatus(str, Enum):
    """Status of report processing."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    EVALUATED = "evaluated"
    FAILED = "failed"

class EvaluationState(TypedDict, total=False):
    """
    Shared state object passed between all LangGraph nodes.
    total=False means fields are optional — nodes add them gradually.
    """
    pdf_path: str
    parsed_report: dict[str, Any]
    review_scores: dict[str, Any]
    critic_report: dict[str, Any]
    final_report: dict[str, Any]
    deliberation_count: int
    is_converged: bool

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


class ParsedReport(BaseModel):
    """Schema for structured document parsing output."""
    title: str = ""
    abstract: str = ""
    introduction: str = ""
    methodology: str = ""
    results: str = ""
    conclusion: str = ""
    bibliography: str = ""
    raw_text: str = ""
    page_count: int = 0
    has_ocr_fallback: bool = False


class CriterionScore(BaseModel):
    """Schema for score and justification of a single rubric criterion."""
    criterion_name: str
    score: float = Field(..., ge=0.0, le=5.0)
    justification: str
    rag_evidence: List[str] = Field(default_factory=list)


class ReviewScores(BaseModel):
    """Schema for the Reviewer agent's score output."""
    scores: List[CriterionScore] = Field(default_factory=list)
    overall_score: float = Field(..., ge=0.0, le=5.0)
    summary: str


class CriticChallenge(BaseModel):
    """Schema for a single score challenge by the Critic agent."""
    criterion_name: str
    original_score: float = Field(..., ge=0.0, le=5.0)
    suggested_score: float = Field(..., ge=0.0, le=5.0)
    challenge_reasoning: str
    supporting_evidence: List[str] = Field(default_factory=list)


class CriticReport(BaseModel):
    """Schema for the Critic agent's feedback output."""
    challenges: List[CriticChallenge] = Field(default_factory=list)
    max_score_delta: float = 0.0
    overall_assessment: str


class FinalEvaluationReport(BaseModel):
    """Schema for the Rapporteur agent's final synthesized evaluation report."""
    project_title: str
    final_scores: List[CriterionScore] = Field(default_factory=list)
    global_grade: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    improvement_recommendations: List[str] = Field(default_factory=list)
    deliberation_rounds: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    status: str
    api: str
    database: dict[str, Any]


class EvaluationResponse(BaseModel):
    request_id: str
    status: str
    processing_time_seconds: float
    report: dict[str, Any]