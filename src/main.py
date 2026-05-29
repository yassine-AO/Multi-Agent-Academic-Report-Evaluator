"""
FastAPI application entrypoint.
Lifespan: startup initializes DB, embeddings, graph.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.db import health_check as db_health_check
from src.models.schemas import EvaluationResponse, HealthResponse
from src.utils import get_logger
from src.workflow import get_graph

logger = get_logger(__name__)

# Store compiled graph and other heavy objects in app state
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting up...")
    
    # Initialize heavy resources
    app.state.graph = get_graph()
    logger.info("LangGraph compiled and stored in app state")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Automated Academic Project Evaluator",
    description="Multi-agent AI system for evaluating student PFE reports",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allow all origins in dev, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """System health check."""
    db_status = db_health_check()
    return {
        "status": "healthy" if db_status["status"] == "healthy" else "degraded",
        "api": "up",
        "database": db_status,
    }


@app.get("/collections")
async def collections():
    """List ChromaDB collections with document counts."""
    from src.db import get_or_create_collection, RUBRICS_COLLECTION, PAST_REPORTS_COLLECTION
    
    result = {}
    for name in [RUBRICS_COLLECTION, PAST_REPORTS_COLLECTION]:
        try:
            col = get_or_create_collection(name)
            result[name] = {"document_count": col.count()}
        except Exception as e:
            result[name] = {"error": str(e)}
    
    return result


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    file: UploadFile = File(..., description="Student PFE report PDF"),
    student_name: str = "",
    program: str = "",
    year: str = "",
):
    """
    Evaluate a student project report.
    
    Upload a PDF file. The system will:
    1. Parse structure
    2. Review and score against rubrics
    3. Critically challenge scores
    4. Deliberate until converged
    5. Return final evaluation report
    """
    import time
    import uuid
    from pathlib import Path
    
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    logger.info(
        f"Evaluation request {request_id}",
        extra={
            "report_filename": file.filename,
            "student": student_name,
            "program": program,
        }
    )
    
    # Save uploaded file to temp location
    temp_path = f"/tmp/eval_{request_id}.pdf"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Run the graph
    graph = app.state.graph
    
    initial_state = {
        "pdf_path": temp_path,
        "parsed_report": {},
        "review_scores": {},
        "critic_report": {},
        "final_report": {},
        "deliberation_count": 0,
        "is_converged": False,
    }
    
    result = graph.invoke(initial_state)
    
    # Build response
    processing_time = round(time.time() - start_time, 2)
    
    response = {
        "request_id": request_id,
        "status": "completed",
        "processing_time_seconds": processing_time,
        "report": result.get("final_report", {}),
    }
    
    logger.info(
        f"Evaluation {request_id} complete",
        extra={"processing_time": processing_time},
    )
    
    return response