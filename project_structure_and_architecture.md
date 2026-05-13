# Multi-Agent Academic Report Evaluator

## 1. Project Overview

**Project Name:** Multi-Agent Academic Report Evaluator

**What is this project?**
The Multi-Agent Academic Report Evaluator is a multi-agent AI system designed to read a student's Projet de Fin d'Études (PFE) report PDF and produce a structured, justified academic evaluation. It simulates the deliberation of a human jury.

Instead of relying on a single AI model guessing a score from memory, this system uses Retrieval-Augmented Generation (RAG) to ground its evaluation in real, local documents. It compares the submitted PDF against a database of past high-scoring reports and the official university grading rubric.

**How is it used in the real world?**
Academic institutions evaluate hundreds of PFE reports every cycle. Juries are often overworked, which can lead to subjective or inconsistent grading across different panels. This system acts as a standardized, transparent baseline assistant. A professor can use it to pre-screen reports to flag weak ones before oral defenses, or students could use it as a self-assessment tool before final submission.

---

## 2. System Interface & API Design

The system is deployed as a backend service, allowing web interfaces or internal university tools to easily communicate with it.

**Framework:** FastAPI

**Why FastAPI is the perfect fit:**
- **Native Pydantic Integration:** LangGraph agents rely heavily on Pydantic models (e.g., `ParsedReport`, `ReviewScores`) to enforce strict rules (like clamping scores between 0.0 and 5.0). FastAPI uses Pydantic natively, meaning agent data structures and API data structures are seamlessly compatible.
- **Asynchronous Execution:** Processing a PDF and running multiple LLM calls takes time. FastAPI handles long-running asynchronous tasks exceptionally well.
- **Auto-Generated Documentation:** FastAPI automatically builds a Swagger UI at `/docs`, making it incredibly easy to test PDF uploads and display structured JSON responses.

**API Endpoints:**
- `POST /evaluate` — Accepts a PDF file upload (`UploadFile`), saves to a temp path, invokes the compiled LangGraph graph, returns the `FinalEvaluationReport` as JSON.
- `GET /health` — Returns system health status (API up, ChromaDB connected, collection count).
- `GET /collections` — Returns metadata about ingested ChromaDB collections (document count, last updated).

---

## 3. The Multi-Agent Architecture

The core intelligence is orchestrated by **LangGraph**. Standard LangChain is a straight pipeline (a Directed Acyclic Graph), but this project requires a feedback loop where a Critic challenges a Reviewer. LangGraph was built exactly for this stateful, cyclical workflow.

### Agent Flow

```
1. Submit PDF
        ↓
   ┌─────────────┐
   │ ParserAgent │  ← Extracts structure using PyMuPDF. Uses OCR if PDF is scanned.
   └─────────────┘
        ↓  ParsedReport
   ┌──────────────┐
   │ ReviewerAgent│  ← Queries RAG for rubric criteria & high-scoring past reports.
   └──────────────┘
        ↓  ReviewScores
   ┌─────────────┐
   │  CriticAgent │  ← Adversarial agent. Queries RAG for low-scoring reports to find flaws.
   └─────────────┘
        ↓  CriticReport
   ┌─────────────────┐
   │ ConvergenceCheck│  ← Loops if score delta > 0.3 AND iterations < 3
   └─────────────────┘
        ↓  Passed (Converged)
   ┌───────────────┐
   │ RapporteurAgent│  ← Synthesizes final evaluation report
   └───────────────┘
        ↓  FinalEvaluationReport
```

### Detailed Agent Roles

| Agent | Role | Output |
|-------|------|--------|
| **Parser** | Extracts the physical structure of the PDF (abstract, methodology, bibliography). Focuses on structural parsing, not semantic chunking. | `ParsedReport` |
| **Reviewer** | The primary grader. Receives the parsed text, queries the RAG pipeline for rubric criteria, and outputs strict `ReviewScores` (0.0 to 5.0) with justifications. | `ReviewScores` |
| **Critic** | The adversarial agent. Independently re-evaluates the Reviewer's scores by looking at low-scoring examples from the database. Outputs a `CriticReport` containing challenges. If the Critic strongly disagrees with the Reviewer, the system loops back to force a re-evaluation. | `CriticReport` |
| **Rapporteur** | The synthesizer. Takes the final, converged scores and generates a clean, comprehensive evaluation report including global grades, strengths, weaknesses, and improvement recommendations. | `FinalEvaluationReport` |

### Conditional Edge Logic (Convergence Check)

```python
convergence_check(state: EvaluationState) -> str:
    if state["critic_report"].max_score_delta > 0.3 AND state["deliberation_count"] < 3:
        return "reviewer"  # loops back for another round
    else:
        state["is_converged"] = True
        return "rapporteur"  # proceeds to final synthesis
```

This is the **cyclic feedback loop** that differentiates LangGraph from plain LangChain.

---

## 4. Advanced Retrieval-Augmented Generation (RAG)

The RAG pipeline is the brain of the operation. It ensures the LLM isn't just guessing, but citing real university standards and historical project data.

### Phase A: Data Ingestion Pipeline

Before the system runs, we must process the university's corpus of past reports and the official rubric.

```
Past PFE Reports & Rubric
        ↓
   Text Extraction (PyMuPDF)
        ↓
   SemanticSplitterNodeParser
        ↓
   OpenAI text-embedding-3-small
        ↓
   ChromaDB
```

**Key Concept:** We do NOT use fixed-size token chunking (which often breaks sentences in half). We use LlamaIndex's `SemanticSplitterNodeParser` that computes embedding similarity between adjacent sentences and splits text only when the semantic topic changes.

### Phase B: Query & Retrieval Pipeline

When the Reviewer or Critic agents need information to justify a score, they use a highly optimized retrieval chain:

```
Agent Query
    ↓
HyDE Transform
    ↓
QueryFusionRetriever
    ↓                    ↓
Dense Vector Search    BM25 Keyword Search
    ↓                    ↓
         ChromaDB
            ↓
    Top 15 Candidate Chunks
            ↓
    Cross-Encoder Reranker
            ↓
    Top 3-5 High Precision Chunks
            ↓
    LLM Context Window
```

**Key Techniques:**
- **HyDE (Hypothetical Document Embeddings):** Instead of searching the database with the agent's raw question, HyDE transforms the query into a hypothetical "ideal answer" before searching. This vastly improves semantic retrieval accuracy.
- **Hybrid Search:** Combines semantic dense vector search (understanding meaning) with BM25 (exact keyword matching).
- **Cross-Encoder Reranker:** We extract 15 documents initially, then use a pre-trained local model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to accurately re-score and filter them down to the absolute best 5 chunks. This prevents blowing out the LLM's token limit and saves cost.

---

## 5. Verified Technology Stack Breakdown

This stack is chosen to reflect modern, production-grade AI engineering. It uses the best tool for every specific job rather than relying on a single monolithic framework.

| Layer | Technology | Role |
|-------|-----------|------|
| **API Framework** | FastAPI | HTTP layer, PDF upload, JSON response, Swagger UI at `/docs` |
| **AI Orchestration** | LangGraph v1 (on top of LangChain) | Stateful multi-agent graph with cyclic Reviewer ↔ Critic loop |
| **RAG Engine** | LlamaIndex | Semantic chunking, HyDE, hybrid retrieval, cross-encoder reranking |
| **Vector Database** | ChromaDB | Persistent local vector store with metadata filtering |
| **LLM** | GPT-4o mini / Claude Haiku | Fast, cost-effective models that excel at strict structured JSON output |
| **Embeddings** | OpenAI `text-embedding-3-small` | Dense vector embeddings for semantic search |
| **Data Validation** | Pydantic v2 | Strict typed schemas, score clamping (0.0–5.0) |
| **PDF Parsing** | PyMuPDF (fitz) + OCR fallback | Text & structure extraction from PDF reports |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local cross-encoder model for precision reranking |
| **Observability** | LangSmith | Full LLM call tracing, prompt/response/token logging |

---

## 6. Complete File Tree

```
Multi-Agent-Academic-Report-Evaluator/
│
├── .env                          # Environment variables (API keys, model config)
├── .env.example                  # Template for .env — committed to Git
├── .gitignore                    # Ignores .env, __pycache__, chroma_db/, etc.
├── README.md                     # Project overview, setup, usage instructions
├── PROJECT_STRUCTURE.md          # This file — engineering map of the codebase
├── requirements.txt              # Pinned Python dependencies
├── pyproject.toml                # Project metadata and build configuration
│
├── app/                          # — MAIN APPLICATION PACKAGE —
│   ├── __init__.py               # Makes `app` a Python package
│   ├── main.py                   # FastAPI application entrypoint & lifespan
│   ├── config.py                 # Centralized configuration loader (reads .env)
│   │
│   ├── api/                      # — HTTP LAYER —
│   │   ├── __init__.py
│   │   ├── router.py             # API route definitions (POST /evaluate, GET /health)
│   │   ├── dependencies.py       # FastAPI dependency injection (graph instance, DB client)
│   │   └── middleware.py         # CORS, request logging, error handling middleware
│   │
│   ├── schemas/                  # — PYDANTIC DATA CONTRACTS —
│   │   ├── __init__.py
│   │   ├── state.py              # EvaluationState — the LangGraph shared state TypedDict
│   │   ├── parsed_report.py      # ParsedReport — output schema of the Parser agent
│   │   ├── review_scores.py      # ReviewScores — output schema of the Reviewer agent
│   │   ├── critic_report.py      # CriticReport — output schema of the Critic agent
│   │   ├── final_report.py       # FinalEvaluationReport — output schema of the Rapporteur agent
│   │   └── api_models.py         # Request/Response models for the FastAPI endpoints
│   │
│   ├── agents/                   # — LANGGRAPH MULTI-AGENT SYSTEM —
│   │   ├── __init__.py
│   │   ├── graph.py              # LangGraph graph definition — wires all nodes & edges
│   │   ├── nodes/                # Individual agent node implementations
│   │   │   ├── __init__.py
│   │   │   ├── parser.py         # Parser Agent — PDF text & structure extraction
│   │   │   ├── reviewer.py       # Reviewer Agent — RAG-grounded scoring
│   │   │   ├── critic.py         # Critic Agent — adversarial challenge
│   │   │   └── rapporteur.py     # Rapporteur Agent — final report synthesis
│   │   ├── edges.py              # Conditional edge logic (convergence check routing)
│   │   └── prompts/              # System prompts for each agent
│   │       ├── __init__.py
│   │       ├── parser_prompt.py    # Prompt template for the Parser agent
│   │       ├── reviewer_prompt.py  # Prompt template for the Reviewer agent
│   │       ├── critic_prompt.py    # Prompt template for the Critic agent
│   │       └── rapporteur_prompt.py # Prompt template for the Rapporteur agent
│   │
│   ├── rag/                      # — RAG PIPELINE (LlamaIndex) —
│   │   ├── __init__.py
│   │   ├── ingestion.py          # Data ingestion pipeline (PDF → chunks → ChromaDB)
│   │   ├── retriever.py          # Query & retrieval pipeline (HyDE + hybrid + rerank)
│   │   ├── chunking.py           # SemanticSplitterNodeParser configuration
│   │   ├── embeddings.py         # Embedding model initialization (text-embedding-3-small)
│   │   └── reranker.py           # Cross-encoder reranker setup (ms-marco-MiniLM-L-6-v2)
│   │
│   ├── db/                       # — VECTOR DATABASE LAYER —
│   │   ├── __init__.py
│   │   ├── chroma_client.py      # ChromaDB client initialization & collection manager
│   │   └── collections.py        # Collection name constants & metadata schema definitions
│   │
│   ├── services/                 # — BUSINESS LOGIC SERVICES —
│   │   ├── __init__.py
│   │   ├── pdf_parser.py         # PyMuPDF-based PDF text extraction with OCR fallback
│   │   └── llm_client.py         # LLM client factory (GPT-4o-mini / Claude Haiku init)
│   │
│   └── utils/                    # — SHARED UTILITIES —
│       ├── __init__.py
│       ├── logging.py            # Structured logging configuration
│       └── helpers.py            # Generic helper functions (score clamping, text cleaning)
│
├── data/                         # — REFERENCE DATA (Git-tracked) —
│   ├── rubrics/                  # Official university grading rubrics (PDF/text)
│   │   └── .gitkeep
│   └── past_reports/             # Corpus of past PFE reports for RAG ingestion
│       └── .gitkeep
│
├── chroma_db/                    # — PERSISTED VECTOR STORE (Git-ignored) —
│   └── .gitkeep                  # Created at runtime by ChromaDB
│
├── scripts/                      # — OPERATIONAL SCRIPTS —
│   ├── ingest.py                 # CLI script to run the full ingestion pipeline
│   └── seed_db.py                # Seeds ChromaDB with initial rubrics + past reports
│
└── tests/                        # — TEST SUITE —
    ├── __init__.py
    ├── conftest.py               # Shared pytest fixtures (mock LLM, test DB, sample PDFs)
    ├── test_api.py               # Integration tests for FastAPI endpoints
    ├── test_agents/              # Unit tests for each agent node
    │   ├── __init__.py
    │   ├── test_parser.py
    │   ├── test_reviewer.py
    │   ├── test_critic.py
    │   └── test_rapporteur.py
    ├── test_rag/                 # Tests for ingestion and retrieval pipelines
    │   ├── __init__.py
    │   ├── test_ingestion.py
    │   └── test_retriever.py
    └── test_services/            # Tests for PDF parsing and LLM client
        ├── __init__.py
        ├── test_pdf_parser.py
        └── test_llm_client.py
```

---

## 7. Detailed Module Breakdown

### Root Files

| File | Purpose |
|------|---------|
| `.env` | Stores secrets: `OPENAI_API_KEY`, `LANGCHAIN_API_KEY` (for LangSmith), `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_PROJECT`, model selection (`LLM_MODEL_NAME`), ChromaDB path, embedding model name. **Never committed to Git.** |
| `.env.example` | Committed template showing all required env vars with placeholder values so any developer can onboard instantly. |
| `.gitignore` | Ignores `.env`, `__pycache__/`, `chroma_db/`, `*.pyc`, `.venv/`, `.idea/`, `.vscode/`, `data/past_reports/*.pdf` (optional, if reports are too large). |
| `requirements.txt` | Pinned versions: `fastapi`, `uvicorn[standard]`, `python-multipart`, `langgraph`, `langchain`, `langchain-openai`, `langchain-anthropic`, `llama-index`, `llama-index-vector-stores-chroma`, `llama-index-embeddings-openai`, `chromadb`, `pymupdf`, `pydantic>=2.0`, `python-dotenv`, `sentence-transformers` (for cross-encoder reranker), `rank-bm25` (for BM25 keyword search), `langsmith`, `pytest`, `httpx` (for async test client). |
| `pyproject.toml` | PEP 621 project metadata — name, version, description, Python version requirement (`>=3.11`), author, license. Optional: tool configs for `pytest`, `ruff`, `mypy`. |

### `app/` — Main Application Package

#### `app/main.py` — FastAPI Entrypoint
- Creates the `FastAPI()` application instance.
- Implements `lifespan` async context manager (startup/shutdown):
  - **Startup:** Initializes ChromaDB client, loads embedding model, compiles the LangGraph graph, stores them in `app.state`.
  - **Shutdown:** Graceful cleanup of resources.
- Mounts the API router from `app.api.router`.
- Configures middleware (CORS, logging).

#### `app/config.py` — Configuration Loader
- Uses `pydantic-settings` (`BaseSettings`) to load and validate all environment variables from `.env`.
- Exposes a typed `Settings` object with fields: `openai_api_key`, `llm_model_name`, `embedding_model_name`, `chroma_db_path`, `langchain_api_key`, `langchain_tracing_v2`, `langchain_project`, `max_deliberation_loops`, `convergence_threshold`.
- Single import point — every module reads config from here, never from `os.environ` directly.

### `app/api/` — HTTP Layer

#### `app/api/router.py` — API Route Definitions
- `POST /evaluate`: Accepts a PDF file upload (`UploadFile`). Saves to a temp path, invokes the compiled LangGraph graph, returns the `FinalEvaluationReport` as JSON.
- `GET /health`: Returns system health status (API up, ChromaDB connected, collection count).
- `GET /collections`: Returns metadata about ingested ChromaDB collections (document count, last updated).

#### `app/api/dependencies.py` — Dependency Injection
- FastAPI `Depends()` functions that provide:
  - The compiled LangGraph `CompiledGraph` instance from `app.state`.
  - The ChromaDB client from `app.state`.
  - The `Settings` object.
- Keeps route handlers thin — business logic stays in agents/services.

#### `app/api/middleware.py` — Middleware Stack
- **CORS middleware:** Allows all origins for development (locked down in production).
- **Request logging middleware:** Logs incoming request method, path, and response time.
- **Global exception handler:** Catches unhandled exceptions and returns structured JSON error responses instead of raw 500s.

### `app/schemas/` — Pydantic Data Contracts

These are the strict data contracts that flow through the entire LangGraph pipeline. Every agent reads from and writes to the shared `EvaluationState`.

#### `app/schemas/state.py` — LangGraph Shared State
- Defines `EvaluationState(TypedDict)` — the single source of truth passed between all agent nodes.
- Fields include: `pdf_path`, `parsed_report`, `review_scores`, `critic_report`, `final_report`, `deliberation_count`, `is_converged`.
- This is what LangGraph uses to route data through the graph.

#### `app/schemas/parsed_report.py` — Parser Output
- `ParsedReport(BaseModel)`: Structured extraction result.
- Fields: `title`, `abstract`, `introduction`, `methodology`, `results`, `conclusion`, `bibliography`, `raw_text`, `page_count`, `has_ocr_fallback`.

#### `app/schemas/review_scores.py` — Reviewer Output
- `CriterionScore(BaseModel)`: Single rubric criterion score.
  - Fields: `criterion_name`, `score` (float, 0.0–5.0, validated with `@field_validator`), `justification`, `rag_evidence` (the retrieved chunks that support this score).
- `ReviewScores(BaseModel)`: Collection of all criterion scores plus `overall_score` and `summary`.

#### `app/schemas/critic_report.py` — Critic Output
- `CriticChallenge(BaseModel)`: A single challenge to a Reviewer score.
  - Fields: `criterion_name`, `original_score`, `suggested_score`, `challenge_reasoning`, `supporting_evidence`.
- `CriticReport(BaseModel)`: List of challenges + `max_score_delta` (used by convergence check) + `overall_assessment`.

#### `app/schemas/final_report.py` — Rapporteur Output
- `FinalEvaluationReport(BaseModel)`: The complete evaluation delivered to the user.
- Fields: `project_title`, `final_scores` (list of `CriterionScore`), `global_grade`, `strengths`, `weaknesses`, `improvement_recommendations`, `deliberation_rounds`, `metadata` (timestamps, model used, tokens consumed).

#### `app/schemas/api_models.py` — API Request/Response Models
- `EvaluationRequest`: Metadata that accompanies the PDF upload (optional student name, program, year).
- `EvaluationResponse`: Wraps `FinalEvaluationReport` with `request_id`, `processing_time_seconds`, `status`.
- `HealthResponse`: API health check response model.

### `app/agents/` — LangGraph Multi-Agent System

This is the **core intelligence** of the system. The architecture uses LangGraph to wire 4 agent nodes into a stateful, cyclic graph.

#### `app/agents/graph.py` — LangGraph Graph Definition
- Builds the `StateGraph(EvaluationState)`.
- Adds nodes: `parser`, `reviewer`, `critic`, `rapporteur`.
- Adds edges:
  - `START → parser`
  - `parser → reviewer`
  - `reviewer → critic`
  - `critic → convergence_check` (conditional edge defined in `edges.py`)
  - `convergence_check` routes to either `reviewer` (loop back) or `rapporteur` (proceed).
  - `rapporteur → END`
- Compiles the graph and exposes it for the FastAPI lifespan to store.

#### `app/agents/nodes/parser.py` — Parser Agent Node
- Receives `EvaluationState` with `pdf_path`.
- Calls `app.services.pdf_parser` to extract raw text.
- Optionally uses an LLM call to identify sections (abstract, methodology, etc.) from the raw text.
- Returns updated state with `parsed_report: ParsedReport`.

#### `app/agents/nodes/reviewer.py` — Reviewer Agent Node
- Receives state with `parsed_report`.
- For each rubric criterion, queries the RAG pipeline (`app.rag.retriever`) to fetch:
  - Relevant rubric criteria descriptions.
  - Matching sections from high-scoring past reports.
- Sends the parsed report sections + RAG context to the LLM with the reviewer system prompt.
- LLM returns structured `ReviewScores` (scores clamped 0.0–5.0 by Pydantic).
- Returns updated state with `review_scores`.

#### `app/agents/nodes/critic.py` — Critic Agent Node
- Receives state with `parsed_report` and `review_scores`.
- Queries the RAG pipeline for **low-scoring** past reports to find counter-examples.
- Sends the Reviewer's scores + counter-evidence to the LLM with the adversarial critic prompt.
- LLM returns `CriticReport` with specific challenges and a `max_score_delta`.
- Returns updated state with `critic_report` and incremented `deliberation_count`.

#### `app/agents/nodes/rapporteur.py` — Rapporteur Agent Node
- Receives the fully converged state (final `review_scores` after deliberation).
- Synthesizes a clean, comprehensive `FinalEvaluationReport`.
- Computes global grade, lists strengths/weaknesses, generates improvement recommendations.
- Returns updated state with `final_report`.

#### `app/agents/edges.py` — Conditional Edge Logic
- Implements `convergence_check(state: EvaluationState) -> str`:
  - If `state["critic_report"].max_score_delta > 0.3` AND `state["deliberation_count"] < 3`:
    - Returns `"reviewer"` → loops back for another round.
  - Else:
    - Sets `state["is_converged"] = True`.
    - Returns `"rapporteur"` → proceeds to final synthesis.

#### `app/agents/prompts/` — Agent System Prompts

Each file contains a system prompt template as a Python string constant. Prompts are separated from logic so they can be iterated on without touching agent code.

| File | Prompt Responsibility |
|------|----------------------|
| `parser_prompt.py` | Instructs the LLM to identify and extract specific academic report sections from raw text. |
| `reviewer_prompt.py` | Instructs the LLM to act as a strict academic reviewer, score each criterion 0.0–5.0, cite RAG evidence, and output valid JSON matching `ReviewScores`. |
| `critic_prompt.py` | Instructs the LLM to act as an adversarial critic, challenge each score with counter-evidence, and output valid JSON matching `CriticReport`. |
| `rapporteur_prompt.py` | Instructs the LLM to synthesize all deliberation data into a clean final report with actionable recommendations. |

### `app/rag/` — RAG Pipeline (LlamaIndex)

The RAG system is split into two distinct phases: **Ingestion** (offline, one-time) and **Retrieval** (runtime, per-query).

#### `app/rag/ingestion.py` — Data Ingestion Pipeline
- Phase A from the architecture diagram.
- Reads PDF files from `data/rubrics/` and `data/past_reports/`.
- Extracts text using PyMuPDF.
- Applies `SemanticSplitterNodeParser` (from `chunking.py`) to split text into semantically coherent chunks.
- Embeds chunks using the configured embedding model (from `embeddings.py`).
- Stores embedded chunks into ChromaDB with metadata tags (`source_type: "rubric" | "past_report"`, `quality_tier: "high" | "low"`, `filename`, `page_number`).

#### `app/rag/retriever.py` — Query & Retrieval Pipeline
- Phase B from the architecture diagram.
- Implements the full retrieval chain:
  1. **HyDE Transform:** Takes the agent's raw query, asks the LLM to generate a hypothetical ideal answer, then uses that as the search query.
  2. **QueryFusionRetriever:** Runs the HyDE-transformed query through both dense vector search (ChromaDB) and BM25 keyword search simultaneously.
  3. **Cross-Encoder Reranking:** Takes the top 15 candidate chunks from hybrid search, re-scores them with `cross-encoder/ms-marco-MiniLM-L-6-v2`, returns only the top 3–5 highest-precision chunks.
- Supports metadata filtering (e.g., Reviewer queries only `quality_tier: "high"`, Critic queries only `quality_tier: "low"`).

#### `app/rag/chunking.py` — Semantic Chunking Configuration
- Configures LlamaIndex's `SemanticSplitterNodeParser`.
- Instead of fixed-size token windows (which break sentences), this computes embedding similarity between adjacent sentences and splits only when the semantic topic changes.
- Parameters: `buffer_size`, `breakpoint_percentile_threshold`, embedding model reference.

#### `app/rag/embeddings.py` — Embedding Model Setup
- Initializes and returns the `OpenAIEmbedding` model configured for `text-embedding-3-small`.
- Central point so the same embedding model is used consistently across ingestion and retrieval.

#### `app/rag/reranker.py` — Cross-Encoder Reranker
- Loads the `cross-encoder/ms-marco-MiniLM-L-6-v2` model from `sentence-transformers`.
- Exposes a `rerank(query, documents, top_k)` function.
- This is a local model — no API calls, no extra cost, runs on CPU.
- Prevents bloating the LLM context window by filtering 15 candidates down to 3–5.

### `app/db/` — Vector Database Layer

#### `app/db/chroma_client.py` — ChromaDB Client
- Initializes a persistent ChromaDB client pointing to `chroma_db/` directory.
- Provides `get_or_create_collection()` with the correct embedding function.
- Handles ChromaDB connection lifecycle.

#### `app/db/collections.py` — Collection Constants
- Defines collection name constants: `RUBRICS_COLLECTION`, `PAST_REPORTS_COLLECTION`.
- Defines metadata schema conventions (what metadata keys each document should have).
- Keeps collection naming centralized to avoid typos across the codebase.

### `app/services/` — Business Logic Services

#### `app/services/pdf_parser.py` — PDF Text Extraction
- Uses **PyMuPDF** (`fitz`) to extract text from uploaded PDF files.
- Implements **OCR fallback:** if a page yields no text (scanned document), falls back to OCR extraction.
- Returns structured text with page boundaries preserved.

#### `app/services/llm_client.py` — LLM Client Factory
- Factory function that returns the appropriate LangChain `ChatModel` based on config:
  - `ChatOpenAI(model="gpt-4o-mini")` or `ChatAnthropic(model="claude-3-haiku")`.
- Configures: temperature (low for structured output), max tokens, response format (JSON mode).
- Single initialization point — all agents use the same LLM instance.

### `app/utils/` — Shared Utilities

#### `app/utils/logging.py` — Logging Configuration
- Sets up structured logging (JSON format for production, pretty format for development).
- Configures log levels per module.
- Integrates with LangSmith tracing when `LANGCHAIN_TRACING_V2=true`.

#### `app/utils/helpers.py` — Generic Helpers
- `clamp_score(value: float, min_val=0.0, max_val=5.0) -> float`: Ensures scores stay in valid range.
- `clean_text(raw: str) -> str`: Normalizes whitespace, removes control characters.
- `calculate_score_delta(review: ReviewScores, critic: CriticReport) -> float`: Computes the maximum absolute difference between reviewer and critic scores (used in convergence check).

### `data/` — Reference Data

| Directory | Contents |
|-----------|----------|
| `data/rubrics/` | Official university grading rubrics in PDF or text format. These define the criteria the Reviewer agent uses. Ingested into ChromaDB with `source_type: "rubric"`. |
| `data/past_reports/` | Corpus of past PFE reports (both high-scoring and low-scoring). Ingested into ChromaDB with `source_type: "past_report"` and `quality_tier: "high"` or `"low"`. |

### `chroma_db/` — Persisted Vector Store
- Git-ignored. Created at runtime when `scripts/ingest.py` runs.
- Contains ChromaDB's internal SQLite database and HNSW index files.
- Persists between server restarts — no need to re-ingest unless data changes.

### `scripts/` — Operational Scripts

#### `scripts/ingest.py` — Ingestion CLI
- Standalone script to run the full ingestion pipeline.
- Reads all PDFs from `data/rubrics/` and `data/past_reports/`.
- Calls `app.rag.ingestion` to process, chunk, embed, and store in ChromaDB.
- Prints progress and final document counts.
- Run with: `python scripts/ingest.py`

#### `scripts/seed_db.py` — Database Seeder
- Seeds ChromaDB with a minimal initial dataset for development/testing.
- Can tag documents with metadata (`quality_tier`, `source_type`).
- Useful for local development without the full corpus.

### `tests/` — Test Suite

| File / Directory | What it tests |
|-----------------|---------------|
| `conftest.py` | Shared fixtures: mock LLM responses, in-memory ChromaDB client, sample PDF bytes, pre-built `EvaluationState` at various pipeline stages. |
| `test_api.py` | Integration tests using FastAPI `TestClient` — tests `/evaluate` endpoint with a real PDF, `/health` endpoint. |
| `test_agents/test_parser.py` | Parser node correctly extracts sections from a sample PDF. |
| `test_agents/test_reviewer.py` | Reviewer node returns valid `ReviewScores` with all criteria scored. |
| `test_agents/test_critic.py` | Critic node returns valid `CriticReport` with challenges. |
| `test_agents/test_rapporteur.py` | Rapporteur node produces a complete `FinalEvaluationReport`. |
| `test_rag/test_ingestion.py` | Ingestion pipeline correctly chunks and stores documents with correct metadata. |
| `test_rag/test_retriever.py` | Retrieval pipeline returns relevant chunks for a given query with correct reranking. |
| `test_services/test_pdf_parser.py` | PyMuPDF extraction works on normal and scanned PDFs. |
| `test_services/test_llm_client.py` | LLM client factory returns correct model type based on config. |

---

## 8. Data Flow Summary

```
CLIENT
POST /evaluate  →  FastAPI  →  LangGraph Engine
                                    ↓
                              ┌─────────┐  ← PyMuPDF+OCR
                              │ Parser  │
                              └─────────┘
                                    ↓ ParsedReport
                              ┌──────────┐  ← RAG (high)
                              │ Reviewer │
                              └──────────┘
                                    ↓ ReviewScores
                          ┌──────────────────┐
                          │  Deliberation    │
                          │      Loop        │
                          │  ┌─────────┐     │  ← RAG (low)
                          │  │  Critic │     │
                          │  └─────────┘     │
                          │        ↓         │
                          │  CriticReport    │
                          │        ↓         │
                          │  ┌─────────────┐ │
                          │  │ Convergence?│ │
                          │  │ delta>0.3 &&│ │
                          │  │ loops<3 →   │ │
                          │  │    LOOP     │ │
                          │  └─────────────┘ │
                          └──────────────────┘
                                    ↓ Converged
                              ┌───────────┐
                              │ Rapporteur│
                              └───────────┘
                                    ↓ FinalEvaluationReport
                          JSON Response  ←  FastAPI  ←
```

---

## 9. Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill environment variables
cp .env.example .env

# 3. Ingest reference documents into ChromaDB
python scripts/ingest.py

# 4. Start the API server
uvicorn app.main:app --reload --port 8000

# 5. Open Swagger UI
# http://localhost:8000/docs

# 6. Run tests
pytest tests/ -v
```
