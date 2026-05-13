# **Technical Architecture Report**

## Automated Academic Project Evaluator **1. Project Overview**

**What** **is** **this** **project?**
The Automated Academic Project Evaluator is a multi-agent AI system designed to read a
student’s Projet de Fin d’Études (PFE) report PDF and produce a structured, justified academic
evaluation. It simulates the deliberation of a human jury panel.
Instead of relying on a single AI model guessing a score from memory, this system uses
**Retrieval-Augmented Generation (RAG)** to ground its evaluation in real, local documents.
It compares the submitted PDF against a database of past high-scoring reports and the official
university grading rubric.


**How** **is** **it** **used** **in** **the** **real** **world?**
Academic institutions evaluate hundreds of PFE reports every cycle. Juries are often overworked, which can lead to subjective or inconsistent grading across different panels. This system acts as a standardized, transparent baseline assistant. A professor can use it to pre-screen
reports to flag weak ones before oral defenses, or students could use it as a self-assessment tool
before final submission.

## **2. System Interface & API Design**


The system will be deployed as a backend service, allowing web interfaces or internal university
tools to easily communicate with it. Regarding your question about the API: **FastAPI** **is**
**absolutely** **the** **best** **choice** **for** **this** **project.**


**Why** **FastAPI** **is** **the** **perfect** **fit:**


- **Native** **Pydantic** **Integration:** Your LangGraph agents rely heavily on Pydantic models
(e.g., `ParsedReport`, `ReviewScores` ) to enforce strict rules (like clamping scores between 0.0
and 5.0). FastAPI uses Pydantic natively, meaning your agent data structures and your API
data structures are seamlessly linked.


- **Asynchronous** **Execution:** Processing a PDF and running multiple LLM calls takes time.
FastAPI handles long-running asynchronous tasks exceptionally well.


- **Auto-Generated** **Documentation:** FastAPI automatically builds a Swagger UI at `/docs`,
making it incredibly easy for you to test PDF uploads and display the structured JSON
responses during your jury presentation without needing to build a complex frontend.


1


## **3. The Multi-Agent Architecture**

The core intelligence is orchestrated by **LangGraph** . Standard LangChain is a straight pipeline
(a Directed Acyclic Graph), but this project requires a **feedback loop** where a Critic challenges
a Reviewer. LangGraph was built exactly for this stateful, cyclical workflow.

























2


**Detailed** **Agent** **Roles**


1. **Agent** **1:** **Parser** . Extracts the physical structure of the PDF (abstract, methodology, bibliography). It focuses on structural parsing, not semantic chunking. Output: `ParsedReport` .


2. **Agent** **2:** **Reviewer** . The primary grader. Receives the parsed text, queries the RAG
pipeline for rubric criteria, and outputs strict `ReviewScores` (0.0 to 5.0) with justifications.


3. **Agent** **3:** **Critic** . The adversarial agent. Independently re-evaluates the Reviewer’s scores
by looking at low-scoring examples from the database. It outputs a `CriticReport` containing
challenges. If the Critic strongly disagrees with the Reviewer, the system loops back to force
a reconciliation.


4. **Agent** **4:** **Rapporteur** . The synthesizer. Takes the final, converged scores and generates a
clean, comprehensive evaluation report including global grades, strengths, weaknesses, and
improvement recommendations.

## **4. Advanced Retrieval-Augmented Generation (RAG)**


The RAG pipeline is the brain of the operation. It ensures the LLM isn’t just guessing, but
citing real university standards and historical project data.


**Phase** **A:** **Data** **Ingestion** **Pipeline**


Before the system runs, we must process the university’s corpus of past reports and the rubric.


**Key** **Concept:** We do not use fixed-size token chunking (which often breaks sentences in
half). We use LlamaIndex’s `SemanticSplitterNodeParser` that computes embedding similarity between adjacent sentences and splits text only when the semantic topic changes.


**Phase** **B:** **Query** **&** **Retrieval** **Pipeline**


When the Reviewer or Critic agents need information to justify a score, they use a highly
optimized retrieval strategy.


3


**Key** **Concepts:**


- **HyDE** **(Hypothetical** **Document** **Embeddings):** Instead of searching the database with
the agent’s raw question, HyDE transforms the query into a hypothetical "ideal answer"
before searching. This vastly improves semantic retrieval accuracy.


- **Hybrid** **Search:** Combines semantic dense vector search (understanding meaning) with
BM25 (exact keyword matching).


- **Cross-Encoder** **Reranker:** We extract 15 documents initially, then use a pre-trained local
model ( `cross-encoder/ms-marco-MiniLM-L-6-v2` ) to accurately re-score and filter them
down to the absolute best 5 chunks. This prevents blowing out the LLM’s token limit and
saves money.

## **5. Verifed Technology Stack Breakdown**


This stack is chosen to reflect modern, production-grade AI engineering. It uses the best tool
for every specific job rather than relying on a single monolithic framework.


4


- **API** **Framework:** **FastAPI** . Selected for its asynchronous speed and native Pydantic
validation.


- **AI** **Orchestration:** **LangGraph** **v1** . Chosen over plain LangChain specifically to allow
the Reviewer _↔_ Critic cyclical feedback loops and state management.


- **RAG** **Engine:** **LlamaIndex** . While LangChain orchestrates the agents, LlamaIndex is
purpose-built specifically for complex ingestion, semantic chunking, and advanced retrieval
logic.


- **LLM:** **GPT-4o** **mini** **or** **Claude** **Haiku** . Fast, cost-effective models that excel at strict
structured JSON output.


- **Vector** **Database:** **ChromaDB** . A persistent, local vector store that supports metadata
filtering (crucial for keeping rubrics and past reports cleanly separated during search).


- **Data** **Validation:** **Pydantic** **v2** . Ensures the LLM outputs exact data structures and
mathematically clamps scores between 0.0 and 5.0 so the API never crashes due to bad LLM
formatting.


- **Observability:** **LangSmith** . Traces every single LLM call, showing exact prompts, responses, and token usage to prove to your jury exactly how the system thinks under the
hood.


5


