"""
Streamlit frontend for the Academic Project Evaluator.
Calls the FastAPI backend for all heavy lifting.
"""

import os
import time

import requests
import streamlit as st

# ─── CONFIG ──────────────────────────────────────────────────────────
API_BASE = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Academic Project Evaluator",
    page_icon="📄",
    layout="centered",
)

# ─── STATE ───────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "loading" not in st.session_state:
    st.session_state.loading = False


# ─── HELPERS ─────────────────────────────────────────────────────────
def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200 and r.json().get("status") == "healthy"
    except Exception:
        return False


def evaluate_pdf(file, student_name: str, program: str, year: str) -> dict | None:
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    data = {
        "student_name": student_name,
        "program": program,
        "year": year,
    }
    try:
        r = requests.post(
            f"{API_BASE}/evaluate",
            files=files,
            data=data,
            timeout=300,  # evaluations take time
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is the backend running?")
        return None
    except requests.exceptions.Timeout:
        st.error("Evaluation timed out. Try again or check backend logs.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code}")
        return None


# ─── UI ──────────────────────────────────────────────────────────────
st.title("📄 Automated Academic Project Evaluator")
st.caption("Multi-agent AI evaluation for student PFE reports")

# Health check
with st.sidebar:
    st.header("System Status")
    if check_api_health():
        st.success("API Online")
        # Show collections
        try:
            cols = requests.get(f"{API_BASE}/collections", timeout=5).json()
            for name, info in cols.items():
                count = info.get("document_count", "?")
                st.write(f"**{name}**: {count} docs")
        except Exception:
            pass
    else:
        st.error("API Offline")
        st.info("Start backend: `uvicorn src.main:app --port 8000`")

    st.divider()
    st.markdown("---")
    st.markdown("Built with FastAPI + LangGraph + Streamlit")

# Upload section
st.subheader("Upload Report")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Student PFE report to evaluate",
    )

with col2:
    student_name = st.text_input("Student Name", "")
    program = st.text_input("Program", "Computer Science")
    year = st.text_input("Year", "2024")

# Evaluate button
evaluate_disabled = uploaded_file is None or st.session_state.loading
if st.button("🚀 Evaluate Report", disabled=evaluate_disabled, type="primary"):
    st.session_state.loading = True
    st.session_state.result = None

    with st.spinner("Agents deliberating... This may take 1-2 minutes."):
        result = evaluate_pdf(uploaded_file, student_name, program, year)
        st.session_state.result = result

    st.session_state.loading = False
    st.rerun()

# Results display
if st.session_state.result:
    result = st.session_state.result

    st.divider()
    st.subheader("📊 Evaluation Result")

    # Metadata
    meta = result.get("report", {}).get("metadata", {})
    cols = st.columns(4)
    cols[0].metric("Request ID", result.get("request_id", "N/A"))
    cols[1].metric("Time", f"{result.get('processing_time_seconds', 0):.1f}s")
    cols[2].metric("Rounds", result.get("report", {}).get("deliberation_rounds", "N/A"))
    cols[3].metric("Model", meta.get("model_used", "N/A"))

    # Grade
    grade = result.get("report", {}).get("global_grade", "N/A")
    st.markdown(f"## 🏆 Final Grade: `{grade}`")

    # Scores
    st.subheader("Criterion Scores")
    scores = result.get("report", {}).get("final_scores", [])
    if scores:
        for s in scores:
            name = s.get("criterion_name", "Unknown")
            score = s.get("score", 0)
            justification = s.get("justification", "")
            
            col_score, col_bar = st.columns([1, 4])
            col_score.write(f"**{name}**")
            col_bar.progress(score / 5.0, text=f"{score}/5.0")
            st.caption(justification[:300])
            st.write("")

    # Strengths / Weaknesses
    col_strengths, col_weaknesses = st.columns(2)
    with col_strengths:
        st.subheader("✅ Strengths")
        for s in result.get("report", {}).get("strengths", []):
            st.write(f"- {s}")

    with col_weaknesses:
        st.subheader("❌ Weaknesses")
        for w in result.get("report", {}).get("weaknesses", []):
            st.write(f"- {w}")

    # Recommendations
    st.subheader("💡 Recommendations")
    for rec in result.get("report", {}).get("improvement_recommendations", []):
        st.info(rec)

    # Raw JSON (collapsible)
    with st.expander("View Raw JSON"):
        st.json(result)