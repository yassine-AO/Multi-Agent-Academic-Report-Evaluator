from src.agents import parser_node, reviewer_node, critic_node, deliberation_node, rapporteur_node

# Test with a mock state
state = {
    "pdf_path": "data/rubrics/project_report.pdf",
    "parsed_report": {},
    "review_scores": {},
    "critic_report": {},
    "final_report": {},
    "deliberation_count": 0,
    "is_converged": False,
}

print("=== Parser ===")
state = parser_node(state)
print(f"Title: {state['parsed_report'].get('title', 'N/A')[:100]}")

print("\n=== Reviewer ===")
state = reviewer_node(state)
print(f"Overall score: {state['review_scores'].get('overall_score', 'N/A')}")

print("\n=== Critic ===")
state = critic_node(state)
print(f"Max delta: {state['critic_report'].get('max_score_delta', 'N/A')}")

print("\n=== Deliberation ===")
next_step = deliberation_node(state)
print(f"Next: {next_step}")

if next_step == "rapporteur":
    print("\n=== Rapporteur ===")
    state = rapporteur_node(state)
    print(f"Global grade: {state['final_report'].get('global_grade', 'N/A')}")

print("\n[OK] Agent chain complete!")