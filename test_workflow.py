from src.workflow import get_graph

print("=== Compiling graph ===")
graph = get_graph()
print(f"Graph type: {type(graph).__name__}")

print("\n=== Graph nodes ===")
print(list(graph.nodes))

print("\n=== Running with test input ===")
initial_state = {
    "pdf_path": "data/rubrics/project_report.pdf",
    "parsed_report": {},
    "review_scores": {},
    "critic_report": {},
    "final_report": {},
    "deliberation_count": 0,
    "is_converged": False,
}

# Run the full graph
result = graph.invoke(initial_state)

print(f"\nFinal report grade: {result['final_report'].get('global_grade', 'N/A')}")
print(f"Deliberation rounds: {result.get('deliberation_count', 0)}")

print("\n✅ Workflow complete!")