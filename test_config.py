from src.config import settings

print("LLM Model:", settings.llm_model_name)
print("Chroma Path:", settings.chroma_db_path)
print("Max Loops:", settings.max_deliberation_loops)
print("Convergence:", settings.convergence_threshold)