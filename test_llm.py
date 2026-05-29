from src.services import get_llm

print("=== Initializing Groq LLM ===")
llm = get_llm()
print(f"Model type: {type(llm).__name__}")
print(f"Model name: {llm.model_name}")

print("\n=== Test invocation ===")
response = llm.invoke("Say 'hello' and nothing else.")
print(f"Response: {response.content}")

print("\n✅ Groq LLM working!")