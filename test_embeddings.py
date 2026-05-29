from src.rag import get_embedding_model, embed_text

print("=== Initializing Google embedding model ===")
model = get_embedding_model()
print(f"Model type: {type(model).__name__}")

print("\n=== Embedding a document ===")
doc = "The student presented a rigorous methodology with clear experimental design."
vector = embed_text(doc)
print(f"Vector length: {len(vector)}")
print(f"First 5 values: {[round(x, 4) for x in vector[:5]]}")

print("\n=== Embedding a query ===")
query = "How strong was the methodology?"
vector2 = model.get_query_embedding(query)
print(f"Query vector length: {len(vector2)}")

print("\n=== Checking consistency ===")
assert len(vector) == 768, f"Expected 768 dimensions, got {len(vector)}"
assert len(vector2) == 768, "Query must also be 768 dimensions"
print("[OK] Both document and query embeddings are 768 dimensions")

print("\n=== Quick similarity check ===")
import math

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(x * x for x in v1))
    norm2 = math.sqrt(sum(x * x for x in v2))
    return dot / (norm1 * norm2)

# Similar sentences should be close
similar_doc = "The approach was scientifically sound and well-structured."
vector3 = embed_text(similar_doc)
sim = cosine_similarity(vector, vector3)
print(f"Similarity between two methodology sentences: {sim:.4f}")
assert sim > 0.6, "Similar sentences should be reasonably close"
print("[OK] Semantic similarity is working")

print("\n[SUCCESS] Google AI Studio embedding system is working!")