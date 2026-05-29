from llama_index.core.schema import Document
from src.rag import split_documents

# Create a fake document with two clear topic shifts
text = """
Introduction to Machine Learning
Machine learning is a subset of artificial intelligence that enables systems to learn from data.

Methodology
We used a convolutional neural network trained on the CIFAR-10 dataset with 50,000 images.

Results
The model achieved 94.2% accuracy on the test set, outperforming the baseline by 3.1%.

Conclusion
Future work includes extending the model to larger datasets and exploring transformer architectures.
"""

doc = Document(text=text, metadata={"filename": "test.pdf"})
nodes = split_documents([doc])

print(f"Number of chunks: {len(nodes)}")
print("=" * 50)

for i, node in enumerate(nodes):
    print(f"\n--- Chunk {i+1} ({len(node.text)} chars) ---")
    print(node.text[:200] + "..." if len(node.text) > 200 else node.text)

# Sanity checks
assert len(nodes) >= 2, "Should split into at least 2 chunks"
print("\n✅ Semantic chunking is working!")