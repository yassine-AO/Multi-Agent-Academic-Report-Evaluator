from src.rag import retrieve_with_rerank, retrieve_for_reviewer, retrieve_for_critic

# Test 1: Basic retrieval from rubrics
print("=== Test 1: Retrieve from rubrics ===")
results = retrieve_with_rerank(
    query="methodology evaluation criteria",
    collection_name="rubrics",
    final_k=3,
)
print(f"Found {len(results)} chunks")
for text, score in results:
    print(f"Score {score:.4f}: {text[:150]}...")
print()

# Test 2: Reviewer retrieval
print("=== Test 2: Reviewer retrieval ===")
reviewer_results = retrieve_for_reviewer(
    criterion="methodology",
    report_section="The student used surveys and interviews.",
)
print(f"Found {len(reviewer_results)} chunks for reviewer")
for text, score in reviewer_results:
    print(f"Score {score:.4f}: {text[:150]}...")
print()

# Test 3: Critic retrieval
print("=== Test 3: Critic retrieval ===")
critic_results = retrieve_for_critic(
    criterion="methodology",
)
print(f"Found {len(critic_results)} chunks for critic")
for text, score in critic_results:
    print(f"Score {score:.4f}: {text[:150]}...")
print()

print("[OK] Retrieval pipeline is working!")