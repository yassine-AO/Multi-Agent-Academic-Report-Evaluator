from src.agents.prompts import (
    PARSER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    RAPPORTEUR_SYSTEM_PROMPT,
)

print("=== Parser prompt (first 200 chars) ===")
print(PARSER_SYSTEM_PROMPT[:200] + "...")

print("\n=== Reviewer prompt (first 200 chars) ===")
print(REVIEWER_SYSTEM_PROMPT[:200] + "...")

print("\n=== Critic prompt (first 200 chars) ===")
print(CRITIC_SYSTEM_PROMPT[:200] + "...")

print("\n=== Rapporteur prompt (first 200 chars) ===")
print(RAPPORTEUR_SYSTEM_PROMPT[:200] + "...")

print("\n✅ All prompts loaded!")