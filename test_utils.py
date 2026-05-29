from src.utils import get_logger, clamp_score, clean_text, calculate_score_delta

# --- Test logging ---
logger = get_logger("test")
logger.info("Testing logging system", extra={"agent": "reviewer"})
logger.warning("This is a warning")

# --- Test clamp_score ---
assert clamp_score(5.7) == 5.0, "Should clamp to max"
assert clamp_score(-0.5) == 0.0, "Should clamp to min"
assert clamp_score(3.8) == 3.8, "Should pass through"
print("[OK] clamp_score works")

# --- Test clean_text ---
messy = "Hello\x00\x01\x02   world\n\n\n!!!  \t\t"
cleaned = clean_text(messy)
print(f"Cleaned: '{cleaned}'")
assert "Hello" in cleaned
assert "world" in cleaned
assert "\x00" not in cleaned
print("[OK] clean_text works")

# --- Test calculate_score_delta ---
rev = {"methodology": 4.0, "results": 3.5, "writing": 4.5}
crit = {"methodology": 3.0, "results": 3.5, "writing": 4.0}
delta = calculate_score_delta(rev, crit)
print(f"Score delta: {delta}")
assert delta == 1.0, f"Expected 1.0, got {delta}"
print("[OK] calculate_score_delta works")

print("\n[SUCCESS] All utility tests passed!")