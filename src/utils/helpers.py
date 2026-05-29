"""
Generic helper functions used across the codebase.
"""

import re


def clamp_score(value: float, min_val: float = 0.0, max_val: float = 5.0) -> float:
    """
    Ensure a score stays within valid academic bounds.
    
    The university grading scale is 0.0 to 5.0.
    This prevents LLM hallucinations from producing invalid scores.
    
    Args:
        value: The raw score from the LLM.
        min_val: Minimum allowed score (default 0.0).
        max_val: Maximum allowed score (default 5.0).
    
    Returns:
        The score clamped to [min_val, max_val].
    
    Examples:
        >>> clamp_score(5.7)
        5.0
        >>> clamp_score(-0.5)
        0.0
        >>> clamp_score(3.8)
        3.8
    """
    return max(min_val, min(max_val, value))


def clean_text(raw: str) -> str:
    """
    Normalize whitespace and remove control characters from extracted PDF text.
    
    PDF extraction often produces weird artifacts: multiple spaces, 
    null bytes, form feeds, etc. This cleans them up before sending 
    text to the LLM or embedding model.
    
    Args:
        raw: The raw text string from PDF extraction.
    
    Returns:
        Cleaned text string.
    """
    if not raw:
        return ""
    
    # Replace multiple whitespace characters (spaces, tabs, newlines) with single space
    cleaned = re.sub(r"\s+", " ", raw)
    
    # Remove null bytes and other control characters except newlines
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    
    # Strip leading/trailing whitespace
    return cleaned.strip()


def calculate_score_delta(reviewer_scores: dict[str, float], critic_scores: dict[str, float]) -> float:
    """
    Compute the maximum absolute difference between reviewer and critic scores.
    
    This is the convergence metric. If the critic disagrees strongly 
    (delta > threshold), the deliberation loop continues.
    
    Args:
        reviewer_scores: Dict mapping criterion_name -> score from Reviewer.
        critic_scores: Dict mapping criterion_name -> score from Critic.
    
    Returns:
        The maximum absolute difference across all shared criteria.
    
    Examples:
        >>> calculate_score_delta(
        ...     {"methodology": 4.0, "results": 3.5},
        ...     {"methodology": 3.0, "results": 3.5}
        ... )
        1.0
    """
    if not reviewer_scores or not critic_scores:
        return 0.0
    
    deltas = []
    for criterion, rev_score in reviewer_scores.items():
        if criterion in critic_scores:
            crit_score = critic_scores[criterion]
            deltas.append(abs(rev_score - crit_score))
    
    return max(deltas) if deltas else 0.0