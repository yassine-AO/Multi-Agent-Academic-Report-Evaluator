"""
LLM client factory using Groq Cloud.
All agents get their LLM from here. Single source of truth.
"""

from langchain_groq import ChatGroq

from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)

# Singleton cache
_llm_instance = None


def get_llm(temperature: float = 0.1, max_tokens: int = 4096) -> ChatGroq:
    """
    Get or create the Groq LLM instance.
    
    Args:
        temperature: 0.0 = deterministic, 1.0 = creative.
                     Low for structured evaluation output.
        max_tokens: Hard limit on response length.
    
    Returns:
        Configured ChatGroq instance.
    """
    global _llm_instance
    if _llm_instance is None:
        logger.info(
            "Initializing Groq LLM",
            extra={
                "model": settings.llm_model_name,
                "temperature": temperature,
            }
        )
        _llm_instance = ChatGroq(
            api_key=settings.groq_api_key,
            model_name=settings.llm_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            # Groq is fast; no need for timeout tweaks usually
        )
    return _llm_instance


def get_structured_llm(temperature: float = 0.0, max_tokens: int = 4096) -> ChatGroq:
    """
    Strict LLM config for JSON-structured output.
    Lower temperature = more deterministic, valid schemas.
    """
    return get_llm(temperature=temperature, max_tokens=max_tokens)


import time
from groq import RateLimitError

def invoke_with_retry(llm, messages, max_retries=3):
    """Invoke LLM with rate limit backoff."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except RateLimitError as e:
            wait = 30 * (attempt + 1)
            logger.warning(f"Rate limit hit, waiting {wait}s...")
            time.sleep(wait)
    raise