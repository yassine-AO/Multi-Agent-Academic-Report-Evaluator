"""
Base helper for all agents.
Reduces boilerplate: load prompt + call LLM + parse JSON.
"""

import json
from typing import TypeVar, Type

from langchain_core.messages import SystemMessage, HumanMessage

from src.services.llm_client import get_structured_llm
from src.utils import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def run_agent(
    system_prompt: str,
    user_content: str,
    output_schema: Type[T],
    temperature: float = 0.0,
) -> T:
    """
    Generic agent runner.
    
    Args:
        system_prompt: The agent's system prompt constant.
        user_content: The dynamic input (report text, scores, etc.).
        output_schema: Pydantic model class to validate against.
        temperature: 0.0 for strict JSON, higher for creative tasks.
    
    Returns:
        Parsed Pydantic model instance.
    """
    llm = get_structured_llm(temperature=temperature)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    
    logger.info(
        "Running agent",
        extra={
            "agent_type": output_schema.__name__,
            "temperature": temperature,
            "input_length": len(user_content),
        }
    )
    
    # Invoke LLM
    try:
        from src.services.llm_client import invoke_with_retry
        from groq import RateLimitError
        response = invoke_with_retry(llm, messages)
    except RateLimitError:
        logger.error("Rate limit exceeded, forcing convergence")
        # Return minimal valid object to break loop
        return output_schema.model_construct()
        
    raw_text = response.content
    
    # Extract JSON from response (Groq may wrap in markdown)
    json_text = _extract_json(raw_text)
    
    # Parse and validate
    try:
        parsed = output_schema.model_validate_json(json_text)
        logger.info(
            "Agent completed successfully",
            extra={"agent_type": output_schema.__name__}
        )
        return parsed
    except Exception as e:
        logger.error(
            f"Failed to parse agent output: {e}",
            extra={"raw_response": raw_text[:500]}
        )
        raise


def _extract_json(text: str) -> str:
    """
    Extract JSON from LLM response.
    Handles markdown code blocks and leading/trailing conversational text.
    """
    text = text.strip()
    
    # Try searching for markdown code blocks first
    if "```" in text:
        first_idx = text.find("```")
        content_start = text.find("\n", first_idx)
        if content_start != -1:
            last_idx = text.find("```", content_start)
            if last_idx != -1:
                candidate = text[content_start:last_idx].strip()
                if (candidate.startswith("{") and candidate.endswith("}")) or \
                   (candidate.startswith("[") and candidate.endswith("]")):
                    return candidate

    # Fallback: Find the first '{' or '[' and the last '}' or ']'
    start_brace = text.find("{")
    start_bracket = text.find("[")
    
    start_idx = -1
    end_idx = -1
    
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start_idx = start_brace
        end_idx = text.rfind("}")
    elif start_bracket != -1:
        start_idx = start_bracket
        end_idx = text.rfind("]")
        
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1]
        
    return text