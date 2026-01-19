"""Ollama API client for LLM interactions."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ..core.config import get_settings
from ..core.exceptions import LLMError

logger = logging.getLogger(__name__)

# Re-export LLMError for backward compatibility
__all__ = ["LLMError", "call_ollama", "extract_json"]


def call_ollama(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1200
) -> str:
    """Call Ollama API with proper error handling.

    Args:
        messages: Chat messages to send
        model: Model to use (defaults to settings.ollama_model if not specified)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
    """
    settings = get_settings()
    effective_model = model or settings.ollama_model
    payload = {
        "model": effective_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.debug(f"Calling Ollama with model={effective_model}, temp={temperature}")

    try:
        response = httpx.post(
            f"{settings.ollama_base_url}/v1/chat/completions",
            json=payload,
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
    except httpx.TimeoutException as e:
        logger.error(f"Ollama request timed out after {settings.request_timeout}s")
        raise LLMError(f"LLM request timed out after {settings.request_timeout}s") from e
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama HTTP error: {e.response.status_code} - {e.response.text[:200]}")
        raise LLMError(f"LLM request failed with status {e.response.status_code}") from e

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Ollama response as JSON: {response.text[:200]}")
        raise LLMError("LLM returned invalid JSON") from e

    # Validate response structure
    if not isinstance(data, dict):
        logger.error(f"Unexpected response type: {type(data)}")
        raise LLMError("LLM response is not a dictionary")

    choices = data.get("choices")
    if not choices or not isinstance(choices, list):
        logger.error(f"Missing or invalid 'choices' in response: {data}")
        raise LLMError("LLM response missing 'choices' array")

    if len(choices) == 0:
        logger.error("Empty 'choices' array in response")
        raise LLMError("LLM returned empty choices")

    message = choices[0].get("message")
    if not message or not isinstance(message, dict):
        logger.error(f"Missing or invalid 'message' in choice: {choices[0]}")
        raise LLMError("LLM response missing message content")

    content = message.get("content", "")
    logger.debug(f"Ollama response length: {len(content)} chars")

    return content


def extract_json(text: str) -> dict[str, Any]:
    """Extract JSON object from LLM response text."""
    if not text:
        logger.warning("Empty text passed to extract_json")
        return {}

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        logger.warning(f"No JSON object found in text: {text[:100]}...")
        return {}

    snippet = text[start : end + 1]
    try:
        result = json.loads(snippet)
        logger.debug(f"Successfully extracted JSON with keys: {list(result.keys())}")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed at position {e.pos}: {e.msg}")
        logger.error(f"Attempted to parse: {snippet[:200]}...")
        return {}
