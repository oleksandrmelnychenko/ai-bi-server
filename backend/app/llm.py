from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any

import httpx

from .config import get_settings
from .prompts import ANSWER_SYSTEM, SQL_GENERATION_SYSTEM, SQL_GENERATION_SQLCODER, TABLE_SELECTION_SYSTEM

logger = logging.getLogger(__name__)

# Maximum rows to send to LLM for answer composition
MAX_ROWS_FOR_LLM = 50


@dataclass
class SelectionResult:
    tables: list[str]
    need_clarification: bool
    clarifying_question: str


class LLMError(Exception):
    """Raised when LLM returns unexpected response format."""
    pass


def _is_sqlcoder_model(model: str | None) -> bool:
    """Check if the model is a SQLCoder variant."""
    if not model:
        return False
    return "sqlcoder" in model.lower()


def _call_ollama(
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


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON object from LLM response text."""
    if not text:
        logger.warning("Empty text passed to _extract_json")
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


def _validate_selection_payload(payload: dict[str, Any]) -> tuple[list[str], bool, str]:
    """Validate and extract fields from table selection response."""
    # Extract tables with type validation
    tables_raw = payload.get("tables")
    if tables_raw is None:
        logger.warning("No 'tables' field in selection payload")
        tables = []
    elif isinstance(tables_raw, list):
        # Filter to only strings
        tables = [t for t in tables_raw if isinstance(t, str)]
        if len(tables) != len(tables_raw):
            logger.warning(f"Filtered out non-string items from tables list")
    elif isinstance(tables_raw, str):
        # LLM returned a single table as string instead of list
        logger.warning(f"'tables' is a string, not list: {tables_raw}")
        tables = [tables_raw]
    else:
        logger.error(f"Unexpected type for 'tables': {type(tables_raw)}")
        tables = []

    # Extract clarification fields
    need_clarification = bool(payload.get("need_clarification", False))
    clarifying_question = payload.get("clarifying_question", "")

    if not isinstance(clarifying_question, str):
        logger.warning(f"clarifying_question is not string: {type(clarifying_question)}")
        clarifying_question = str(clarifying_question) if clarifying_question else ""

    return tables, need_clarification, clarifying_question


def select_tables(question: str, table_keys: list[str]) -> SelectionResult:
    """Ask LLM to select relevant tables for the question."""
    table_list = "\n".join(f"- {name}" for name in table_keys)
    user_prompt = f"Question:\n{question}\n\nAvailable tables:\n{table_list}\n"

    logger.info(f"Selecting tables for question: {question[:100]}...")

    content = _call_ollama(
        [
            {"role": "system", "content": TABLE_SELECTION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=800,
    )

    payload = _extract_json(content)
    tables, need_clarification, clarifying_question = _validate_selection_payload(payload)

    # Filter to valid tables only
    valid_tables = set(table_keys)
    filtered = [t for t in tables if t in valid_tables]
    invalid = [t for t in tables if t not in valid_tables]

    if invalid:
        logger.warning(f"LLM selected invalid tables (not in schema): {invalid}")

    logger.info(f"Selected {len(filtered)} tables, need_clarification={need_clarification}")

    # Use meaningful fallback for clarification
    if need_clarification and not clarifying_question:
        clarifying_question = "Будь ласка, уточніть ваш запит. Які саме дані вас цікавлять?"

    return SelectionResult(
        tables=filtered,
        need_clarification=need_clarification,
        clarifying_question=clarifying_question
    )


def generate_sql(
    question: str,
    table_details: str,
    join_hints: str,
    max_rows: int,
    model: str | None = None
) -> str:
    """Generate SQL query for the question using selected tables.

    Args:
        question: User's question
        table_details: Schema information for selected tables
        join_hints: Join conditions between tables
        max_rows: Row limit to apply
        model: Optional model override (e.g., sqlcoder for specialized SQL generation)
    """
    logger.info(f"Generating SQL for question: {question[:100]}...")
    logger.debug(f"Table details length: {len(table_details)}, join hints length: {len(join_hints)}")

    # Detect SQLCoder and use optimized prompt format + temperature
    if _is_sqlcoder_model(model):
        logger.info("Using SQLCoder-optimized prompt format")
        # SQLCoder: structured prompt with ### sections (no system message)
        schema_with_hints = (
            f"{table_details}\n\n"
            f"-- Join hints:\n{join_hints}\n"
            f"-- Row limit: TOP ({max_rows})"
        )
        user_prompt = SQL_GENERATION_SQLCODER.format(
            question=question,
            schema=schema_with_hints
        )
        messages = [{"role": "user", "content": user_prompt}]
        temperature = 0.0  # Deterministic for SQLCoder
    else:
        # Qwen/general models: system + user message format
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Table details:\n{table_details}\n\n"
            f"Join hints:\n{join_hints}\n\n"
            f"Row limit: TOP ({max_rows})\n"
        )
        messages = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        temperature = 0.05

    content = _call_ollama(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=1200,
    )

    sql = content.strip()
    logger.info(f"Generated SQL length: {len(sql)} chars")
    logger.debug(f"Generated SQL: {sql[:200]}...")

    return sql


def compose_answer(
    question: str,
    sql: str,
    columns: list[str],
    rows: list[list[Any]],
    max_rows_for_llm: int = MAX_ROWS_FOR_LLM
) -> str:
    """Compose natural language answer from query results."""
    # Truncate rows to avoid token overflow
    truncated = len(rows) > max_rows_for_llm
    sample_rows = rows[:max_rows_for_llm]

    if truncated:
        logger.info(f"Truncating results from {len(rows)} to {max_rows_for_llm} rows for LLM")

    sample = {"columns": columns, "rows": sample_rows}

    # Add truncation note to prompt if needed
    truncation_note = ""
    if truncated:
        truncation_note = f"\n\nNote: Showing first {max_rows_for_llm} of {len(rows)} total rows."

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"SQL:\n{sql}\n\n"
        f"Data:\n{json.dumps(sample, ensure_ascii=False)}"
        f"{truncation_note}"
    )

    logger.info(f"Composing answer for {len(sample_rows)} rows, {len(columns)} columns")

    content = _call_ollama(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=900,
    )

    answer = content.strip()
    logger.info(f"Composed answer length: {len(answer)} chars")

    return answer
