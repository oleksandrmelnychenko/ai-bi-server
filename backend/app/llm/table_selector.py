"""Table selection using LLM."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from .client import call_ollama, extract_json
from .prompts import TABLE_SELECTION_SYSTEM

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    tables: list[str]
    need_clarification: bool
    clarifying_question: str


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

    content = call_ollama(
        [
            {"role": "system", "content": TABLE_SELECTION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=800,
    )

    payload = extract_json(content)
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
