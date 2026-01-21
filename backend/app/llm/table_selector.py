"""Table selection using lexical matching.

Selects relevant tables based on token matching between the question
and table/column names. Simple, fast, and effective for most queries.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
import re
from typing import Any, Iterable

from ..core.config import get_settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|\d+")

_TABLE_TOKEN_CACHE: dict[str, Any] = {
    "loaded_at": None,
    "tokens": {},
}


@dataclass
class SelectionResult:
    tables: list[str]
    need_clarification: bool
    clarifying_question: str


def _split_identifier(text: str) -> list[str]:
    """Split an identifier into tokens (handles CamelCase)."""
    cleaned = text.replace("[", "").replace("]", "")
    parts = _TOKEN_RE.findall(cleaned)
    tokens: list[str] = []
    for part in parts:
        for token in _CAMEL_RE.findall(part):
            token = token.lower()
            if token:
                tokens.append(token)
    return tokens


def _normalize_identifier(raw: str) -> str:
    """Normalize table/column identifier."""
    cleaned = raw.strip().strip(",;")
    cleaned = cleaned.replace("[", "").replace("]", "").replace('"', "")
    return cleaned


def _tokenize_question(question: str) -> set[str]:
    """Extract tokens from user question."""
    tokens = set()
    for token in _split_identifier(question):
        if len(token) > 1:
            tokens.add(token)
    return tokens


def _build_alias_map(table_keys: Iterable[str]) -> dict[str, list[str]]:
    """Build mapping from normalized names to original table keys."""
    alias_map: dict[str, list[str]] = defaultdict(list)
    for key in table_keys:
        if not key:
            continue
        cleaned = _normalize_identifier(key).lower()
        base = cleaned.split(".")[-1]
        if cleaned not in alias_map:
            alias_map[cleaned] = []
        if base not in alias_map:
            alias_map[base] = []
        alias_map[cleaned].append(key)
        if base != cleaned:
            alias_map[base].append(key)
    return alias_map


def _get_table_tokens(schema_cache: Any, table_keys: list[str]) -> dict[str, set[str]]:
    """Get tokens for each table (from name + column names)."""
    loaded_at = getattr(schema_cache, "loaded_at", None)
    cache_loaded_at = _TABLE_TOKEN_CACHE.get("loaded_at")
    if cache_loaded_at == loaded_at and _TABLE_TOKEN_CACHE.get("tokens"):
        return _TABLE_TOKEN_CACHE["tokens"]

    tokens_by_table: dict[str, set[str]] = {}
    for key in table_keys:
        table = schema_cache.table_info(key) if schema_cache else None
        tokens = set(_split_identifier(key))
        if table:
            for column in table.columns:
                tokens.update(_split_identifier(column.name))
        tokens_by_table[key] = {t for t in tokens if len(t) > 1}

    _TABLE_TOKEN_CACHE["loaded_at"] = loaded_at
    _TABLE_TOKEN_CACHE["tokens"] = tokens_by_table
    return tokens_by_table


def _score_tables(
    question: str,
    table_keys: list[str],
    schema_cache: Any,
) -> dict[str, float]:
    """Score tables using lexical token matching.

    Args:
        question: User question
        table_keys: List of table keys to score
        schema_cache: Schema cache for column info

    Returns:
        Dict mapping table keys to scores
    """
    tokens = _tokenize_question(question)
    if not tokens:
        return {}

    token_weight = 0.2
    exact_bonus = 1.0

    scores: dict[str, float] = defaultdict(float)
    table_tokens = _get_table_tokens(schema_cache, table_keys) if schema_cache else {}
    question_lower = question.lower()

    for key in table_keys:
        base = key.split(".")[-1].lower()
        overlap = tokens & table_tokens.get(key, set())
        if overlap:
            scores[key] += len(overlap) * token_weight
        if base and base in question_lower:
            scores[key] += exact_bonus

    return scores


def select_tables(
    question: str,
    table_keys: list[str],
    schema_cache: Any | None = None,
) -> SelectionResult:
    """Select relevant tables using lexical matching.

    Args:
        question: User's question
        table_keys: Available table keys
        schema_cache: Optional schema cache for column info

    Returns:
        SelectionResult with selected tables
    """
    if not question or not table_keys:
        return SelectionResult(tables=[], need_clarification=False, clarifying_question="")

    settings = get_settings()
    max_tables = max(settings.table_selection_max_tables, 1)
    min_score = max(settings.table_selection_min_score, 0.0)

    # Score tables using lexical matching
    scores = _score_tables(question, table_keys, schema_cache)

    # Rank by score
    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    # Select tables meeting threshold
    selected = [key for key, score in ranked if score >= min_score][:max_tables]

    # Fallback: if no tables meet threshold, take top scoring ones
    if not selected and ranked:
        selected = [key for key, score in ranked if score > 0][:max_tables]
        if selected:
            logger.info(f"Fallback: selected {len(selected)} tables below threshold")

    logger.info(f"Selected {len(selected)} tables from {len(table_keys)} available")

    return SelectionResult(
        tables=selected,
        need_clarification=False,
        clarifying_question="",
    )
