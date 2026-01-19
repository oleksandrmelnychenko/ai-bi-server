"""Table selection using vector + lexical retrieval."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
import re
from typing import Any, Iterable

from ..core.config import get_settings
from ..retrieval.schema_hints import is_available as schema_vectors_available, search_schema

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
    cleaned = raw.strip().strip(",;")
    cleaned = cleaned.replace("[", "").replace("]", "").replace('"', "")
    return cleaned


def _tokenize_question(question: str) -> set[str]:
    tokens = set()
    for token in _split_identifier(question):
        if len(token) > 1:
            tokens.add(token)
    return tokens


def _build_alias_map(table_keys: Iterable[str]) -> dict[str, list[str]]:
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


def _resolve_table_keys(name: str, alias_map: dict[str, list[str]]) -> list[str]:
    cleaned = _normalize_identifier(name).lower()
    if not cleaned:
        return []
    base = cleaned.split(".")[-1]
    resolved: list[str] = []
    for key in alias_map.get(cleaned, []):
        if key not in resolved:
            resolved.append(key)
    if base != cleaned:
        for key in alias_map.get(base, []):
            if key not in resolved:
                resolved.append(key)
    return resolved


def _get_table_tokens(schema_cache: Any, table_keys: list[str]) -> dict[str, set[str]]:
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


def _score_from_vectors(
    question: str,
    alias_map: dict[str, list[str]],
    min_similarity: float,
    top_k: int,
    weights: dict[str, float],
) -> dict[str, float]:
    """Score tables using vector similarity search.

    Args:
        question: User question
        alias_map: Table name alias mapping
        min_similarity: Minimum cosine similarity threshold
        top_k: Max results from vector search
        weights: Dict with 'table', 'column', 'rel' weight values

    Returns:
        Dict mapping table keys to scores, empty dict if vector search unavailable/fails
    """
    if not schema_vectors_available():
        logger.info("Schema vectors not available, skipping vector scoring")
        return {}

    scores: dict[str, float] = defaultdict(float)
    try:
        results = search_schema(
            question,
            top_k=top_k,
            min_similarity=min_similarity,
            entry_types=None,
        )
    except Exception as exc:
        logger.warning(f"Schema vector search failed: {exc}")
        return {}

    for entry in results:
        entry_type = entry.type
        if entry_type == "table":
            weight = weights.get("table", 1.0)
            targets = _resolve_table_keys(entry.name, alias_map)
        elif entry_type == "column":
            weight = weights.get("column", 0.6)
            table_name = entry.data.get("table") if isinstance(entry.data, dict) else None
            if not table_name and entry.name:
                table_name = entry.name.split(".")[0]
            targets = _resolve_table_keys(table_name or "", alias_map)
        elif entry_type == "relationship":
            weight = weights.get("rel", 0.4)
            targets = []
            if isinstance(entry.data, dict):
                for name in (entry.data.get("from"), entry.data.get("to")):
                    targets.extend(_resolve_table_keys(name or "", alias_map))
        else:
            continue

        for table_key in targets:
            scores[table_key] += entry.similarity * weight

    return scores


def _score_from_lexical(
    question: str,
    table_keys: list[str],
    schema_cache: Any,
    weights: dict[str, float],
) -> dict[str, float]:
    """Score tables using lexical token matching.

    Args:
        question: User question
        table_keys: List of table keys to score
        schema_cache: Schema cache for column info
        weights: Dict with 'token' and 'exact' weight values

    Returns:
        Dict mapping table keys to scores
    """
    tokens = _tokenize_question(question)
    if not tokens:
        return {}

    token_weight = weights.get("token", 0.2)
    exact_bonus = weights.get("exact", 1.0)

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
    """Select relevant tables using vectors + lexical matching (no LLM).

    Uses a hybrid approach:
    1. Vector similarity search on schema embeddings
    2. Lexical token matching on table/column names
    3. Combined scoring with configurable weights

    Falls back to lexical-only if vector search is unavailable.
    """
    if not question or not table_keys:
        return SelectionResult(tables=[], need_clarification=False, clarifying_question="")

    settings = get_settings()
    max_tables = max(settings.table_selection_max_tables, 1)
    min_score = max(settings.table_selection_min_score, 0.0)
    top_k = max(settings.table_selection_vector_top_k, max_tables)

    # Build weight dicts from config
    vector_weights = {
        "table": settings.table_selection_vector_table_weight,
        "column": settings.table_selection_vector_column_weight,
        "rel": settings.table_selection_vector_rel_weight,
    }
    lexical_weights = {
        "token": settings.table_selection_lexical_token_weight,
        "exact": settings.table_selection_lexical_exact_bonus,
    }

    alias_map = _build_alias_map(table_keys)
    combined_scores: dict[str, float] = defaultdict(float)

    # Try vector scoring first
    vector_scores = _score_from_vectors(
        question,
        alias_map,
        min_similarity=settings.table_selection_vector_min_similarity,
        top_k=top_k,
        weights=vector_weights,
    )

    # Always run lexical scoring (fallback + supplement)
    lexical_scores = _score_from_lexical(question, table_keys, schema_cache, lexical_weights)

    # Log which methods contributed
    has_vector = bool(vector_scores)
    has_lexical = bool(lexical_scores)

    if not has_vector and has_lexical:
        logger.info("Using lexical-only scoring (vector search unavailable)")
    elif has_vector and not has_lexical:
        logger.info("Using vector-only scoring (no lexical matches)")
    elif has_vector and has_lexical:
        logger.info("Using combined vector + lexical scoring")
    else:
        logger.warning("No scores from either method")

    # Combine scores
    for key, score in vector_scores.items():
        combined_scores[key] += score
    for key, score in lexical_scores.items():
        combined_scores[key] += score

    ranked = sorted(
        combined_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    selected = [key for key, score in ranked if score >= min_score][:max_tables]

    # Fallback: if no tables meet threshold, take top scoring ones
    if not selected and ranked:
        selected = [key for key, score in ranked if score > 0][:max_tables]
        if selected:
            logger.info(f"Fallback: selected {len(selected)} tables below threshold")

    logger.info(f"Selected {len(selected)} tables (vector={has_vector}, lexical={has_lexical})")

    return SelectionResult(
        tables=selected,
        need_clarification=False,
        clarifying_question="",
    )
