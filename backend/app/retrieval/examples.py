"""SQL examples loader for few-shot learning.

This module loads curated SQL examples from YAML configuration and provides
them as context to the LLM for generating SQL queries that match the
patterns and conventions of the GBA repository.

Usage:
    from app.retrieval import get_relevant_examples

    # In generate_sql():
    examples = get_relevant_examples(question, table_keys=selected_tables)
    # Include examples in the prompt
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from ..core.config import get_settings

logger = logging.getLogger(__name__)

# Module-level cache for loaded examples
_examples_cache: dict[str, Any] = {}
_extracted_cache: dict[str, Any] = {}
_index_conn: sqlite3.Connection | None = None
_index_path: str | None = None
_index_fts_enabled: bool = False

_TABLE_REF_RE = re.compile(r"\b(?:FROM|JOIN)\s+([^\s,;]+)", re.IGNORECASE)
_CTE_NAME_RE = re.compile(r"(?:\bWITH\b|,)\s*([^\s,]+)\s+AS\s*\(", re.IGNORECASE)
_TOKEN_RE = re.compile(r"\w+")


def _strip_identifier(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = cleaned.rstrip(",;")
    cleaned = cleaned.strip("()")
    cleaned = cleaned.replace("[", "").replace("]", "").replace('"', "")
    return cleaned.strip()


def _normalize_table_key(name: str) -> list[str]:
    cleaned = _strip_identifier(name)
    if not cleaned:
        return []
    if cleaned.startswith(("#", "@")):
        return []

    parts = [part for part in cleaned.split(".") if part]
    if not parts:
        return []

    if len(parts) >= 2:
        schema = parts[-2]
        table = parts[-1]
        return [f"{schema}.{table}".lower(), table.lower()]

    return [parts[-1].lower()]


def _extract_cte_names(sql: str) -> set[str]:
    names: set[str] = set()
    for match in _CTE_NAME_RE.finditer(sql):
        name = _strip_identifier(match.group(1))
        if name:
            names.add(name.lower())
    return names


def extract_table_names(sql: str) -> list[str]:
    """Extract normalized table names from SQL text."""
    if not sql:
        return []

    cte_names = _extract_cte_names(sql)
    tables: set[str] = set()

    for match in _TABLE_REF_RE.finditer(sql):
        raw = match.group(1)
        cleaned = _strip_identifier(raw)
        if not cleaned:
            continue
        upper = cleaned.upper()
        if upper.startswith("SELECT") or upper.startswith("WITH"):
            continue

        normalized = _normalize_table_key(cleaned)
        if not normalized:
            continue

        base_names = {name.split(".")[-1] for name in normalized}
        if base_names & cte_names:
            continue

        tables.update(normalized)

    return sorted(tables)


def _normalize_table_list(tables: list[str]) -> set[str]:
    normalized: set[str] = set()
    for name in tables:
        if not isinstance(name, str):
            continue
        normalized.update(_normalize_table_key(name))
    return normalized


def _get_example_tables(example: dict[str, Any]) -> set[str]:
    cached = example.get("_tables_normalized")
    if isinstance(cached, list):
        return set(cached)
    if isinstance(cached, set):
        return cached

    tables = example.get("tables")
    if isinstance(tables, list) and tables:
        normalized = _normalize_table_list(tables)
    else:
        normalized = set(extract_table_names(example.get("sql", "")))

    example["_tables_normalized"] = sorted(normalized)
    return normalized


def _normalize_table_keys(table_keys: list[str] | None) -> set[str]:
    if not table_keys:
        return set()
    normalized: set[str] = set()
    for key in table_keys:
        if not isinstance(key, str):
            continue
        normalized.update(_normalize_table_key(key))
    return normalized


def _tokenize_text(text: str) -> set[str]:
    tokens = {token for token in _TOKEN_RE.findall(text.lower()) if len(token) > 1}
    return tokens


def _get_example_tokens(example: dict[str, Any]) -> set[str]:
    cached = example.get("_tokens_normalized")
    if isinstance(cached, list):
        return set(cached)
    if isinstance(cached, set):
        return cached

    tokens: set[str] = set()
    question = example.get("question", "")
    if isinstance(question, str) and question:
        tokens.update(_tokenize_text(question))

    tables = _get_example_tables(example)
    for table in tables:
        tokens.update(_tokenize_text(table.replace(".", " ")))

    example["_tokens_normalized"] = sorted(tokens)
    return tokens


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().lower()


def _score_example(
    example: dict[str, Any],
    selected_tables: set[str],
    query_tokens: set[str]
) -> float:
    score = 0.0
    tables = _get_example_tables(example)

    if selected_tables and tables:
        overlap = len(tables & selected_tables)
        if overlap:
            score += 4.0 * overlap

    if query_tokens:
        tokens = _get_example_tokens(example)
        if tokens:
            overlap = len(tokens & query_tokens)
            if overlap:
                score += overlap / max(len(query_tokens), 1)

    if example.get("question"):
        score += 0.2

    return score


def _rank_examples(
    examples: list[dict[str, Any]],
    selected_tables: set[str],
    query_tokens: set[str]
) -> list[dict[str, Any]]:
    scored = []
    for ex in examples:
        score = _score_example(ex, selected_tables, query_tokens)
        sql_len = len(ex.get("sql", ""))
        scored.append((score, -sql_len, ex))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored]


def _index_exists() -> Path | None:
    settings = get_settings()
    if not settings.sql_index_enabled:
        return None
    index_path = Path(settings.sql_index_path)
    if not index_path.is_absolute():
        index_path = Path(__file__).parent.parent.parent / index_path
    if not index_path.exists():
        return None
    return index_path


def _get_index_connection() -> sqlite3.Connection | None:
    global _index_conn, _index_path, _index_fts_enabled
    index_path = _index_exists()
    if not index_path:
        return None

    path_str = str(index_path)
    if _index_conn is None or _index_path != path_str:
        conn = sqlite3.connect(index_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _index_conn = conn
        _index_path = path_str
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='examples_fts'"
        ).fetchone()
        _index_fts_enabled = bool(row)
    return _index_conn


def _fetch_index_rows(
    conn: sqlite3.Connection,
    category: str,
    query_tokens: set[str],
    selected_tables: set[str],
    limit: int,
    require_table_match: bool
) -> list[sqlite3.Row]:
    params: list[Any] = []
    table_filter = ""

    if require_table_match and selected_tables:
        placeholders = ", ".join("?" for _ in selected_tables)
        table_filter = (
            " AND EXISTS ("
            "SELECT 1 FROM example_tables t "
            "WHERE t.example_id = e.id AND t.table_key IN (" + placeholders + ")"
            ")"
        )
        params.extend(sorted(selected_tables))

    if _index_fts_enabled and query_tokens:
        fts_query = " OR ".join(sorted(query_tokens))
        sql = (
            "SELECT e.sql_hash, e.category, e.description, e.question, e.sql, "
            "e.tables, e.curated, bm25(examples_fts) AS rank "
            "FROM examples_fts "
            "JOIN examples e ON e.id = examples_fts.rowid "
            "WHERE examples_fts MATCH ? AND e.category = ?"
            + table_filter +
            " ORDER BY e.curated DESC, rank ASC "
            "LIMIT ?"
        )
        params = [fts_query, category] + params + [limit]
    else:
        sql = (
            "SELECT e.sql_hash, e.category, e.description, e.question, e.sql, "
            "e.tables, e.curated, LENGTH(e.sql) AS rank "
            "FROM examples e "
            "WHERE e.category = ?"
            + table_filter +
            " ORDER BY e.curated DESC, rank ASC "
            "LIMIT ?"
        )
        params = [category] + params + [limit]

    return list(conn.execute(sql, params))


def _get_index_category_descriptions(
    conn: sqlite3.Connection,
    categories: list[str],
) -> dict[str, str]:
    if not categories:
        return {}
    placeholders = ", ".join("?" for _ in categories)
    rows = conn.execute(
        f"SELECT name, description FROM categories WHERE name IN ({placeholders})",
        categories,
    ).fetchall()
    return {row["name"]: row["description"] for row in rows}


def _build_examples_from_index(
    question: str,
    categories: list[str],
    selected_tables: set[str],
    max_per_category: int,
    max_total: int | None,
) -> str:
    conn = _get_index_connection()
    if not conn or not categories:
        return ""

    query_tokens = _tokenize_text(question)
    descriptions = _get_index_category_descriptions(conn, categories)
    lines = ["## SQL Examples (follow these patterns exactly):", ""]
    total_added = 0
    seen_hashes: set[str] = set()

    for category in categories:
        rows = _fetch_index_rows(
            conn,
            category,
            query_tokens,
            selected_tables,
            max_per_category,
            require_table_match=True,
        )
        if selected_tables and len(rows) < max_per_category:
            rows.extend(
                _fetch_index_rows(
                    conn,
                    category,
                    query_tokens,
                    set(),
                    max_per_category - len(rows),
                    require_table_match=False,
                )
            )

        if not rows:
            continue

        pattern_desc = descriptions.get(category) or category
        for row in rows:
            sql_hash = row["sql_hash"]
            if sql_hash in seen_hashes:
                continue
            seen_hashes.add(sql_hash)

            sql = row["sql"]
            if not sql:
                continue

            lines.append(f"-- Pattern: {pattern_desc}")
            question_text = row["question"] or ""
            if question_text:
                lines.append(f"-- Question: {question_text}")
            lines.append(sql)
            lines.append("")
            total_added += 1

            if max_total is not None and total_added >= max_total:
                return "\n".join(lines)

    if total_added:
        return "\n".join(lines)
    return ""


def _select_examples_by_priority(
    curated_examples: list[dict[str, Any]],
    extracted_examples: list[dict[str, Any]],
    selected_tables: set[str],
    max_per_category: int,
    query_tokens: set[str]
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    seen_sql: set[str] = set()

    def add_examples(examples: list[dict[str, Any]], require_table_match: bool) -> None:
        ranked = _rank_examples(examples, selected_tables, query_tokens)
        for ex in ranked:
            if len(chosen) >= max_per_category:
                return

            sql = ex.get("sql", "").strip()
            if not sql:
                continue

            if require_table_match and selected_tables:
                tables = _get_example_tables(ex)
                if not (tables & selected_tables):
                    continue

            normalized_sql = _normalize_sql(sql)
            if normalized_sql in seen_sql:
                continue

            seen_sql.add(normalized_sql)
            chosen.append(ex)

    add_examples(curated_examples, require_table_match=True)
    add_examples(extracted_examples, require_table_match=True)

    if selected_tables and len(chosen) < max_per_category:
        add_examples(curated_examples, require_table_match=False)
        add_examples(extracted_examples, require_table_match=False)

    return chosen


def _build_examples(
    categories: list[str],
    curated_data: dict[str, Any],
    extracted_data: dict[str, Any],
    max_categories: int,
    max_per_category: int,
    selected_tables: set[str],
    max_total: int | None,
    query_tokens: set[str]
) -> str:
    lines = ["## SQL Examples (follow these patterns exactly):", ""]
    has_examples = False
    total_added = 0

    for cat_name in categories[:max_categories]:
        curated_cat = curated_data.get("categories", {}).get(cat_name, {})
        extracted_cat = extracted_data.get("categories", {}).get(cat_name, {})
        curated_examples = curated_cat.get("examples", [])
        extracted_examples = extracted_cat.get("examples", [])

        if not curated_examples and not extracted_examples:
            continue

        chosen = _select_examples_by_priority(
            curated_examples,
            extracted_examples,
            selected_tables,
            max_per_category,
            query_tokens,
        )
        if not chosen:
            continue

        description = curated_cat.get("description") or extracted_cat.get("description") or cat_name
        for ex in chosen:
            question_text = ex.get("question", "")
            sql = ex.get("sql", "").strip()

            if not sql:
                continue

            has_examples = True
            lines.append(f"-- Pattern: {description}")
            if question_text:
                lines.append(f"-- Question: {question_text}")
            lines.append(sql)
            lines.append("")
            total_added += 1

            if max_total is not None and total_added >= max_total:
                return "\n".join(lines)

    if has_examples:
        return "\n".join(lines)
    return ""



def load_examples(path: str | None = None) -> dict[str, Any]:
    """Load SQL examples from YAML file with caching.

    Args:
        path: Relative path from backend directory to YAML file

    Returns:
        Dictionary with 'categories' key containing example data
    """
    global _examples_cache

    if not _examples_cache:
        settings = get_settings()
        effective_path = path or settings.sql_examples_path
        examples_path = Path(effective_path)
        if not examples_path.is_absolute():
            examples_path = Path(__file__).parent.parent.parent / effective_path
        if not examples_path.exists():
            logger.warning(f"SQL examples file not found: {examples_path}")
            return {"categories": {}}

        try:
            with open(examples_path, encoding="utf-8-sig") as f:
                _examples_cache = yaml.safe_load(f) or {"categories": {}}
            logger.info(f"Loaded {len(_examples_cache.get('categories', {}))} example categories")
        except Exception as e:
            logger.error(f"Failed to load SQL examples: {e}")
            return {"categories": {}}

    return _examples_cache


def load_extracted_examples(path: str | None = None) -> dict[str, Any]:
    """Load extracted SQL examples from YAML file with caching.

    Args:
        path: Relative path from backend directory to extracted YAML file.
            Comma-separated paths are supported.

    Returns:
        Dictionary with 'categories' key containing example data
    """
    global _extracted_cache

    if not _extracted_cache:
        settings = get_settings()
        effective_path = path or settings.sql_examples_extracted_path
        paths = [p.strip() for p in effective_path.split(",") if p.strip()]
        merged: dict[str, Any] = {"categories": {}}

        for raw_path in paths:
            examples_path = Path(raw_path)
            if not examples_path.is_absolute():
                examples_path = Path(__file__).parent.parent.parent / raw_path
            if not examples_path.exists():
                logger.info(f"Extracted SQL examples file not found: {examples_path}")
                continue

            try:
                with open(examples_path, encoding="utf-8-sig") as f:
                    data = yaml.safe_load(f) or {"categories": {}}
            except Exception as e:
                logger.error(f"Failed to load extracted SQL examples: {e}")
                continue

            for cat_name, cat_data in data.get("categories", {}).items():
                merged_cat = merged["categories"].setdefault(cat_name, {})
                if not merged_cat.get("description"):
                    merged_cat["description"] = cat_data.get("description", "")
                merged_cat.setdefault("examples", [])
                merged_cat["examples"].extend(cat_data.get("examples", []))

        _extracted_cache = merged
        logger.info(
            f"Loaded {len(_extracted_cache.get('categories', {}))} extracted example categories"
        )

    return _extracted_cache


def classify_question(
    question: str,
    examples_data: dict[str, Any] | None = None,
    fallback_to_simple: bool = True
) -> list[str]:
    """Classify question into categories using keywords from YAML config.

    Args:
        question: User's question in Ukrainian
        examples_data: Optional examples payload to use for classification

    Returns:
        List of matched category names, ordered by match relevance
    """
    if examples_data is None:
        examples_data = load_examples()
    categories = examples_data.get("categories", {})
    q_lower = question.lower()

    matched = []
    for cat_name, cat_data in categories.items():
        keywords = cat_data.get("keywords", [])
        if any(kw in q_lower for kw in keywords):
            matched.append(cat_name)
            logger.debug(f"Question matched category '{cat_name}' by keywords")

    # Default to simple_select if no specific match
    if fallback_to_simple and not matched and "simple_select" in categories:
        matched.append("simple_select")

    return matched


def _categories_by_table_overlap(
    selected_tables: set[str],
    curated_data: dict[str, Any],
    extracted_data: dict[str, Any],
    max_categories: int
) -> list[str]:
    if not selected_tables:
        return []

    category_scores: dict[str, int] = {}
    all_categories = set(curated_data.get("categories", {}).keys())
    all_categories.update(extracted_data.get("categories", {}).keys())

    for cat_name in all_categories:
        curated_examples = curated_data.get("categories", {}).get(cat_name, {}).get("examples", [])
        extracted_examples = extracted_data.get("categories", {}).get(cat_name, {}).get("examples", [])
        examples = curated_examples + extracted_examples
        score = 0
        for ex in examples:
            tables = _get_example_tables(ex)
            if tables & selected_tables:
                score += 1
        if score:
            category_scores[cat_name] = score

    ranked = sorted(category_scores.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:max_categories]]


def get_relevant_examples(
    question: str,
    max_categories: int = 10,
    max_per_category: int = 20,
    max_total: int | None = None,
    table_keys: list[str] | None = None
) -> str:
    """Get formatted examples relevant to the question.

    With local Ollama (Qwen 128K context) we can include MANY examples.
    More examples = better pattern learning for the model.

    Args:
        question: User's question in Ukrainian
        max_categories: Maximum categories to include (default: 10 = all categories)
        max_per_category: Maximum examples per category (default: 20)
        max_total: Optional max examples to include across all categories
        table_keys: Optional list of selected table names to bias examples

    Returns:
        Formatted string with SQL examples for the prompt, or empty string
        if no examples are found
    """
    if max_total is None:
        max_total = get_settings().sql_examples_max_total
    if max_total is not None and max_total <= 0:
        max_total = None
    curated_data = load_examples()
    extracted_data = load_extracted_examples()
    selected_tables = _normalize_table_keys(table_keys)
    categories = classify_question(
        question,
        examples_data=curated_data,
        fallback_to_simple=False
    )
    if not categories:
        categories = _categories_by_table_overlap(
            selected_tables,
            curated_data,
            extracted_data,
            max_categories,
        )
    if not categories:
        categories = classify_question(question, examples_data=curated_data)

    if not categories:
        return ""

    categories = categories[:max_categories]
    indexed = _build_examples_from_index(
        question,
        categories,
        selected_tables,
        max_per_category,
        max_total,
    )
    if indexed:
        return indexed

    query_tokens = _tokenize_text(question)
    content = _build_examples(
        categories,
        curated_data,
        extracted_data,
        max_categories,
        max_per_category,
        selected_tables,
        max_total,
        query_tokens,
    )
    if content:
        return content

    if selected_tables:
        logger.info("No table-matched examples found; falling back to unfiltered examples")
        return _build_examples(
            categories,
            curated_data,
            extracted_data,
            max_categories,
            max_per_category,
            set(),
            max_total,
            query_tokens,
        )

    return ""


def reload_examples() -> None:
    """Force reload of examples from file.

    Call this after updating sql_examples.yaml to refresh cached data.
    """
    global _examples_cache, _extracted_cache, _index_conn, _index_path, _index_fts_enabled
    _examples_cache = {}
    _extracted_cache = {}
    if _index_conn is not None:
        _index_conn.close()
        _index_conn = None
        _index_path = None
        _index_fts_enabled = False
    load_examples()
    logger.info("SQL examples cache reloaded")


def get_all_categories(include_extracted: bool = False) -> list[str]:
    """Get list of all available example categories.

    Returns:
        List of category names
    """
    examples_data = load_examples()
    categories = list(examples_data.get("categories", {}).keys())
    if include_extracted:
        extracted = load_extracted_examples()
        for name in extracted.get("categories", {}).keys():
            if name not in categories:
                categories.append(name)
    return categories


def get_category_info(category: str, include_extracted: bool = False) -> dict[str, Any]:
    """Get information about a specific category.

    Args:
        category: Category name
        include_extracted: Include extracted examples in counts

    Returns:
        Dictionary with 'description', 'keywords', and 'examples' count
    """
    examples_data = load_examples()
    cat_data = examples_data.get("categories", {}).get(category, {})

    extracted_count = 0
    if include_extracted:
        extracted = load_extracted_examples().get("categories", {}).get(category, {})
        extracted_count = len(extracted.get("examples", []))

    return {
        "description": cat_data.get("description", ""),
        "keywords": cat_data.get("keywords", []),
        "example_count": len(cat_data.get("examples", [])) + extracted_count
    }
