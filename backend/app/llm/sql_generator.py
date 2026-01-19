"""SQL query generation using LLM."""

from __future__ import annotations

import logging

from ..core.config import get_settings
from .client import call_ollama
from .prompts import SQL_GENERATION_SQLCODER, SQL_GENERATION_SYSTEM

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_VECTOR_SEARCH_AVAILABLE = None
_SCHEMA_VECTORS_AVAILABLE = None


def _get_retrieval_modules():
    """Lazy load retrieval modules to avoid import issues during setup."""
    global _VECTOR_SEARCH_AVAILABLE, _SCHEMA_VECTORS_AVAILABLE

    if _VECTOR_SEARCH_AVAILABLE is None:
        try:
            from ..retrieval.vector_search import DEFAULT_VECTOR_DB
            from ..retrieval.vector_search import get_relevant_examples as get_vector_examples
            _VECTOR_SEARCH_AVAILABLE = DEFAULT_VECTOR_DB.exists()
        except ImportError:
            _VECTOR_SEARCH_AVAILABLE = False
            get_vector_examples = None
    else:
        if _VECTOR_SEARCH_AVAILABLE:
            from ..retrieval.vector_search import get_relevant_examples as get_vector_examples
        else:
            get_vector_examples = None

    if _SCHEMA_VECTORS_AVAILABLE is None:
        try:
            from ..retrieval.schema_hints import (
                format_schema_hints,
                get_schema_hints,
                is_available as schema_vectors_available,
            )
            _SCHEMA_VECTORS_AVAILABLE = schema_vectors_available()
        except ImportError:
            _SCHEMA_VECTORS_AVAILABLE = False
            get_schema_hints = None
            format_schema_hints = None
    else:
        if _SCHEMA_VECTORS_AVAILABLE:
            from ..retrieval.schema_hints import format_schema_hints, get_schema_hints
        else:
            get_schema_hints = None
            format_schema_hints = None

    return (
        _VECTOR_SEARCH_AVAILABLE,
        get_vector_examples if _VECTOR_SEARCH_AVAILABLE else None,
        _SCHEMA_VECTORS_AVAILABLE,
        get_schema_hints if _SCHEMA_VECTORS_AVAILABLE else None,
        format_schema_hints if _SCHEMA_VECTORS_AVAILABLE else None,
    )


def _format_vector_examples(examples: list[dict], max_total: int = 10) -> str:
    """Format vector search results into prompt format."""
    if not examples:
        return ""

    lines = ["-- SQL Examples from GBA Repository (semantic match):\n"]

    for i, ex in enumerate(examples[:max_total]):
        category = ex.get("category", "unknown")
        sql = ex.get("sql", "").strip()
        tables = ex.get("tables", [])
        similarity = ex.get("similarity", 0)

        if not sql:
            continue

        lines.append(f"-- Example {i+1}: [{category}] (similarity: {similarity:.2f})")
        if tables:
            clean_tables = [t for t in tables[:5] if t.lower() not in ("and", "or", "on")]
            if clean_tables:
                lines.append(f"-- Tables: {', '.join(clean_tables)}")
        lines.append(sql)
        lines.append("")

    return "\n".join(lines) if len(lines) > 1 else ""


def get_relevant_examples(
    question: str,
    table_keys: list[str] | None = None,
    max_examples: int = 10,
    use_vector_search: bool = True
) -> str:
    """Get relevant SQL examples using vector search with keyword fallback.

    Args:
        question: User's question
        table_keys: List of selected tables to boost relevant examples
        max_examples: Maximum number of examples to return
        use_vector_search: Whether to use vector search (default True)

    Returns:
        Formatted string with SQL examples for the prompt
    """
    settings = get_settings()

    (
        vector_available,
        get_vector_examples,
        schema_available,
        _get_schema_hints,
        _format_schema_hints,
    ) = _get_retrieval_modules()

    # Try vector search first if available
    if use_vector_search and vector_available and get_vector_examples:
        try:
            examples = get_vector_examples(
                question=question,
                tables=table_keys,
                max_examples=max_examples
            )
            if examples:
                logger.info(f"Vector search returned {len(examples)} examples")
                return _format_vector_examples(examples, max_total=max_examples)
        except Exception as e:
            logger.warning(f"Vector search failed, falling back to keyword: {e}")

    # Fall back to keyword-based matching
    try:
        from ..retrieval.examples import get_relevant_examples as get_keyword_examples
        logger.info("Using keyword-based example matching")
        return get_keyword_examples(
            question,
            max_categories=settings.sql_examples_max_categories,
            max_per_category=settings.sql_examples_max_per_category,
            max_total=settings.sql_examples_max_total,
            table_keys=table_keys
        )
    except ImportError:
        logger.warning("Keyword example retrieval not available")
        return ""


def _is_sqlcoder_model(model: str | None) -> bool:
    """Check if the model is a SQLCoder variant."""
    if not model:
        return False
    return "sqlcoder" in model.lower()


def generate_sql(
    question: str,
    table_details: str,
    join_hints: str,
    max_rows: int,
    selected_tables: list[str] | None = None,
    model: str | None = None
) -> str:
    """Generate SQL query for the question using selected tables.

    Args:
        question: User's question
        table_details: Schema information for selected tables
        join_hints: Join conditions between tables
        max_rows: Row limit to apply
        selected_tables: Optional list of selected tables to bias example selection
        model: Optional model override (e.g., sqlcoder for specialized SQL generation)
    """
    logger.info(f"Generating SQL for question: {question[:100]}...")
    logger.debug(f"Table details length: {len(table_details)}, join hints length: {len(join_hints)}")

    settings = get_settings()

    # Try to get domain-specific query templates
    domain_example = ""
    try:
        from ..domain import extract_parameters, format_for_prompt, get_query, is_client_question

        # Extract parameters from question (exchange rate, currency, etc.)
        extracted_params = extract_parameters(question)
        if extracted_params.exchange_rate is not None:
            logger.info(f"Extracted exchange rate from question: {extracted_params.exchange_rate}")
        if extracted_params.currency:
            logger.info(f"Extracted currency from question: {extracted_params.currency}")

        # Check for domain-specific query templates (client-related)
        if is_client_question(question):
            client_query = get_query(question)
            if client_query:
                domain_example = format_for_prompt(client_query, params=extracted_params)
                logger.info(f"Using client query template: {client_query.id} ({client_query.name_uk})")
    except ImportError:
        logger.debug("Domain module not available for query templates")
        extracted_params = None

    # Get relevant few-shot examples using vector search (with keyword fallback)
    examples = get_relevant_examples(
        question=question,
        table_keys=selected_tables,
        max_examples=settings.sql_examples_max_total,
        use_vector_search=True
    )

    # Combine domain example (priority) with vector search results
    if domain_example:
        examples = f"-- RECOMMENDED PATTERN (use this as primary reference):\n{domain_example}\n\n{examples}"

    if examples:
        logger.debug(f"Including few-shot examples")

    # Get schema hints using vector search (if available)
    schema_hints = ""
    (
        vector_available,
        _get_vector_examples,
        schema_available,
        get_schema_hints,
        format_schema_hints,
    ) = _get_retrieval_modules()

    if schema_available and get_schema_hints and format_schema_hints:
        try:
            hints = get_schema_hints(question, max_tables=5, max_columns=8, max_functions=3)
            schema_hints = format_schema_hints(hints)
            if schema_hints:
                logger.info(f"Including schema hints: {len(hints.get('tables', []))} tables, {len(hints.get('columns', []))} columns, {len(hints.get('functions', []))} functions")
        except Exception as e:
            logger.warning(f"Schema hints retrieval failed: {e}")

    # Detect SQLCoder and use optimized prompt format + temperature
    if _is_sqlcoder_model(model):
        logger.info("Using SQLCoder-optimized prompt format")
        # SQLCoder: structured prompt with ### sections (no system message)
        # Inject examples before schema for better pattern recognition
        schema_with_hints = (
            f"{examples}\n\n" if examples else ""
        ) + (
            f"{schema_hints}\n" if schema_hints else ""
        ) + (
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
        # Inject examples after question for context
        examples_section = f"{examples}\n" if examples else ""
        schema_hints_section = f"{schema_hints}\n" if schema_hints else ""
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"{examples_section}"
            f"{schema_hints_section}"
            f"Table details:\n{table_details}\n\n"
            f"Join hints:\n{join_hints}\n\n"
            f"Row limit: TOP ({max_rows})\n"
        )
        messages = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        temperature = 0.05

    content = call_ollama(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=1200,
    )

    sql = content.strip()
    logger.info(f"Generated SQL length: {len(sql)} chars")
    logger.debug(f"Generated SQL: {sql[:200]}...")

    return sql
