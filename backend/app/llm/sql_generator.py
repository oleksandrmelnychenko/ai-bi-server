"""SQL query generation using LLM with schema knowledge injection.

Strategy:
1. Inject actual column definitions from schema_knowledge.yaml into the prompt
2. Use curated queries from domain catalogs as few-shot examples
3. LLM understands schema meaning → generates correct SQL

This approach prevents column hallucination by showing the LLM exactly what columns exist.
"""

from __future__ import annotations

import logging
import re

from ..core.config import get_settings
from .client import call_ollama
from .prompts import SQL_GENERATION_SQLCODER, SQL_GENERATION_SYSTEM

logger = logging.getLogger(__name__)


def _is_sqlcoder_model(model: str | None) -> bool:
    """Check if the model is a SQLCoder variant."""
    if not model:
        return False
    return "sqlcoder" in model.lower()


def _get_schema_context(table_names: list[str]) -> str:
    """Get schema documentation for selected tables.

    This injects actual column definitions with descriptions so the LLM
    knows exactly what columns exist and what they mean.
    """
    try:
        from ..retrieval.schema_knowledge import get_full_context_for_tables
        context = get_full_context_for_tables(table_names)
        if context:
            logger.info(f"Injecting schema knowledge for {len(table_names)} tables")
            return context
    except ImportError:
        logger.debug("Schema knowledge module not available")
    except Exception as e:
        logger.warning(f"Failed to get schema context: {e}")

    return ""


def _get_curated_examples(question: str) -> str:
    """Get relevant curated SQL examples based on the question.

    These are real working queries from the codebase, not synthetic examples.
    """
    try:
        from ..retrieval.curated_queries import get_relevant_examples
        examples = get_relevant_examples(question)
        if examples:
            logger.info("Found matching curated query example")
            return examples
    except ImportError:
        logger.debug("Curated queries module not available")
    except Exception as e:
        logger.warning(f"Failed to get curated examples: {e}")

    return ""


def _adjust_top_limit(sql: str, max_rows: int) -> str:
    """Adjust or add TOP limit to SQL query."""
    sql = sql.strip()

    # Check if TOP already exists
    top_pattern = r'\bTOP\s*\(\s*\d+\s*\)'
    if re.search(top_pattern, sql, re.IGNORECASE):
        # Replace existing TOP
        sql = re.sub(top_pattern, f'TOP ({max_rows})', sql, count=1, flags=re.IGNORECASE)
    else:
        # Add TOP after SELECT (handle CTE case)
        if re.search(r'^\s*WITH\b', sql, re.IGNORECASE):
            # CTE: find last SELECT and add TOP there
            matches = list(re.finditer(r'\bSELECT\s+(DISTINCT\s+)?', sql, re.IGNORECASE))
            if matches:
                last_match = matches[-1]
                distinct = last_match.group(1) or ""
                start, end = last_match.span()
                sql = sql[:start] + f"SELECT {distinct}TOP ({max_rows}) " + sql[end:]
        else:
            # Regular query
            sql = re.sub(
                r'^(SELECT\s+)',
                f'SELECT TOP ({max_rows}) ',
                sql,
                count=1,
                flags=re.IGNORECASE
            )

    return sql


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
        table_details: Schema information for selected tables (from schema_cache)
        join_hints: Join conditions between tables (from join_graph)
        max_rows: Row limit to apply
        selected_tables: List of selected tables (used for schema knowledge injection)
        model: Optional model override
    """
    logger.info(f"Generating SQL for question: {question[:100]}...")

    # === 1. GET SCHEMA KNOWLEDGE ===
    # Inject actual column definitions with descriptions
    schema_knowledge = ""
    if selected_tables:
        schema_knowledge = _get_schema_context(selected_tables)

    # === 2. GET CURATED EXAMPLES ===
    # Real SQL queries from the codebase as few-shot examples
    curated_examples = _get_curated_examples(question)

    # === 3. BUILD PROMPT ===
    if _is_sqlcoder_model(model):
        logger.info("Using SQLCoder-optimized prompt format")

        # SQLCoder format: structured sections
        schema_section = ""

        if schema_knowledge:
            schema_section += f"{schema_knowledge}\n\n"

        if curated_examples:
            schema_section += f"{curated_examples}\n\n"

        schema_section += (
            f"{table_details}\n\n"
            f"-- Join hints:\n{join_hints}\n"
            f"-- Row limit: TOP ({max_rows})"
        )

        user_prompt = SQL_GENERATION_SQLCODER.format(
            question=question,
            schema=schema_section
        )
        messages = [{"role": "user", "content": user_prompt}]
        temperature = 0.0

    else:
        # Qwen/general models: system + user message format
        logger.info("Using general model prompt format")

        # Build user prompt with all context
        prompt_parts = []

        prompt_parts.append(f"Question:\n{question}")

        if curated_examples:
            prompt_parts.append(f"\n{curated_examples}")

        if schema_knowledge:
            prompt_parts.append(f"\n{schema_knowledge}")

        prompt_parts.append(f"\nTable details (from database):\n{table_details}")
        prompt_parts.append(f"\nJoin hints:\n{join_hints}")
        prompt_parts.append(f"\nRow limit: TOP ({max_rows})")

        user_prompt = "\n".join(prompt_parts)

        messages = [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        temperature = 0.05

    # === 4. CALL LLM ===
    content = call_ollama(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=1500,
    )

    sql = content.strip()
    logger.info(f"Generated SQL length: {len(sql)} chars")
    logger.debug(f"Generated SQL: {sql[:200]}...")

    # === 5. POST-PROCESS ===
    # Extract SQL from markdown if needed
    sql = _extract_sql(sql)

    # Validate and auto-correct
    sql = _validate_and_correct_sql(sql)

    # Ensure TOP limit
    sql = _adjust_top_limit(sql, max_rows)

    return sql


def _extract_sql(text: str) -> str:
    """Extract SQL from LLM response, handling various output formats."""
    if not text:
        return ""

    # Try ```sql block first
    fenced = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # Try generic ``` block
    fenced = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # Try SQLCoder [SQL]...[/SQL] format
    sqlcoder = re.search(r"\[SQL\]\s*(.*?)(?:\[/SQL\]|$)", text, re.IGNORECASE | re.DOTALL)
    if sqlcoder:
        return sqlcoder.group(1).strip()

    # Try to find SELECT statement directly
    select_match = re.search(r"(SELECT\s+.*?)(?:;|\Z)", text, re.IGNORECASE | re.DOTALL)
    if select_match:
        return select_match.group(1).strip()

    # Try to find WITH (CTE) statement
    with_match = re.search(r"(WITH\s+.*?)(?:;|\Z)", text, re.IGNORECASE | re.DOTALL)
    if with_match:
        return with_match.group(1).strip()

    return text.strip()


def _validate_and_correct_sql(sql: str) -> str:
    """Validate SQL columns and auto-correct if possible."""
    try:
        from ..security.sql_validator import validate_sql_columns, auto_correct_sql

        is_valid, warnings, errors = validate_sql_columns(sql)

        if warnings:
            for w in warnings:
                logger.warning(f"SQL validation warning: {w}")

        if not is_valid:
            logger.warning(f"SQL has {len(errors)} invalid column(s), attempting auto-correction")
            for table, col, suggestions in errors:
                logger.warning(f"  Invalid: {table}.{col} -> suggestions: {suggestions}")

            corrected_sql, corrections = auto_correct_sql(sql)

            if corrections:
                logger.info(f"Auto-corrected {len(corrections)} column(s): {corrections}")
                return corrected_sql
            else:
                logger.warning("Could not auto-correct SQL, returning original")

    except ImportError:
        logger.debug("SQL validator not available")
    except Exception as e:
        logger.warning(f"SQL validation failed: {e}")

    return sql
