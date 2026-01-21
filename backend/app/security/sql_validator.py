"""SQL validation with column checking and auto-correction.

Uses curated SQL examples to build a whitelist of valid table.column combinations.
Can validate LLM-generated SQL and suggest corrections for invalid columns.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from difflib import get_close_matches
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Cache for valid columns
_VALID_COLUMNS: dict[str, set[str]] | None = None
_COLUMN_ALIASES: dict[str, str] = {}  # Maps common wrong names to correct ones


def _load_valid_columns() -> dict[str, set[str]]:
    """Extract valid columns from SQL examples."""
    global _VALID_COLUMNS

    if _VALID_COLUMNS is not None:
        return _VALID_COLUMNS

    _VALID_COLUMNS = defaultdict(set)

    # Load from sql_examples_joins.yaml
    examples_path = Path(__file__).parent.parent.parent / "schema" / "sql_examples_joins.yaml"

    if not examples_path.exists():
        logger.warning(f"SQL examples not found: {examples_path}")
        return _VALID_COLUMNS

    try:
        with open(examples_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Extract columns from SQL: [Table].Column or Table.Column
        column_pattern = r'\[?(\w+)\]?\.(\w+)'

        for cat_data in data.get("categories", {}).values():
            for ex in cat_data.get("examples", []):
                sql = ex.get("sql", "")
                matches = re.findall(column_pattern, sql)
                for table, column in matches:
                    # Skip SQL keywords
                    if column.upper() not in ('ON', 'AND', 'OR', 'JOIN', 'LEFT', 'OUTER', 'WHERE', 'FROM', 'SELECT'):
                        _VALID_COLUMNS[table.lower()].add(column)

        logger.info(f"Loaded {len(_VALID_COLUMNS)} tables with valid columns")

    except Exception as e:
        logger.error(f"Failed to load SQL examples: {e}")

    return _VALID_COLUMNS


def _build_column_aliases():
    """Build common wrong column name -> correct column name mappings."""
    global _COLUMN_ALIASES

    # Common hallucinated columns -> actual columns
    _COLUMN_ALIASES = {
        # Client table
        "client.fullname": "Client.TIN",
        "client.name": "Client.TIN",
        "client.phone": "Client.EmailAddress",
        "client.code": "Client.USREOU",
        "client.address": "Client.TIN",

        # Product table
        "product.code": "Product.MainOriginalNumber",
        "product.sku": "Product.MainOriginalNumber",
        "product.productcode": "Product.MainOriginalNumber",

        # General
        "status": "ID",
        "total": "Amount",
        "sum": "Amount",
    }


def get_valid_columns(table: str) -> set[str]:
    """Get valid column names for a table."""
    columns = _load_valid_columns()
    return columns.get(table.lower(), set())


def validate_sql_columns(sql: str) -> tuple[bool, list[str], list[tuple[str, str, list[str]]]]:
    """Validate SQL columns against known valid columns.

    Returns:
        (is_valid, warnings, errors)
        - is_valid: True if all columns are valid
        - warnings: List of warning messages
        - errors: List of (table, invalid_column, suggestions) tuples
    """
    valid_columns = _load_valid_columns()

    # Extract table.column references from SQL
    column_pattern = r'\[?(\w+)\]?\.(\w+)'
    matches = re.findall(column_pattern, sql)

    warnings = []
    errors = []

    for table, column in matches:
        table_lower = table.lower()

        # Skip SQL keywords
        if column.upper() in ('ON', 'AND', 'OR', 'JOIN', 'LEFT', 'OUTER', 'WHERE', 'FROM', 'SELECT', 'TOP'):
            continue

        # Check if table exists in our examples
        if table_lower not in valid_columns:
            warnings.append(f"Unknown table: {table}")
            continue

        # Check if column exists for this table
        table_cols = valid_columns[table_lower]
        if column not in table_cols and column != 'ID':  # ID is always valid
            # Find similar column names
            suggestions = get_close_matches(column, list(table_cols), n=3, cutoff=0.4)
            errors.append((table, column, suggestions))

    is_valid = len(errors) == 0
    return is_valid, warnings, errors


def suggest_column_fix(table: str, wrong_column: str) -> str | None:
    """Suggest a fix for an invalid column name.

    Returns the suggested correct column name, or None if no suggestion.
    """
    _build_column_aliases()

    # Check aliases first
    alias_key = f"{table.lower()}.{wrong_column.lower()}"
    if alias_key in _COLUMN_ALIASES:
        return _COLUMN_ALIASES[alias_key]

    # Check general aliases
    if wrong_column.lower() in _COLUMN_ALIASES:
        return _COLUMN_ALIASES[wrong_column.lower()]

    # Find closest match in valid columns
    valid_cols = get_valid_columns(table)
    if valid_cols:
        matches = get_close_matches(wrong_column, list(valid_cols), n=1, cutoff=0.5)
        if matches:
            return matches[0]

    return None


def _inject_top_for_cte(sql: str, limit_val: str) -> str:
    """Inject TOP into a CTE query's main SELECT (not the CTE's internal SELECT)."""
    # For CTE like: WITH cte AS (SELECT ...) SELECT * FROM cte
    # We need to find the main SELECT (after the CTE definition)

    # Find the position after the last CTE definition closes
    # Match pattern: WITH name AS (...) [, name AS (...)]* SELECT
    # Strategy: Find all SELECTs and use the LAST one that's not inside parentheses

    # Simple approach: Find the last SELECT in the query
    matches = list(re.finditer(r'\bSELECT\s+(DISTINCT\s+)?', sql, re.IGNORECASE))
    if not matches:
        return sql

    # The last SELECT is typically the main query in a CTE
    last_match = matches[-1]
    distinct = last_match.group(1) or ""
    start, end = last_match.span()

    # Replace the last SELECT with SELECT TOP
    return sql[:start] + f"SELECT {distinct}TOP ({limit_val}) " + sql[end:]


def auto_correct_sql(sql: str) -> tuple[str, list[str]]:
    """Attempt to auto-correct invalid column names and syntax in SQL.

    Returns:
        (corrected_sql, corrections_made)
    """
    corrected_sql = sql
    corrections = []

    # === FIX SQL DIALECT ISSUES ===
    # Remove trailing semicolons first
    corrected_sql = corrected_sql.rstrip().rstrip(';').rstrip()

    # Convert LIMIT to TOP (MySQL/PostgreSQL -> SQL Server)
    limit_match = re.search(r'\bLIMIT\s+(\d+)\s*$', corrected_sql, re.IGNORECASE)
    if limit_match:
        limit_val = limit_match.group(1)
        # Remove LIMIT clause
        corrected_sql = re.sub(r'\s*LIMIT\s+\d+\s*$', '', corrected_sql, flags=re.IGNORECASE)
        # Add TOP after SELECT if not already present
        if not re.search(r'\bTOP\s*\(', corrected_sql, re.IGNORECASE):
            # Handle CTE queries - add TOP to main query SELECT, not the CTE SELECT
            if re.search(r'^\s*WITH\b', corrected_sql, re.IGNORECASE):
                # Find all SELECT positions and inject TOP into the LAST one (main query)
                # The main query SELECT comes after the CTE definition's closing paren
                corrected_sql = _inject_top_for_cte(corrected_sql, limit_val)
            else:
                # Regular query - add TOP after first SELECT
                corrected_sql = re.sub(
                    r'^(SELECT)\s+',
                    f'SELECT TOP ({limit_val}) ',
                    corrected_sql,
                    count=1,
                    flags=re.IGNORECASE
                )
        corrections.append(f"LIMIT {limit_val} -> TOP ({limit_val})")

    # Convert OFFSET/FETCH to TOP (if simple case)
    if re.search(r'\bOFFSET\s+\d+\s+ROWS?\s+FETCH\s+', corrected_sql, re.IGNORECASE):
        fetch_match = re.search(r'FETCH\s+(?:NEXT\s+)?(\d+)\s+ROWS?\s+ONLY', corrected_sql, re.IGNORECASE)
        if fetch_match:
            fetch_val = fetch_match.group(1)
            corrected_sql = re.sub(r'\s*OFFSET\s+\d+\s+ROWS?\s+FETCH.*$', '', corrected_sql, flags=re.IGNORECASE)
            corrected_sql = re.sub(
                r'^(SELECT)\s+',
                f'SELECT TOP ({fetch_val}) ',
                corrected_sql,
                count=1,
                flags=re.IGNORECASE
            )
            corrections.append(f"OFFSET/FETCH -> TOP ({fetch_val})")

    # === FIX COLUMN NAMES ===
    is_valid, warnings, errors = validate_sql_columns(corrected_sql)

    for table, wrong_col, suggestions in errors:
        # Try to find a fix
        fix = suggest_column_fix(table, wrong_col)

        if not fix and suggestions:
            fix = suggestions[0]

        if fix:
            # Replace wrong column with fix
            # Match [Table].WrongCol or Table.WrongCol
            pattern = rf'\[?{re.escape(table)}\]?\.{re.escape(wrong_col)}\b'
            replacement = f"[{table}].{fix}"
            corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)
            corrections.append(f"{table}.{wrong_col} -> {table}.{fix}")

    return corrected_sql, corrections


def get_table_column_hints(tables: list[str]) -> str:
    """Get column hints for specific tables to include in prompt.

    Returns a formatted string with valid columns for the given tables.
    """
    valid_columns = _load_valid_columns()

    lines = ["-- VALID COLUMNS (use ONLY these):"]

    for table in tables:
        # Handle both "dbo.Table" and "Table" formats
        table_name = table.split(".")[-1] if "." in table else table
        table_lower = table_name.lower()

        if table_lower in valid_columns:
            cols = sorted(valid_columns[table_lower])[:10]  # Limit to 10 most important
            lines.append(f"-- {table_name}: {', '.join(cols)}")

    return "\n".join(lines)
