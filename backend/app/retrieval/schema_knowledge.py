"""Schema knowledge loader - provides actual column definitions for LLM prompts.

Instead of pattern matching against examples, we inject the actual schema
with column descriptions so the LLM understands what columns exist and mean.
"""

from __future__ import annotations

import logging
from pathlib import Path
from functools import lru_cache

import yaml

logger = logging.getLogger(__name__)

# Cache for loaded schema
_SCHEMA_KNOWLEDGE: dict | None = None


def _get_schema_path() -> Path:
    """Get path to schema_knowledge.yaml."""
    return Path(__file__).parent.parent.parent / "schema" / "schema_knowledge.yaml"


def _load_schema() -> dict:
    """Load schema knowledge from YAML."""
    global _SCHEMA_KNOWLEDGE

    if _SCHEMA_KNOWLEDGE is not None:
        return _SCHEMA_KNOWLEDGE

    schema_path = _get_schema_path()
    if not schema_path.exists():
        logger.warning(f"Schema knowledge not found: {schema_path}")
        return {"tables": [], "functions": [], "relationships": []}

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            _SCHEMA_KNOWLEDGE = yaml.safe_load(f)
        logger.info(f"Loaded schema knowledge: {len(_SCHEMA_KNOWLEDGE.get('tables', []))} tables")
        return _SCHEMA_KNOWLEDGE
    except Exception as e:
        logger.error(f"Failed to load schema knowledge: {e}")
        return {"tables": [], "functions": [], "relationships": []}


def get_table_schema(table_name: str) -> dict | None:
    """Get schema for a specific table.

    Args:
        table_name: Table name (with or without 'dbo.' prefix)

    Returns:
        Dict with table info including columns, or None if not found
    """
    schema = _load_schema()

    # Normalize table name (remove dbo. prefix if present)
    clean_name = table_name.replace("dbo.", "").strip("[]")

    for table in schema.get("tables", []):
        if table.get("name", "").lower() == clean_name.lower():
            return table

    return None


def get_tables_schema(table_names: list[str]) -> list[dict]:
    """Get schema for multiple tables.

    Args:
        table_names: List of table names

    Returns:
        List of table schema dicts
    """
    result = []
    for name in table_names:
        table = get_table_schema(name)
        if table:
            result.append(table)
    return result


def format_table_schema(table: dict, include_keywords: bool = False) -> str:
    """Format a table schema for injection into LLM prompt.

    Args:
        table: Table schema dict from schema_knowledge.yaml
        include_keywords: Whether to include Ukrainian keywords

    Returns:
        Formatted string for prompt
    """
    lines = []

    name = table.get("name", "Unknown")
    description = table.get("description", "")

    lines.append(f"-- Table: [{name}]")
    if description:
        lines.append(f"-- Description: {description}")

    if include_keywords:
        keywords = table.get("keywords_uk", [])
        if keywords:
            lines.append(f"-- Keywords: {', '.join(keywords)}")

    lines.append(f"-- Columns:")

    for col in table.get("columns", []):
        col_name = col.get("name", "")
        col_type = col.get("type", "")
        col_desc = col.get("description", "")

        # Format: name (type) - description
        col_line = f"--   [{name}].[{col_name}]"
        if col_type:
            col_line += f" ({col_type})"
        if col_desc:
            col_line += f" - {col_desc}"

        lines.append(col_line)

    return "\n".join(lines)


def format_tables_for_prompt(table_names: list[str], include_keywords: bool = True) -> str:
    """Format multiple tables' schemas for LLM prompt.

    This is the main function to use - it generates the schema documentation
    that should be injected into the SQL generation prompt.

    Args:
        table_names: List of table names to include
        include_keywords: Whether to include Ukrainian keywords

    Returns:
        Formatted schema documentation string
    """
    tables = get_tables_schema(table_names)

    if not tables:
        return ""

    sections = []
    sections.append("=" * 60)
    sections.append("AVAILABLE COLUMNS (use ONLY these columns!):")
    sections.append("=" * 60)

    for table in tables:
        sections.append("")
        sections.append(format_table_schema(table, include_keywords))

    return "\n".join(sections)


def get_function_docs(function_names: list[str] | None = None) -> str:
    """Get UDF function documentation.

    Args:
        function_names: Optional list of specific functions. If None, returns all.

    Returns:
        Formatted UDF documentation string
    """
    schema = _load_schema()
    functions = schema.get("functions", [])

    if not functions:
        return ""

    lines = []
    lines.append("-- Available UDF Functions:")

    for func in functions:
        name = func.get("name", "")

        # Filter if specific functions requested
        if function_names and name not in function_names:
            continue

        desc = func.get("description", "")
        params = func.get("parameters", [])
        returns = func.get("returns", "")

        lines.append(f"--   {name}")
        if desc:
            lines.append(f"--     Description: {desc}")
        if params:
            param_strs = [f"{p.get('name', '')}: {p.get('description', '')}" for p in params]
            lines.append(f"--     Parameters: {'; '.join(param_strs)}")
        if returns:
            lines.append(f"--     Returns: {returns}")

    return "\n".join(lines)


def get_relationships(table_names: list[str]) -> str:
    """Get relationship documentation for tables.

    Args:
        table_names: List of table names

    Returns:
        Formatted relationships string
    """
    schema = _load_schema()
    relationships = schema.get("relationships", [])

    if not relationships:
        return ""

    # Normalize table names
    clean_names = {name.replace("dbo.", "").strip("[]").lower() for name in table_names}

    lines = []
    lines.append("-- Table Relationships:")

    for rel in relationships:
        from_table = rel.get("from", "").lower()
        to_table = rel.get("to", "").lower()

        # Include if either table is in our list
        if from_table in clean_names or to_table in clean_names:
            via = rel.get("via", "")
            desc = rel.get("description", "")

            line = f"--   {rel.get('from', '')} -> {rel.get('to', '')}"
            if via:
                line += f" ({via})"
            if desc:
                line += f" // {desc}"
            lines.append(line)

    return "\n".join(lines) if len(lines) > 1 else ""


def get_full_context_for_tables(table_names: list[str]) -> str:
    """Get complete schema context for a set of tables.

    This combines:
    - Table schemas with columns
    - Relationships between tables
    - Relevant UDF functions

    Args:
        table_names: List of table names

    Returns:
        Complete schema context for prompt injection
    """
    parts = []

    # Table schemas
    table_docs = format_tables_for_prompt(table_names, include_keywords=True)
    if table_docs:
        parts.append(table_docs)

    # Relationships
    rel_docs = get_relationships(table_names)
    if rel_docs:
        parts.append("")
        parts.append(rel_docs)

    # UDF functions (include common ones)
    func_docs = get_function_docs()
    if func_docs:
        parts.append("")
        parts.append(func_docs)

    return "\n".join(parts)


# Quick test
if __name__ == "__main__":
    # Test loading
    tables = ["Client", "ClientInDebt", "Debt", "Agreement"]
    print(get_full_context_for_tables(tables))
