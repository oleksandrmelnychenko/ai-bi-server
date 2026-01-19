"""Extract SQL queries from database objects for few-shot learning.

Usage:
    cd backend
    python -m tools.extract_db_examples
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from app.db import DatabaseError, get_connection
from app.sql_examples import extract_table_names
from tools.extract_gba_examples import (
    ExtractedQuery,
    categorize_sql,
    contains_sql_keywords,
    export_to_yaml,
    normalize_sql,
)

DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent / "schema" / "sql_examples_extracted_db.yaml"
OBJECT_TYPE_MAP = {
    "V": "view",
    "P": "procedure",
    "FN": "scalar_function",
    "IF": "inline_table_function",
    "TF": "table_function",
}


def fetch_db_definitions() -> list[tuple[str, str, str, str]]:
    """Fetch SQL module definitions from the database."""
    query = """
        SELECT o.type, s.name AS schema_name, o.name AS object_name, m.definition
        FROM sys.objects o
        JOIN sys.schemas s ON o.schema_id = s.schema_id
        JOIN sys.sql_modules m ON o.object_id = m.object_id
        WHERE o.type IN ('V', 'P', 'FN', 'IF', 'TF')
        ORDER BY o.type, s.name, o.name;
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]
    except DatabaseError as exc:
        raise exc


def extract_sql_from_definition(definition: str) -> list[str]:
    """Extract SQL statements from a view/proc/function definition."""
    if not definition:
        return []

    statements = re.split(r"^\s*GO\s*$", definition, flags=re.IGNORECASE | re.MULTILINE)
    extracted: list[str] = []

    for statement in statements:
        if not contains_sql_keywords(statement):
            continue

        match = re.search(r"\b(SELECT|WITH)\b", statement, re.IGNORECASE)
        if not match:
            continue

        sql = normalize_sql(statement[match.start():])
        if sql and len(sql) >= 30:
            extracted.append(sql)

    return extracted


def extract_db_queries() -> dict[str, list[ExtractedQuery]]:
    """Extract SQL queries from database objects into categorized buckets."""
    all_queries: dict[str, list[ExtractedQuery]] = {}
    rows = fetch_db_definitions()

    for obj_type, schema_name, object_name, definition in rows:
        source_label = f"db:{schema_name}.{object_name}"
        source_type = f"db_{OBJECT_TYPE_MAP.get(obj_type, 'object')}"
        sql_statements = extract_sql_from_definition(definition)

        for sql in sql_statements:
            category = categorize_sql(sql)
            tables = extract_table_names(sql)
            query = ExtractedQuery(
                sql=sql,
                source_file=source_label,
                method_name=OBJECT_TYPE_MAP.get(obj_type, "object"),
                tables=tables,
                category=category,
                source_type=source_type,
            )
            all_queries.setdefault(category, []).append(query)

    return all_queries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SQL queries from database objects for few-shot learning"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output YAML file path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit per category (0 = no limit)",
    )
    args = parser.parse_args()

    try:
        queries = extract_db_queries()
    except DatabaseError as exc:
        print(f"Database error: {exc}")
        return

    total = sum(len(v) for v in queries.values())
    print(f"Found {total} SQL statements across {len(queries)} categories")

    if total > 0:
        limit_per_category = args.limit if args.limit > 0 else None
        export_to_yaml(queries, args.output, limit_per_category=limit_per_category)
        print(f"Exported DB examples to {args.output}")
    else:
        print("No SQL statements found in database objects.")


if __name__ == "__main__":
    main()
