"""Validate SQL examples against database schema.

This tool checks that SQL examples reference valid tables and columns
from the database schema, helping identify broken or outdated examples.

Usage:
    cd backend
    python -m tools.validate_examples
    python -m tools.validate_examples --output schema/validation_report.json
    python -m tools.validate_examples --no-db  # Skip database connection, use SQLite index
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from app.config import get_settings
from app.sql_examples import extract_table_names


@dataclass
class ValidationError:
    """Single validation error for an example."""
    error_type: str  # "unknown_table", "unknown_column", "syntax_warning"
    message: str
    table: str | None = None
    column: str | None = None


@dataclass
class ValidationResult:
    """Result of validating a single SQL example."""
    sql_hash: str
    category: str
    question: str
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    tables_found: list[str] = field(default_factory=list)
    tables_validated: int = 0
    columns_checked: int = 0
    quality_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sql_hash": self.sql_hash,
            "category": self.category,
            "question": self.question,
            "valid": self.valid,
            "errors": [asdict(e) for e in self.errors],
            "tables_found": self.tables_found,
            "tables_validated": self.tables_validated,
            "columns_checked": self.columns_checked,
            "quality_score": self.quality_score,
        }


@dataclass
class ValidationReport:
    """Summary of all validation results."""
    total_examples: int = 0
    valid_examples: int = 0
    invalid_examples: int = 0
    unknown_tables: int = 0
    unknown_columns: int = 0
    results: list[ValidationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_examples": self.total_examples,
                "valid_examples": self.valid_examples,
                "invalid_examples": self.invalid_examples,
                "unknown_tables": self.unknown_tables,
                "unknown_columns": self.unknown_columns,
                "validity_rate": round(self.valid_examples / max(self.total_examples, 1), 3),
            },
            "results": [r.to_dict() for r in self.results],
        }


# Regex patterns for column extraction
_COLUMN_REF_PATTERNS = [
    # [Table].Column or [Table].[Column]
    re.compile(r"\[([^\]]+)\]\.\[?(\w+)\]?", re.IGNORECASE),
    # Table.Column (without brackets)
    re.compile(r"\b(\w+)\.(\w+)\b", re.IGNORECASE),
    # SELECT Column AS ... or SELECT Column,
    re.compile(r"SELECT\s+(?:\[?(\w+)\]?\.)?(\w+)", re.IGNORECASE),
]

# SQL keywords to exclude from column detection
_SQL_KEYWORDS = {
    "select", "from", "where", "join", "left", "right", "inner", "outer",
    "on", "and", "or", "not", "in", "is", "null", "as", "order", "by",
    "group", "having", "top", "distinct", "union", "all", "case", "when",
    "then", "else", "end", "like", "between", "exists", "count", "sum",
    "avg", "min", "max", "round", "isnull", "coalesce", "cast", "convert",
    "dateadd", "datediff", "year", "month", "day", "getdate", "getutcdate",
    "desc", "asc", "with", "over", "partition", "row_number", "rank",
    "delete", "insert", "update", "set", "values", "into", "create", "drop",
    "table", "index", "view", "procedure", "function", "trigger", "schema",
}


def _normalize_table_key(name: str) -> str:
    """Normalize table name to schema.table format."""
    cleaned = name.strip().lower()
    cleaned = cleaned.replace("[", "").replace("]", "").replace('"', "")
    if "." not in cleaned:
        cleaned = f"dbo.{cleaned}"
    return cleaned


def _extract_column_refs(sql: str) -> list[tuple[str, str]]:
    """Extract (table, column) pairs from SQL.

    Returns list of tuples where first element is table name (or empty string)
    and second is column name.
    """
    refs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for pattern in _COLUMN_REF_PATTERNS:
        for match in pattern.finditer(sql):
            groups = match.groups()
            if len(groups) >= 2:
                table = groups[0] or ""
                column = groups[1] or ""
            else:
                table = ""
                column = groups[0] if groups else ""

            if not column:
                continue

            # Skip SQL keywords
            if column.lower() in _SQL_KEYWORDS:
                continue

            # Skip numeric values
            if column.isdigit():
                continue

            # Clean up names
            table = table.strip().lower().replace("[", "").replace("]", "")
            column = column.strip().lower().replace("[", "").replace("]", "")

            key = (table, column)
            if key not in seen:
                seen.add(key)
                refs.append(key)

    return refs


class SchemaValidator:
    """Validates SQL against database schema."""

    def __init__(self) -> None:
        self.tables: dict[str, set[str]] = {}  # table_key -> set of column names
        self._loaded = False

    def load_from_database(self) -> None:
        """Load schema from SQL Server database."""
        from app.schema_cache import SchemaCache

        cache = SchemaCache()
        cache.load()

        for table_key, table_info in cache.tables.items():
            columns = {col.name.lower() for col in table_info.columns}
            self.tables[table_key.lower()] = columns
            # Also add without schema prefix for matching
            table_name = table_info.name.lower()
            if table_name not in self.tables:
                self.tables[table_name] = columns

        self._loaded = True

    def load_from_sqlite_index(self, index_path: Path) -> None:
        """Load schema from SQLite index file."""
        if not index_path.exists():
            raise FileNotFoundError(f"SQLite index not found: {index_path}")

        conn = sqlite3.connect(index_path)
        conn.row_factory = sqlite3.Row

        # Load tables and columns
        tables_query = "SELECT table_key FROM schema_tables"
        columns_query = "SELECT table_key, column_name FROM schema_columns"

        # Initialize all tables
        for row in conn.execute(tables_query):
            self.tables[row["table_key"].lower()] = set()

        # Add columns to tables
        for row in conn.execute(columns_query):
            table_key = row["table_key"].lower()
            column_name = row["column_name"].lower()
            if table_key in self.tables:
                self.tables[table_key].add(column_name)
            # Also add short table name
            short_name = table_key.split(".")[-1]
            if short_name not in self.tables:
                self.tables[short_name] = set()
            self.tables[short_name].add(column_name)

        conn.close()
        self._loaded = True

    def is_loaded(self) -> bool:
        """Check if schema has been loaded."""
        return self._loaded

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in schema."""
        normalized = _normalize_table_key(table_name)
        short_name = table_name.lower().replace("[", "").replace("]", "")
        return normalized in self.tables or short_name in self.tables

    def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if column exists in table."""
        normalized = _normalize_table_key(table_name)
        short_name = table_name.lower().replace("[", "").replace("]", "")
        column_lower = column_name.lower()

        # Check both full key and short name
        for key in (normalized, short_name):
            if key in self.tables:
                if column_lower in self.tables[key]:
                    return True
        return False

    def get_table_columns(self, table_name: str) -> set[str]:
        """Get columns for a table."""
        normalized = _normalize_table_key(table_name)
        short_name = table_name.lower().replace("[", "").replace("]", "")

        for key in (normalized, short_name):
            if key in self.tables:
                return self.tables[key]
        return set()


def validate_example(
    sql: str,
    sql_hash: str,
    category: str,
    question: str,
    validator: SchemaValidator,
) -> ValidationResult:
    """Validate a single SQL example against schema.

    Args:
        sql: The SQL query text
        sql_hash: Unique hash of the SQL
        category: Example category name
        question: Example question text
        validator: SchemaValidator with loaded schema

    Returns:
        ValidationResult with validation details
    """
    result = ValidationResult(
        sql_hash=sql_hash,
        category=category,
        question=question,
        valid=True,
    )

    # Extract table names from SQL
    tables = extract_table_names(sql)
    result.tables_found = tables

    if not validator.is_loaded():
        # Can't validate without schema, assume valid
        return result

    # Validate each table exists
    validated_tables: set[str] = set()
    for table in tables:
        if validator.table_exists(table):
            validated_tables.add(table)
            result.tables_validated += 1
        else:
            result.valid = False
            result.errors.append(ValidationError(
                error_type="unknown_table",
                message=f"Table not found in schema: {table}",
                table=table,
            ))

    # Extract and validate column references
    column_refs = _extract_column_refs(sql)
    for table_ref, column in column_refs:
        result.columns_checked += 1

        if not table_ref:
            # Column without table prefix - check against all found tables
            found = False
            for table in validated_tables:
                if validator.column_exists(table, column):
                    found = True
                    break
            if not found and validated_tables:
                # Only warn if we have validated tables to check against
                result.errors.append(ValidationError(
                    error_type="syntax_warning",
                    message=f"Column '{column}' not found in any matched table",
                    column=column,
                ))
        else:
            # Column with table prefix
            if table_ref in [t.split(".")[-1].lower() for t in validated_tables]:
                if not validator.column_exists(table_ref, column):
                    result.valid = False
                    result.errors.append(ValidationError(
                        error_type="unknown_column",
                        message=f"Column not found: {table_ref}.{column}",
                        table=table_ref,
                        column=column,
                    ))

    # Calculate quality score
    # Start at 1.0, deduct for errors
    score = 1.0
    for error in result.errors:
        if error.error_type == "unknown_table":
            score -= 0.3  # Major penalty
        elif error.error_type == "unknown_column":
            score -= 0.15  # Medium penalty
        elif error.error_type == "syntax_warning":
            score -= 0.05  # Minor penalty

    # Bonus for having a question
    if question:
        score += 0.1

    # Bonus for validating multiple tables (indicates complex query)
    if result.tables_validated > 2:
        score += 0.1

    result.quality_score = max(0.0, min(1.0, score))

    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file."""
    if not path.exists():
        return {"categories": {}}
    with open(path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {"categories": {}}


def _hash_sql(sql: str) -> str:
    """Generate hash for SQL."""
    import hashlib
    normalized = " ".join(sql.split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def validate_examples_file(
    yaml_path: Path,
    validator: SchemaValidator,
) -> ValidationReport:
    """Validate all examples in a YAML file.

    Args:
        yaml_path: Path to sql_examples.yaml or sql_examples_extracted.yaml
        validator: SchemaValidator with loaded schema

    Returns:
        ValidationReport with all results
    """
    report = ValidationReport()
    data = _load_yaml(yaml_path)

    for category, cat_data in data.get("categories", {}).items():
        for example in cat_data.get("examples", []):
            sql = example.get("sql", "").strip()
            if not sql:
                continue

            question = example.get("question", "")
            sql_hash = _hash_sql(sql)

            result = validate_example(
                sql=sql,
                sql_hash=sql_hash,
                category=category,
                question=question,
                validator=validator,
            )

            report.total_examples += 1
            if result.valid:
                report.valid_examples += 1
            else:
                report.invalid_examples += 1
                for error in result.errors:
                    if error.error_type == "unknown_table":
                        report.unknown_tables += 1
                    elif error.error_type == "unknown_column":
                        report.unknown_columns += 1

            report.results.append(result)

    return report


def main() -> None:
    """Main entry point for validation tool."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Validate SQL examples against database schema"
    )
    parser.add_argument(
        "--curated",
        type=Path,
        default=Path(settings.sql_examples_path),
        help="Path to curated examples YAML",
    )
    parser.add_argument(
        "--extracted",
        type=str,
        default=settings.sql_examples_extracted_path,
        help="Comma-separated paths to extracted examples YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON report path (prints to stdout if not specified)",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip database connection, use SQLite index instead",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(settings.sql_index_path),
        help="SQLite index path (used with --no-db)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed validation results",
    )
    args = parser.parse_args()

    # Initialize validator
    validator = SchemaValidator()

    if args.no_db:
        print(f"Loading schema from SQLite index: {args.index}")
        try:
            validator.load_from_sqlite_index(args.index)
            print(f"Loaded {len(validator.tables)} tables from index")
        except FileNotFoundError:
            print(f"Warning: SQLite index not found at {args.index}")
            print("Validation will skip table/column checks")
    else:
        print("Loading schema from database...")
        try:
            validator.load_from_database()
            print(f"Loaded {len(validator.tables)} tables from database")
        except Exception as e:
            print(f"Warning: Could not connect to database: {e}")
            print("Validation will skip table/column checks")

    # Validate curated examples
    combined_report = ValidationReport()

    if args.curated.exists():
        print(f"\nValidating curated examples: {args.curated}")
        report = validate_examples_file(args.curated, validator)
        combined_report.total_examples += report.total_examples
        combined_report.valid_examples += report.valid_examples
        combined_report.invalid_examples += report.invalid_examples
        combined_report.unknown_tables += report.unknown_tables
        combined_report.unknown_columns += report.unknown_columns
        combined_report.results.extend(report.results)
        print(f"  - {report.total_examples} examples, {report.valid_examples} valid, {report.invalid_examples} invalid")

    # Validate extracted examples
    for raw_path in [p.strip() for p in args.extracted.split(",") if p.strip()]:
        path = Path(raw_path)
        if not path.exists():
            continue

        print(f"\nValidating extracted examples: {path}")
        report = validate_examples_file(path, validator)
        combined_report.total_examples += report.total_examples
        combined_report.valid_examples += report.valid_examples
        combined_report.invalid_examples += report.invalid_examples
        combined_report.unknown_tables += report.unknown_tables
        combined_report.unknown_columns += report.unknown_columns
        combined_report.results.extend(report.results)
        print(f"  - {report.total_examples} examples, {report.valid_examples} valid, {report.invalid_examples} invalid")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total examples:   {combined_report.total_examples}")
    print(f"Valid examples:   {combined_report.valid_examples}")
    print(f"Invalid examples: {combined_report.invalid_examples}")
    print(f"Unknown tables:   {combined_report.unknown_tables}")
    print(f"Unknown columns:  {combined_report.unknown_columns}")
    if combined_report.total_examples > 0:
        rate = combined_report.valid_examples / combined_report.total_examples * 100
        print(f"Validity rate:    {rate:.1f}%")

    # Print verbose details
    if args.verbose:
        print("\n" + "-" * 60)
        print("INVALID EXAMPLES:")
        print("-" * 60)
        for result in combined_report.results:
            if not result.valid:
                print(f"\n[{result.category}] {result.question or 'No question'}")
                print(f"  Hash: {result.sql_hash}")
                for error in result.errors:
                    print(f"  - {error.error_type}: {error.message}")

    # Write output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(combined_report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()
