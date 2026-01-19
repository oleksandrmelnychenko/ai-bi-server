"""Build a SQLite index for SQL examples and schema metadata.

Usage:
    cd backend
    python -m tools.build_sql_index
    python -m tools.build_sql_index --drop  # Drop and rebuild
    python -m tools.build_sql_index --validate  # Include validation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from app.config import get_settings
from app.join_rules import load_join_rules
from app.schema_cache import SchemaCache
from app.sql_examples import extract_table_names

INDEX_VERSION = "2"  # Bumped for validation columns


def _normalize_identifier(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = cleaned.rstrip(",;")
    cleaned = cleaned.strip("()")
    cleaned = cleaned.replace("[", "").replace("]", "").replace('"', "")
    return cleaned.strip()


def _normalize_table_key(name: str) -> list[str]:
    cleaned = _normalize_identifier(name)
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


def _normalize_table_list(tables: list[str]) -> list[str]:
    normalized: set[str] = set()
    for name in tables:
        if not isinstance(name, str):
            continue
        normalized.update(_normalize_table_key(name))
    return sorted(normalized)


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split()).strip().lower()


def _hash_sql(sql: str) -> str:
    normalized = _normalize_sql(sql)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"categories": {}}
    with open(path, encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle) or {"categories": {}}


def _ensure_fts(conn: sqlite3.Connection) -> None:
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS examples_fts
            USING fts5(question, sql, tables, category, content='examples', content_rowid='id')
            """
        )
    except sqlite3.OperationalError as exc:
        raise RuntimeError("SQLite FTS5 is not available; rebuild Python with FTS5 support.") from exc


def _create_schema(conn: sqlite3.Connection, drop_existing: bool, reset_data: bool) -> None:
    if drop_existing:
        conn.executescript(
            """
            DROP TABLE IF EXISTS examples_fts;
            DROP TABLE IF EXISTS example_tables;
            DROP TABLE IF EXISTS examples;
            DROP TABLE IF EXISTS categories;
            DROP TABLE IF EXISTS schema_columns;
            DROP TABLE IF EXISTS schema_tables;
            DROP TABLE IF EXISTS foreign_keys;
            DROP TABLE IF EXISTS join_rules;
            DROP TABLE IF EXISTS meta;
            """
        )

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS categories (
            name TEXT PRIMARY KEY,
            description TEXT,
            keywords TEXT
        );

        CREATE TABLE IF NOT EXISTS examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            description TEXT,
            question TEXT,
            sql TEXT,
            source TEXT,
            source_type TEXT,
            tables TEXT,
            sql_hash TEXT UNIQUE,
            curated INTEGER,
            validated INTEGER DEFAULT 0,
            validation_errors TEXT,
            quality_score REAL DEFAULT 1.0
        );

        CREATE TABLE IF NOT EXISTS example_tables (
            example_id INTEGER,
            table_key TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_example_tables_key ON example_tables(table_key);
        CREATE INDEX IF NOT EXISTS idx_examples_category ON examples(category);

        CREATE TABLE IF NOT EXISTS schema_tables (
            table_key TEXT PRIMARY KEY,
            schema_name TEXT,
            table_name TEXT,
            role TEXT,
            default_filters TEXT,
            primary_key TEXT
        );

        CREATE TABLE IF NOT EXISTS schema_columns (
            table_key TEXT,
            column_name TEXT,
            data_type TEXT,
            nullable INTEGER,
            max_length INTEGER,
            PRIMARY KEY (table_key, column_name)
        );

        CREATE TABLE IF NOT EXISTS foreign_keys (
            fk_table TEXT,
            fk_column TEXT,
            pk_table TEXT,
            pk_column TEXT,
            name TEXT
        );

        CREATE TABLE IF NOT EXISTS join_rules (
            name TEXT,
            left_table TEXT,
            right_table TEXT,
            left_column TEXT,
            right_column TEXT,
            weight INTEGER,
            enabled INTEGER
        );
        """
    )
    _ensure_fts(conn)

    if reset_data:
        conn.executescript(
            """
            DELETE FROM examples_fts;
            DELETE FROM example_tables;
            DELETE FROM examples;
            DELETE FROM categories;
            DELETE FROM schema_columns;
            DELETE FROM schema_tables;
            DELETE FROM foreign_keys;
            DELETE FROM join_rules;
            DELETE FROM meta;
            """
        )


def _collect_examples(payload: dict[str, Any], curated: bool) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    categories = payload.get("categories", {}) or {}

    for category, cat_data in categories.items():
        description = cat_data.get("description", category)
        for example in cat_data.get("examples", []) or []:
            sql = (example.get("sql") or "").strip()
            if not sql:
                continue

            sql_hash = _hash_sql(sql)
            if sql_hash in merged:
                if curated and not merged[sql_hash]["curated"]:
                    merged[sql_hash] = {
                        "category": category,
                        "description": description,
                        "question": example.get("question", "") or "",
                        "sql": sql,
                        "source": example.get("source", "") or "",
                        "source_type": example.get("source_type", "") or "",
                        "tables": example.get("tables", []) or [],
                        "sql_hash": sql_hash,
                        "curated": True,
                    }
                continue

            merged[sql_hash] = {
                "category": category,
                "description": description,
                "question": example.get("question", "") or "",
                "sql": sql,
                "source": example.get("source", "") or "",
                "source_type": example.get("source_type", "") or "",
                "tables": example.get("tables", []) or [],
                "sql_hash": sql_hash,
                "curated": curated,
            }

    return merged


def _insert_categories(
    conn: sqlite3.Connection,
    curated_payload: dict[str, Any],
    extracted_payload: dict[str, Any],
) -> None:
    categories: dict[str, dict[str, Any]] = {}

    for payload in (extracted_payload, curated_payload):
        for name, cat_data in (payload.get("categories", {}) or {}).items():
            entry = categories.setdefault(name, {"description": "", "keywords": []})
            if not entry["description"]:
                entry["description"] = cat_data.get("description", "") or name
            if not entry["keywords"]:
                entry["keywords"] = cat_data.get("keywords", []) or []

    for name, data in categories.items():
        conn.execute(
            "INSERT OR REPLACE INTO categories (name, description, keywords) VALUES (?, ?, ?)",
            (name, data["description"], json.dumps(data["keywords"])),
        )


def _insert_examples(
    conn: sqlite3.Connection,
    examples: dict[str, dict[str, Any]],
    validator: Any = None,
) -> None:
    for record in examples.values():
        sql = record["sql"]
        tables = record["tables"]
        if isinstance(tables, list) and tables:
            normalized_tables = _normalize_table_list(tables)
        else:
            normalized_tables = extract_table_names(sql)

        tables_text = " ".join(normalized_tables)

        # Validate if validator is provided
        validated = 0
        validation_errors = None
        quality_score = 1.0

        if validator is not None:
            from tools.validate_examples import validate_example

            result = validate_example(
                sql=sql,
                sql_hash=record["sql_hash"],
                category=record["category"],
                question=record["question"],
                validator=validator,
            )
            validated = 1 if result.valid else 0
            if result.errors:
                validation_errors = json.dumps(
                    [{"type": e.error_type, "message": e.message} for e in result.errors]
                )
            quality_score = result.quality_score

        conn.execute(
            """
            INSERT INTO examples (
                category, description, question, sql, source, source_type,
                tables, sql_hash, curated, validated, validation_errors, quality_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["category"],
                record["description"],
                record["question"],
                sql,
                record["source"],
                record["source_type"],
                tables_text,
                record["sql_hash"],
                1 if record["curated"] else 0,
                validated,
                validation_errors,
                quality_score,
            ),
        )
        example_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        for table_key in normalized_tables:
            conn.execute(
                "INSERT INTO example_tables (example_id, table_key) VALUES (?, ?)",
                (example_id, table_key),
            )

        conn.execute(
            """
            INSERT INTO examples_fts (rowid, question, sql, tables, category)
            VALUES (?, ?, ?, ?, ?)
            """,
            (example_id, record["question"], sql, tables_text, record["category"]),
        )


def _insert_schema(conn: sqlite3.Connection, join_rules_path: str, skip_db: bool) -> None:
    if skip_db:
        return

    schema_cache = SchemaCache()
    schema_cache.load()
    join_rules = load_join_rules(join_rules_path)

    for table_key, table in schema_cache.tables.items():
        table_rule = join_rules.tables.get(table_key)
        role = table_rule.role if table_rule else "unknown"
        default_filters = list(table_rule.default_filters) if table_rule else []
        primary_key = list(table.primary_key) if table.primary_key else []

        conn.execute(
            """
            INSERT OR REPLACE INTO schema_tables (
                table_key, schema_name, table_name, role, default_filters, primary_key
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                table_key,
                table.schema,
                table.name,
                role,
                json.dumps(default_filters),
                json.dumps(primary_key),
            ),
        )

        for column in table.columns:
            conn.execute(
                """
                INSERT OR REPLACE INTO schema_columns (
                    table_key, column_name, data_type, nullable, max_length
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    table_key,
                    column.name,
                    column.data_type,
                    1 if column.nullable else 0,
                    column.max_length if column.max_length is not None else None,
                ),
            )

    for fk in schema_cache.foreign_keys:
        conn.execute(
            """
            INSERT INTO foreign_keys (
                fk_table, fk_column, pk_table, pk_column, name
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                fk.fk_key,
                fk.fk_column,
                fk.pk_key,
                fk.pk_column,
                fk.name,
            ),
        )

    for join in join_rules.joins:
        for column in join.columns:
            conn.execute(
                """
                INSERT INTO join_rules (
                    name, left_table, right_table, left_column, right_column, weight, enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    join.name,
                    join.left,
                    join.right,
                    column.left,
                    column.right,
                    join.weight,
                    1 if join.enabled else 0,
                ),
            )


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Build a SQLite index for SQL examples and schema metadata"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.sql_index_path),
        help="Output SQLite path",
    )
    parser.add_argument(
        "--curated",
        type=Path,
        default=Path(settings.sql_examples_path),
        help="Curated YAML path",
    )
    parser.add_argument(
        "--extracted",
        type=str,
        default=settings.sql_examples_extracted_path,
        help="Comma-separated extracted YAML paths",
    )
    parser.add_argument(
        "--join-rules",
        type=Path,
        default=Path(settings.join_rules_path),
        help="Join rules YAML path",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip DB schema introspection",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing index tables before rebuilding",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate examples against database schema",
    )
    args = parser.parse_args()

    curated_payload = _load_yaml(args.curated)
    extracted_payload = {"categories": {}}
    for raw_path in [p.strip() for p in args.extracted.split(",") if p.strip()]:
        payload = _load_yaml(Path(raw_path))
        for name, cat_data in (payload.get("categories", {}) or {}).items():
            merged_cat = extracted_payload["categories"].setdefault(name, {})
            if not merged_cat.get("description"):
                merged_cat["description"] = cat_data.get("description", "")
            merged_cat.setdefault("keywords", [])
            merged_cat.setdefault("examples", [])
            merged_cat["examples"].extend(cat_data.get("examples", []) or [])

    curated_examples = _collect_examples(curated_payload, curated=True)
    extracted_examples = _collect_examples(extracted_payload, curated=False)
    for sql_hash, record in extracted_examples.items():
        if sql_hash not in curated_examples:
            curated_examples[sql_hash] = record

    # Initialize validator if --validate is set
    validator = None
    if args.validate and not args.no_db:
        from tools.validate_examples import SchemaValidator

        print("Loading schema for validation...")
        validator = SchemaValidator()
        try:
            validator.load_from_database()
            print(f"Loaded {len(validator.tables)} tables for validation")
        except Exception as e:
            print(f"Warning: Could not load schema for validation: {e}")
            validator = None

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(output_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    _create_schema(conn, args.drop, reset_data=True)
    _insert_categories(conn, curated_payload, extracted_payload)
    _insert_examples(conn, curated_examples, validator)
    _insert_schema(conn, str(args.join_rules), args.no_db)

    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("version", INDEX_VERSION))
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("examples", str(len(curated_examples))))

    # Add validation stats to meta
    if validator is not None:
        valid_count = conn.execute("SELECT COUNT(*) FROM examples WHERE validated = 1").fetchone()[0]
        invalid_count = conn.execute("SELECT COUNT(*) FROM examples WHERE validated = 0").fetchone()[0]
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("validated_count", str(valid_count)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("invalid_count", str(invalid_count)),
        )
        print(f"Validation: {valid_count} valid, {invalid_count} invalid examples")

    conn.commit()
    conn.close()

    print(f"Index written to {output_path}")


if __name__ == "__main__":
    main()
