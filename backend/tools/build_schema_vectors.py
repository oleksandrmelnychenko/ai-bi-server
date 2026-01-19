"""Build vector embeddings for schema knowledge base.

This enables semantic search to find relevant tables/columns for user questions.

Usage:
    cd backend
    python -m tools.build_schema_vectors
    python -m tools.build_schema_vectors --no-db  # Use YAML only
    python -m tools.build_schema_vectors --no-yaml  # Use DB only
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from app.core.config import get_settings
from app.core.db import get_connection
from app.schema.cache import SchemaCache
from app.schema.join_rules import load_join_rules

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
SCHEMA_YAML = Path(__file__).parent.parent / "schema" / "schema_knowledge.yaml"
OUTPUT_DB = Path(__file__).parent.parent / "schema" / "schema_vectors.sqlite"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"tables": [], "functions": [], "relationships": []}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"tables": [], "functions": [], "relationships": []}


def _normalize_name(name: str) -> str:
    cleaned = name.strip().lower()
    cleaned = cleaned.replace("[", "").replace("]", "").replace('"', "")
    return cleaned


def _name_variants(name: str) -> list[str]:
    cleaned = _normalize_name(name)
    if not cleaned:
        return []
    base = cleaned.split(".")[-1]
    if base != cleaned:
        return [cleaned, base]
    return [cleaned]


def _find_table(
    name: str,
    by_full: dict[str, dict[str, Any]],
    by_base: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    cleaned = _normalize_name(name)
    if cleaned in by_full:
        return by_full[cleaned]
    base = cleaned.split(".")[-1]
    candidates = by_base.get(base, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    for candidate in candidates:
        if candidate.get("name", "").lower().startswith("dbo."):
            return candidate
    return candidates[0]


def _merge_keywords(existing: list[str], incoming: list[str]) -> list[str]:
    seen = set(k for k in existing if isinstance(k, str))
    merged = [k for k in existing if isinstance(k, str)]
    for item in incoming:
        if not isinstance(item, str):
            continue
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


def _merge_schema(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    base_tables = base.get("tables", []) or []
    base_functions = base.get("functions", []) or []
    base_relationships = base.get("relationships", []) or []

    by_full: dict[str, dict[str, Any]] = {}
    by_base: dict[str, list[dict[str, Any]]] = {}
    for table in base_tables:
        name = table.get("name", "")
        for variant in _name_variants(name):
            if variant not in by_full:
                by_full[variant] = table
        base = _normalize_name(name).split(".")[-1]
        by_base.setdefault(base, []).append(table)

    for table in overlay.get("tables", []) or []:
        name = table.get("name", "")
        match = _find_table(name, by_full, by_base)
        if match is None:
            base_tables.append(table)
            continue
        if table.get("description") and not match.get("description"):
            match["description"] = table.get("description", "")
        match["keywords_uk"] = _merge_keywords(
            match.get("keywords_uk", []) or [],
            table.get("keywords_uk", []) or [],
        )
        base_columns = {c.get("name", "").lower(): c for c in match.get("columns", []) or []}
        for column in table.get("columns", []) or []:
            col_name = column.get("name", "").lower()
            if not col_name:
                continue
            target = base_columns.get(col_name)
            if target is None:
                match.setdefault("columns", []).append(column)
                continue
            if column.get("description") and not target.get("description"):
                target["description"] = column.get("description", "")
            target["keywords_uk"] = _merge_keywords(
                target.get("keywords_uk", []) or [],
                column.get("keywords_uk", []) or [],
            )

    functions_by_name = {f.get("name", "").lower(): f for f in base_functions}
    for func in overlay.get("functions", []) or []:
        name = func.get("name", "").lower()
        if not name:
            continue
        if name not in functions_by_name:
            base_functions.append(func)
            continue
        target = functions_by_name[name]
        if func.get("description") and not target.get("description"):
            target["description"] = func.get("description", "")
        target["keywords_uk"] = _merge_keywords(
            target.get("keywords_uk", []) or [],
            func.get("keywords_uk", []) or [],
        )
        if func.get("parameters") and not target.get("parameters"):
            target["parameters"] = func.get("parameters", [])

    rel_keys = {
        (
            _normalize_name(r.get("from", "")),
            _normalize_name(r.get("to", "")),
            _normalize_name(r.get("via", "")),
        )
        for r in base_relationships
    }
    for rel in overlay.get("relationships", []) or []:
        key = (
            _normalize_name(rel.get("from", "")),
            _normalize_name(rel.get("to", "")),
            _normalize_name(rel.get("via", "")),
        )
        if key in rel_keys:
            continue
        rel_keys.add(key)
        base_relationships.append(rel)

    base["tables"] = base_tables
    base["functions"] = base_functions
    base["relationships"] = base_relationships
    return base


def _load_db_functions() -> list[dict[str, Any]]:
    functions: list[dict[str, Any]] = []
    query = """
        SELECT s.name AS schema_name, o.name AS object_name
        FROM sys.objects o
        JOIN sys.schemas s ON o.schema_id = s.schema_id
        WHERE o.type IN ('FN', 'TF', 'IF') AND o.is_ms_shipped = 0
        ORDER BY s.name, o.name;
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        for schema_name, object_name in cursor.fetchall():
            name = f"{schema_name}.{object_name}"
            functions.append({
                "name": name,
                "description": "",
                "keywords_uk": [],
                "parameters": [],
            })
    return functions


def _schema_from_db(join_rules_path: Path | None) -> dict[str, Any]:
    schema_cache = SchemaCache()
    schema_cache.load()

    tables: list[dict[str, Any]] = []
    for table in schema_cache.tables.values():
        columns = []
        for col in table.columns:
            columns.append({
                "name": col.name,
                "type": col.data_type,
                "description": "",
                "keywords_uk": [],
            })
        tables.append({
            "name": table.key,
            "description": "",
            "keywords_uk": [],
            "object_type": getattr(table, "object_type", "table"),
            "columns": columns,
        })

    relationships: list[dict[str, Any]] = []
    for fk in schema_cache.foreign_keys:
        relationships.append({
            "from": fk.fk_key,
            "to": fk.pk_key,
            "via": f"{fk.fk_key}.{fk.fk_column} = {fk.pk_key}.{fk.pk_column}",
            "description": "",
        })

    if join_rules_path and join_rules_path.exists():
        join_rules = load_join_rules(str(join_rules_path))
        for join in join_rules.joins:
            if not join.enabled:
                continue
            for column in join.columns:
                relationships.append({
                    "from": join.left,
                    "to": join.right,
                    "via": f"{join.left}.{column.left} = {join.right}.{column.right}",
                    "description": join.name,
                })

    functions = []
    try:
        functions = _load_db_functions()
    except Exception as exc:
        print(f"Warning: Could not load DB functions: {exc}")

    return {
        "tables": tables,
        "functions": functions,
        "relationships": relationships,
    }


def create_searchable_entries(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Create searchable entries from schema."""
    entries = []

    # Process tables and columns
    for table in schema.get("tables", []) or []:
        table_name = table.get("name", "")
        table_desc = table.get("description", "")
        table_keywords = table.get("keywords_uk", []) or []
        object_type = table.get("object_type", "table")

        # Table-level entry
        searchable = (
            f"Table {table_name} ({object_type}): {table_desc}. "
            f"Keywords: {', '.join(table_keywords)}"
        )
        entries.append({
            "type": "table",
            "name": table_name,
            "searchable_text": searchable,
            "description": table_desc,
            "keywords_uk": table_keywords,
            "data": json.dumps({"table": table_name, "object_type": object_type}),
        })

        # Column-level entries
        for col in table.get("columns", []) or []:
            col_name = col.get("name", "")
            col_type = col.get("type", "")
            col_desc = col.get("description", "")
            col_keywords = col.get("keywords_uk", []) or []

            searchable = (
                f"Column {table_name}.{col_name} ({col_type}): {col_desc}. "
                f"Keywords: {', '.join(col_keywords)}"
            )
            entries.append({
                "type": "column",
                "name": f"{table_name}.{col_name}",
                "searchable_text": searchable,
                "description": col_desc,
                "keywords_uk": col_keywords,
                "data": json.dumps({
                    "table": table_name,
                    "column": col_name,
                    "type": col_type,
                }),
            })

    # Process functions
    for func in schema.get("functions", []) or []:
        func_name = func.get("name", "")
        func_desc = func.get("description", "")
        func_keywords = func.get("keywords_uk", []) or []
        params = func.get("parameters", []) or []
        param_str = ", ".join(p.get("name", "") for p in params)

        searchable = (
            f"Function {func_name}({param_str}): {func_desc}. "
            f"Keywords: {', '.join(func_keywords)}"
        )
        entries.append({
            "type": "function",
            "name": func_name,
            "searchable_text": searchable,
            "description": func_desc,
            "keywords_uk": func_keywords,
            "data": json.dumps({"function": func_name, "params": params}),
        })

    # Process relationships
    for rel in schema.get("relationships", []) or []:
        rel_from = rel.get("from", "")
        rel_to = rel.get("to", "")
        rel_via = rel.get("via", "")
        rel_desc = rel.get("description", "")

        searchable = f"Relationship {rel_from} -> {rel_to}: {rel_desc}. Join: {rel_via}"
        entries.append({
            "type": "relationship",
            "name": f"{rel_from}->{rel_to}",
            "searchable_text": searchable,
            "description": rel_desc,
            "keywords_uk": [],
            "data": json.dumps({"from": rel_from, "to": rel_to, "via": rel_via}),
        })

    return entries


def create_database(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database for schema vectors."""
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE schema_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            name TEXT NOT NULL,
            searchable_text TEXT NOT NULL,
            description TEXT,
            keywords_uk TEXT,
            data TEXT,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("CREATE INDEX idx_type ON schema_vectors(type)")
    cursor.execute("CREATE INDEX idx_name ON schema_vectors(name)")

    cursor.execute("""
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()
    return conn


def build_vectors(
    model_name: str = DEFAULT_MODEL,
    output_db: Path = OUTPUT_DB,
    schema_yaml: Path = SCHEMA_YAML,
    join_rules: Path | None = None,
    use_db: bool = True,
    use_yaml: bool = True,
) -> None:
    """Build vector index for schema knowledge."""
    schema: dict[str, Any] = {"tables": [], "functions": [], "relationships": []}

    if use_db:
        print("Loading schema from database...")
        try:
            schema = _schema_from_db(join_rules)
            print(f"Loaded {len(schema.get('tables', []))} tables from DB")
        except Exception as exc:
            print(f"Warning: Could not load schema from DB: {exc}")
            if not use_yaml:
                raise

    if use_yaml and schema_yaml and schema_yaml.exists():
        print(f"Loading schema knowledge from {schema_yaml}...")
        yaml_schema = _load_yaml(schema_yaml)
        if schema.get("tables"):
            schema = _merge_schema(schema, yaml_schema)
        else:
            schema = yaml_schema

    print("Creating searchable entries...")
    entries = create_searchable_entries(schema)
    print(f"Created {len(entries)} entries")

    # Count by type
    from collections import Counter
    type_counts = Counter(e["type"] for e in entries)
    for entry_type, count in type_counts.items():
        print(f"  {entry_type}: {count}")

    print(f"\nLoading embedding model: {model_name}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}")

    print("\nGenerating embeddings...")
    texts = [e["searchable_text"] for e in entries]

    # For multilingual-e5, prefix with "query: "
    if "e5" in model_name.lower():
        texts = [f"query: {t}" for t in texts]

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print(f"\nSaving to {output_db}...")
    output_db.parent.mkdir(parents=True, exist_ok=True)
    conn = create_database(output_db)
    cursor = conn.cursor()

    for entry, embedding in zip(entries, embeddings):
        cursor.execute("""
            INSERT INTO schema_vectors
            (type, name, searchable_text, description, keywords_uk, data, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            entry["type"],
            entry["name"],
            entry["searchable_text"],
            entry["description"],
            json.dumps(entry["keywords_uk"]),
            entry["data"],
            embedding.astype(np.float32).tobytes(),
        ))

    # Store metadata
    cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                   ("model_name", model_name))
    cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                   ("embedding_dim", str(embedding_dim)))
    cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                   ("entry_count", str(len(entries))))

    conn.commit()
    conn.close()

    print("\nBuilt schema vector index:")
    print(f"  Entries: {len(entries)}")
    print(f"  Model: {model_name}")
    print(f"  Output: {output_db}")
    print(f"  Size: {output_db.stat().st_size / 1024:.1f} KB")


def main() -> None:
    settings = get_settings()
    default_output = Path(settings.schema_vectors_path) if settings.schema_vectors_path else OUTPUT_DB
    parser = argparse.ArgumentParser(
        description="Build vector embeddings for schema knowledge"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Embedding model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output SQLite path (default: {OUTPUT_DB})",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=SCHEMA_YAML,
        help=f"Schema knowledge YAML path (default: {SCHEMA_YAML})",
    )
    parser.add_argument(
        "--join-rules",
        type=Path,
        default=Path(settings.join_rules_path),
        help="Join rules path for relationship hints",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip DB schema introspection (use YAML only)",
    )
    parser.add_argument(
        "--no-yaml",
        action="store_true",
        help="Skip YAML merge (use DB only)",
    )
    args = parser.parse_args()

    build_vectors(
        model_name=args.model,
        output_db=args.output,
        schema_yaml=args.yaml,
        join_rules=args.join_rules,
        use_db=not args.no_db,
        use_yaml=not args.no_yaml,
    )


if __name__ == "__main__":
    main()
