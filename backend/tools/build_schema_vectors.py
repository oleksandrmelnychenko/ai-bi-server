"""Build vector embeddings for schema knowledge base.

This enables semantic search to find relevant tables/columns for user questions.

Usage:
    cd backend
    python -m tools.build_schema_vectors
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import yaml

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
SCHEMA_YAML = Path(__file__).parent.parent / "schema" / "schema_knowledge.yaml"
OUTPUT_DB = Path(__file__).parent.parent / "schema" / "schema_vectors.sqlite"


def load_schema() -> dict:
    """Load schema knowledge from YAML."""
    with open(SCHEMA_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_searchable_entries(schema: dict) -> list[dict]:
    """Create searchable entries from schema."""
    entries = []

    # Process tables and columns
    for table in schema.get("tables", []):
        table_name = table["name"]
        table_desc = table.get("description", "")
        table_keywords = table.get("keywords_uk", [])

        # Table-level entry
        searchable = f"Table {table_name}: {table_desc}. Keywords: {', '.join(table_keywords)}"
        entries.append({
            "type": "table",
            "name": table_name,
            "searchable_text": searchable,
            "description": table_desc,
            "keywords_uk": table_keywords,
            "data": json.dumps({"table": table_name})
        })

        # Column-level entries
        for col in table.get("columns", []):
            col_name = col["name"]
            col_type = col.get("type", "")
            col_desc = col.get("description", "")
            col_keywords = col.get("keywords_uk", [])

            searchable = f"Column {table_name}.{col_name} ({col_type}): {col_desc}. Keywords: {', '.join(col_keywords)}"
            entries.append({
                "type": "column",
                "name": f"{table_name}.{col_name}",
                "searchable_text": searchable,
                "description": col_desc,
                "keywords_uk": col_keywords,
                "data": json.dumps({
                    "table": table_name,
                    "column": col_name,
                    "type": col_type
                })
            })

    # Process functions
    for func in schema.get("functions", []):
        func_name = func["name"]
        func_desc = func.get("description", "")
        func_keywords = func.get("keywords_uk", [])
        params = func.get("parameters", [])
        param_str = ", ".join(p["name"] for p in params)

        searchable = f"Function {func_name}({param_str}): {func_desc}. Keywords: {', '.join(func_keywords)}"
        entries.append({
            "type": "function",
            "name": func_name,
            "searchable_text": searchable,
            "description": func_desc,
            "keywords_uk": func_keywords,
            "data": json.dumps({"function": func_name, "params": params})
        })

    # Process relationships
    for rel in schema.get("relationships", []):
        rel_from = rel["from"]
        rel_to = rel["to"]
        rel_via = rel.get("via", "")
        rel_desc = rel.get("description", "")

        searchable = f"Relationship {rel_from} -> {rel_to}: {rel_desc}. Join: {rel_via}"
        entries.append({
            "type": "relationship",
            "name": f"{rel_from}->{rel_to}",
            "searchable_text": searchable,
            "description": rel_desc,
            "keywords_uk": [],
            "data": json.dumps({"from": rel_from, "to": rel_to, "via": rel_via})
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


def build_vectors(model_name: str = DEFAULT_MODEL):
    """Build vector index for schema knowledge."""
    print(f"Loading schema from {SCHEMA_YAML}...")
    schema = load_schema()

    print("Creating searchable entries...")
    entries = create_searchable_entries(schema)
    print(f"Created {len(entries)} entries")

    # Count by type
    from collections import Counter
    type_counts = Counter(e["type"] for e in entries)
    for t, c in type_counts.items():
        print(f"  {t}: {c}")

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

    print(f"\nSaving to {OUTPUT_DB}...")
    conn = create_database(OUTPUT_DB)
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
            embedding.astype(np.float32).tobytes()
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

    print(f"\nBuilt schema vector index:")
    print(f"  Entries: {len(entries)}")
    print(f"  Model: {model_name}")
    print(f"  Output: {OUTPUT_DB}")
    print(f"  Size: {OUTPUT_DB.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build_vectors()
