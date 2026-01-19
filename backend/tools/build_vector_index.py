"""Build vector embeddings index for SQL examples.

This tool generates embeddings for extracted SQL queries to enable
semantic search for few-shot example retrieval.

Usage:
    cd backend
    pip install sentence-transformers numpy
    python -m tools.build_vector_index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# Model options (multilingual-e5-base recommended for Ukrainian)
DEFAULT_MODEL = "intfloat/multilingual-e5-base"
FALLBACK_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DEFAULT_INPUT_PATH = Path(__file__).parent.parent / "schema" / "sql_examples_extracted.yaml"
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent / "schema" / "sql_vectors.sqlite"

# SQL pattern keywords for description generation
SQL_PATTERNS = {
    "cte_pagination": "CTE with ROW_NUMBER pagination",
    "cte_aggregation": "CTE with aggregation",
    "cte_union": "CTE with UNION",
    "aggregation": "GROUP BY aggregation",
    "top_aggregation": "TOP N with aggregation",
    "complex_join": "Complex multi-table JOIN (5+ tables)",
    "multi_join": "Multiple table JOINs",
    "subquery": "Subquery pattern",
    "date_calculation": "Date calculations",
    "udf_call": "UDF function calls",
    "simple_select": "Basic SELECT query",
}


@dataclass
class SQLExample:
    """Represents an SQL example with metadata for embedding."""
    id: str
    sql: str
    source_file: str
    method_name: str
    tables: list[str]
    category: str
    searchable_text: str
    embedding: Optional[np.ndarray] = None


def generate_sql_id(sql: str, source: str) -> str:
    """Generate unique ID for SQL example."""
    content = f"{source}:{sql}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def extract_sql_keywords(sql: str) -> list[str]:
    """Extract key SQL patterns from query."""
    keywords = []
    sql_upper = sql.upper()

    if "WITH" in sql_upper and "AS" in sql_upper:
        keywords.append("CTE")
    if "ROW_NUMBER" in sql_upper:
        keywords.append("pagination")
    if "GROUP BY" in sql_upper:
        keywords.append("aggregation")
    if "SUM(" in sql_upper or "COUNT(" in sql_upper or "AVG(" in sql_upper:
        keywords.append("aggregate_function")
    if "LEFT JOIN" in sql_upper or "LEFT OUTER JOIN" in sql_upper:
        keywords.append("left_join")
    if "INNER JOIN" in sql_upper:
        keywords.append("inner_join")
    if "ORDER BY" in sql_upper:
        keywords.append("sorting")
    if "TOP" in sql_upper:
        keywords.append("top_n")
    if "LIKE" in sql_upper:
        keywords.append("text_search")
    if "BETWEEN" in sql_upper or "DATEDIFF" in sql_upper:
        keywords.append("date_filter")
    if "dbo.Get" in sql:
        keywords.append("udf_call")
    if "UNION" in sql_upper:
        keywords.append("union")
    if "EXISTS" in sql_upper or "IN (SELECT" in sql_upper:
        keywords.append("subquery")

    return keywords


def generate_searchable_text(example: dict, category: str) -> str:
    """Generate text to embed for semantic search.

    Creates a rich description combining:
    - Source context (file, method)
    - Tables involved
    - SQL pattern type
    - Key operations
    """
    source = example.get("source", "unknown")
    tables = example.get("tables", [])
    sql = example.get("sql", "")

    # Parse source into file and method
    if "::" in source:
        file_part, method_part = source.split("::", 1)
        file_name = file_part.split("/")[-1].split("\\")[-1]
    else:
        file_name = source
        method_part = "unknown"

    # Extract SQL patterns
    keywords = extract_sql_keywords(sql)

    # Build searchable text
    parts = []

    # Pattern description
    pattern_desc = SQL_PATTERNS.get(category, category)
    parts.append(f"Pattern: {pattern_desc}")

    # Tables
    if tables:
        # Filter out noise
        clean_tables = [t for t in tables if t.lower() not in ("and", "or", "on", "as")]
        if clean_tables:
            parts.append(f"Tables: {', '.join(clean_tables[:10])}")

    # Source context
    if "Repository" in file_name:
        entity = file_name.replace("Repository.cs", "").replace(".cs", "")
        parts.append(f"Entity: {entity}")

    # Method name (often descriptive)
    if method_part != "unknown" and method_part != "sql_file":
        # Convert camelCase to words
        method_words = re.sub(r"([a-z])([A-Z])", r"\1 \2", method_part)
        parts.append(f"Operation: {method_words}")

    # SQL patterns
    if keywords:
        parts.append(f"Features: {', '.join(keywords)}")

    return " | ".join(parts)


def load_examples(input_path: Path) -> list[SQLExample]:
    """Load SQL examples from YAML file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    examples = []
    seen_ids = set()

    for category, cat_data in data.get("categories", {}).items():
        for ex in cat_data.get("examples", []):
            source = ex.get("source", "unknown")
            sql = ex.get("sql", "")

            if not sql or len(sql) < 30:
                continue

            # Generate unique ID
            ex_id = generate_sql_id(sql, source)
            if ex_id in seen_ids:
                continue
            seen_ids.add(ex_id)

            # Parse source
            if "::" in source:
                file_part, method_part = source.split("::", 1)
            else:
                file_part = source
                method_part = "unknown"

            # Generate searchable text
            searchable_text = generate_searchable_text(ex, category)

            examples.append(SQLExample(
                id=ex_id,
                sql=sql,
                source_file=file_part,
                method_name=method_part,
                tables=ex.get("tables", []),
                category=category,
                searchable_text=searchable_text,
            ))

    return examples


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize numpy array to bytes for SQLite storage."""
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes, dim: int = 768) -> np.ndarray:
    """Deserialize bytes to numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


def create_database(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database with vector storage schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sql_vectors (
            id TEXT PRIMARY KEY,
            sql TEXT NOT NULL,
            source_file TEXT,
            method_name TEXT,
            tables TEXT,  -- JSON array
            category TEXT,
            searchable_text TEXT,
            embedding BLOB,
            embedding_dim INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_category ON sql_vectors(category)
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()
    return conn


def build_index(
    input_path: Path,
    output_path: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    show_progress: bool = True
) -> None:
    """Build vector index from SQL examples."""
    print(f"Loading examples from {input_path}...")
    examples = load_examples(input_path)
    print(f"Loaded {len(examples)} unique examples")

    print(f"\nLoading embedding model: {model_name}")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embedding_dim = model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {embedding_dim}")
    except Exception as e:
        print(f"Failed to load {model_name}, trying fallback...")
        model_name = FALLBACK_MODEL
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embedding_dim = model.get_sentence_embedding_dimension()
        print(f"Fallback model loaded. Embedding dimension: {embedding_dim}")

    # Generate embeddings in batches
    print(f"\nGenerating embeddings for {len(examples)} examples...")
    texts = [ex.searchable_text for ex in examples]

    # For multilingual-e5, prefix with "query: " for better results
    if "e5" in model_name.lower():
        texts = [f"query: {t}" for t in texts]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )

    for ex, emb in zip(examples, embeddings):
        ex.embedding = emb

    # Store in database
    print(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing database to rebuild
    if output_path.exists():
        output_path.unlink()

    conn = create_database(output_path)
    cursor = conn.cursor()

    # Insert examples
    for ex in examples:
        cursor.execute("""
            INSERT INTO sql_vectors
            (id, sql, source_file, method_name, tables, category, searchable_text, embedding, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ex.id,
            ex.sql,
            ex.source_file,
            ex.method_name,
            json.dumps(ex.tables),
            ex.category,
            ex.searchable_text,
            serialize_embedding(ex.embedding),
            embedding_dim
        ))

    # Store metadata
    cursor.execute("""
        INSERT OR REPLACE INTO metadata (key, value)
        VALUES ('model_name', ?), ('embedding_dim', ?), ('example_count', ?)
    """, (model_name, str(embedding_dim), str(len(examples))))

    conn.commit()
    conn.close()

    print(f"\nBuilt vector index with {len(examples)} examples")
    print(f"  Model: {model_name}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Output: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Print category distribution
    from collections import Counter
    cat_counts = Counter(ex.category for ex in examples)
    print("\nCategory distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build vector embeddings index for SQL examples"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input YAML file (default: {DEFAULT_INPUT_PATH})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output SQLite database (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Embedding model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)"
    )

    args = parser.parse_args()
    build_index(args.input, args.output, args.model, args.batch_size)


if __name__ == "__main__":
    main()
