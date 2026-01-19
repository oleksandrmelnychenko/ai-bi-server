"""Vector-based SQL example retrieval using semantic search.

This module provides semantic search for SQL examples using embeddings,
replacing keyword-based matching with similarity-based retrieval.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Default paths
DEFAULT_VECTOR_DB = Path(__file__).parent.parent.parent / "schema" / "sql_vectors.sqlite"

# Cache for embeddings and model
_cache: dict = {
    "model": None,
    "model_name": None,
    "embeddings": None,  # numpy array of all embeddings
    "examples": None,    # list of example metadata
    "db_path": None
}


@dataclass
class SQLExampleResult:
    """Result from vector search."""
    id: str
    sql: str
    source_file: str
    method_name: str
    tables: list[str]
    category: str
    similarity: float


def _get_model(model_name: str = None):
    """Load or get cached embedding model."""
    if _cache["model"] is not None and (model_name is None or model_name == _cache["model_name"]):
        return _cache["model"]

    from sentence_transformers import SentenceTransformer

    if model_name is None:
        # Load from database metadata
        db_path = _cache.get("db_path") or DEFAULT_VECTOR_DB
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM metadata WHERE key = 'model_name'")
            row = cursor.fetchone()
            conn.close()
            if row:
                model_name = row[0]

    model_name = model_name or "intfloat/multilingual-e5-base"
    print(f"Loading embedding model: {model_name}")
    _cache["model"] = SentenceTransformer(model_name)
    _cache["model_name"] = model_name
    return _cache["model"]


def _load_embeddings(db_path: Path = None) -> tuple[np.ndarray, list[dict]]:
    """Load all embeddings from database into memory."""
    db_path = db_path or DEFAULT_VECTOR_DB

    if _cache["embeddings"] is not None and _cache["db_path"] == db_path:
        return _cache["embeddings"], _cache["examples"]

    if not db_path.exists():
        raise FileNotFoundError(f"Vector database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get embedding dimension
    cursor.execute("SELECT value FROM metadata WHERE key = 'embedding_dim'")
    row = cursor.fetchone()
    embedding_dim = int(row[0]) if row else 768

    # Load all examples
    cursor.execute("""
        SELECT id, sql, source_file, method_name, tables, category, embedding
        FROM sql_vectors
        WHERE embedding IS NOT NULL
    """)

    examples = []
    embeddings_list = []

    for row in cursor.fetchall():
        ex_id, sql, source_file, method_name, tables_json, category, embedding_blob = row
        tables = json.loads(tables_json) if tables_json else []
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)

        examples.append({
            "id": ex_id,
            "sql": sql,
            "source_file": source_file,
            "method_name": method_name,
            "tables": tables,
            "category": category
        })
        embeddings_list.append(embedding)

    conn.close()

    embeddings_array = np.vstack(embeddings_list)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / norms

    _cache["embeddings"] = embeddings_array
    _cache["examples"] = examples
    _cache["db_path"] = db_path

    return embeddings_array, examples


def cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all embeddings."""
    # Query is already normalized by model, embeddings pre-normalized
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    return np.dot(embeddings, query_norm)


def search_similar(
    question: str,
    top_k: int = 20,
    min_similarity: float = 0.3,
    required_tables: Optional[list[str]] = None,
    db_path: Path = None
) -> list[SQLExampleResult]:
    """Find SQL examples similar to the question using semantic search.

    Args:
        question: User's question in Ukrainian or English
        top_k: Maximum number of results to return
        min_similarity: Minimum similarity score threshold
        required_tables: Optional list of tables to boost results for
        db_path: Optional path to vector database

    Returns:
        List of SQLExampleResult sorted by relevance
    """
    db_path = db_path or DEFAULT_VECTOR_DB

    # Load model and embeddings
    model = _get_model()
    embeddings, examples = _load_embeddings(db_path)

    # Generate query embedding
    # For multilingual-e5, prefix with "query: " for better results
    if "e5" in _cache.get("model_name", "").lower():
        query_text = f"query: {question}"
    else:
        query_text = question

    query_embedding = model.encode(query_text, convert_to_numpy=True)

    # Compute similarities
    similarities = cosine_similarity(query_embedding, embeddings)

    # Boost by table overlap if required_tables provided
    if required_tables:
        required_set = set(t.lower() for t in required_tables)
        for i, ex in enumerate(examples):
            ex_tables = set(t.lower() for t in ex["tables"])
            overlap = len(required_set & ex_tables)
            if overlap > 0:
                # Boost similarity by table overlap (up to 0.2 bonus)
                boost = min(0.2, overlap * 0.05)
                similarities[i] += boost

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get extra for filtering

    results = []
    seen_sql_hashes = set()

    for idx in top_indices:
        if len(results) >= top_k:
            break

        sim = similarities[idx]
        if sim < min_similarity:
            continue

        ex = examples[idx]

        # Deduplicate by SQL content (first 200 chars)
        sql_hash = hash(ex["sql"][:200])
        if sql_hash in seen_sql_hashes:
            continue
        seen_sql_hashes.add(sql_hash)

        results.append(SQLExampleResult(
            id=ex["id"],
            sql=ex["sql"],
            source_file=ex["source_file"],
            method_name=ex["method_name"],
            tables=ex["tables"],
            category=ex["category"],
            similarity=float(sim)
        ))

    return results


def get_relevant_examples(
    question: str,
    tables: Optional[list[str]] = None,
    max_examples: int = 10,
    db_path: Path = None
) -> list[dict]:
    """Get relevant SQL examples for few-shot learning.

    This is the main API for integration with the LLM prompt builder.

    Args:
        question: User's question
        tables: List of tables the query will use (for boosting)
        max_examples: Maximum number of examples to return
        db_path: Optional path to vector database

    Returns:
        List of example dictionaries with 'sql', 'tables', 'category' keys
    """
    results = search_similar(
        question=question,
        top_k=max_examples,
        required_tables=tables,
        db_path=db_path
    )

    # Convert to dict format expected by prompts
    examples = []
    for r in results:
        examples.append({
            "sql": r.sql,
            "tables": r.tables,
            "category": r.category,
            "source": f"{r.source_file}::{r.method_name}",
            "similarity": r.similarity
        })

    return examples


def get_stats(db_path: Path = None) -> dict:
    """Get statistics about the vector index."""
    db_path = db_path or DEFAULT_VECTOR_DB

    if not db_path.exists():
        return {"error": "Vector database not found"}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Count examples
    cursor.execute("SELECT COUNT(*) FROM sql_vectors")
    count = cursor.fetchone()[0]

    # Get metadata
    cursor.execute("SELECT key, value FROM metadata")
    metadata = dict(cursor.fetchall())

    # Category distribution
    cursor.execute("SELECT category, COUNT(*) FROM sql_vectors GROUP BY category ORDER BY COUNT(*) DESC")
    categories = dict(cursor.fetchall())

    conn.close()

    return {
        "total_examples": count,
        "model": metadata.get("model_name", "unknown"),
        "embedding_dim": int(metadata.get("embedding_dim", 0)),
        "categories": categories,
        "db_size_mb": db_path.stat().st_size / 1024 / 1024
    }


def clear_cache() -> None:
    """Clear the in-memory cache."""
    _cache["model"] = None
    _cache["model_name"] = None
    _cache["embeddings"] = None
    _cache["examples"] = None
    _cache["db_path"] = None
