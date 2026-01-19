"""Vector-based schema knowledge retrieval using semantic search.

This module provides semantic search for schema elements (tables, columns,
functions, relationships) using embeddings to help LLM understand the database.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_SCHEMA_VECTOR_DB = Path(__file__).parent.parent.parent / "schema" / "schema_vectors.sqlite"

# Cache for embeddings and model
_cache: dict = {
    "model": None,
    "model_name": None,
    "embeddings": None,
    "entries": None,
    "db_path": None
}


@dataclass
class SchemaEntry:
    """Result from schema vector search."""
    id: int
    type: str           # table, column, function, relationship
    name: str
    description: str
    keywords_uk: list[str]
    data: dict          # Additional structured data
    similarity: float


def _get_model(model_name: str = None):
    """Load or get cached embedding model."""
    if _cache["model"] is not None and (model_name is None or model_name == _cache["model_name"]):
        return _cache["model"]

    from sentence_transformers import SentenceTransformer

    if model_name is None:
        # Load from database metadata
        db_path = _cache.get("db_path") or DEFAULT_SCHEMA_VECTOR_DB
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM metadata WHERE key = 'model_name'")
            row = cursor.fetchone()
            conn.close()
            if row:
                model_name = row[0]

    model_name = model_name or "intfloat/multilingual-e5-base"
    logger.info(f"Loading schema embedding model: {model_name}")
    _cache["model"] = SentenceTransformer(model_name)
    _cache["model_name"] = model_name
    return _cache["model"]


def _load_embeddings(db_path: Path = None) -> tuple[np.ndarray, list[dict]]:
    """Load all embeddings from database into memory."""
    db_path = db_path or DEFAULT_SCHEMA_VECTOR_DB

    if _cache["embeddings"] is not None and _cache["db_path"] == db_path:
        return _cache["embeddings"], _cache["entries"]

    if not db_path.exists():
        raise FileNotFoundError(f"Schema vector database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get embedding dimension
    cursor.execute("SELECT value FROM metadata WHERE key = 'embedding_dim'")
    row = cursor.fetchone()
    embedding_dim = int(row[0]) if row else 768

    # Load all entries
    cursor.execute("""
        SELECT id, type, name, description, keywords_uk, data, embedding
        FROM schema_vectors
        WHERE embedding IS NOT NULL
    """)

    entries = []
    embeddings_list = []

    for row in cursor.fetchall():
        entry_id, entry_type, name, description, keywords_json, data_json, embedding_blob = row
        keywords = json.loads(keywords_json) if keywords_json else []
        data = json.loads(data_json) if data_json else {}
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)

        entries.append({
            "id": entry_id,
            "type": entry_type,
            "name": name,
            "description": description,
            "keywords_uk": keywords,
            "data": data
        })
        embeddings_list.append(embedding)

    conn.close()

    embeddings_array = np.vstack(embeddings_list)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / norms

    _cache["embeddings"] = embeddings_array
    _cache["entries"] = entries
    _cache["db_path"] = db_path

    logger.info(f"Loaded {len(entries)} schema entries from {db_path}")

    return embeddings_array, entries


def cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all embeddings."""
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    return np.dot(embeddings, query_norm)


def search_schema(
    question: str,
    top_k: int = 10,
    min_similarity: float = 0.3,
    entry_types: Optional[list[str]] = None,
    db_path: Path = None
) -> list[SchemaEntry]:
    """Find schema elements similar to the question using semantic search.

    Args:
        question: User's question in Ukrainian or English
        top_k: Maximum number of results to return
        min_similarity: Minimum similarity score threshold
        entry_types: Optional filter for entry types (table, column, function, relationship)
        db_path: Optional path to vector database

    Returns:
        List of SchemaEntry sorted by relevance
    """
    db_path = db_path or DEFAULT_SCHEMA_VECTOR_DB

    # Load model and embeddings
    model = _get_model()
    embeddings, entries = _load_embeddings(db_path)

    # Generate query embedding
    # For multilingual-e5, prefix with "query: " for better results
    if "e5" in _cache.get("model_name", "").lower():
        query_text = f"query: {question}"
    else:
        query_text = question

    query_embedding = model.encode(query_text, convert_to_numpy=True)

    # Compute similarities
    similarities = cosine_similarity(query_embedding, embeddings)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k * 3]  # Get extra for filtering

    results = []

    for idx in top_indices:
        if len(results) >= top_k:
            break

        sim = similarities[idx]
        if sim < min_similarity:
            continue

        entry = entries[idx]

        # Filter by type if specified
        if entry_types and entry["type"] not in entry_types:
            continue

        results.append(SchemaEntry(
            id=entry["id"],
            type=entry["type"],
            name=entry["name"],
            description=entry["description"],
            keywords_uk=entry["keywords_uk"],
            data=entry["data"],
            similarity=float(sim)
        ))

    return results


def get_schema_hints(
    question: str,
    max_tables: int = 5,
    max_columns: int = 8,
    max_functions: int = 3,
    db_path: Path = None
) -> dict:
    """Get relevant schema hints for LLM prompt.

    This is the main API for integration with the LLM prompt builder.

    Args:
        question: User's question
        max_tables: Maximum table hints to return
        max_columns: Maximum column hints to return
        max_functions: Maximum function hints to return
        db_path: Optional path to vector database

    Returns:
        Dictionary with 'tables', 'columns', 'functions', 'relationships' keys
    """
    db_path = db_path or DEFAULT_SCHEMA_VECTOR_DB

    if not db_path.exists():
        logger.warning(f"Schema vector database not found: {db_path}")
        return {"tables": [], "columns": [], "functions": [], "relationships": []}

    try:
        # Search for each type separately to ensure balanced results
        tables = search_schema(question, top_k=max_tables, entry_types=["table"], db_path=db_path)
        columns = search_schema(question, top_k=max_columns, entry_types=["column"], db_path=db_path)
        functions = search_schema(question, top_k=max_functions, entry_types=["function"], db_path=db_path)
        relationships = search_schema(question, top_k=3, entry_types=["relationship"], db_path=db_path)

        return {
            "tables": [{"name": t.name, "description": t.description, "similarity": t.similarity} for t in tables],
            "columns": [{"name": c.name, "description": c.description, "table": c.data.get("table", ""), "type": c.data.get("type", ""), "similarity": c.similarity} for c in columns],
            "functions": [{"name": f.name, "description": f.description, "params": f.data.get("params", []), "similarity": f.similarity} for f in functions],
            "relationships": [{"from": r.data.get("from", ""), "to": r.data.get("to", ""), "via": r.data.get("via", ""), "description": r.description} for r in relationships]
        }
    except Exception as e:
        logger.error(f"Schema vector search failed: {e}")
        return {"tables": [], "columns": [], "functions": [], "relationships": []}


def format_schema_hints(hints: dict) -> str:
    """Format schema hints as a prompt section.

    Args:
        hints: Dictionary from get_schema_hints()

    Returns:
        Formatted string for LLM prompt
    """
    if not any(hints.values()):
        return ""

    lines = ["-- Relevant Schema Elements:\n"]

    # Tables
    if hints["tables"]:
        lines.append("-- Tables:")
        for t in hints["tables"]:
            lines.append(f"--   {t['name']}: {t['description']}")
        lines.append("")

    # Columns
    if hints["columns"]:
        lines.append("-- Important Columns:")
        for c in hints["columns"]:
            col_info = f"{c['table']}.{c['name']}" if c['table'] else c['name']
            lines.append(f"--   {col_info} ({c['type']}): {c['description']}")
        lines.append("")

    # Functions
    if hints["functions"]:
        lines.append("-- UDF Functions:")
        for f in hints["functions"]:
            params = ", ".join(p.get("name", "") for p in f["params"]) if f["params"] else ""
            lines.append(f"--   {f['name']}({params}): {f['description']}")
        lines.append("")

    # Relationships (join hints)
    if hints["relationships"]:
        lines.append("-- Key Relationships:")
        for r in hints["relationships"]:
            lines.append(f"--   {r['from']} -> {r['to']}: {r['via']}")
        lines.append("")

    return "\n".join(lines)


def is_available() -> bool:
    """Check if schema vector search is available."""
    return DEFAULT_SCHEMA_VECTOR_DB.exists()


def get_stats(db_path: Path = None) -> dict:
    """Get statistics about the schema vector index."""
    db_path = db_path or DEFAULT_SCHEMA_VECTOR_DB

    if not db_path.exists():
        return {"error": "Schema vector database not found"}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Count entries
    cursor.execute("SELECT COUNT(*) FROM schema_vectors")
    count = cursor.fetchone()[0]

    # Get metadata
    cursor.execute("SELECT key, value FROM metadata")
    metadata = dict(cursor.fetchall())

    # Type distribution
    cursor.execute("SELECT type, COUNT(*) FROM schema_vectors GROUP BY type ORDER BY COUNT(*) DESC")
    types = dict(cursor.fetchall())

    conn.close()

    return {
        "total_entries": count,
        "model": metadata.get("model_name", "unknown"),
        "embedding_dim": int(metadata.get("embedding_dim", 0)),
        "types": types,
        "db_size_kb": db_path.stat().st_size / 1024
    }


def clear_cache() -> None:
    """Clear the in-memory cache."""
    _cache["model"] = None
    _cache["model_name"] = None
    _cache["embeddings"] = None
    _cache["entries"] = None
    _cache["db_path"] = None
