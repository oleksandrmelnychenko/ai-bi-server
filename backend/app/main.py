from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
from pydantic import BaseModel

from .core import (
    DatabaseError,
    get_settings,
    ChatRequest,
    ChatResponse,
    DatabaseType,
    test_connection,
    test_all_connections,
    get_available_databases,
)
from .core.exceptions import LLMError
from .schema import (
    SchemaCache,
    JoinRules,
    build_adjacency,
    build_join_plan,
    edges_from_foreign_keys,
    load_join_rules,
)
from .core.db import fetch_rows
from .llm import compose_answer, generate_sql, select_tables
from .security import apply_row_limit, extract_sql, is_safe_sql

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Concord BI Chat", version="0.1.0")
settings = get_settings()

# Thread-safe schema state
_state_lock = threading.RLock()
schema_cache = SchemaCache()
join_rules = JoinRules()

# CORS configuration
cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --- Structured Error Response ---

class ErrorDetail(BaseModel):
    error_code: str
    message: str
    details: dict[str, Any] | None = None


def raise_error(status_code: int, error_code: str, message: str, details: dict | None = None):
    """Raise HTTPException with structured error detail."""
    raise HTTPException(
        status_code=status_code,
        detail=ErrorDetail(error_code=error_code, message=message, details=details).model_dump()
    )


# --- Input Sanitization ---

# Maximum message length to prevent abuse
MAX_MESSAGE_LENGTH = 2000

# Patterns that might indicate prompt injection attempts
SUSPICIOUS_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+instructions?",
    r"disregard\s+(previous|all|above)",
    r"forget\s+(everything|all)",
    r"new\s+instructions?:",
    r"system\s*:",
    r"assistant\s*:",
    r"```\s*(sql|python|bash|sh)\s*\n.*?(drop|delete|truncate|alter|insert|update|exec)",
]


def sanitize_user_input(message: str) -> str:
    """
    Sanitize user input to mitigate prompt injection attacks.

    This is defense-in-depth; the SQL guardrails are the primary protection.
    """
    if not message:
        return ""

    # Truncate to prevent token overflow attacks
    sanitized = message[:MAX_MESSAGE_LENGTH]

    # Remove code fences that might confuse the LLM
    sanitized = re.sub(r"```", "", sanitized)

    # Log if suspicious patterns detected (but still allow the message)
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, sanitized, re.IGNORECASE | re.DOTALL):
            logger.warning(f"Suspicious pattern detected in user input: {pattern}")
            break

    return sanitized.strip()


# --- Startup Events ---

@app.on_event("startup")
def _load_schema() -> None:
    global join_rules
    logger.info("Loading schema on startup...")
    with _state_lock:
        schema_cache.load()
        join_rules = load_join_rules(settings.join_rules_path)
    logger.info(f"Schema loaded: {len(schema_cache.tables)} tables, {len(join_rules.joins)} join rules")


# --- API Endpoints ---

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/schema/summary")
def schema_summary() -> dict[str, int | str]:
    with _state_lock:
        return {
            "tables": len(schema_cache.tables),
            "foreign_keys": len(schema_cache.foreign_keys),
            "join_rules": len(join_rules.joins),
            "loaded_at": schema_cache.loaded_at.isoformat() if schema_cache.loaded_at else "",
        }


@app.post("/api/schema/refresh")
def schema_refresh() -> dict[str, int | str]:
    global join_rules
    logger.info("Refreshing schema...")
    with _state_lock:
        schema_cache.load()
        join_rules = load_join_rules(settings.join_rules_path)
    logger.info(f"Schema refreshed: {len(schema_cache.tables)} tables")
    return schema_summary()


# --- Database Endpoints ---

@app.get("/api/databases")
def list_databases() -> dict[str, Any]:
    """List all configured databases and their status."""
    databases = get_available_databases()
    return {
        "databases": databases,
        "default": settings.default_database.value,
        "count": len(databases),
    }


@app.get("/api/databases/test")
def test_all_databases() -> dict[str, Any]:
    """Test connections to all configured databases."""
    results = test_all_connections()
    all_connected = all(r.get("connected", False) for r in results.values())
    return {
        "all_connected": all_connected,
        "results": results,
    }


@app.get("/api/databases/{db_type}/test")
def test_database(db_type: str) -> dict[str, Any]:
    """Test connection to a specific database."""
    try:
        db_enum = DatabaseType(db_type)
    except ValueError:
        raise_error(400, "invalid_database", f"Unknown database type: {db_type}")

    result = test_connection(db_enum)
    if not result["connected"]:
        raise_error(503, "connection_failed", f"Failed to connect to {db_type}", result)

    return result


# --- Query Endpoint (GET) ---

@app.get("/api/query", response_model=ChatResponse)
def query_llm(
    q: str = Query(..., min_length=1, max_length=2000, description="Question in Ukrainian or English"),
    db: str = Query(default="local", description="Database to query (local, identity)"),
    skip_answer: bool = Query(default=False, description="Skip LLM answer generation for faster response"),
) -> ChatResponse:
    """
    Query the database using natural language.

    - **q**: Your question (e.g., "Який борг клієнта Acme?")
    - **db**: Target database (default: local)
    - **skip_answer**: Skip answer generation (faster, returns data only)

    Returns SQL query, results, and natural language answer (unless skip_answer=true).
    """
    # Validate database type
    try:
        db_type = DatabaseType(db)
    except ValueError:
        raise_error(400, "invalid_database", f"Unknown database: {db}. Use: local, identity")

    # Sanitize input
    message = sanitize_user_input(q)
    if not message:
        raise_error(400, "empty_query", "Query parameter 'q' is required")

    logger.info(f"GET query request: {message[:100]}... (db={db})")

    # Ensure schema is loaded
    with _state_lock:
        if not schema_cache.tables:
            logger.info("Schema not loaded, loading now...")
            schema_cache.load()

    # Stage 1: Table Selection
    try:
        with _state_lock:
            table_keys = schema_cache.table_keys()
        selection = select_tables(message, table_keys, schema_cache=schema_cache)
    except LLMError as exc:
        logger.error(f"LLM error during table selection: {exc}")
        raise_error(502, "llm_error", f"LLM service error: {exc}")
    except httpx.HTTPError as exc:
        logger.error(f"HTTP error during table selection: {exc}")
        raise_error(502, "llm_connection_error", f"Failed to connect to LLM: {exc}")

    if selection.need_clarification:
        logger.info("Returning clarification request")
        return ChatResponse(answer=selection.clarifying_question, warnings=["clarification"])

    if not selection.tables:
        logger.warning("No tables selected for query")
        return ChatResponse(
            answer="Не вдалося визначити потрібні таблиці. Будь ласка, уточніть ваш запит.",
            warnings=["no_tables"],
        )

    # Build join plan
    with _state_lock:
        available_edges = join_rules.to_edges() if join_rules.joins else edges_from_foreign_keys(schema_cache.foreign_keys)
    adjacency = build_adjacency(available_edges)
    join_edges, missing = build_join_plan(adjacency, selection.tables)
    join_tables = _tables_from_join_edges(join_edges)
    detail_tables = _merge_table_keys(selection.tables, join_tables)
    table_details = _format_table_details(detail_tables)
    join_hints = _format_join_hints(join_edges)

    logger.info(f"Selected {len(selection.tables)} tables, {len(join_edges)} joins")

    # Stage 2: SQL Generation
    sql_model = settings.ollama_sql_model or None
    try:
        sql_raw = generate_sql(
            message,
            table_details,
            join_hints,
            settings.max_rows,
            selected_tables=detail_tables,
            model=sql_model,
        )
    except LLMError as exc:
        logger.error(f"LLM error during SQL generation: {exc}")
        raise_error(502, "llm_error", f"LLM service error: {exc}")
    except httpx.HTTPError as exc:
        logger.error(f"HTTP error during SQL generation: {exc}")
        raise_error(502, "llm_connection_error", f"Failed to connect to LLM: {exc}")

    # Extract and validate SQL
    sql = extract_sql(sql_raw)
    sql = apply_row_limit(sql, settings.max_rows)

    ok, reason = is_safe_sql(sql)
    if not ok:
        logger.warning(f"SQL validation failed: {reason}")
        raise_error(400, "invalid_sql", f"Generated SQL failed validation: {reason}")

    # Execute query on specified database
    try:
        columns, rows = fetch_rows(sql, max_rows=settings.max_rows, db_type=db_type)
        logger.info(f"Query returned {len(rows)} rows from {db}")
    except DatabaseError as exc:
        logger.error(f"Database error: {exc}")
        raise_error(500, "database_error", f"Database query failed: {exc}")
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        raise_error(500, "query_error", f"Query execution failed: {exc}")

    # Stage 3: Answer Composition (optional)
    if skip_answer:
        answer = f"Запит виконано. Отримано {len(rows)} рядків."
        logger.info("Skipping answer composition (skip_answer=true)")
    else:
        try:
            answer = compose_answer(message, sql, columns, rows)
        except LLMError as exc:
            logger.error(f"LLM error during answer composition: {exc}")
            raise_error(502, "llm_error", f"LLM service error: {exc}")
        except httpx.HTTPError as exc:
            logger.error(f"HTTP error during answer composition: {exc}")
            raise_error(502, "llm_connection_error", f"Failed to connect to LLM: {exc}")

    # Collect warnings
    warnings: list[str] = []
    if missing:
        warnings.append(f"missing_join_path: {', '.join(missing)}")
    if len(rows) == settings.max_rows:
        warnings.append(f"results_truncated: {settings.max_rows}")
    if db != "local":
        warnings.append(f"database: {db}")
    if skip_answer:
        warnings.append("answer_skipped")

    logger.info("GET query complete")

    return ChatResponse(answer=answer, sql=sql, columns=columns, rows=rows, warnings=warnings)


# --- Helper Functions ---

def _format_table_details(table_keys: list[str]) -> str:
    """Format table metadata for LLM context."""
    lines: list[str] = []
    with _state_lock:
        for key in table_keys:
            table = schema_cache.table_info(key)
            if not table:
                continue
            table_rule = join_rules.tables.get(key)
            role = table_rule.role if table_rule else "unknown"
            default_filters = ", ".join(table_rule.default_filters) if table_rule else ""
            filters = default_filters if default_filters else "none"
            columns = ", ".join(f"{col.name} ({col.data_type})" for col in table.columns)
            primary_key = ", ".join(table.primary_key) if table.primary_key else "none"
            object_type = getattr(table, "object_type", "table")
            lines.append(
                f"{table.key} ({object_type}): role: {role}; default_filters: {filters}; columns: {columns}; primary_key: {primary_key}"
            )
    return "\n".join(lines)


def _tables_from_join_edges(edges: list) -> list[str]:
    tables: list[str] = []
    for edge in edges:
        for name in (edge.left_table, edge.right_table):
            if name and name not in tables:
                tables.append(name)
    return tables


def _merge_table_keys(primary: list[str], additional: list[str]) -> list[str]:
    merged = list(primary)
    for name in additional:
        if name not in merged:
            merged.append(name)
    return merged


def _format_join_hints(edges: list) -> str:
    """Format join hints for LLM context."""
    if not edges:
        return "(no join hints available)"
    return "\n".join(
        " AND ".join(
            f"{edge.left_table}.{left} = {edge.right_table}.{right}" for left, right in edge.columns
        )
        for edge in edges
    )


# --- Main Chat Endpoint ---

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # Sanitize and validate input
    message = sanitize_user_input(request.message)
    if not message:
        raise_error(400, "empty_message", "Message is required")

    logger.info(f"Chat request: {message[:100]}...")

    # Ensure schema is loaded
    with _state_lock:
        if not schema_cache.tables:
            logger.info("Schema not loaded, loading now...")
            schema_cache.load()

    # Stage 1: Table Selection
    try:
        with _state_lock:
            table_keys = schema_cache.table_keys()
        selection = select_tables(message, table_keys, schema_cache=schema_cache)
    except LLMError as exc:
        logger.error(f"LLM error during table selection: {exc}")
        raise_error(502, "llm_error", f"LLM service error: {exc}")
    except httpx.HTTPError as exc:
        logger.error(f"HTTP error during table selection: {exc}")
        raise_error(502, "llm_connection_error", f"Failed to connect to LLM: {exc}")

    if selection.need_clarification:
        logger.info("Returning clarification request")
        return ChatResponse(answer=selection.clarifying_question, warnings=["clarification"])

    if not selection.tables:
        logger.warning("No tables selected for query")
        return ChatResponse(
            answer="Не вдалося визначити потрібні таблиці. Будь ласка, уточніть ваш запит.",
            warnings=["no_tables"],
        )

    # Build join plan
    with _state_lock:
        available_edges = join_rules.to_edges() if join_rules.joins else edges_from_foreign_keys(schema_cache.foreign_keys)
    adjacency = build_adjacency(available_edges)
    join_edges, missing = build_join_plan(adjacency, selection.tables)
    join_tables = _tables_from_join_edges(join_edges)
    detail_tables = _merge_table_keys(selection.tables, join_tables)
    table_details = _format_table_details(detail_tables)
    join_hints = _format_join_hints(join_edges)

    logger.info(f"Selected {len(selection.tables)} tables, {len(join_edges)} joins")

    # Stage 2: SQL Generation (use specialized SQL model if configured)
    sql_model = settings.ollama_sql_model or None  # Empty string becomes None (use default)
    if sql_model:
        logger.info(f"Using specialized SQL model: {sql_model}")
    try:
        sql_raw = generate_sql(
            message,
            table_details,
            join_hints,
            settings.max_rows,
            selected_tables=detail_tables,
            model=sql_model,
        )
    except LLMError as exc:
        logger.error(f"LLM error during SQL generation: {exc}")
        raise_error(502, "llm_error", f"LLM service error: {exc}")
    except httpx.HTTPError as exc:
        logger.error(f"HTTP error during SQL generation: {exc}")
        raise_error(502, "llm_connection_error", f"Failed to connect to LLM: {exc}")

    # Extract and validate SQL
    sql = extract_sql(sql_raw)
    sql = apply_row_limit(sql, settings.max_rows)

    logger.debug(f"Generated SQL: {sql[:200]}...")

    ok, reason = is_safe_sql(sql)
    if not ok:
        logger.warning(f"SQL validation failed: {reason}")
        raise_error(400, "invalid_sql", f"Generated SQL failed validation: {reason}")

    # Execute query
    try:
        columns, rows = fetch_rows(sql, max_rows=settings.max_rows)
        logger.info(f"Query returned {len(rows)} rows, {len(columns)} columns")
    except DatabaseError as exc:
        logger.error(f"Database error: {exc}")
        raise_error(500, "database_error", f"Database query failed: {exc}")
    except Exception as exc:
        logger.error(f"Unexpected error during query execution: {exc}")
        raise_error(500, "query_error", f"Query execution failed: {exc}")

    # Stage 3: Answer Composition (optional)
    if request.skip_answer:
        answer = f"Запит виконано. Отримано {len(rows)} рядків."
        logger.info("Skipping answer composition (skip_answer=true)")
    else:
        try:
            answer = compose_answer(message, sql, columns, rows)
        except LLMError as exc:
            logger.error(f"LLM error during answer composition: {exc}")
            raise_error(502, "llm_error", f"LLM service error: {exc}")
        except httpx.HTTPError as exc:
            logger.error(f"HTTP error during answer composition: {exc}")
            raise_error(502, "llm_connection_error", f"Failed to connect to LLM: {exc}")

    # Collect warnings
    warnings: list[str] = []
    if missing:
        warnings.append(f"missing_join_path: {', '.join(missing)}")
    if len(rows) == settings.max_rows:
        warnings.append(f"results_truncated: {settings.max_rows}")
    if request.skip_answer:
        warnings.append("answer_skipped")

    logger.info("Chat response complete")

    return ChatResponse(answer=answer, sql=sql, columns=columns, rows=rows, warnings=warnings)
