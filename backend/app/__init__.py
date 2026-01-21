"""Concord Insight BI Chat Assistant.

This package provides a local BI chat assistant that converts Ukrainian-language
questions into SQL Server queries using Ollama (Qwen model).

Package Structure:
    core/       - Core infrastructure (config, db, models, exceptions)
    schema/     - Database schema management (cache, join graph, join rules)
    llm/        - LLM interaction (client, prompts, table selection, SQL generation)
    retrieval/  - SQL example retrieval (YAML-based, vector search)
    security/   - SQL validation and guardrails
"""

# Backward compatibility exports
# These allow existing code to import from app.* directly

# Core
from .core.config import (
    Settings,
    get_settings,
    get_cached_settings,
    DatabaseType,
    DatabaseConnection,
    ConnectionStrings,
)
from .core.db import (
    fetch_rows,
    get_connection,
    get_db_connection,
    test_connection,
    test_all_connections,
    get_available_databases,
)
from .core.exceptions import DatabaseError, LLMError, ValidationError
from .core.models import ChatMessage, ChatRequest, ChatResponse

# Schema
from .schema.cache import ColumnInfo, ForeignKeyInfo, SchemaCache, TableInfo
from .schema.join_graph import (
    INFINITY,
    JoinEdge,
    build_adjacency,
    build_join_plan,
    edges_from_foreign_keys,
)
from .schema.join_rules import (
    JoinColumnRule,
    JoinRule,
    JoinRules,
    TableRule,
    load_join_rules,
)

# LLM
from .llm.answer_composer import MAX_ROWS_FOR_LLM, compose_answer
from .llm.client import call_ollama, extract_json
from .llm.prompts import (
    ANSWER_SYSTEM,
    SQL_GENERATION_SQLCODER,
    SQL_GENERATION_SYSTEM,
    TABLE_SELECTION_SYSTEM,
)
from .llm.sql_generator import generate_sql
from .llm.table_selector import SelectionResult, select_tables

# Security
from .security.sql_guard import apply_row_limit, extract_sql, is_safe_sql

__all__ = [
    # Core - Config
    "Settings",
    "get_settings",
    "get_cached_settings",
    "DatabaseType",
    "DatabaseConnection",
    "ConnectionStrings",
    # Core - Database
    "DatabaseError",
    "fetch_rows",
    "get_connection",
    "get_db_connection",
    "test_connection",
    "test_all_connections",
    "get_available_databases",
    # Core - Exceptions & Models
    "LLMError",
    "ValidationError",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    # Schema
    "ColumnInfo",
    "ForeignKeyInfo",
    "SchemaCache",
    "TableInfo",
    "INFINITY",
    "JoinEdge",
    "build_adjacency",
    "build_join_plan",
    "edges_from_foreign_keys",
    "JoinColumnRule",
    "JoinRule",
    "JoinRules",
    "TableRule",
    "load_join_rules",
    # LLM
    "MAX_ROWS_FOR_LLM",
    "compose_answer",
    "call_ollama",
    "extract_json",
    "ANSWER_SYSTEM",
    "SQL_GENERATION_SQLCODER",
    "SQL_GENERATION_SYSTEM",
    "TABLE_SELECTION_SYSTEM",
    "generate_sql",
    "SelectionResult",
    "select_tables",
    # Security
    "apply_row_limit",
    "extract_sql",
    "is_safe_sql",
]
