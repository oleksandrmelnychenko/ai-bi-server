"""Core infrastructure module.

Contains configuration, database connection, models, and exceptions.
"""

from .config import (
    Settings,
    get_settings,
    get_cached_settings,
    clear_settings_cache,
    DatabaseType,
    DatabaseConnection,
    ConnectionStrings,
)
from .db import (
    fetch_rows,
    get_connection,
    get_db_connection,
    execute_non_query,
    test_connection,
    test_all_connections,
    get_available_databases,
)
from .exceptions import DatabaseError, LLMError, ValidationError
from .models import ChatMessage, ChatRequest, ChatResponse

__all__ = [
    # Config
    "Settings",
    "get_settings",
    "get_cached_settings",
    "clear_settings_cache",
    "DatabaseType",
    "DatabaseConnection",
    "ConnectionStrings",
    # Database
    "DatabaseError",
    "fetch_rows",
    "get_connection",
    "get_db_connection",
    "execute_non_query",
    "test_connection",
    "test_all_connections",
    "get_available_databases",
    # Exceptions
    "LLMError",
    "ValidationError",
    # Models
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
]
