"""Database connection and query execution with multi-database support."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
import logging
from typing import Any, Iterable, Optional, Generator

import pyodbc

from .config import get_settings, DatabaseType, DatabaseConnection
from .exceptions import DatabaseError

logger = logging.getLogger(__name__)

# Connection pool (simple caching)
_connection_pool: dict[DatabaseType, pyodbc.Connection] = {}


def get_connection(db_type: Optional[DatabaseType] = None) -> pyodbc.Connection:
    """Get a database connection.

    Args:
        db_type: Database type to connect to. If None, uses default database.

    Returns:
        pyodbc.Connection object

    Raises:
        DatabaseError: If connection fails or database type not configured
    """
    settings = get_settings()

    if db_type is None:
        db_type = settings.default_database

    conn_config = settings.connections.get(db_type)
    if not conn_config:
        raise DatabaseError(f"Database '{db_type.value}' is not configured")

    if not conn_config.enabled:
        raise DatabaseError(f"Database '{db_type.value}' is disabled")

    try:
        return pyodbc.connect(conn_config.connection_string, timeout=conn_config.timeout)
    except pyodbc.Error as e:
        logger.error(f"Failed to connect to database {db_type.value}: {e}")
        raise DatabaseError(f"Failed to connect to database {db_type.value}: {e}") from e


@contextmanager
def get_db_connection(db_type: Optional[DatabaseType] = None) -> Generator[pyodbc.Connection, None, None]:
    """Context manager for database connections.

    Args:
        db_type: Database type to connect to. If None, uses default database.

    Yields:
        pyodbc.Connection object

    Example:
        with get_db_connection(DatabaseType.LOCAL) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
    """
    conn = get_connection(db_type)
    try:
        yield conn
    finally:
        conn.close()


def _normalize_value(value: Any) -> Any:
    """Normalize database values for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        # Try to decode as UTF-8, otherwise return hex representation
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, memoryview):
        return bytes(value).hex()
    return value


def fetch_rows(
    sql: str,
    params: Iterable[Any] | None = None,
    max_rows: int | None = None,
    db_type: Optional[DatabaseType] = None
) -> tuple[list[str], list[list[Any]]]:
    """Execute a SQL query and fetch results.

    Args:
        sql: The SQL query to execute
        params: Optional query parameters
        max_rows: Maximum number of rows to fetch
        db_type: Database to query. If None, uses default database.

    Returns:
        Tuple of (column_names, rows)

    Raises:
        DatabaseError: If the query fails
    """
    settings = get_settings()
    limit = max_rows if max_rows is not None else settings.max_rows
    target_db = db_type or settings.default_database

    logger.debug(f"Executing SQL on {target_db.value} (limit={limit}): {sql[:200]}...")

    try:
        with get_db_connection(target_db) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params or [])
            columns = [col[0] for col in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(limit) if limit else cursor.fetchall()
            normalized = [[_normalize_value(value) for value in row] for row in rows]

            logger.debug(f"Query returned {len(normalized)} rows from {target_db.value}")
            return columns, normalized

    except pyodbc.ProgrammingError as e:
        logger.error(f"SQL programming error on {target_db.value}: {e}")
        raise DatabaseError(f"Invalid SQL query: {e}") from e
    except pyodbc.DataError as e:
        logger.error(f"SQL data error on {target_db.value}: {e}")
        raise DatabaseError(f"Data error in query: {e}") from e
    except pyodbc.OperationalError as e:
        logger.error(f"Database operational error on {target_db.value}: {e}")
        raise DatabaseError(f"Database operation failed: {e}") from e
    except pyodbc.Error as e:
        logger.error(f"Database error on {target_db.value}: {e}")
        raise DatabaseError(f"Database error: {e}") from e


def execute_non_query(
    sql: str,
    params: Iterable[Any] | None = None,
    db_type: Optional[DatabaseType] = None
) -> int:
    """Execute a SQL statement that doesn't return results (INSERT, UPDATE, DELETE).

    Args:
        sql: The SQL statement to execute
        params: Optional query parameters
        db_type: Database to execute on. If None, uses default database.

    Returns:
        Number of rows affected

    Raises:
        DatabaseError: If the statement fails
    """
    settings = get_settings()
    target_db = db_type or settings.default_database

    logger.debug(f"Executing non-query on {target_db.value}: {sql[:200]}...")

    try:
        with get_db_connection(target_db) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params or [])
            rowcount = cursor.rowcount
            conn.commit()

            logger.debug(f"Statement affected {rowcount} rows on {target_db.value}")
            return rowcount

    except pyodbc.Error as e:
        logger.error(f"Database error on {target_db.value}: {e}")
        raise DatabaseError(f"Database error: {e}") from e


def test_connection(db_type: Optional[DatabaseType] = None) -> dict[str, Any]:
    """Test database connection and return info.

    Args:
        db_type: Database to test. If None, uses default database.

    Returns:
        Dictionary with connection status and info
    """
    settings = get_settings()
    target_db = db_type or settings.default_database
    conn_config = settings.connections.get(target_db)

    result = {
        "database": target_db.value,
        "host": conn_config.host if conn_config else "N/A",
        "catalog": conn_config.database if conn_config else "N/A",
        "connected": False,
        "error": None,
        "server_version": None,
    }

    if not conn_config:
        result["error"] = "Database not configured"
        return result

    try:
        with get_db_connection(target_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT @@VERSION")
            row = cursor.fetchone()
            result["connected"] = True
            result["server_version"] = row[0] if row else "Unknown"
    except Exception as e:
        result["error"] = str(e)

    return result


def test_all_connections() -> dict[str, dict[str, Any]]:
    """Test all configured database connections.

    Returns:
        Dictionary mapping database type to connection test results
    """
    settings = get_settings()
    results = {}

    for db_type, conn in settings.connections.get_enabled():
        results[db_type.value] = test_connection(db_type)

    return results


def get_available_databases() -> list[dict[str, str]]:
    """Get list of available (configured and enabled) databases.

    Returns:
        List of dictionaries with database info
    """
    settings = get_settings()
    databases = []

    for db_type, conn in settings.connections.get_enabled():
        databases.append({
            "type": db_type.value,
            "name": conn.name,
            "host": conn.host,
            "database": conn.database,
            "is_default": db_type == settings.default_database,
        })

    return databases
