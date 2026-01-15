from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import logging
from typing import Any, Iterable

import pyodbc

from .config import get_settings

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Raised when a database operation fails."""
    pass


def get_connection() -> pyodbc.Connection:
    """Get a database connection."""
    settings = get_settings()
    try:
        return pyodbc.connect(settings.db_connection_string, timeout=10)
    except pyodbc.Error as e:
        logger.error(f"Failed to connect to database: {e}")
        raise DatabaseError(f"Failed to connect to database: {e}") from e


def _normalize_value(value: Any) -> Any:
    """Normalize database values for JSON serialization."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def fetch_rows(
    sql: str,
    params: Iterable[Any] | None = None,
    max_rows: int | None = None
) -> tuple[list[str], list[list[Any]]]:
    """
    Execute a SQL query and fetch results.

    Args:
        sql: The SQL query to execute
        params: Optional query parameters
        max_rows: Maximum number of rows to fetch

    Returns:
        Tuple of (column_names, rows)

    Raises:
        DatabaseError: If the query fails
    """
    settings = get_settings()
    limit = max_rows if max_rows is not None else settings.max_rows

    logger.debug(f"Executing SQL (limit={limit}): {sql[:200]}...")

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params or [])
            columns = [col[0] for col in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(limit) if limit else cursor.fetchall()
            normalized = [[_normalize_value(value) for value in row] for row in rows]

            logger.debug(f"Query returned {len(normalized)} rows")
            return columns, normalized

    except pyodbc.ProgrammingError as e:
        logger.error(f"SQL programming error: {e}")
        raise DatabaseError(f"Invalid SQL query: {e}") from e
    except pyodbc.DataError as e:
        logger.error(f"SQL data error: {e}")
        raise DatabaseError(f"Data error in query: {e}") from e
    except pyodbc.OperationalError as e:
        logger.error(f"Database operational error: {e}")
        raise DatabaseError(f"Database operation failed: {e}") from e
    except pyodbc.Error as e:
        logger.error(f"Database error: {e}")
        raise DatabaseError(f"Database error: {e}") from e
