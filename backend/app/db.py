from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable

import pyodbc

from .config import get_settings


def get_connection() -> pyodbc.Connection:
    settings = get_settings()
    return pyodbc.connect(settings.db_connection_string, timeout=10)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def fetch_rows(sql: str, params: Iterable[Any] | None = None, max_rows: int | None = None) -> tuple[list[str], list[list[Any]]]:
    settings = get_settings()
    limit = max_rows if max_rows is not None else settings.max_rows
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params or [])
        columns = [col[0] for col in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(limit) if limit else cursor.fetchall()
        normalized = [[_normalize_value(value) for value in row] for row in rows]
        return columns, normalized
