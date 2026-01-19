"""Base dataclasses for domain queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExtractedParams:
    """Parameters extracted from user question."""

    exchange_rate: Optional[float] = None
    client_name: Optional[str] = None
    date: Optional[str] = None  # e.g., "вчора", "2024-01-15"
    currency: Optional[str] = None  # e.g., "EUR", "USD"


@dataclass
class ClientQuery:
    """A client query template."""

    id: str
    context: str  # debt, cash_flow, purchases, etc.
    name_uk: str
    sql: str
    tables: list[str]
    notes: str
