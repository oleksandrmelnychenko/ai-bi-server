"""Custom exceptions for the application."""

from __future__ import annotations


class DatabaseError(Exception):
    """Raised when a database operation fails."""

    pass


class LLMError(Exception):
    """Raised when LLM returns unexpected response format or fails."""

    pass


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass
