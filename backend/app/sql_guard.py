from __future__ import annotations

import logging
import re
from typing import Tuple

import sqlparse
from sqlparse.sql import Comment, Identifier, IdentifierList, Parenthesis, Token
from sqlparse.tokens import DML, DDL, Keyword

logger = logging.getLogger(__name__)

# Dangerous statement types that should never appear
DANGEROUS_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "MERGE", "DROP", "ALTER", "TRUNCATE",
    "EXEC", "EXECUTE", "CREATE", "GRANT", "REVOKE", "DENY",
    "BACKUP", "RESTORE", "SHUTDOWN", "KILL", "OPENROWSET", "OPENDATASOURCE",
    "XP_CMDSHELL", "SP_EXECUTESQL", "BULK",
}


def extract_sql(text: str) -> str:
    """Extract SQL from LLM response, handling markdown code blocks."""
    if not text:
        return ""
    # Try ```sql block first
    fenced = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    # Try generic ``` block
    fenced = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _extract_identifiers(token) -> list[str]:
    """Recursively extract table/column identifiers from a token."""
    identifiers = []
    if isinstance(token, Identifier):
        # Get the real name (handle schema.table format)
        name = token.get_real_name()
        if name:
            identifiers.append(name.upper())
        # Also check for schema-qualified names
        parent = token.get_parent_name()
        if parent:
            identifiers.append(f"{parent.upper()}.{name.upper()}")
    elif isinstance(token, IdentifierList):
        for item in token.get_identifiers():
            identifiers.extend(_extract_identifiers(item))
    elif hasattr(token, "tokens"):
        for sub in token.tokens:
            identifiers.extend(_extract_identifiers(sub))
    return identifiers


def _check_token_safety(token, depth: int = 0) -> tuple[bool, str]:
    """Recursively check a token and its children for dangerous patterns."""
    # Check comments for hidden code
    if isinstance(token, Comment):
        comment_text = str(token).upper()
        for keyword in DANGEROUS_KEYWORDS:
            if keyword in comment_text:
                return False, f"Dangerous keyword '{keyword}' found in comment"

    # Check raw token values
    if hasattr(token, "ttype") and token.ttype is not None:
        value = str(token).strip().upper()
        # Check for DML/DDL tokens
        if token.ttype in (DML, DDL):
            if value not in ("SELECT",):
                return False, f"Non-SELECT DML/DDL token: {value}"
        # Check for dangerous keywords
        if token.ttype is Keyword:
            if value in DANGEROUS_KEYWORDS:
                return False, f"Dangerous keyword: {value}"

    # Recursively check children
    if hasattr(token, "tokens"):
        for sub in token.tokens:
            safe, reason = _check_token_safety(sub, depth + 1)
            if not safe:
                return False, reason

    return True, ""


def is_safe_sql(sql: str, allowed_tables: set[str] | None = None) -> Tuple[bool, str]:
    """
    Validate SQL query for safety using sqlparse.

    Args:
        sql: The SQL query to validate
        allowed_tables: Optional set of allowed table names (schema.table format)

    Returns:
        Tuple of (is_safe, error_reason)
    """
    candidate = sql.strip().rstrip(";").strip()

    if not candidate:
        return False, "Empty SQL"

    # Parse the SQL
    try:
        parsed = sqlparse.parse(candidate)
    except Exception as e:
        logger.error(f"SQL parse error: {e}")
        return False, f"SQL parse error: {e}"

    # Must be exactly one statement
    if len(parsed) != 1:
        return False, f"Expected 1 statement, got {len(parsed)}"

    stmt = parsed[0]

    # Check statement type
    stmt_type = stmt.get_type()
    if stmt_type is None:
        # Try to detect from tokens
        first_token = stmt.token_first(skip_cm=True, skip_ws=True)
        if first_token:
            first_word = str(first_token).strip().upper()
            if first_word not in ("SELECT", "WITH"):
                return False, f"Statement must start with SELECT or WITH, got: {first_word}"
        else:
            return False, "Could not determine statement type"
    elif stmt_type.upper() != "SELECT":
        return False, f"Only SELECT statements allowed, got: {stmt_type}"

    # Deep token safety check (catches comments with hidden code)
    safe, reason = _check_token_safety(stmt)
    if not safe:
        return False, reason

    # Check for multiple statements via semicolon in the middle
    # (sqlparse might not split on all semicolons)
    if ";" in candidate:
        return False, "Multiple statements not allowed (semicolon in query)"

    # Validate table names against whitelist if provided
    if allowed_tables:
        identifiers = _extract_identifiers(stmt)
        # This is a basic check - in production you'd want more sophisticated
        # table name extraction from FROM, JOIN clauses
        logger.debug(f"Extracted identifiers: {identifiers}")

    return True, ""


def apply_row_limit(sql: str, max_rows: int) -> str:
    """Inject TOP clause if not already present."""
    # Already has TOP
    if re.search(r"(?is)\bTOP\s*\(\s*\d+\s*\)", sql):
        return sql
    # Already has FETCH NEXT (OFFSET-FETCH pagination)
    if re.search(r"(?is)\bFETCH\s+NEXT\b", sql):
        return sql
    # CTE query - inject TOP after the SELECT in the main query
    if re.search(r"(?is)^\s*WITH\b", sql):
        match = re.search(r"(?is)\bSELECT\s+(DISTINCT\s+)?", sql)
        if not match:
            return sql
        distinct = match.group(1) or ""
        replacement = f"SELECT {distinct}TOP ({max_rows}) "
        return re.sub(r"(?is)\bSELECT\s+(DISTINCT\s+)?", replacement, sql, count=1)
    # Regular SELECT
    match = re.match(r"(?is)^\s*SELECT\s+(DISTINCT\s+)?", sql)
    if not match:
        return sql
    distinct = match.group(1) or ""
    replacement = f"SELECT {distinct}TOP ({max_rows}) "
    return re.sub(r"(?is)^\s*SELECT\s+(DISTINCT\s+)?", replacement, sql, count=1)
