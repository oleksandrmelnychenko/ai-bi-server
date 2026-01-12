from __future__ import annotations

import re
from typing import Tuple

BAD_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|TRUNCATE|EXEC|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def extract_sql(text: str) -> str:
    if not text:
        return ""
    fenced = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    fenced = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def is_safe_sql(sql: str) -> Tuple[bool, str]:
    candidate = sql.strip().rstrip(";")
    if not candidate:
        return False, "Empty SQL"
    if ";" in candidate:
        return False, "Multiple statements are not allowed"
    if BAD_KEYWORDS.search(candidate):
        return False, "Only SELECT queries are allowed"
    if not re.match(r"(?is)^\s*(WITH\b.*?SELECT\b|SELECT\b)", candidate):
        return False, "Only SELECT queries are allowed"
    return True, ""


def apply_row_limit(sql: str, max_rows: int) -> str:
    if re.search(r"(?is)\bTOP\s*\(\s*\d+\s*\)", sql):
        return sql
    if re.search(r"(?is)\bFETCH\s+NEXT\b", sql):
        return sql
    if re.search(r"(?is)^\s*WITH\b", sql):
        match = re.search(r"(?is)\bSELECT\s+(DISTINCT\s+)?", sql)
        if not match:
            return sql
        distinct = match.group(1) or ""
        replacement = f"SELECT {distinct}TOP ({max_rows}) "
        return re.sub(r"(?is)\bSELECT\s+(DISTINCT\s+)?", replacement, sql, count=1)
    match = re.match(r"(?is)^\s*SELECT\s+(DISTINCT\s+)?", sql)
    if not match:
        return sql
    distinct = match.group(1) or ""
    replacement = f"SELECT {distinct}TOP ({max_rows}) "
    return re.sub(r"(?is)^\s*SELECT\s+(DISTINCT\s+)?", replacement, sql, count=1)
