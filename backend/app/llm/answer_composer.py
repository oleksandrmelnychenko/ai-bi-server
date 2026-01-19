"""Answer composition using LLM."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from .client import call_ollama
from .prompts import ANSWER_SYSTEM

logger = logging.getLogger(__name__)


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles database types safely."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode('utf-8')
            except UnicodeDecodeError:
                return obj.hex()
        if isinstance(obj, memoryview):
            return bytes(obj).hex()
        # Fallback for any other unknown type
        try:
            return str(obj)
        except Exception:
            return repr(obj)

# Maximum rows to send to LLM for answer composition
MAX_ROWS_FOR_LLM = 50


def compose_answer(
    question: str,
    sql: str,
    columns: list[str],
    rows: list[list[Any]],
    max_rows_for_llm: int = MAX_ROWS_FOR_LLM
) -> str:
    """Compose natural language answer from query results."""
    # Truncate rows to avoid token overflow
    truncated = len(rows) > max_rows_for_llm
    sample_rows = rows[:max_rows_for_llm]

    if truncated:
        logger.info(f"Truncating results from {len(rows)} to {max_rows_for_llm} rows for LLM")

    sample = {"columns": columns, "rows": sample_rows}

    # Add truncation note to prompt if needed
    truncation_note = ""
    if truncated:
        truncation_note = f"\n\nNote: Showing first {max_rows_for_llm} of {len(rows)} total rows."

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"SQL:\n{sql}\n\n"
        f"Data:\n{json.dumps(sample, ensure_ascii=False, cls=SafeJSONEncoder)}"
        f"{truncation_note}"
    )

    logger.info(f"Composing answer for {len(sample_rows)} rows, {len(columns)} columns")

    content = call_ollama(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=900,
    )

    answer = content.strip()
    logger.info(f"Composed answer length: {len(answer)} chars")

    return answer
