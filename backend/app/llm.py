from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

import httpx

from .config import get_settings
from .prompts import ANSWER_SYSTEM, SQL_GENERATION_SYSTEM, TABLE_SELECTION_SYSTEM


@dataclass
class SelectionResult:
    tables: list[str]
    need_clarification: bool
    clarifying_question: str


def _call_ollama(messages: list[dict[str, str]], temperature: float = 0.1, max_tokens: int = 1200) -> str:
    settings = get_settings()
    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = httpx.post(
        f"{settings.ollama_base_url}/v1/chat/completions",
        json=payload,
        timeout=settings.request_timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _extract_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass
    return {}


def select_tables(question: str, table_keys: list[str]) -> SelectionResult:
    table_list = "\n".join(f"- {name}" for name in table_keys)
    user_prompt = f"Question:\n{question}\n\nAvailable tables:\n{table_list}\n"
    content = _call_ollama(
        [
            {"role": "system", "content": TABLE_SELECTION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=800,
    )
    payload = _extract_json(content)
    tables = payload.get("tables") or []
    valid_tables = set(table_keys)
    filtered = [t for t in tables if t in valid_tables]
    need_clarification = bool(payload.get("need_clarification"))
    clarifying_question = payload.get("clarifying_question") or ""
    if need_clarification and not clarifying_question:
        clarifying_question = "Уточніть, будь ласка, який саме період або критерії ви маєте на увазі?"
    return SelectionResult(tables=filtered, need_clarification=need_clarification, clarifying_question=clarifying_question)


def generate_sql(question: str, table_details: str, join_hints: str, max_rows: int) -> str:
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Table details:\n{table_details}\n\n"
        f"Join hints:\n{join_hints}\n\n"
        f"Row limit: TOP ({max_rows})\n"
    )
    content = _call_ollama(
        [
            {"role": "system", "content": SQL_GENERATION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.05,
        max_tokens=1200,
    )
    return content.strip()


def compose_answer(question: str, sql: str, columns: list[str], rows: list[list[Any]]) -> str:
    sample = {"columns": columns, "rows": rows}
    user_prompt = (
        f"Question:\n{question}\n\nSQL:\n{sql}\n\nData:\n{json.dumps(sample, ensure_ascii=False)}"
    )
    content = _call_ollama(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=900,
    )
    return content.strip()
