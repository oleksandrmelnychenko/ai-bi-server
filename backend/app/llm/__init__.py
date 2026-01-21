"""LLM interaction module.

Contains Ollama client, prompts, and specialized LLM operations:
- Table selection
- SQL generation
- Answer composition
"""

from .answer_composer import MAX_ROWS_FOR_LLM, compose_answer
from .client import LLMError, call_ollama, extract_json
from .prompts import ANSWER_SYSTEM, SQL_GENERATION_SQLCODER, SQL_GENERATION_SYSTEM, TABLE_SELECTION_SYSTEM
from .sql_generator import generate_sql
from .table_selector import SelectionResult, select_tables

__all__ = [
    "MAX_ROWS_FOR_LLM",
    "compose_answer",
    "LLMError",
    "call_ollama",
    "extract_json",
    "ANSWER_SYSTEM",
    "SQL_GENERATION_SQLCODER",
    "SQL_GENERATION_SYSTEM",
    "TABLE_SELECTION_SYSTEM",
    "generate_sql",
    "SelectionResult",
    "select_tables",
]
