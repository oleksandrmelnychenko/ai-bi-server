"""SQL example retrieval module.

Contains YAML-based example loading, keyword matching, and vector search.
"""

from .examples import (
    classify_question,
    extract_table_names,
    get_all_categories,
    get_category_info,
    get_relevant_examples,
    load_examples,
    load_extracted_examples,
    reload_examples,
)
from .schema_hints import (
    format_schema_hints,
    get_schema_hints,
    is_available as schema_vectors_available,
)
from .vector_search import (
    SQLExampleResult,
    get_relevant_examples as get_vector_examples,
    get_stats as get_vector_stats,
    search_similar,
)

__all__ = [
    # examples.py
    "classify_question",
    "extract_table_names",
    "get_all_categories",
    "get_category_info",
    "get_relevant_examples",
    "load_examples",
    "load_extracted_examples",
    "reload_examples",
    # schema_hints.py
    "format_schema_hints",
    "get_schema_hints",
    "schema_vectors_available",
    # vector_search.py
    "SQLExampleResult",
    "get_vector_examples",
    "get_vector_stats",
    "search_similar",
]
