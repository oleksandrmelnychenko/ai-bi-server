"""Retrieval module for schema knowledge and curated queries.

This module provides:
- schema_knowledge: Loads actual column definitions from schema_knowledge.yaml
- curated_queries: Loads real SQL examples from domain catalogs (debt_catalog.yaml, etc.)

The approach is to inject actual schema documentation into the LLM prompt
so it understands what columns exist, rather than pattern matching against examples.
"""

from .schema_knowledge import (
    format_table_schema,
    format_tables_for_prompt,
    get_full_context_for_tables,
    get_function_docs,
    get_relationships,
    get_table_schema,
    get_tables_schema,
)

from .curated_queries import (
    format_query_as_example,
    get_available_domains,
    get_domain_keywords,
    get_domain_queries,
    get_few_shot_examples,
    get_relevant_examples,
    match_question_to_query,
)

__all__ = [
    # schema_knowledge.py - actual column definitions
    "format_table_schema",
    "format_tables_for_prompt",
    "get_full_context_for_tables",
    "get_function_docs",
    "get_relationships",
    "get_table_schema",
    "get_tables_schema",
    # curated_queries.py - real SQL examples
    "format_query_as_example",
    "get_available_domains",
    "get_domain_keywords",
    "get_domain_queries",
    "get_few_shot_examples",
    "get_relevant_examples",
    "match_question_to_query",
]
