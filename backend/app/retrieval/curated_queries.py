"""Curated queries loader - provides real SQL examples from the codebase.

These are actual working SQL queries extracted from the GBA repository,
organized by domain (debt, sales, etc.) with Ukrainian question patterns.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Cache for loaded catalogs
_CATALOGS: dict[str, dict] = {}


def _get_catalog_path(domain: str) -> Path:
    """Get path to a domain catalog YAML file."""
    return Path(__file__).parent.parent.parent / "schema" / f"{domain}_catalog.yaml"


def _load_catalog(domain: str) -> dict:
    """Load a domain catalog from YAML."""
    if domain in _CATALOGS:
        return _CATALOGS[domain]

    catalog_path = _get_catalog_path(domain)
    if not catalog_path.exists():
        logger.debug(f"Catalog not found: {catalog_path}")
        return {}

    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = yaml.safe_load(f)
        _CATALOGS[domain] = catalog
        logger.info(f"Loaded {domain} catalog: {len(catalog.get('queries', {}))} queries")
        return catalog
    except Exception as e:
        logger.error(f"Failed to load catalog {domain}: {e}")
        return {}


def get_available_domains() -> list[str]:
    """Get list of available domain catalogs."""
    schema_dir = Path(__file__).parent.parent.parent / "schema"
    catalogs = list(schema_dir.glob("*_catalog.yaml"))
    return [p.stem.replace("_catalog", "") for p in catalogs]


def get_domain_queries(domain: str) -> dict:
    """Get all queries for a domain.

    Args:
        domain: Domain name (e.g., 'debt', 'sales')

    Returns:
        Dict of query_name -> query_info
    """
    catalog = _load_catalog(domain)
    return catalog.get("queries", {})


def get_domain_keywords(domain: str) -> dict:
    """Get keyword patterns for question classification.

    Args:
        domain: Domain name

    Returns:
        Dict of keyword categories
    """
    catalog = _load_catalog(domain)
    return catalog.get("keywords", {})


def match_question_to_query(question: str, domain: str) -> dict | None:
    """Try to match a Ukrainian question to a curated query.

    Uses keyword matching to find the most relevant query pattern.

    Args:
        question: User's question in Ukrainian
        domain: Domain to search in

    Returns:
        Query info dict if matched, None otherwise
    """
    catalog = _load_catalog(domain)
    queries = catalog.get("queries", {})
    keywords = catalog.get("keywords", {})

    if not queries:
        return None

    question_lower = question.lower()

    # Score each query based on keyword matches
    best_match = None
    best_score = 0

    for query_name, query_info in queries.items():
        score = 0

        # Check question patterns
        for pattern in query_info.get("questions", []):
            # Remove parameter placeholders like {name}
            clean_pattern = pattern.lower()
            for word in clean_pattern.split():
                if not word.startswith("{"):
                    if word in question_lower:
                        score += 2

        # Check domain keywords
        for keyword_cat, keyword_list in keywords.items():
            if isinstance(keyword_list, list):
                for kw in keyword_list:
                    if kw.lower() in question_lower:
                        score += 1
            elif isinstance(keyword_list, dict):
                for sub_list in keyword_list.values():
                    if isinstance(sub_list, list):
                        for kw in sub_list:
                            if kw.lower() in question_lower:
                                score += 1

        if score > best_score:
            best_score = score
            best_match = query_info

    # Require minimum score to match
    if best_score >= 3:
        return best_match

    return None


def format_query_as_example(query_info: dict) -> str:
    """Format a curated query as a few-shot example.

    Args:
        query_info: Query dict from catalog

    Returns:
        Formatted example for prompt
    """
    lines = []

    name = query_info.get("name", "")
    questions = query_info.get("questions", [])
    sql = query_info.get("sql", "")
    notes = query_info.get("notes", [])

    if name:
        lines.append(f"-- Example: {name}")

    if questions:
        # Show first 2 question patterns
        for q in questions[:2]:
            lines.append(f"-- Question: {q}")

    if sql:
        lines.append(sql.strip())

    if notes:
        for note in notes[:2]:
            lines.append(f"-- Note: {note}")

    return "\n".join(lines)


def get_few_shot_examples(domain: str, max_examples: int = 3) -> str:
    """Get few-shot examples from a domain catalog.

    Args:
        domain: Domain name
        max_examples: Maximum number of examples to include

    Returns:
        Formatted few-shot examples string
    """
    queries = get_domain_queries(domain)

    if not queries:
        return ""

    lines = []
    lines.append("=" * 60)
    lines.append(f"FEW-SHOT EXAMPLES ({domain.upper()} domain):")
    lines.append("=" * 60)

    for i, (name, query_info) in enumerate(queries.items()):
        if i >= max_examples:
            break

        lines.append("")
        lines.append(format_query_as_example(query_info))

    return "\n".join(lines)


def get_relevant_examples(question: str, max_examples: int = 2) -> str:
    """Get relevant few-shot examples based on question keywords.

    Searches all domain catalogs for matching patterns.

    Args:
        question: User's question
        max_examples: Maximum examples to return

    Returns:
        Formatted examples string
    """
    question_lower = question.lower()

    # Detect domain from keywords
    domain_scores = {}

    for domain in get_available_domains():
        keywords = get_domain_keywords(domain)
        score = 0

        # Check all keyword lists
        for kw_list in keywords.values():
            if isinstance(kw_list, list):
                for kw in kw_list:
                    if kw.lower() in question_lower:
                        score += 1
            elif isinstance(kw_list, dict):
                for sub_list in kw_list.values():
                    if isinstance(sub_list, list):
                        for kw in sub_list:
                            if kw.lower() in question_lower:
                                score += 1

        if score > 0:
            domain_scores[domain] = score

    if not domain_scores:
        return ""

    # Get top domain
    top_domain = max(domain_scores, key=domain_scores.get)

    # Try to match specific query
    matched_query = match_question_to_query(question, top_domain)

    if matched_query:
        lines = []
        lines.append("=" * 60)
        lines.append("RELEVANT EXAMPLE (matched to your question):")
        lines.append("=" * 60)
        lines.append("")
        lines.append(format_query_as_example(matched_query))
        return "\n".join(lines)

    # Fall back to general examples from domain
    return get_few_shot_examples(top_domain, max_examples)


# Quick test
if __name__ == "__main__":
    print("Available domains:", get_available_domains())
    print()

    # Test debt domain
    print(get_few_shot_examples("debt", max_examples=2))
    print()

    # Test question matching
    question = "Який борг у клієнта Іванов?"
    print(f"Matching question: {question}")
    print(get_relevant_examples(question))
