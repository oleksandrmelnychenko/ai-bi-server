"""Extract SQL queries from GBA repository for few-shot learning.

This tool scans C# repository files and extracts SQL queries for use as
few-shot examples in the AI-BI system's SQL generation prompts.

Usage:
    cd backend
    python -m tools.extract_gba_examples
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.sql_examples import extract_table_names

# Default path to GBA repository
DEFAULT_GBA_PATH = Path(r"C:\Users\123\RiderProjects\gba-server\src")
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent / "schema" / "sql_examples_extracted.yaml"
SQL_KEYWORDS = ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH")
SQL_KEYWORD_RE = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|WITH)\b", re.IGNORECASE)
STRING_LITERAL_RE = re.compile(
    r'(\$@\"(?:[^\"]|\"\")*\"|\$\"(?:[^\"\\]|\\.)*\"|@\"(?:[^\"]|\"\")*\"|\"(?:[^\"\\]|\\.)*\")',
    re.DOTALL,
)
# Pattern to find SQL string variable assignments (multiline)
# Matches: string sqlExpr = "...", var sql = "...", etc.
SQL_VAR_ASSIGNMENT_RE = re.compile(
    r'(?:string|var)\s+(\w*(?:sql|query|expression|command)\w*)\s*=\s*',
    re.IGNORECASE
)
# Pattern to find += additions to SQL variables
SQL_VAR_APPEND_RE = re.compile(
    r'(\w*(?:sql|query|expression|command)\w*)\s*\+=\s*',
    re.IGNORECASE
)
SKIP_DIRS = {".git", ".vs", "bin", "obj", "node_modules"}


@dataclass
class ExtractedQuery:
    """Represents an extracted SQL query with metadata."""
    sql: str
    source_file: str
    method_name: str
    tables: list[str] = field(default_factory=list)
    category: str = "unknown"
    description: str = ""
    source_type: str = "repo_cs"


def read_text_robust(path: Path) -> str:
    """Read text with fallback encodings for mixed repo content."""
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_sql(sql: str) -> str:
    """Normalize SQL formatting while preserving line breaks."""
    lines = sql.splitlines()
    cleaned_lines = []
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def contains_sql_keywords(text: str) -> bool:
    return bool(SQL_KEYWORD_RE.search(text or ""))


def decode_csharp_string(literal: str) -> str:
    """Decode a C# string literal into plain text."""
    raw = literal.strip()
    if raw.startswith("$@") or raw.startswith("@$"):
        raw = raw.replace("$", "", 1)

    if raw.startswith("@"):
        content = raw[2:-1]
        content = content.replace('""', '"')
    else:
        content = raw[1:-1]
        content = content.replace("\\\\", "\\")
        content = content.replace('\\"', '"')
        content = content.replace("\\n", "\n")
        content = content.replace("\\t", " ")
        content = content.replace("\\r", "")

    if raw.startswith("$"):
        content = re.sub(r"\{[^}]*\}", "@param", content)
    return content


def find_statement_end(content: str, start: int) -> int:
    """Find the end of a C# statement starting at position start.

    Handles nested parentheses, string literals, and finds the closing semicolon.
    """
    pos = start
    depth = 0  # Track parentheses depth
    in_string = False
    string_char = None
    is_verbatim = False

    while pos < len(content):
        char = content[pos]

        # Handle string literal detection
        if not in_string:
            if char == '"':
                in_string = True
                string_char = '"'
                # Check for verbatim string (@" or $@")
                if pos > 0 and content[pos-1] == '@':
                    is_verbatim = True
                elif pos > 1 and content[pos-2:pos] == '$@':
                    is_verbatim = True
                else:
                    is_verbatim = False
            elif char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ';' and depth <= 0:
                return pos + 1
        else:
            # Inside string literal
            if is_verbatim:
                # Verbatim strings: "" is escape for "
                if char == '"':
                    if pos + 1 < len(content) and content[pos + 1] == '"':
                        pos += 1  # Skip escaped quote
                    else:
                        in_string = False
            else:
                # Regular strings: \" is escape for "
                if char == '\\' and pos + 1 < len(content):
                    pos += 1  # Skip escape sequence
                elif char == '"':
                    in_string = False

        pos += 1

    return len(content)  # Reached end of content


def extract_sql_variable_complete(content: str, var_name: str, start_pos: int) -> tuple[str, int]:
    """Extract complete SQL from variable assignment including += continuations.

    Returns tuple of (combined_sql, end_position).
    """
    all_parts = []

    # Find the end of initial assignment
    end_pos = find_statement_end(content, start_pos)
    statement = content[start_pos:end_pos]

    # Extract literals from initial assignment
    literals = list(STRING_LITERAL_RE.finditer(statement))
    for lit in literals:
        all_parts.append(decode_csharp_string(lit.group(0)))

    # Now look for += continuations for this variable
    # Search from end_pos onwards for varName +=
    search_pos = end_pos
    append_pattern = re.compile(
        rf'\b{re.escape(var_name)}\s*\+=\s*',
        re.IGNORECASE
    )

    while search_pos < len(content):
        # Look for the next += for this variable
        append_match = append_pattern.search(content, search_pos)
        if not append_match:
            break

        # Check if there's another variable assignment between current pos and append
        # If so, this += might belong to a different scope
        intervening = content[search_pos:append_match.start()]

        # Simple heuristic: if we see another function declaration, stop
        if re.search(r'\b(public|private|protected|internal)\s+(?:async\s+)?(?:[\w<>\[\],\s]+)\s+\w+\s*\(', intervening):
            break

        # Find end of this += statement
        append_start = append_match.end()
        append_end = find_statement_end(content, append_start)
        append_statement = content[append_start:append_end]

        # Extract literals from this += statement
        append_literals = list(STRING_LITERAL_RE.finditer(append_statement))
        for lit in append_literals:
            all_parts.append(decode_csharp_string(lit.group(0)))

        search_pos = append_end
        end_pos = append_end

    combined = "".join(all_parts)
    return combined, end_pos


def extract_literal_sequences(content: str) -> list[str]:
    """Extract SQL candidates from concatenated string literals.

    Improved version that finds SQL variable assignments and extracts
    complete statements including multiline string concatenations
    and += continuations.
    """
    candidates = []
    processed_vars = {}  # Track variable positions we've processed

    # Method 1: Find SQL variable assignments (string sqlExpression = ...)
    # and follow their += continuations
    for match in SQL_VAR_ASSIGNMENT_RE.finditer(content):
        var_name = match.group(1)
        start_pos = match.end()

        # Extract complete SQL including += continuations
        combined, end_pos = extract_sql_variable_complete(content, var_name, start_pos)

        if contains_sql_keywords(combined) and len(combined) >= 30:
            candidates.append(combined)
            processed_vars[var_name] = (match.start(), end_pos)

    # Method 2: Find standalone string literals that look like SQL
    # (for cases where variable name doesn't match our pattern)
    seen_positions = set()
    for var_name, (start, end) in processed_vars.items():
        for i in range(start, end):
            seen_positions.add(i)

    # Also mark += statements for known variables
    for match in SQL_VAR_APPEND_RE.finditer(content):
        var_name = match.group(1)
        if var_name.lower() in [v.lower() for v in processed_vars.keys()]:
            end_pos = find_statement_end(content, match.end())
            for i in range(match.start(), end_pos):
                seen_positions.add(i)

    # Find string literals we haven't processed yet
    current_sequence: list[str] = []
    last_end = None

    for match in STRING_LITERAL_RE.finditer(content):
        if match.start() in seen_positions:
            continue

        # Check if this string is connected to previous by + operator
        if last_end is not None:
            between = content[last_end:match.start()]
            # Allow whitespace, newlines, and + operator between strings
            if re.match(r'^[\s\r\n]*\+[\s\r\n]*$', between):
                current_sequence.append(match.group(0))
            else:
                # Process current sequence
                if current_sequence:
                    decoded = "".join(decode_csharp_string(lit) for lit in current_sequence)
                    if contains_sql_keywords(decoded) and len(decoded) >= 30:
                        candidates.append(decoded)
                current_sequence = [match.group(0)]
        else:
            current_sequence = [match.group(0)]

        last_end = match.end()

    # Process final sequence
    if current_sequence:
        decoded = "".join(decode_csharp_string(lit) for lit in current_sequence)
        if contains_sql_keywords(decoded) and len(decoded) >= 30:
            candidates.append(decoded)

    return candidates


def extract_sql_from_string_builders(content: str) -> list[str]:
    """Extract SQL assembled via StringBuilder.Append/AppendLine."""
    builder_pattern = re.compile(
        r"(?:var|StringBuilder)\s+(\w+)\s*=\s*new\s+StringBuilder",
        re.IGNORECASE,
    )
    builders = {m.group(1) for m in builder_pattern.finditer(content)}
    if not builders:
        return []

    candidates: list[str] = []
    for name in builders:
        event_re = re.compile(
            rf"{re.escape(name)}\.(AppendLine|Append|Clear)\(([^;]*)\);",
            re.IGNORECASE,
        )
        buffer_parts: list[str] = []
        for match in event_re.finditer(content):
            method = match.group(1).lower()
            if method == "clear":
                if buffer_parts:
                    candidate = "\n".join(buffer_parts)
                    if contains_sql_keywords(candidate):
                        candidates.append(candidate)
                    buffer_parts = []
                continue

            args = match.group(2)
            for literal in STRING_LITERAL_RE.finditer(args):
                buffer_parts.append(decode_csharp_string(literal.group(0)))

        if buffer_parts:
            candidate = "\n".join(buffer_parts)
            if contains_sql_keywords(candidate):
                candidates.append(candidate)

    return candidates


def categorize_sql(sql: str) -> str:
    """Auto-categorize SQL query based on content patterns."""
    sql_upper = sql.upper()

    # Check for CTEs first
    if "WITH" in sql_upper and "SELECT" in sql_upper:
        if "ROW_NUMBER" in sql_upper:
            return "cte_pagination"
        elif "UNION" in sql_upper:
            return "cte_union"
        else:
            return "cte_aggregation"

    # Aggregation patterns
    if "GROUP BY" in sql_upper:
        if "TOP" in sql_upper:
            return "top_aggregation"
        return "aggregation"

    # Join complexity
    if "LEFT OUTER JOIN" in sql_upper or "LEFT JOIN" in sql_upper:
        join_count = sql_upper.count("JOIN")
        if join_count >= 5:
            return "complex_join"
        return "multi_join"

    # Subqueries
    if "EXISTS" in sql_upper or "IN (SELECT" in sql_upper:
        return "subquery"

    # Date calculations
    if "DATEDIFF" in sql_upper or "GETUTCDATE" in sql_upper or "DATEADD" in sql_upper:
        return "date_calculation"

    # UDF calls
    if "dbo.Get" in sql:
        return "udf_call"

    # DML operations (we'll filter these out but still categorize)
    if "INSERT" in sql_upper:
        return "insert"
    if "UPDATE" in sql_upper:
        return "update"
    if "DELETE" in sql_upper:
        return "delete"

    return "simple_select"


def extract_sql_from_csharp(content: str, file_path: str) -> list[ExtractedQuery]:
    """Extract SQL strings from C# repository files.

    Looks for SQL patterns in string literals and extracts them along
    with the containing method name for context.
    """
    queries = []
    seen_sql: set[str] = set()

    # Find method context using method declarations
    method_pattern = re.compile(
        r'(?:public|private|protected|internal)\s+(?:async\s+)?(?:[\w<>\[\],\s]+)\s+(\w+)\s*\([^)]*\)\s*\{',
        re.MULTILINE
    )
    methods = list(method_pattern.finditer(content))

    candidates = []
    candidates.extend(extract_literal_sequences(content))
    candidates.extend(extract_sql_from_string_builders(content))

    for sql_raw in candidates:
        # Clean up the SQL string
        sql = normalize_sql(sql_raw)
        if not sql:
            continue

        normalized_sql = sql.lower()
        if normalized_sql in seen_sql:
            continue
        seen_sql.add(normalized_sql)

        # Skip too short or invalid SQL
        if not sql or len(sql) < 30:
            continue

        # Skip if it doesn't look like valid SQL
        if not contains_sql_keywords(sql):
            continue

        # Find which method this belongs to
        method_name = "unknown"
        match_pos = content.find(sql_raw)
        for m in methods:
            if m.start() < match_pos:
                method_name = m.group(1)

        # Auto-categorize
        category = categorize_sql(sql)
        tables = extract_table_names(sql)

        queries.append(ExtractedQuery(
            sql=sql,
            source_file=file_path,
            method_name=method_name,
            tables=tables,
            category=category,
            source_type="repo_cs",
        ))

    return queries


def extract_sql_from_sql_file(content: str, file_path: str) -> list[ExtractedQuery]:
    """Extract SQL statements from .sql files."""
    queries: list[ExtractedQuery] = []
    seen_sql: set[str] = set()

    batches = re.split(r"^\s*GO\s*$", content, flags=re.IGNORECASE | re.MULTILINE)
    for batch in batches:
        if not contains_sql_keywords(batch):
            continue

        match = re.search(r"\b(SELECT|WITH)\b", batch, re.IGNORECASE)
        if not match:
            continue

        sql = normalize_sql(batch[match.start():])
        if not sql or len(sql) < 30:
            continue

        normalized_sql = sql.lower()
        if normalized_sql in seen_sql:
            continue
        seen_sql.add(normalized_sql)

        category = categorize_sql(sql)
        tables = extract_table_names(sql)

        queries.append(ExtractedQuery(
            sql=sql,
            source_file=file_path,
            method_name="sql_file",
            tables=tables,
            category=category,
            source_type="repo_sql",
        ))

    return queries


def scan_repository(repo_path: Path) -> dict[str, list[ExtractedQuery]]:
    """Scan entire GBA repository and extract all SQL queries.

    Args:
        repo_path: Path to the repository directory

    Returns:
        Dictionary mapping category names to lists of extracted queries
    """
    all_queries: dict[str, list[ExtractedQuery]] = {}

    if not repo_path.exists():
        print(f"Warning: Repository path does not exist: {repo_path}")
        return all_queries

    # Find all C# and SQL files (skip build artifacts)
    cs_files = [
        p for p in repo_path.rglob("*.cs")
        if not any(part.lower() in SKIP_DIRS for part in p.parts)
    ]
    sql_files = [
        p for p in repo_path.rglob("*.sql")
        if not any(part.lower() in SKIP_DIRS for part in p.parts)
    ]
    print(f"Found {len(cs_files)} C# files and {len(sql_files)} SQL files to scan")

    for cs_file in cs_files:
        try:
            content = read_text_robust(cs_file)
            relative_path = str(cs_file.relative_to(repo_path))
            queries = extract_sql_from_csharp(content, relative_path)

            for q in queries:
                if q.category not in all_queries:
                    all_queries[q.category] = []
                all_queries[q.category].append(q)

        except Exception as e:
            print(f"Error processing {cs_file}: {e}")

    for sql_file in sql_files:
        try:
            content = read_text_robust(sql_file)
            relative_path = str(sql_file.relative_to(repo_path))
            queries = extract_sql_from_sql_file(content, relative_path)

            for q in queries:
                if q.category not in all_queries:
                    all_queries[q.category] = []
                all_queries[q.category].append(q)
        except Exception as e:
            print(f"Error processing {sql_file}: {e}")

    return all_queries


def export_to_yaml(
    queries: dict[str, list[ExtractedQuery]],
    output_path: Path,
    limit_per_category: int | None = None
) -> None:
    """Export extracted queries to YAML format for review and curation.

    Args:
        queries: Dictionary of category -> list of ExtractedQuery
        output_path: Path to write the YAML file
    """
    category_descriptions = {
        "simple_select": "Basic SELECT with soft-delete filter",
        "multi_join": "Multiple table JOINs (2-4 tables)",
        "complex_join": "Complex JOINs (5+ tables)",
        "aggregation": "GROUP BY with aggregations",
        "top_aggregation": "TOP N with GROUP BY",
        "cte_pagination": "CTE with ROW_NUMBER pagination",
        "cte_union": "CTE with UNION of multiple sources",
        "cte_aggregation": "CTE with aggregation logic",
        "subquery": "Queries with subqueries",
        "date_calculation": "Date/time calculations",
        "udf_call": "Queries using UDF functions",
        "insert": "INSERT statements (for reference only)",
        "update": "UPDATE statements (for reference only)",
        "delete": "DELETE statements (for reference only)",
    }

    output: dict = {"categories": {}}

    # Filter out DML operations - we only want SELECT examples
    select_categories = {k: v for k, v in queries.items()
                        if k not in ("insert", "update", "delete")}

    for category, query_list in sorted(select_categories.items()):
        if limit_per_category:
            query_list = query_list[:limit_per_category]
        output["categories"][category] = {
            "description": category_descriptions.get(category, category),
            "keywords": [],  # To be filled manually during curation
            "examples": [
                {
                    "source": f"{q.source_file}::{q.method_name}",
                    "source_type": q.source_type,
                    "tables": q.tables,
                    "sql": q.sql,
                    "question": ""  # To be filled manually during curation
                }
                for q in query_list
            ]
        }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(output, f, allow_unicode=True, default_flow_style=False, width=120)

    print(f"\nExported to {output_path}")
    print("\nCategories extracted:")
    for cat, items in sorted(queries.items()):
        if cat not in ("insert", "update", "delete"):
            print(f"  {cat}: {len(items)} queries")


def main() -> None:
    """Main entry point for the extraction tool."""
    parser = argparse.ArgumentParser(
        description="Extract SQL queries from GBA repository for few-shot learning"
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=DEFAULT_GBA_PATH,
        help=f"Path to GBA repository (default: {DEFAULT_GBA_PATH})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output YAML file path (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit per category (0 = no limit)"
    )
    args = parser.parse_args()

    print(f"Scanning GBA repository at {args.repo}...")
    queries = scan_repository(args.repo)

    total = sum(len(v) for v in queries.values())
    print(f"\nFound {total} total queries across {len(queries)} categories")

    if total > 0:
        limit_per_category = args.limit if args.limit > 0 else None
        export_to_yaml(queries, args.output, limit_per_category=limit_per_category)
        print("\nNext steps:")
        print("1. Review the extracted examples in sql_examples_extracted.yaml")
        print("2. Add Ukrainian keywords to each category")
        print("3. Add question descriptions for the best examples")
        print("4. Remove low-quality or duplicate queries")
        print("5. Save the curated version as sql_examples.yaml")
    else:
        print("No queries found. Check the repository path.")


if __name__ == "__main__":
    main()
