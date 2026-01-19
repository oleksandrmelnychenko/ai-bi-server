"""Extract ALL SQL queries from GBA repository directly to SQLite.

This tool scans C# repository files and extracts all SQL queries,
storing them in a SQLite database with full metadata for:
- Few-shot learning (vector search)
- UI display and browsing
- Analysis and reporting

Usage:
    cd backend
    python -m tools.extract_sql_to_sqlite
    python -m tools.extract_sql_to_sqlite --embed  # Also generate embeddings
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Default paths
DEFAULT_GBA_PATH = Path(r"C:\Users\123\RiderProjects\gba-server\src")
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent / "schema" / "sql_knowledge.sqlite"

# SQL patterns
SQL_KEYWORDS = ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "MERGE", "EXEC")
SQL_KEYWORD_RE = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|WITH|MERGE|EXEC)\b", re.IGNORECASE)
STRING_LITERAL_RE = re.compile(
    r'(\$@\"(?:[^\"]|\"\")*\"|\$\"(?:[^\"\\]|\\.)*\"|@\"(?:[^\"]|\"\")*\"|\"(?:[^\"\\]|\\.)*\")',
    re.DOTALL,
)
SQL_VAR_ASSIGNMENT_RE = re.compile(
    r'(?:string|var)\s+(\w*(?:sql|query|expression|command|script)\w*)\s*=\s*[\r\n\s]*',
    re.IGNORECASE
)
METHOD_PATTERN = re.compile(
    r'(?:public|private|protected|internal)\s+(?:static\s+)?(?:async\s+)?(?:[\w<>\[\],\s]+)\s+(\w+)\s*\([^)]*\)\s*\{',
    re.MULTILINE
)
SKIP_DIRS = {".git", ".vs", "bin", "obj", "node_modules", "packages"}

# Table extraction
TABLE_REF_RE = re.compile(r"\b(?:FROM|JOIN|INTO|UPDATE)\s+([^\s,;(]+)", re.IGNORECASE)
CTE_NAME_RE = re.compile(r"(?:\bWITH\b|,)\s*(\[?\w+\]?)\s+AS\s*\(", re.IGNORECASE)


@dataclass
class ExtractedSQL:
    """Represents an extracted SQL query with full metadata."""
    id: str
    sql: str
    sql_type: str  # SELECT, INSERT, UPDATE, DELETE, WITH, OTHER
    source_file: str
    method_name: str
    variable_name: str
    line_number: int
    tables: list[str] = field(default_factory=list)
    category: str = "unknown"
    patterns: list[str] = field(default_factory=list)
    char_count: int = 0
    join_count: int = 0
    is_complete: bool = True


def generate_sql_id(sql: str, source: str, method: str) -> str:
    """Generate unique ID for SQL query."""
    content = f"{source}:{method}:{sql[:500]}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def read_text_robust(path: Path) -> str:
    """Read text with fallback encodings."""
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


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
    """Find the end of a C# statement (handles strings and nested parens)."""
    pos = start
    depth = 0
    in_string = False
    is_verbatim = False

    while pos < len(content):
        char = content[pos]

        if not in_string:
            if char == '"':
                in_string = True
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
            if is_verbatim:
                if char == '"':
                    if pos + 1 < len(content) and content[pos + 1] == '"':
                        pos += 1
                    else:
                        in_string = False
            else:
                if char == '\\' and pos + 1 < len(content):
                    pos += 1
                elif char == '"':
                    in_string = False
        pos += 1

    return len(content)


def extract_tables(sql: str) -> list[str]:
    """Extract table names from SQL."""
    cte_names = set()
    for match in CTE_NAME_RE.finditer(sql):
        name = match.group(1).strip("[]").lower()
        if name:
            cte_names.add(name)

    tables = set()
    for match in TABLE_REF_RE.finditer(sql):
        raw = match.group(1).strip("[]()").strip()
        if not raw or raw.upper() in ("SELECT", "WITH", "@"):
            continue
        # Remove schema prefix for comparison
        base_name = raw.split(".")[-1].strip("[]").lower()
        if base_name and base_name not in cte_names:
            tables.add(base_name)

    return sorted(tables)


def detect_sql_type(sql: str) -> str:
    """Detect the type of SQL statement."""
    sql_upper = sql.strip().upper()
    if sql_upper.startswith("SELECT") or sql_upper.startswith(";WITH") or sql_upper.startswith("WITH"):
        return "SELECT"
    if sql_upper.startswith("INSERT"):
        return "INSERT"
    if sql_upper.startswith("UPDATE"):
        return "UPDATE"
    if sql_upper.startswith("DELETE"):
        return "DELETE"
    if sql_upper.startswith("MERGE"):
        return "MERGE"
    if sql_upper.startswith("EXEC"):
        return "EXEC"
    if "SELECT" in sql_upper:
        return "SELECT"
    return "OTHER"


def detect_patterns(sql: str) -> list[str]:
    """Detect SQL patterns in query."""
    patterns = []
    sql_upper = sql.upper()

    if "WITH" in sql_upper and "AS" in sql_upper:
        patterns.append("CTE")
    if "ROW_NUMBER" in sql_upper:
        patterns.append("ROW_NUMBER")
    if "RANK()" in sql_upper or "DENSE_RANK" in sql_upper:
        patterns.append("RANKING")
    if "GROUP BY" in sql_upper:
        patterns.append("GROUP_BY")
    if "HAVING" in sql_upper:
        patterns.append("HAVING")
    if re.search(r"\bSUM\s*\(", sql_upper):
        patterns.append("SUM")
    if re.search(r"\bCOUNT\s*\(", sql_upper):
        patterns.append("COUNT")
    if re.search(r"\bAVG\s*\(", sql_upper):
        patterns.append("AVG")
    if re.search(r"\bMAX\s*\(", sql_upper):
        patterns.append("MAX")
    if re.search(r"\bMIN\s*\(", sql_upper):
        patterns.append("MIN")
    if "LEFT JOIN" in sql_upper or "LEFT OUTER JOIN" in sql_upper:
        patterns.append("LEFT_JOIN")
    if "INNER JOIN" in sql_upper:
        patterns.append("INNER_JOIN")
    if "CROSS JOIN" in sql_upper:
        patterns.append("CROSS_JOIN")
    if "UNION" in sql_upper:
        patterns.append("UNION")
    if "ORDER BY" in sql_upper:
        patterns.append("ORDER_BY")
    if "TOP" in sql_upper:
        patterns.append("TOP")
    if "DISTINCT" in sql_upper:
        patterns.append("DISTINCT")
    if "CASE WHEN" in sql_upper:
        patterns.append("CASE_WHEN")
    if "COALESCE" in sql_upper or "ISNULL" in sql_upper:
        patterns.append("NULL_HANDLING")
    if "dbo.Get" in sql:
        patterns.append("UDF")
    if "GETUTCDATE" in sql_upper or "GETDATE" in sql_upper:
        patterns.append("CURRENT_DATE")
    if "DATEDIFF" in sql_upper or "DATEADD" in sql_upper:
        patterns.append("DATE_CALC")
    if "EXISTS" in sql_upper:
        patterns.append("EXISTS")
    if re.search(r"IN\s*\(\s*SELECT", sql_upper):
        patterns.append("SUBQUERY")
    if "LIKE" in sql_upper:
        patterns.append("LIKE")
    if "BETWEEN" in sql_upper:
        patterns.append("BETWEEN")
    if "PIVOT" in sql_upper:
        patterns.append("PIVOT")
    if "SCOPE_IDENTITY" in sql_upper:
        patterns.append("IDENTITY")

    return patterns


def categorize_sql(sql: str, sql_type: str) -> str:
    """Categorize SQL query based on patterns."""
    if sql_type != "SELECT":
        return sql_type.lower()

    sql_upper = sql.upper()
    join_count = sql_upper.count("JOIN")

    if "WITH" in sql_upper and "AS" in sql_upper:
        if "ROW_NUMBER" in sql_upper:
            return "cte_pagination"
        if "UNION" in sql_upper:
            return "cte_union"
        return "cte_aggregation"

    if "GROUP BY" in sql_upper:
        if "TOP" in sql_upper:
            return "top_aggregation"
        return "aggregation"

    if join_count >= 5:
        return "complex_join"
    if join_count >= 2:
        return "multi_join"

    if "EXISTS" in sql_upper or "IN (SELECT" in sql_upper:
        return "subquery"

    if "DATEDIFF" in sql_upper or "DATEADD" in sql_upper:
        return "date_calculation"

    if "dbo.Get" in sql:
        return "udf_call"

    return "simple_select"


def normalize_sql(sql: str) -> str:
    """Normalize SQL formatting."""
    lines = sql.splitlines()
    cleaned = []
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


def extract_sql_variable_complete(content: str, var_name: str, start_pos: int) -> tuple[str, int]:
    """Extract complete SQL from variable assignment including += continuations."""
    all_parts = []

    end_pos = find_statement_end(content, start_pos)
    statement = content[start_pos:end_pos]

    for lit in STRING_LITERAL_RE.finditer(statement):
        all_parts.append(decode_csharp_string(lit.group(0)))

    search_pos = end_pos
    append_pattern = re.compile(rf'\b{re.escape(var_name)}\s*\+=\s*', re.IGNORECASE)

    while search_pos < len(content):
        append_match = append_pattern.search(content, search_pos)
        if not append_match:
            break

        intervening = content[search_pos:append_match.start()]
        if re.search(r'\b(public|private|protected|internal)\s+(?:async\s+)?(?:[\w<>\[\],\s]+)\s+\w+\s*\(', intervening):
            break

        append_start = append_match.end()
        append_end = find_statement_end(content, append_start)
        append_statement = content[append_start:append_end]

        for lit in STRING_LITERAL_RE.finditer(append_statement):
            all_parts.append(decode_csharp_string(lit.group(0)))

        search_pos = append_end
        end_pos = append_end

    return "".join(all_parts), end_pos


def find_line_number(content: str, pos: int) -> int:
    """Find line number for a position in content."""
    return content[:pos].count('\n') + 1


def find_method_at_position(content: str, pos: int, methods: list) -> str:
    """Find which method contains the given position."""
    method_name = "unknown"
    for m in methods:
        if m.start() < pos:
            method_name = m.group(1)
    return method_name


def extract_string_sequences(content: str) -> list[tuple[str, int]]:
    """Extract all string literal sequences connected by + operators.

    Returns list of (combined_string, start_position) tuples.
    """
    sequences = []
    current_parts = []
    current_start = None
    last_end = None

    for match in STRING_LITERAL_RE.finditer(content):
        if last_end is not None:
            between = content[last_end:match.start()]
            # Check if connected by + operator (allowing whitespace/newlines)
            if re.match(r'^[\s\r\n]*\+[\s\r\n]*$', between):
                current_parts.append(match.group(0))
            else:
                # Save current sequence
                if current_parts:
                    combined = "".join(decode_csharp_string(p) for p in current_parts)
                    if combined and SQL_KEYWORD_RE.search(combined):
                        sequences.append((combined, current_start))
                current_parts = [match.group(0)]
                current_start = match.start()
        else:
            current_parts = [match.group(0)]
            current_start = match.start()

        last_end = match.end()

    # Don't forget the last sequence
    if current_parts:
        combined = "".join(decode_csharp_string(p) for p in current_parts)
        if combined and SQL_KEYWORD_RE.search(combined):
            sequences.append((combined, current_start))

    return sequences


def extract_from_file(file_path: Path, repo_path: Path) -> list[ExtractedSQL]:
    """Extract all SQL from a single C# file."""
    content = read_text_robust(file_path)
    relative_path = str(file_path.relative_to(repo_path))

    queries = []
    seen_sql = set()
    methods = list(METHOD_PATTERN.finditer(content))
    processed_positions = set()

    # Method 1: Find SQL variable assignments (with += continuations)
    for match in SQL_VAR_ASSIGNMENT_RE.finditer(content):
        var_name = match.group(1)
        start_pos = match.end()
        line_number = find_line_number(content, match.start())

        sql_raw, end_pos = extract_sql_variable_complete(content, var_name, start_pos)
        sql = normalize_sql(sql_raw)

        if not sql or len(sql) < 20:
            continue
        if not SQL_KEYWORD_RE.search(sql):
            continue

        sql_hash = hashlib.md5(sql.lower().encode()).hexdigest()
        if sql_hash in seen_sql:
            continue
        seen_sql.add(sql_hash)

        # Mark positions as processed
        for i in range(match.start(), end_pos):
            processed_positions.add(i)

        method_name = find_method_at_position(content, match.start(), methods)
        sql_type = detect_sql_type(sql)
        tables = extract_tables(sql)
        patterns = detect_patterns(sql)
        category = categorize_sql(sql, sql_type)

        queries.append(ExtractedSQL(
            id=generate_sql_id(sql, relative_path, method_name),
            sql=sql,
            sql_type=sql_type,
            source_file=relative_path,
            method_name=method_name,
            variable_name=var_name,
            line_number=line_number,
            tables=tables,
            category=category,
            patterns=patterns,
            char_count=len(sql),
            join_count=sql.upper().count("JOIN"),
            is_complete=not sql.rstrip().endswith("+")
        ))

    # Method 2: Find ANY string concatenation that looks like SQL
    for sql_raw, start_pos in extract_string_sequences(content):
        # Skip if already processed via variable assignment
        if start_pos in processed_positions:
            continue

        sql = normalize_sql(sql_raw)
        if not sql or len(sql) < 30:
            continue

        sql_hash = hashlib.md5(sql.lower().encode()).hexdigest()
        if sql_hash in seen_sql:
            continue
        seen_sql.add(sql_hash)

        line_number = find_line_number(content, start_pos)
        method_name = find_method_at_position(content, start_pos, methods)
        sql_type = detect_sql_type(sql)
        tables = extract_tables(sql)
        patterns = detect_patterns(sql)
        category = categorize_sql(sql, sql_type)

        queries.append(ExtractedSQL(
            id=generate_sql_id(sql, relative_path, method_name),
            sql=sql,
            sql_type=sql_type,
            source_file=relative_path,
            method_name=method_name,
            variable_name="inline",
            line_number=line_number,
            tables=tables,
            category=category,
            patterns=patterns,
            char_count=len(sql),
            join_count=sql.upper().count("JOIN"),
            is_complete=not sql.rstrip().endswith("+")
        ))

    # Method 3: StringBuilder patterns
    builder_pattern = re.compile(r"(?:var|StringBuilder)\s+(\w+)\s*=\s*new\s+StringBuilder", re.IGNORECASE)
    for builder_match in builder_pattern.finditer(content):
        builder_name = builder_match.group(1)
        line_number = find_line_number(content, builder_match.start())

        append_re = re.compile(rf"{re.escape(builder_name)}\.(?:AppendLine|Append)\(([^;]*)\);", re.IGNORECASE)
        parts = []
        for append_match in append_re.finditer(content):
            for lit in STRING_LITERAL_RE.finditer(append_match.group(1)):
                parts.append(decode_csharp_string(lit.group(0)))

        if parts:
            sql = normalize_sql("\n".join(parts))
            if sql and len(sql) >= 20 and SQL_KEYWORD_RE.search(sql):
                sql_hash = hashlib.md5(sql.lower().encode()).hexdigest()
                if sql_hash not in seen_sql:
                    seen_sql.add(sql_hash)
                    method_name = find_method_at_position(content, builder_match.start(), methods)
                    sql_type = detect_sql_type(sql)

                    queries.append(ExtractedSQL(
                        id=generate_sql_id(sql, relative_path, method_name),
                        sql=sql,
                        sql_type=sql_type,
                        source_file=relative_path,
                        method_name=method_name,
                        variable_name=builder_name,
                        line_number=line_number,
                        tables=extract_tables(sql),
                        category=categorize_sql(sql, sql_type),
                        patterns=detect_patterns(sql),
                        char_count=len(sql),
                        join_count=sql.upper().count("JOIN"),
                        is_complete=True
                    ))

    return queries


def create_database(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database with schema."""
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE sql_queries (
            id TEXT PRIMARY KEY,
            sql TEXT NOT NULL,
            sql_type TEXT NOT NULL,
            source_file TEXT NOT NULL,
            method_name TEXT,
            variable_name TEXT,
            line_number INTEGER,
            tables TEXT,  -- JSON array
            category TEXT,
            patterns TEXT,  -- JSON array
            char_count INTEGER,
            join_count INTEGER,
            is_complete INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE embeddings (
            sql_id TEXT PRIMARY KEY,
            embedding BLOB,
            embedding_dim INTEGER,
            searchable_text TEXT,
            FOREIGN KEY (sql_id) REFERENCES sql_queries(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX idx_sql_type ON sql_queries(sql_type)")
    cursor.execute("CREATE INDEX idx_category ON sql_queries(category)")
    cursor.execute("CREATE INDEX idx_source_file ON sql_queries(source_file)")
    cursor.execute("CREATE INDEX idx_method_name ON sql_queries(method_name)")
    cursor.execute("CREATE INDEX idx_char_count ON sql_queries(char_count)")

    conn.commit()
    return conn


def extract_all(repo_path: Path, output_path: Path, verbose: bool = True) -> int:
    """Extract all SQL from repository to SQLite."""
    if verbose:
        print(f"Scanning repository: {repo_path}")

    # Find all C# files
    cs_files = [
        p for p in repo_path.rglob("*.cs")
        if not any(part.lower() in SKIP_DIRS for part in p.parts)
    ]

    if verbose:
        print(f"Found {len(cs_files)} C# files")

    # Create database
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn = create_database(output_path)
    cursor = conn.cursor()

    # Extract from each file
    total_queries = 0
    stats = {"by_type": {}, "by_category": {}, "by_file": {}}

    for i, cs_file in enumerate(cs_files):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{len(cs_files)}...")

        try:
            queries = extract_from_file(cs_file, repo_path)

            for q in queries:
                cursor.execute("""
                    INSERT OR IGNORE INTO sql_queries
                    (id, sql, sql_type, source_file, method_name, variable_name,
                     line_number, tables, category, patterns, char_count, join_count, is_complete)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    q.id, q.sql, q.sql_type, q.source_file, q.method_name, q.variable_name,
                    q.line_number, json.dumps(q.tables), q.category, json.dumps(q.patterns),
                    q.char_count, q.join_count, 1 if q.is_complete else 0
                ))

                # Update stats
                stats["by_type"][q.sql_type] = stats["by_type"].get(q.sql_type, 0) + 1
                stats["by_category"][q.category] = stats["by_category"].get(q.category, 0) + 1

                file_name = q.source_file.split("\\")[-1].split("/")[-1]
                stats["by_file"][file_name] = stats["by_file"].get(file_name, 0) + 1

                total_queries += 1

        except Exception as e:
            if verbose:
                print(f"  Error processing {cs_file}: {e}")

    # Store metadata
    cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                   ("repo_path", str(repo_path)))
    cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                   ("extracted_at", datetime.utcnow().isoformat()))
    cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                   ("total_queries", str(total_queries)))
    cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                   ("stats_by_type", json.dumps(stats["by_type"])))
    cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                   ("stats_by_category", json.dumps(stats["by_category"])))

    conn.commit()
    conn.close()

    if verbose:
        print(f"\nExtracted {total_queries} SQL queries to {output_path}")
        print(f"\nBy type:")
        for t, c in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")
        print(f"\nBy category:")
        for cat, c in sorted(stats["by_category"].items(), key=lambda x: -x[1])[:15]:
            print(f"  {cat}: {c}")
        print(f"\nTop source files:")
        for f, c in sorted(stats["by_file"].items(), key=lambda x: -x[1])[:10]:
            print(f"  {f}: {c}")

    return total_queries


def add_embeddings(db_path: Path, model_name: str = "intfloat/multilingual-e5-base", batch_size: int = 32) -> None:
    """Add embeddings to existing database."""
    print(f"Loading embedding model: {model_name}")
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all queries
    cursor.execute("""
        SELECT id, sql, source_file, method_name, tables, category, patterns
        FROM sql_queries
        WHERE sql_type = 'SELECT'
    """)
    rows = cursor.fetchall()
    print(f"Generating embeddings for {len(rows)} SELECT queries...")

    # Generate searchable text and embeddings
    texts = []
    ids = []
    for row in rows:
        sql_id, sql, source_file, method_name, tables_json, category, patterns_json = row
        tables = json.loads(tables_json) if tables_json else []
        patterns = json.loads(patterns_json) if patterns_json else []

        # Build searchable text
        file_name = source_file.split("\\")[-1].split("/")[-1].replace(".cs", "")
        parts = [f"Pattern: {category}"]
        if tables:
            parts.append(f"Tables: {', '.join(tables[:8])}")
        if method_name and method_name != "unknown":
            method_words = re.sub(r"([a-z])([A-Z])", r"\1 \2", method_name)
            parts.append(f"Operation: {method_words}")
        if patterns:
            parts.append(f"Features: {', '.join(patterns[:5])}")

        searchable_text = " | ".join(parts)

        # For e5 models, prefix with "query: "
        if "e5" in model_name.lower():
            texts.append(f"query: {searchable_text}")
        else:
            texts.append(searchable_text)
        ids.append((sql_id, searchable_text))

    # Generate embeddings in batches
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    # Store embeddings
    for (sql_id, searchable_text), embedding in zip(ids, embeddings):
        cursor.execute("""
            INSERT OR REPLACE INTO embeddings (sql_id, embedding, embedding_dim, searchable_text)
            VALUES (?, ?, ?, ?)
        """, (sql_id, embedding.astype(np.float32).tobytes(), embedding_dim, searchable_text))

    # Update metadata
    cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                   ("embedding_model", model_name))
    cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                   ("embedding_dim", str(embedding_dim)))
    cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                   ("embeddings_count", str(len(ids))))

    conn.commit()
    conn.close()
    print(f"Added {len(ids)} embeddings")


def main():
    parser = argparse.ArgumentParser(description="Extract SQL from GBA repository to SQLite")
    parser.add_argument("--repo", type=Path, default=DEFAULT_GBA_PATH,
                        help=f"Path to GBA repository (default: {DEFAULT_GBA_PATH})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                        help=f"Output SQLite file (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("--embed", action="store_true",
                        help="Also generate embeddings after extraction")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-base",
                        help="Embedding model to use")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    count = extract_all(args.repo, args.output, verbose=not args.quiet)

    if args.embed and count > 0:
        print("\nGenerating embeddings...")
        add_embeddings(args.output, args.model)

    print(f"\nDone! Database: {args.output}")
    print(f"Size: {args.output.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
