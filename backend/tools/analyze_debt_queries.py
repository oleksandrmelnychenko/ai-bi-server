"""Analyze all debt-related SQL queries from the knowledge base."""

import sqlite3
import json
from pathlib import Path
from collections import defaultdict

DB_PATH = Path(__file__).parent.parent / "schema" / "sql_vectors.sqlite"


def analyze_debt_queries():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Find ALL debt-related queries
    cur.execute("""
        SELECT sql, source_file, category, tables
        FROM sql_vectors
        WHERE LOWER(sql) LIKE '%debt%'
           OR LOWER(tables) LIKE '%debt%'
           OR LOWER(sql) LIKE '%clientindebt%'
        ORDER BY source_file, LENGTH(sql) DESC
    """)

    results = cur.fetchall()
    print(f"Found {len(results)} debt-related queries\n")

    # Group by source file
    by_source = defaultdict(list)
    for sql, source_file, category, tables in results:
        # Clean source path
        source = source_file.replace("GBA.Domain\\Repositories\\", "").replace(".cs", "")
        by_source[source].append({
            "sql": sql,
            "category": category,
            "tables": json.loads(tables) if tables else []
        })

    print("=== DEBT QUERIES BY REPOSITORY ===\n")
    for source, queries in sorted(by_source.items(), key=lambda x: -len(x[1])):
        print(f"{source}: {len(queries)} queries")

    # Analyze ClientRepository debt queries in detail
    print("\n" + "=" * 60)
    print("=== CLIENT REPOSITORY DEBT QUERIES ===")
    print("=" * 60 + "\n")

    client_queries = []
    for source, queries in by_source.items():
        if "Client" in source and "Sync" not in source:
            client_queries.extend(queries)

    # Categorize by pattern
    patterns = {
        "debt_totals": [],      # SUM/aggregation of debt
        "debt_by_date": [],     # Date-filtered debt
        "debt_by_agreement": [],  # Per-agreement debt
        "debt_details": [],     # Individual debt records
        "debt_days": [],        # Days overdue calculations
    }

    for q in client_queries:
        sql_lower = q["sql"].lower()

        if "datediff" in sql_lower and "debt" in sql_lower:
            patterns["debt_days"].append(q)
        elif "sum(" in sql_lower and "debt" in sql_lower:
            patterns["debt_totals"].append(q)
        elif "getutcdate()" in sql_lower or "@date" in sql_lower:
            patterns["debt_by_date"].append(q)
        elif "agreementid" in sql_lower and "debt" in sql_lower:
            patterns["debt_by_agreement"].append(q)
        else:
            patterns["debt_details"].append(q)

    for pattern_name, queries in patterns.items():
        if queries:
            print(f"\n--- {pattern_name.upper()} ({len(queries)} queries) ---")
            for i, q in enumerate(queries[:2]):  # Show first 2 examples
                print(f"\nExample {i+1} [{q['category']}]:")
                print(f"Tables: {', '.join(q['tables'][:8])}")
                # Show key part of SQL
                sql_preview = q["sql"][:500].replace("\n", " ")
                print(f"SQL: {sql_preview}...")

    conn.close()
    return by_source


if __name__ == "__main__":
    analyze_debt_queries()
