"""Analyze ALL client-related queries from the knowledge base."""

import sqlite3
import json
import sys
from pathlib import Path
from collections import defaultdict

DB_PATH = Path(__file__).parent.parent / "schema" / "sql_vectors.sqlite"


def analyze_client_repos():
    """Find all client-related repositories."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Find ALL client-related repositories
    cur.execute("""
        SELECT source_file, COUNT(*) as cnt
        FROM sql_vectors
        WHERE LOWER(source_file) LIKE '%client%'
        GROUP BY source_file
        ORDER BY cnt DESC
    """)

    print("=== CLIENT-RELATED REPOSITORIES ===\n")
    total = 0
    repos = []
    for source_file, cnt in cur.fetchall():
        # Clean the path
        name = source_file.split("\\")[-1].replace(".cs", "")
        print(f"{cnt:4d} - {name}")
        total += cnt
        repos.append((name, cnt))

    print(f"\nTotal: {total} queries from {len(repos)} repositories")
    return repos


def analyze_client_contexts():
    """Analyze client queries by business context."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Get all client-related queries
    cur.execute("""
        SELECT sql, source_file, category, tables
        FROM sql_vectors
        WHERE LOWER(source_file) LIKE '%client%'
    """)

    # Categorize by business context based on keywords in SQL
    contexts = {
        "debt": {"keywords": ["debt", "clientindebt", "заборгован"], "queries": []},
        "cash_flow": {"keywords": ["cashflow", "payment", "income", "outcome", "getexchanged"], "queries": []},
        "agreements": {"keywords": ["agreement", "договор", "clientagreement"], "queries": []},
        "purchases": {"keywords": ["sale", "order", "orderitem", "purchase"], "queries": []},
        "invoices": {"keywords": ["invoice", "накладн", "saleinvoice"], "queries": []},
        "products": {"keywords": ["product", "товар", "productincome"], "queries": []},
        "structure": {"keywords": ["subclient", "rootclient", "clientsubclient"], "queries": []},
        "contacts": {"keywords": ["contact", "address", "phone", "email"], "queries": []},
        "profile": {"keywords": ["userprofile", "manager", "clienttype", "role"], "queries": []},
        "pricing": {"keywords": ["pricing", "price", "discount", "знижк"], "queries": []},
        "region": {"keywords": ["region", "country", "регіон"], "queries": []},
        "bank": {"keywords": ["bank", "iban", "account"], "queries": []},
    }

    for sql, source_file, category, tables in cur.fetchall():
        sql_lower = sql.lower()
        tables_lower = (tables or "").lower()
        combined = sql_lower + " " + tables_lower

        matched = False
        for ctx_name, ctx_data in contexts.items():
            for kw in ctx_data["keywords"]:
                if kw in combined:
                    ctx_data["queries"].append({
                        "sql": sql,
                        "source": source_file.split("\\")[-1],
                        "category": category,
                        "tables": json.loads(tables) if tables else []
                    })
                    matched = True
                    break
            if matched:
                break

        if not matched:
            # Put in general
            if "general" not in contexts:
                contexts["general"] = {"keywords": [], "queries": []}
            contexts["general"]["queries"].append({
                "sql": sql,
                "source": source_file.split("\\")[-1],
                "category": category
            })

    print("\n" + "=" * 60)
    print("=== CLIENT QUERIES BY BUSINESS CONTEXT ===")
    print("=" * 60)

    for ctx_name, ctx_data in sorted(contexts.items(), key=lambda x: -len(x[1]["queries"])):
        queries = ctx_data["queries"]
        if queries:
            print(f"\n{ctx_name.upper()}: {len(queries)} queries")
            # Show unique sources
            sources = set(q["source"].replace(".cs", "") for q in queries)
            print(f"  Sources: {', '.join(sorted(sources)[:5])}")

    conn.close()
    return contexts


def show_context_examples(context_name: str, max_examples: int = 3):
    """Show example queries for a specific context."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    context_keywords = {
        "cash_flow": "getexchanged",
        "agreements": "clientagreement",
        "purchases": "sale",
        "invoices": "invoice",
        "structure": "subclient",
    }

    kw = context_keywords.get(context_name, context_name)

    cur.execute(f"""
        SELECT sql, source_file, category, tables
        FROM sql_vectors
        WHERE LOWER(source_file) LIKE '%client%'
          AND (LOWER(sql) LIKE '%{kw}%' OR LOWER(tables) LIKE '%{kw}%')
        ORDER BY LENGTH(sql) DESC
        LIMIT {max_examples}
    """)

    print(f"\n=== {context_name.upper()} EXAMPLES ===\n")
    for sql, source, category, tables in cur.fetchall():
        print(f"Source: {source.split(chr(92))[-1]}")
        print(f"Category: {category}")
        print(f"SQL preview: {sql[:300].replace(chr(10), ' ')}...")
        print()

    conn.close()


if __name__ == "__main__":
    analyze_client_repos()
    contexts = analyze_client_contexts()
