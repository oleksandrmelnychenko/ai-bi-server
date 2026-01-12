from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import sys

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.join_graph import edges_from_foreign_keys
from app.schema_cache import SchemaCache


def build_payload(schema_cache: SchemaCache) -> dict:
    tables = []
    for table in sorted(schema_cache.tables.values(), key=lambda item: item.key):
        tables.append(
            {
                "name": table.key,
                "primary_key": table.primary_key or [],
                "role": "unknown",
                "default_filters": [],
            }
        )

    joins = []
    for edge in edges_from_foreign_keys(schema_cache.foreign_keys):
        joins.append(
            {
                "name": edge.name or f"{edge.left_table}->{edge.right_table}",
                "left": edge.left_table,
                "right": edge.right_table,
                "columns": [{"left": left, "right": right} for left, right in edge.columns],
                "weight": edge.weight,
                "enabled": True,
            }
        )

    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tables": tables,
        "joins": joins,
    }


def export_join_rules(output_path: str) -> str:
    schema_cache = SchemaCache()
    schema_cache.load()

    payload = build_payload(schema_cache)
    resolved = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)
    return resolved


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Export join rules from SQL Server schema")
    parser.add_argument(
        "--output",
        default=settings.join_rules_path,
        help="Path to write join_rules.yaml",
    )
    args = parser.parse_args()

    output_path = export_join_rules(args.output)
    print(f"Wrote join rules to {output_path}")


if __name__ == "__main__":
    main()
