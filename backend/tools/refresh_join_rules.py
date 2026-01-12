from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from tools.apply_join_overrides import apply_overrides_to_path
from tools.apply_table_rules import apply_table_rules_to_path
from tools.export_join_rules import export_join_rules


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Export join rules and apply overrides/table rules")
    parser.add_argument(
        "--output",
        default=settings.join_rules_path,
        help="Path to write join_rules.yaml",
    )
    args = parser.parse_args()

    output_path = Path(export_join_rules(args.output))
    join_changes = apply_overrides_to_path(output_path)
    table_changes = apply_table_rules_to_path(output_path)

    print(
        f"Refreshed join rules at {output_path} (join overrides: {join_changes}, table updates: {table_changes})"
    )


if __name__ == "__main__":
    main()
