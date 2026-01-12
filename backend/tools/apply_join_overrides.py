from __future__ import annotations

import argparse
from pathlib import Path
import yaml


AUDIT_USER_COLUMNS = {
    "CreatedByID",
    "UpdatedByID",
    "DeletedByID",
    "ResponsibleID",
    "ResponsibleId",
    "ResponsibleUserID",
    "OfferProcessingStatusChangedByID",
    "ChangedToInvoiceByID",
    "ChangedByID",
    "ProcessedByID",
    "CanceledByID",
    "UpdateUserID",
    "AddedByID",
    "LockedByID",
    "LastViewedByID",
    "DiscountUpdatedByID",
    "MoneyReturnedByID",
    "CommissionHeadID",
    "ManagerID",
    "MainManagerID",
    "ColleagueID",
    "DepreciatedToID",
}

DISABLED_USER_COLUMNS = {"DeletedByID"}

CLIENT_HIERARCHY_COLUMNS = {"MainClientID", "RootClientID", "SubClientID"}


def _is_history_table(name: str) -> bool:
    return name.endswith("History") or name.startswith("History")


def _other_side_columns(join: dict, table: str) -> list[str]:
    columns = join.get("columns", []) or []
    if join.get("left") == table:
        return [col.get("right") for col in columns if col.get("right")]
    return [col.get("left") for col in columns if col.get("left")]


def _bump_weight(join: dict, target: int) -> None:
    current = int(join.get("weight", 1) or 1)
    join["weight"] = max(current, target)


def apply_overrides(payload: dict) -> int:
    joins = payload.get("joins", []) or []
    changed = 0

    for join in joins:
        left = join.get("left")
        right = join.get("right")
        if not left or not right:
            continue

        # Self joins create ambiguous loops.
        if left == right:
            before = join.get("weight")
            _bump_weight(join, 7)
            if join.get("weight") != before:
                changed += 1

        # Penalize history tables as bridge nodes.
        if _is_history_table(left) or _is_history_table(right):
            before = join.get("weight")
            _bump_weight(join, 4)
            if join.get("weight") != before:
                changed += 1

        # User audit joins are ambiguous; deprioritize or disable.
        if left == "dbo.User" or right == "dbo.User":
            fk_cols = _other_side_columns(join, "dbo.User")
            if any(col in DISABLED_USER_COLUMNS for col in fk_cols):
                if join.get("enabled", True) is not False:
                    join["enabled"] = False
                    changed += 1
            elif any(col in AUDIT_USER_COLUMNS for col in fk_cols):
                before = join.get("weight")
                _bump_weight(join, 9)
                if join.get("weight") != before:
                    changed += 1
            else:
                before = join.get("weight")
                _bump_weight(join, 3)
                if join.get("weight") != before:
                    changed += 1

        # Client hierarchy joins are optional; penalize.
        if left == "dbo.Client" or right == "dbo.Client":
            fk_cols = _other_side_columns(join, "dbo.Client")
            if any(col in CLIENT_HIERARCHY_COLUMNS for col in fk_cols):
                before = join.get("weight")
                _bump_weight(join, 6)
                if join.get("weight") != before:
                    changed += 1

    return changed


def apply_overrides_to_path(path: Path) -> int:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    changed = apply_overrides(payload)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply manual join overrides to join_rules.yaml")
    parser.add_argument(
        "--path",
        default=r"C:\Users\123\AI-BI-Server\backend\schema\join_rules.yaml",
        help="Path to join_rules.yaml",
    )
    args = parser.parse_args()

    path = Path(args.path)
    changed = apply_overrides_to_path(path)
    print(f"Applied overrides to {path} (changed {changed} joins)")


if __name__ == "__main__":
    main()
