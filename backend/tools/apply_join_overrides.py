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

# Weight penalties for different join types
# Higher weight = less preferred in Dijkstra path finding
class JoinWeights:
    """
    Weight penalties applied to join edges.
    Higher values make the join less likely to be used.
    """
    SELF_JOIN = 7           # Self-referential joins create ambiguous loops
    HISTORY_TABLE = 4       # History tables shouldn't be bridge nodes
    AUDIT_USER = 9          # Audit columns (CreatedByID, etc.) are rarely needed for joins
    GENERIC_USER = 3        # Generic User joins, slightly deprioritized
    CLIENT_HIERARCHY = 6    # Client hierarchy columns (MainClientID, etc.)


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
            _bump_weight(join, JoinWeights.SELF_JOIN)
            if join.get("weight") != before:
                changed += 1

        # Penalize history tables as bridge nodes.
        if _is_history_table(left) or _is_history_table(right):
            before = join.get("weight")
            _bump_weight(join, JoinWeights.HISTORY_TABLE)
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
                _bump_weight(join, JoinWeights.AUDIT_USER)
                if join.get("weight") != before:
                    changed += 1
            else:
                before = join.get("weight")
                _bump_weight(join, JoinWeights.GENERIC_USER)
                if join.get("weight") != before:
                    changed += 1

        # Client hierarchy joins are optional; penalize.
        if left == "dbo.Client" or right == "dbo.Client":
            fk_cols = _other_side_columns(join, "dbo.Client")
            if any(col in CLIENT_HIERARCHY_COLUMNS for col in fk_cols):
                before = join.get("weight")
                _bump_weight(join, JoinWeights.CLIENT_HIERARCHY)
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
    # Use relative path from tools directory
    default_path = Path(__file__).resolve().parent.parent / "schema" / "join_rules.yaml"

    parser = argparse.ArgumentParser(description="Apply manual join overrides to join_rules.yaml")
    parser.add_argument(
        "--path",
        default=str(default_path),
        help="Path to join_rules.yaml",
    )
    args = parser.parse_args()

    path = Path(args.path)
    changed = apply_overrides_to_path(path)
    print(f"Applied overrides to {path} (changed {changed} joins)")


if __name__ == "__main__":
    main()
