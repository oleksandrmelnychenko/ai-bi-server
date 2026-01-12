from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.schema_cache import SchemaCache


DIMENSION_SUFFIXES = (
    "Translation",
    "Type",
    "Status",
    "StatusName",
    "Category",
    "Group",
    "Role",
    "Permission",
    "Region",
    "RegionCode",
    "Country",
    "Currency",
    "MeasureUnit",
    "PriceType",
    "CalculationType",
    "Incoterm",
    "VatRate",
    "TransporterType",
    "TermsOfDelivery",
    "ChartMonth",
    "FilterItem",
    "FilterOperationItem",
    "AgreementType",
    "ClientType",
    "TaxAccountingScheme",
    "Number",
    "Numerator",
)

FACT_KEYWORDS = (
    "Sale",
    "Order",
    "Invoice",
    "Payment",
    "Supply",
    "Shipment",
    "Consignment",
    "Delivery",
    "Return",
    "TaxFree",
    "Act",
    "ReSale",
    "Transfer",
    "Income",
    "Outcome",
    "Packing",
    "BillOfLading",
)

SOFT_DELETE_FLAGS = {"isdeleted", "deleted"}
SOFT_DELETE_DATES = {"deletedat", "deletedon", "deleteddate", "deletedutc", "deletedtime"}
ROLE_OVERRIDES = {
    "dbo.Client": "dimension",
    "dbo.Product": "dimension",
    "dbo.User": "dimension",
}


def infer_role(table_name: str, fk_columns: set[str], columns: list[str], primary_keys: set[str]) -> str:
    name = table_name.split(".")[-1]

    if name.endswith(DIMENSION_SUFFIXES):
        return "dimension"

    non_key_columns = [col for col in columns if col not in fk_columns and col not in primary_keys]
    if len(columns) <= 6 and len(fk_columns) >= 2 and not non_key_columns:
        return "bridge"

    if any(keyword in name for keyword in FACT_KEYWORDS):
        return "fact"

    return "unknown"


def build_default_filters(columns: list[str]) -> list[str]:
    filters: list[str] = []
    for col in columns:
        lower = col.lower()
        if lower in SOFT_DELETE_FLAGS:
            filters.append(f"{col} = 0")
        elif lower in SOFT_DELETE_DATES:
            filters.append(f"{col} IS NULL")
    return filters


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply table roles and default filters to join_rules.yaml")
    parser.add_argument(
        "--path",
        default=r"C:\Users\123\AI-BI-Server\backend\schema\join_rules.yaml",
        help="Path to join_rules.yaml",
    )
    args = parser.parse_args()

    path = Path(args.path)
    updated = apply_table_rules_to_path(path)
    print(f"Applied table rules to {path} (updated {updated} tables)")


def apply_table_rules_to_path(path: Path) -> int:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    schema_cache = SchemaCache()
    schema_cache.load()

    fk_columns_by_table: dict[str, set[str]] = {}
    for fk in schema_cache.foreign_keys:
        fk_columns_by_table.setdefault(fk.fk_key, set()).add(fk.fk_column)

    updated = 0
    for table in payload.get("tables", []) or []:
        name = table.get("name")
        if not name:
            continue
        schema_table = schema_cache.table_info(name)
        if not schema_table:
            continue
        columns = [col.name for col in schema_table.columns]
        primary_keys = set(schema_table.primary_key)
        fk_cols = fk_columns_by_table.get(name, set())

        override_role = ROLE_OVERRIDES.get(name)
        if override_role:
            if table.get("role") != override_role:
                table["role"] = override_role
                updated += 1
        else:
            role = table.get("role") or "unknown"
            if role == "unknown":
                inferred = infer_role(name, fk_cols, columns, primary_keys)
                if inferred != role:
                    table["role"] = inferred
                    updated += 1

        default_filters = list(table.get("default_filters") or [])
        if not default_filters:
            inferred_filters = build_default_filters(columns)
            if inferred_filters:
                table["default_filters"] = inferred_filters
                updated += 1

    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    return updated


if __name__ == "__main__":
    main()
