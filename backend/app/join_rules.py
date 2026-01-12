from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any

import yaml

from .join_graph import JoinEdge


@dataclass(frozen=True)
class JoinColumnRule:
    left: str
    right: str


@dataclass(frozen=True)
class JoinRule:
    name: str
    left: str
    right: str
    columns: tuple[JoinColumnRule, ...]
    weight: int = 1
    enabled: bool = True


@dataclass(frozen=True)
class TableRule:
    name: str
    primary_key: tuple[str, ...] = ()
    role: str = "unknown"
    default_filters: tuple[str, ...] = ()


@dataclass
class JoinRules:
    tables: dict[str, TableRule] = field(default_factory=dict)
    joins: list[JoinRule] = field(default_factory=list)

    def to_edges(self) -> list[JoinEdge]:
        edges: list[JoinEdge] = []
        for join in self.joins:
            if not join.enabled:
                continue
            columns = tuple((col.left, col.right) for col in join.columns)
            edges.append(
                JoinEdge(
                    left_table=join.left,
                    right_table=join.right,
                    columns=columns,
                    weight=join.weight,
                    name=join.name,
                )
            )
        return edges


def _load_payload(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        if path.lower().endswith(".json"):
            return json.load(handle)
        return yaml.safe_load(handle) or {}


def load_join_rules(path: str | None) -> JoinRules:
    if not path:
        return JoinRules()
    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        return JoinRules()

    payload = _load_payload(resolved)
    tables: dict[str, TableRule] = {}
    for item in payload.get("tables", []) or []:
        name = item.get("name")
        if not name:
            continue
        primary_key = tuple(item.get("primary_key", []) or [])
        role = item.get("role") or "unknown"
        default_filters = tuple(item.get("default_filters", []) or [])
        tables[name] = TableRule(
            name=name,
            primary_key=primary_key,
            role=role,
            default_filters=default_filters,
        )

    joins: list[JoinRule] = []
    for item in payload.get("joins", []) or []:
        left = item.get("left")
        right = item.get("right")
        if not left or not right:
            continue
        name = item.get("name") or f"{left}->{right}"
        columns_payload = item.get("columns", []) or []
        columns = tuple(
            JoinColumnRule(left=col.get("left"), right=col.get("right"))
            for col in columns_payload
            if col.get("left") and col.get("right")
        )
        if not columns:
            continue
        weight = int(item.get("weight", 1) or 1)
        enabled = bool(item.get("enabled", True))
        joins.append(
            JoinRule(
                name=name,
                left=left,
                right=right,
                columns=columns,
                weight=weight,
                enabled=enabled,
            )
        )

    return JoinRules(tables=tables, joins=joins)
