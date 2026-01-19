from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Iterable

from .cache import ForeignKeyInfo

# Infinity constant for Dijkstra's algorithm (unreachable nodes)
INFINITY = 1_000_000_000


@dataclass(frozen=True)
class JoinEdge:
    left_table: str
    right_table: str
    columns: tuple[tuple[str, str], ...]
    weight: int = 1
    name: str | None = None


def edges_from_foreign_keys(foreign_keys: list[ForeignKeyInfo]) -> list[JoinEdge]:
    grouped: dict[tuple[str, str, str, str, str], list[tuple[str, str]]] = {}
    for fk in foreign_keys:
        key = (fk.name, fk.fk_schema, fk.fk_table, fk.pk_schema, fk.pk_table)
        grouped.setdefault(key, []).append((fk.fk_column, fk.pk_column))

    edges: list[JoinEdge] = []
    for (name, fk_schema, fk_table, pk_schema, pk_table), columns in sorted(grouped.items()):
        edges.append(
            JoinEdge(
                left_table=f"{fk_schema}.{fk_table}",
                right_table=f"{pk_schema}.{pk_table}",
                columns=tuple(columns),
                weight=1,
                name=name,
            )
        )
    return edges


def build_adjacency(edges: Iterable[JoinEdge]) -> dict[str, list[JoinEdge]]:
    adjacency: dict[str, list[JoinEdge]] = {}
    for edge in edges:
        adjacency.setdefault(edge.left_table, []).append(edge)
        reverse_columns = tuple((right, left) for left, right in edge.columns)
        adjacency.setdefault(edge.right_table, []).append(
            JoinEdge(
                left_table=edge.right_table,
                right_table=edge.left_table,
                columns=reverse_columns,
                weight=edge.weight,
                name=edge.name,
            )
        )
    return adjacency


def _find_path(adjacency: dict[str, list[JoinEdge]], start: str, goal: str) -> tuple[list[JoinEdge], int]:
    if start == goal:
        return [], 0

    distances: dict[str, int] = {start: 0}
    prev: dict[str, str] = {}
    prev_edge: dict[str, JoinEdge] = {}
    heap: list[tuple[int, str]] = [(0, start)]

    while heap:
        cost, current = heapq.heappop(heap)
        if current == goal:
            break
        if cost != distances.get(current, 0):
            continue
        for edge in adjacency.get(current, []):
            next_table = edge.right_table
            edge_cost = max(edge.weight, 1)
            next_cost = cost + edge_cost
            if next_cost < distances.get(next_table, INFINITY):
                distances[next_table] = next_cost
                prev[next_table] = current
                prev_edge[next_table] = edge
                heapq.heappush(heap, (next_cost, next_table))

    if goal not in distances:
        return [], INFINITY

    path: list[JoinEdge] = []
    cursor = goal
    while cursor != start:
        edge = prev_edge[cursor]
        path.append(edge)
        cursor = prev[cursor]
    path.reverse()
    return path, distances[goal]


def build_join_plan(adjacency: dict[str, list[JoinEdge]], table_keys: list[str]) -> tuple[list[JoinEdge], list[str]]:
    if not table_keys:
        return [], []
    root = table_keys[0]
    joined = {root}
    plan_edges: list[JoinEdge] = []
    missing: list[str] = []

    for target in table_keys[1:]:
        if target in joined:
            continue
        best_path: list[JoinEdge] | None = None
        best_cost = INFINITY
        for source in list(joined):
            path, cost = _find_path(adjacency, source, target)
            if not path:
                continue
            if cost < best_cost:
                best_path = path
                best_cost = cost
        if not best_path:
            missing.append(target)
            continue
        for edge in best_path:
            if edge not in plan_edges:
                plan_edges.append(edge)
            joined.add(edge.left_table)
            joined.add(edge.right_table)

    return plan_edges, missing
