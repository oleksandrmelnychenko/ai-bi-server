"""Schema management module.

Contains schema caching, join graph algorithms, and join rules.
"""

from .cache import ColumnInfo, ForeignKeyInfo, SchemaCache, TableInfo
from .join_graph import INFINITY, JoinEdge, build_adjacency, build_join_plan, edges_from_foreign_keys
from .join_rules import JoinColumnRule, JoinRule, JoinRules, TableRule, load_join_rules

__all__ = [
    "ColumnInfo",
    "ForeignKeyInfo",
    "SchemaCache",
    "TableInfo",
    "INFINITY",
    "JoinEdge",
    "build_adjacency",
    "build_join_plan",
    "edges_from_foreign_keys",
    "JoinColumnRule",
    "JoinRule",
    "JoinRules",
    "TableRule",
    "load_join_rules",
]
