from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .db import get_connection


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool
    max_length: int | None


@dataclass
class TableInfo:
    schema: str
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.schema}.{self.name}"


@dataclass
class ForeignKeyInfo:
    fk_schema: str
    fk_table: str
    fk_column: str
    pk_schema: str
    pk_table: str
    pk_column: str
    name: str

    @property
    def fk_key(self) -> str:
        return f"{self.fk_schema}.{self.fk_table}"

    @property
    def pk_key(self) -> str:
        return f"{self.pk_schema}.{self.pk_table}"


class SchemaCache:
    def __init__(self) -> None:
        self.tables: dict[str, TableInfo] = {}
        self.foreign_keys: list[ForeignKeyInfo] = []
        self.loaded_at: datetime | None = None

    def load(self) -> None:
        self.tables.clear()
        self.foreign_keys.clear()
        with get_connection() as conn:
            cursor = conn.cursor()
            # Combined query: tables + columns + primary keys in a single round trip
            cursor.execute(
                """
                SELECT
                    s.name AS schema_name,
                    t.name AS table_name,
                    c.name AS column_name,
                    ty.name AS data_type,
                    c.max_length,
                    c.is_nullable,
                    CASE WHEN pk_cols.column_id IS NOT NULL THEN 1 ELSE 0 END AS is_primary_key,
                    pk_cols.key_ordinal
                FROM sys.tables t
                JOIN sys.schemas s ON t.schema_id = s.schema_id
                JOIN sys.columns c ON c.object_id = t.object_id
                JOIN sys.types ty ON c.user_type_id = ty.user_type_id
                LEFT JOIN (
                    SELECT ic.object_id, ic.column_id, ic.key_ordinal
                    FROM sys.indexes i
                    JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
                    WHERE i.is_primary_key = 1
                ) pk_cols ON pk_cols.object_id = t.object_id AND pk_cols.column_id = c.column_id
                ORDER BY s.name, t.name, c.column_id;
                """
            )
            # Track primary key columns to add in correct order after all columns processed
            pk_columns: dict[str, list[tuple[str, int]]] = {}

            for row in cursor.fetchall():
                schema_name, table_name, column_name, data_type, max_length, is_nullable, is_pk, pk_ordinal = row
                key = f"{schema_name}.{table_name}"

                # Create table if not exists
                if key not in self.tables:
                    self.tables[key] = TableInfo(schema=schema_name, name=table_name)

                # Add column
                self.tables[key].columns.append(
                    ColumnInfo(
                        name=column_name,
                        data_type=data_type,
                        nullable=bool(is_nullable),
                        max_length=max_length if max_length is not None else None,
                    )
                )

                # Track primary key columns with their ordinal for correct ordering
                if is_pk:
                    pk_columns.setdefault(key, []).append((column_name, pk_ordinal or 0))

            # Add primary keys in correct order
            for key, pk_list in pk_columns.items():
                pk_list.sort(key=lambda x: x[1])
                self.tables[key].primary_key = [col for col, _ in pk_list]

            # Foreign keys query (separate due to different structure)
            cursor.execute(
                """
                SELECT sch1.name AS fk_schema,
                       t1.name AS fk_table,
                       c1.name AS fk_column,
                       sch2.name AS pk_schema,
                       t2.name AS pk_table,
                       c2.name AS pk_column,
                       fk.name AS fk_name
                FROM sys.foreign_key_columns fkc
                JOIN sys.foreign_keys fk ON fkc.constraint_object_id = fk.object_id
                JOIN sys.tables t1 ON fkc.parent_object_id = t1.object_id
                JOIN sys.schemas sch1 ON t1.schema_id = sch1.schema_id
                JOIN sys.columns c1 ON fkc.parent_object_id = c1.object_id AND fkc.parent_column_id = c1.column_id
                JOIN sys.tables t2 ON fkc.referenced_object_id = t2.object_id
                JOIN sys.schemas sch2 ON t2.schema_id = sch2.schema_id
                JOIN sys.columns c2 ON fkc.referenced_object_id = c2.object_id AND fkc.referenced_column_id = c2.column_id
                ORDER BY sch1.name, t1.name, fk.name, fkc.constraint_column_id;
                """
            )
            for fk_schema, fk_table, fk_column, pk_schema, pk_table, pk_column, fk_name in cursor.fetchall():
                self.foreign_keys.append(
                    ForeignKeyInfo(
                        fk_schema=fk_schema,
                        fk_table=fk_table,
                        fk_column=fk_column,
                        pk_schema=pk_schema,
                        pk_table=pk_table,
                        pk_column=pk_column,
                        name=fk_name,
                    )
                )

        self.loaded_at = datetime.utcnow()

    def table_keys(self) -> list[str]:
        return sorted(self.tables.keys())

    def table_info(self, key: str) -> TableInfo | None:
        return self.tables.get(key)

    def foreign_key_edges(self) -> list[ForeignKeyInfo]:
        return list(self.foreign_keys)
