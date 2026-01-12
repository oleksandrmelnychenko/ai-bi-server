from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx

from .config import get_settings
from .db import fetch_rows
from .join_graph import build_adjacency, build_join_plan, edges_from_foreign_keys
from .join_rules import JoinRules, load_join_rules
from .llm import compose_answer, generate_sql, select_tables
from .models import ChatRequest, ChatResponse
from .schema_cache import SchemaCache
from .sql_guard import apply_row_limit, extract_sql, is_safe_sql

app = FastAPI(title="Concord BI Chat", version="0.1.0")
settings = get_settings()
schema_cache = SchemaCache()
join_rules = JoinRules()

cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_schema() -> None:
    global join_rules
    schema_cache.load()
    join_rules = load_join_rules(settings.join_rules_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/schema/summary")
def schema_summary() -> dict[str, int | str]:
    return {
        "tables": len(schema_cache.tables),
        "foreign_keys": len(schema_cache.foreign_keys),
        "join_rules": len(join_rules.joins),
        "loaded_at": schema_cache.loaded_at.isoformat() if schema_cache.loaded_at else "",
    }


@app.post("/api/schema/refresh")
def schema_refresh() -> dict[str, int | str]:
    global join_rules
    schema_cache.load()
    join_rules = load_join_rules(settings.join_rules_path)
    return schema_summary()


def _format_table_details(table_keys: list[str]) -> str:
    lines: list[str] = []
    for key in table_keys:
        table = schema_cache.table_info(key)
        if not table:
            continue
        table_rule = join_rules.tables.get(key)
        role = table_rule.role if table_rule else "unknown"
        default_filters = ", ".join(table_rule.default_filters) if table_rule else ""
        filters = default_filters if default_filters else "none"
        columns = ", ".join(f"{col.name} ({col.data_type})" for col in table.columns)
        primary_key = ", ".join(table.primary_key) if table.primary_key else "none"
        lines.append(
            f"{table.key}: role: {role}; default_filters: {filters}; columns: {columns}; primary_key: {primary_key}"
        )
    return "\n".join(lines)


def _format_join_hints(edges: list) -> str:
    if not edges:
        return "(no join hints available)"
    return "\n".join(
        " AND ".join(
            f\"{edge.left_table}.{left} = {edge.right_table}.{right}\" for left, right in edge.columns
        )
        for edge in edges
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")

    if not schema_cache.tables:
        schema_cache.load()

    try:
        selection = select_tables(request.message, schema_cache.table_keys())
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    if selection.need_clarification:
        return ChatResponse(answer=selection.clarifying_question, warnings=["clarification"])

    if not selection.tables:
        return ChatResponse(
            answer="Unable to determine required tables. Please clarify your request.",
            warnings=["no_tables"],
        )

    available_edges = join_rules.to_edges() if join_rules.joins else edges_from_foreign_keys(schema_cache.foreign_keys)
    adjacency = build_adjacency(available_edges)
    join_edges, missing = build_join_plan(adjacency, selection.tables)
    table_details = _format_table_details(selection.tables)
    join_hints = _format_join_hints(join_edges)

    try:
        sql_raw = generate_sql(request.message, table_details, join_hints, settings.max_rows)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    sql = extract_sql(sql_raw)
    sql = apply_row_limit(sql, settings.max_rows)
    ok, reason = is_safe_sql(sql)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    columns, rows = fetch_rows(sql, max_rows=settings.max_rows)

    try:
        answer = compose_answer(request.message, sql, columns, rows)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    warnings: list[str] = []
    if missing:
        warnings.append(f"missing_join_path: {', '.join(missing)}")

    return ChatResponse(answer=answer, sql=sql, columns=columns, rows=rows, warnings=warnings)
