# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Concord Insight is a local BI chat assistant that converts Ukrainian-language questions into SQL Server queries using Ollama (Qwen model) and returns grounded answers. It's designed for local-first operation with MS SQL Server and Ollama.

## Development Commands

### Backend (FastAPI + Python)
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend (React + TypeScript + Vite)
```powershell
cd frontend
npm install
npm run dev        # Development server at http://localhost:5173
npm run build      # Production build
```

### Schema Management Tools
```powershell
cd backend
python -m tools.refresh_join_rules        # Full refresh: export + overrides + table rules
python -m tools.export_join_rules         # Export FKs from SQL Server to YAML
python -m tools.apply_join_overrides      # Apply weight penalties to join_rules.yaml
python -m tools.apply_table_rules         # Infer table roles and default filters
```

## Architecture

### Request Flow
1. User sends Ukrainian question via `/api/chat`
2. `select_tables()` in `llm.py` asks Ollama which tables are needed
3. `build_join_plan()` in `join_graph.py` computes join paths using Dijkstra's algorithm over weighted FK graph
4. `generate_sql()` creates a SELECT query with schema/join context
5. `sql_guard.py` validates query (SELECT-only, no multiple statements, applies TOP limit)
6. `db.py` executes query and returns results
7. `compose_answer()` generates Ukrainian summary of results

### Backend Modules (`backend/app/`)
- **main.py**: FastAPI app, routes, orchestrates the chat flow
- **schema_cache.py**: Loads and caches SQL Server metadata (tables, columns, PKs, FKs) from `sys.*` views
- **join_graph.py**: Builds weighted adjacency graph, finds shortest join paths via Dijkstra
- **join_rules.py**: Parses YAML/JSON join rules into typed dataclasses (JoinRule, TableRule)
- **llm.py**: Ollama API calls for table selection, SQL generation, answer composition
- **sql_guard.py**: SQL validation (blocks non-SELECT), extracts SQL from markdown, enforces row limits
- **prompts.py**: System prompts for each LLM call (table selection, SQL generation, answer)
- **config.py**: Environment-based settings loaded via python-dotenv
- **db.py**: pyodbc connection and query execution

### Backend Tools (`backend/tools/`)
- **export_join_rules.py**: Exports FK relationships from SQL Server to `schema/join_rules.yaml`
- **apply_join_overrides.py**: Applies weight penalties based on domain heuristics:
  - Self-joins: weight 7
  - History tables: weight 4
  - Audit user columns (CreatedByID, UpdatedByID, etc.): weight 9
  - Client hierarchy columns (MainClientID, RootClientID): weight 6
  - DeletedByID joins: disabled entirely
- **apply_table_rules.py**: Infers table roles and soft-delete filters:
  - `dimension`: Tables ending in Type, Status, Category, Translation, etc.
  - `fact`: Tables containing Sale, Order, Invoice, Payment, Shipment, etc.
  - `bridge`: Small tables with mostly FK columns
  - Overrides: dbo.Client, dbo.Product, dbo.User always dimension
  - Detects `Deleted = 0` / `DeletedAt IS NULL` filters
- **refresh_join_rules.py**: Runs export → apply_join_overrides → apply_table_rules in sequence

### Join Rules System (`backend/schema/join_rules.yaml`)
YAML config loaded at startup containing:
- **tables**: name, primary_key, role (fact/dimension/bridge/unknown), default_filters
- **joins**: left/right tables, column mappings, weight (higher = less preferred), enabled flag

The weighted graph allows fine-tuning which join paths the LLM sees, avoiding ambiguous audit/history relationships.

### Frontend (`frontend/src/`)
- Single-page React app with chat interface
- Displays SQL queries and result tables in expandable details
- Sends message + last 6 messages as history to backend
- Vite proxies `/api` requests to backend at port 8000

## Key Design Decisions

- **SELECT-only guardrail**: `sql_guard.py` blocks INSERT/UPDATE/DELETE/DROP etc. via regex
- **Automatic TOP limit**: Queries without TOP/FETCH get `TOP (max_rows)` injected
- **Weighted join hints**: LLM receives join conditions from lowest-weight paths via Dijkstra
- **Ukrainian responses**: All LLM answer prompts specify Ukrainian output
- **Soft-delete awareness**: Default filters like `Deleted = 0` are passed to LLM with table details

## Environment Configuration

Copy `backend/.env.example` to `backend/.env` and adjust:
- `DB_*`: SQL Server connection (default uses ODBC Driver 17)
- `OLLAMA_MODEL`: Model name from `ollama list` (default: qwen2.5-coder:30b)
- `MAX_ROWS`: Query result limit (default: 200)
- `REQUEST_TIMEOUT`: Ollama request timeout in seconds (default: 90)
- `JOIN_RULES_PATH`: Path to join_rules.yaml (default: backend/schema/join_rules.yaml)

## API Endpoints

- `GET /health` - Health check
- `GET /api/schema/summary` - Table/FK/join rule counts and load timestamp
- `POST /api/schema/refresh` - Reload schema and join rules from database/YAML
- `POST /api/chat` - Main chat endpoint (accepts `message` and `history`)
