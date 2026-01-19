# Concord Insight

A local BI chat assistant that turns Ukrainian questions into SQL Server queries using Ollama (Qwen) and returns grounded answers.

## What is included
- FastAPI backend with schema introspection, join-graph hints, SQL guardrails, and Ollama integration
- React + TypeScript chat UI
- Local-first setup for MS SQL Server + Ollama

## Architecture
- Backend: `backend/app`
  - Loads schema (tables, columns, PKs, FKs)
  - Selects tables via schema vectors + lexical matching (no LLM)
  - Builds join hints from foreign keys
  - Generates SQL (SELECT only) + enforces row limits
  - Executes SQL and summarizes results in Ukrainian
- Frontend: `frontend/src`
  - Chat UI with SQL and data preview panels

## Prerequisites
- SQL Server accessible at `localhost` with ODBC driver 17 or 18
- Ollama running locally (`http://localhost:11434`)
- Node 18+ and Python 3.10+

## Backend setup
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Update OLLAMA_MODEL based on `ollama list` if needed
uvicorn app.main:app --reload --port 8000
```

## Frontend setup
```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in a browser.

## API endpoints
- `GET /health`
- `GET /api/schema/summary`
- `POST /api/schema/refresh`
- `POST /api/chat`

## Join rules
- Auto-generated at `backend/schema/join_rules.yaml` from all foreign keys.
- Regenerate after schema changes:
```powershell
cd backend
python tools/refresh_join_rules.py
```
- Or run the steps manually:
```powershell
cd backend
python tools/export_join_rules.py
python tools/apply_join_overrides.py
python tools/apply_table_rules.py
```
- Manual overrides are applied to the full schema:
  - Audit user joins are penalized (`weight` 9).
  - Generic user joins use `weight` 3 to avoid accidental bridging.
  - Self-joins and client hierarchy joins are penalized.
  - `DeletedByID` joins are disabled by default.
- Table rules are inferred:
  - Soft-delete filters are added when `IsDeleted` or `DeletedAt/DeletedOn/DeletedDate` columns exist.
  - Roles are inferred (dimension/bridge/fact) and can be edited.
- Role overrides: `dbo.Client`, `dbo.Product`, and `dbo.User` are forced to `dimension`.
- You can edit `weight` or `enabled` to prefer or block join paths.

## Notes
- The backend blocks non-SELECT statements and enforces `TOP` limits.
- For production, use a read-only SQL user instead of `sa`.

## SQL examples
- Curated few-shot examples live in `backend/schema/sql_examples.yaml`.
- Extracted examples can be generated from:
  - Repo sources: `python -m tools.extract_gba_examples --repo C:\Users\123\RiderProjects\gba-server\src`
  - DB objects: `python -m tools.extract_db_examples`
- Runtime can load multiple extracted YAML files via `SQL_EXAMPLES_EXTRACTED_PATH` (comma-separated), e.g.
  `backend/schema/sql_examples_extracted.yaml,backend/schema/sql_examples_extracted_db.yaml`.
- Build a SQLite index (examples + schema metadata) for fast retrieval:
  `python -m tools.build_sql_index`
- Enable index usage at runtime with `SQL_INDEX_ENABLED=1`.
- Build schema vectors (tables/columns + optional YAML merge) for semantic table selection:
  `python -m tools.build_schema_vectors`
- If SQL joins look off, add more context to your question or create reporting views.
