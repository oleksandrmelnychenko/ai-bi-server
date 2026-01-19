# Vector-Based SQL Example Retrieval System

## Problem Statement

Current keyword-based classification has limitations:
- Misses semantic intent ("Хто найбільше купує?" doesn't match any keyword)
- Falls back to generic examples too often
- Manual curation doesn't scale (only 27 examples with questions)
- 78 out of 129 extracted SQL queries are truncated/incomplete

**Goal**: Build a semantic search system that finds relevant SQL examples based on the meaning of user questions, not just keyword matching.

## Data Sources

### GBA Repository Statistics
- **648 Repository files** (`*Repository.cs`)
- **~8,000+ SELECT statements**
- **~324 CTE queries** (complex patterns)
- Rich variety of join patterns, aggregations, UDF calls

### Current Extraction Issues
1. String concatenation (`"SELECT " + "FROM "`) not properly joined
2. Multi-line strings partially captured
3. No method context preserved with SQL
4. Missing question/intent descriptions

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OFFLINE PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │ GBA Repository  │───→│ SQL Extractor   │───→│ SQL Processor   │        │
│  │ (.cs files)     │    │ (complete SQL)  │    │ (parse/validate)│        │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘        │
│                                                          │                  │
│                                                          ▼                  │
│                         ┌─────────────────────────────────────────┐        │
│                         │           SQL Knowledge Base            │        │
│                         │  ┌─────────────────────────────────┐   │        │
│                         │  │ sql_id: "abc123"                │   │        │
│                         │  │ sql: "SELECT [Client]..."       │   │        │
│                         │  │ source: "ClientRepository.cs"   │   │        │
│                         │  │ method: "GetAllFiltered"        │   │        │
│                         │  │ tables: ["Client", "Agreement"] │   │        │
│                         │  │ pattern: "cte_pagination"       │   │        │
│                         │  │ description: "Get filtered..."  │   │        │
│                         │  └─────────────────────────────────┘   │        │
│                         └───────────────────┬─────────────────────┘        │
│                                             │                               │
│                                             ▼                               │
│                         ┌─────────────────────────────────────────┐        │
│                         │        Embedding Generator              │        │
│                         │   (multilingual-e5-base / Ollama)       │        │
│                         └───────────────────┬─────────────────────┘        │
│                                             │                               │
│                                             ▼                               │
│                         ┌─────────────────────────────────────────┐        │
│                         │         Vector Index (SQLite)           │        │
│                         │  - sql_id                               │        │
│                         │  - embedding (768 dim)                  │        │
│                         │  - metadata (tables, pattern, etc.)     │        │
│                         └─────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           ONLINE RETRIEVAL                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Question                                                              │
│  "Покажи топ 10 клієнтів з найбільшим боргом"                             │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────┐                                                       │
│  │ Embed Question  │  (same model as offline)                              │
│  └────────┬────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────┐                               │
│  │     Vector Similarity Search            │                               │
│  │     (cosine similarity, top-50)         │                               │
│  └────────┬────────────────────────────────┘                               │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────┐                               │
│  │     Re-rank by Table Overlap            │                               │
│  │     (boost if tables match schema)      │                               │
│  └────────┬────────────────────────────────┘                               │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────┐                               │
│  │     Select Top-K Examples (5-10)        │                               │
│  │     (diverse patterns)                  │                               │
│  └────────┬────────────────────────────────┘                               │
│           │                                                                 │
│           ▼                                                                 │
│  Few-shot examples for LLM prompt                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Embedding Model Options

| Model | Size | Ukrainian Support | Speed | Notes |
|-------|------|-------------------|-------|-------|
| `intfloat/multilingual-e5-base` | 278M | Excellent | Fast | Recommended |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118M | Good | Faster | Lighter alternative |
| Ollama (local) | varies | Depends | Slower | No external dependency |

**Recommendation**: Use `multilingual-e5-base` with `sentence-transformers` library. It handles Ukrainian well and produces high-quality embeddings.

### 2. What to Embed

For each SQL query, create a searchable text:
```
Source: ClientRepository.cs
Method: GetAllFiltered
Pattern: CTE with pagination and debt calculation
Tables: Client, ClientAgreement, Agreement, Debt, ClientInDebt
Description: Get filtered list of clients with their total debt in EUR
```

This text captures:
- **Context** (where the SQL comes from)
- **Intent** (what it does)
- **Structure** (tables involved)

### 3. Storage Format

SQLite with numpy-serialized vectors (simple, no extra dependencies):

```sql
CREATE TABLE sql_vectors (
    id TEXT PRIMARY KEY,
    sql TEXT NOT NULL,
    source_file TEXT,
    method_name TEXT,
    tables TEXT,  -- JSON array
    pattern TEXT,
    description TEXT,  -- Auto-generated or manual
    embedding BLOB,  -- numpy array serialized
    created_at TIMESTAMP
);

CREATE INDEX idx_pattern ON sql_vectors(pattern);
```

### 4. Similarity Search

For SQLite, use brute-force cosine similarity (fast enough for <10K vectors):
```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_similar(query_embedding, top_k=50):
    # Load all embeddings (cached in memory)
    # Compute similarities
    # Return top-k
```

For larger scale, consider:
- **ChromaDB** - Simple vector DB, Python-native
- **Qdrant** - More features, REST API
- **FAISS** - Facebook's library, very fast

## Implementation Plan

### Phase 1: Improved SQL Extraction (1 day)
1. Fix string concatenation handling in `extract_gba_examples.py`
2. Extract complete SQL queries with full context
3. Parse SQL to extract tables reliably
4. Store in enhanced YAML format

**Output**: `sql_examples_complete.yaml` with 2000+ queries

### Phase 2: Auto-Description Generation (0.5 day)
1. Use LLM to generate descriptions for each SQL
2. Create searchable text combining method name + tables + description
3. Manual review of top 100 most common patterns

**Output**: SQL knowledge base with descriptions

### Phase 3: Embedding Pipeline (0.5 day)
1. Install `sentence-transformers`
2. Generate embeddings for all SQL descriptions
3. Store in SQLite with vector blob
4. Build simple retrieval function

**Output**: `sql_vectors.sqlite` with embeddings

### Phase 4: Integration (0.5 day)
1. Replace `get_relevant_examples()` with vector search
2. Add table-overlap re-ranking
3. Test with various Ukrainian questions
4. Benchmark latency and quality

**Output**: Working vector-based retrieval

### Phase 5: Evaluation & Tuning (ongoing)
1. Log queries and retrieved examples
2. Measure SQL generation success rate
3. Fine-tune retrieval parameters
4. Add feedback loop for improvement

## File Structure

```
backend/
├── app/
│   ├── sql_examples.py      # Current (keep for fallback)
│   └── sql_vectors.py       # NEW: Vector-based retrieval
├── tools/
│   ├── extract_gba_examples.py  # Improve extraction
│   ├── build_sql_index.py       # Current index builder
│   ├── build_vector_index.py    # NEW: Vector index builder
│   └── generate_descriptions.py # NEW: LLM description generator
├── schema/
│   ├── sql_examples.yaml         # Curated examples
│   ├── sql_examples_extracted.yaml  # Current extracted
│   ├── sql_knowledge_base.yaml   # NEW: Complete extraction
│   └── sql_vectors.sqlite        # NEW: Vector index
```

## Dependencies

```
# requirements.txt additions
sentence-transformers>=2.2.0  # For embeddings
numpy>=1.24.0                 # Vector operations
```

No external vector DB needed - SQLite handles everything.

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Embedding model too large | Use MiniLM variant (118MB) |
| Slow retrieval | Cache embeddings in memory on startup |
| Poor Ukrainian understanding | Test with e5-multilingual, fallback to keyword |
| Too many similar results | Add diversity sampling in top-K selection |

## Success Metrics

1. **Retrieval Quality**: Relevant examples in top-5 for 80%+ of questions
2. **Latency**: <200ms for retrieval (including embedding)
3. **Coverage**: 2000+ unique SQL patterns indexed
4. **SQL Generation**: 10% improvement in valid SQL rate

## Next Steps

1. [ ] Decide on embedding model
2. [ ] Fix SQL extraction to get complete queries
3. [ ] Generate descriptions (LLM or template-based)
4. [ ] Build vector index
5. [ ] Integrate with `generate_sql()` flow
