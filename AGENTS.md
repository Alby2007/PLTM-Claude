# PLTM — Working Notes (Windows)

## Branch layout
- `main` — public repost. 6 core modules ship as `*.cpython-311-darwin.so` (macOS-only Cython binaries). **Cannot run on Windows/Linux.** Shielded: `src/memory/{memory_jury,memory_pipeline,phi_rms,session_continuity}.py`, `src/analysis/{epistemic_v2,pltm_self}.py` (sources gitignored; rebuild via `build_shield.py`).
- `dev-with-source` — `main` + commits with all real `.py` sources. **Develop here.**

## Setup (Windows)
`requirements.txt` is now a clean working set (the original macOS freeze dump is kept as `requirements-macos-freeze.txt`):

```powershell
uv venv .venv --python 3.11 --seed
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

- **Pin `mcp==1.30.0`** — mcp 2.x removed the `@app.list_tools()`/`@app.call_tool()` decorators this server uses.
- Embedding model `all-MiniLM-L6-v2` downloads to `~/.cache/huggingface` on first semantic call (~90MB); first call is slow, cached after.
- `GROQ_API_KEY` in `.env` enables LLM-backed tools (ingestion, fact-check); core memory works without it.

## Verify

```powershell
# Full suite (462 tests, ~30s). test_vector_store.py is excluded — dead
# pgvector path needing asyncpg; project moved to sqlite-vec.
.venv/Scripts/python.exe -m pytest tests/ -q --timeout 120 --ignore=tests/unit/test_vector_store.py

# Server smoke (module-level list_tools/call_tool functions):
.venv/Scripts/python.exe -c "import asyncio, mcp_server.pltm_server as s; asyncio.run(s.initialize_pltm()); print(len(asyncio.run(s.list_tools())))"
```

Real data lives in `data/pltm_mcp.db` (user_id is `claude`, ~1.9k memories). It's committed to git — don't push personal data.

## Architecture notes (post-debug, Oct 2026)

- Two memory systems coexist: legacy **atom graph** (`src/core`, `src/storage/sqlite_store.py`, pipeline+jury+reconciliation) and **typed memory** (`src/memory/memory_types.py` — episodic/semantic/belief/procedural).
- Jury approval ≠ promotion: `WriteLane` promotes approved atoms to `SUBSTANTIATED`. `deliberate_batch` returns 3 lists `(approved, rejected, quarantined)`.
- `atoms` table has `UNIQUE(subject, predicate, object, graph)` — identical triples are reassertions (upsert), not duplicate rows.
- Extraction is clause-aware multi-atom (splits on `.!?;,` + ` and | but | because`, never inside version numbers/URLs/decimals).
- Conflict detection flags potential conflicts; the **resolver** decides supersede/contextualize/coexist. Non-exclusive different objects are not conflicts; `located_at`/`lives_in` are exclusive-equivalent.
- `auto_prune` compares `decayed_strength()` (no floor) — `current_strength()` clamps at type floors.
- Internal `TypedMemoryStore.store()` calls all pass `bypass_jury=True`; only user-facing writes go through the jury.
- Emergence detector is wired to real `SelfOrganizedCriticality`, `PhiIntegrationCalculator`, `KnowledgeNetworkGraph` (registry: `_reg.criticality/phi_calc/knowledge_graph`).
- Handlers resolve DB via `_get_db_path()` / `db_config.set_db_path()` — both honor `initialize_pltm(custom_db_path=...)`.
- Sensory: camera/mic emit abbreviated keys (`ts`, `speech`, `text`, `conf`); `SensoryMemoryIntegrator` accepts both schemas. `sensory_look`/`sensory_listen` persist meaningful observations to atoms.
- `epistemic_monitor` tables are created lazily on first `EpistemicMonitor()` — import is side-effect-free.

## Remaining known issues
- `tests/unit/test_vector_store.py` — dead Postgres/pgvector path (`asyncpg` not installed).
- `src/memory/hierarchical_memory.py` has a hardcoded 7-domain keyword taxonomy baked around the original author's interests.
- Dead code removed (Oct 2026): `src/api/`, `src/auth/`, `src/middleware/`, `src/monitoring/`, `src/workers/`, `src/observability/`, `src/provenance/`, `src/storage/{neo4j_store,redis_cache}.py`, `src/reconciliation/semantic_*.py` (3 benchmark experiments), `src/agents/{adaptive_prompts,lifelong_learning_agent,multi_agent_workspace}.py`. Recover via git if needed.
- **`database is locked` gotcha**: sync `sqlite3.connect` writers (ToolAnalytics, epistemic monitor, snapshots) run inside the event loop — now fail fast (timeout ≤2s). If you see persistent lock failures, check for zombie `python.exe` processes holding the DB before suspecting code.
- `contextual_retrieve` requires `messages: string[]`, `capture_session_state` requires `session_id`.
