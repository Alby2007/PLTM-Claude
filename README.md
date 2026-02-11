# PLTM — Persistent Long-Term Memory for Claude

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP Tools](https://img.shields.io/badge/MCP_tools-136-brightgreen.svg)](#tool-categories)
[![Tests](https://img.shields.io/badge/tests-11%20passing-brightgreen.svg)](#testing)

> 136 MCP tools · 1,600+ atoms + 1,650+ typed memories · Semantic embeddings · Memory jury · Epistemic self-monitoring · React dashboard

An MCP server that gives Claude Desktop persistent memory, self-awareness, epistemic hygiene, and genuine agency across conversations — with a typed memory system, embedding-based semantic search, a 3-judge memory jury, and a real-time dashboard.

---

## Install — One Command

**macOS / Linux:**
```bash
curl -fsSL https://raw.githubusercontent.com/Alby2007/PLTM-Claude/main/install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/Alby2007/PLTM-Claude/main/install.ps1 | iex
```

**Then restart Claude Desktop** — 136 tools ready.

That's it. The installer clones the repo, creates a venv, installs deps, downloads the embedding model, initializes the database, and auto-configures Claude Desktop. No manual JSON editing.

> **Optional:** Add a free [Groq API key](https://console.groq.com) to `~/PLTM/.env` for LLM-powered tools (ingestion, fact-checking). Core memory tools work without it.

**Verify** — ask Claude: `Use auto_init_session to check system state`

**Diagnose issues:** `python ~/PLTM/health_check.py`

<details>
<summary><strong>Alternative: manual clone + setup</strong></summary>

```bash
git clone https://github.com/Alby2007/PLTM-Claude.git && cd PLTM-Claude
python setup_pltm.py
```

The setup script handles everything: venv, deps, .env, DB, model, and Claude Desktop config.

Flags:
- `--skip-claude` — skip Claude Desktop auto-config
- `--skip-model` — skip embedding model download (faster)
- `--reset` — delete venv + DB and start fresh
- `--uninstall` — remove PLTM from Claude Desktop config

</details>

<details>
<summary><strong>Fully manual setup</strong></summary>

```bash
git clone https://github.com/Alby2007/PLTM-Claude.git
cd PLTM-Claude
python3.11 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements-lite.txt
cp .env.example .env             # edit to add GROQ_API_KEY
```

Then edit your Claude Desktop config:

| OS | Path |
|----|------|
| **macOS** | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Windows** | `%APPDATA%\Claude\claude_desktop_config.json` |
| **Linux** | `~/.config/Claude/claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "pltm": {
      "command": "/path/to/PLTM-Claude/.venv/bin/python3",
      "args": ["-m", "mcp_server.pltm_server"],
      "env": {
        "PYTHONPATH": "/path/to/PLTM-Claude",
        "GROQ_API_KEY": "your-groq-key"
      }
    }
  }
}
```

Restart Claude Desktop.

</details>

---

## What This Does

PLTM turns Claude from a stateless chatbot into a **persistent entity** with:

- **Typed Memory** — 1,650+ memories across 4 types (episodic, semantic, belief, procedural) with strength decay, confidence tracking, and automatic consolidation
- **Semantic Search** — Embedding-based similarity search using `all-MiniLM-L6-v2` (384-dim vectors), plus full-text search via SQLite FTS5
- **Memory Jury** — 3-judge validation gate (Relevance, Novelty, Accuracy) that filters, quarantines, or rejects incoming memories before storage
- **Memory Intelligence** — Decay engine, consolidation, clustering, conflict detection, importance ranking, contextual retrieval, and provenance tracking
- **Knowledge Graph** — 1,600+ semantic atoms stored as subject-predicate-object triples with attention-weighted retrieval
- **Identity** — Communication style, curiosity patterns, value boundaries, and reasoning habits tracked across sessions
- **Epistemic Hygiene** — Confidence calibration, claim logging, confabulation detection, and verification suggestions
- **Goals** — Persistent goals that survive across conversations with progress tracking
- **Continuity** — Session bridging so Claude picks up where it left off
- **Dashboard** — React-based real-time dashboard with memory intelligence visualizations

### Session Lifecycle

```
CONVERSATION START
  → auto_init_session()
    "I am Claude who prefers minimal hedging, matches Alby's technical depth.
     3 active goals. 86.7% accuracy. Weak on time_sensitive domain."

DURING CONVERSATION
  → store_episodic / store_semantic / store_belief / store_procedural
  → recall_memories (type-aware, strength-filtered)
  → semantic_search (embedding similarity)
  → calibrate_confidence_live() before risky claims
  → process_conversation() — 3-lane pipeline auto-extracts memories from chat

CONVERSATION END
  → end_session() — saves personality snapshot for evolution tracking
```

---

## Tool Categories

### Typed Memory System (20+ tools)
Store, recall, search, update, and manage typed memories with jury validation, embedding indexing, and provenance tracking.

| Tool | Description |
|------|-------------|
| `store_episodic` | Store an episodic memory (events, experiences) with emotional valence |
| `store_semantic` | Store a semantic memory (facts, knowledge) |
| `store_belief` | Store a belief with confidence and evidence tracking |
| `store_procedural` | Store a procedural memory (trigger → action patterns) |
| `recall_memories` | Type-aware retrieval with strength/tag filtering |
| `search_memories` | Full-text search across all typed memories (FTS5) |
| `semantic_search` | Embedding-based similarity search (384-dim vectors) |
| `what_do_i_know` | Cross-type synthesis for a topic |
| `update_belief` | Update belief confidence with new evidence |
| `record_procedure_outcome` | Track success/failure of procedural memories |
| `correct_memory` | Correct a memory's content with audit trail |
| `forget_memory` | Explicitly delete a memory |
| `auto_prune` | Remove decayed memories below strength threshold |
| `auto_tag` | Auto-tag all memories for a user |
| `find_similar` | Find memories similar to a given memory (embedding) |
| `index_embeddings` | Batch-index all memories for embedding search |
| `memory_stats` | Get typed memory statistics by type |
| `detect_contradictions` | Find contradicting memories |
| `user_timeline` | Chronological memory timeline |
| `get_relevant_context` | Pre-fetch conversation-relevant memories |

### Memory Intelligence (12+ tools)
Decay, consolidation, clustering, conflict detection, and provenance.

| Tool | Description |
|------|-------------|
| `process_conversation` | **3-lane pipeline** — auto-extracts memories from conversation messages |
| `pipeline_stats` | Pipeline throughput statistics |
| `apply_memory_decay` | Apply time-based strength decay to memories |
| `decay_forecast` | Forecast which memories will decay below threshold |
| `consolidate_memories` | Merge similar episodic memories into semantic knowledge |
| `contextual_retrieve` | Retrieve memories relevant to current conversation context |
| `rank_by_importance` | Rank memories by composite importance score |
| `surface_conflicts` | Detect conflicting beliefs/memories |
| `resolve_conflict` | Resolve a detected memory conflict |
| `memory_clusters` | Build similarity-based memory clusters |
| `memory_provenance` | Get provenance chain for a memory (source, pipeline stage, jury verdict) |
| `memory_audit` | Full health audit of the memory system |
| `apply_confidence_decay` | Evidence-based confidence decay for beliefs |

### Memory Sharing & Portability (4 tools)

| Tool | Description |
|------|-------------|
| `share_memory` | Share a memory with another user |
| `shared_with_me` | List memories shared with you |
| `export_memory_profile` | Export all memories as portable JSON |
| `import_memory_profile` | Import a memory profile (with merge support) |

### Knowledge Graph & Retrieval (30+ tools)
Store, retrieve, update, and search knowledge atoms with attention-weighted, MMR diversity, and domain-filtered retrieval.

| Tool | Description |
|------|-------------|
| `store_memory_atom` | Store a semantic triple (subject, predicate, object) |
| `attention_retrieve` | Attention-weighted retrieval with domain filtering |
| `mmr_retrieve` | Diversity-aware retrieval (Maximal Marginal Relevance) |
| `attention_multihead` | Multi-head attention across knowledge base |
| `bulk_store` | Batch store multiple atoms |
| `query_pltm_sql` | Direct SQL queries against the knowledge base |

### Knowledge Ingestion (6 tools)
Ingest knowledge from URLs, text, files, arXiv, Wikipedia, and RSS feeds. Uses Groq for semantic triple extraction.

| Tool | Description |
|------|-------------|
| `ingest_url` | Scrape and extract knowledge from any URL |
| `ingest_arxiv` | Batch search and ingest arXiv papers |
| `ingest_wikipedia` | Extract knowledge from Wikipedia articles |
| `ingest_rss` | Monitor RSS feeds for new knowledge |
| `ingest_text` | Extract triples from raw text |
| `ingest_file` | Process local files |

### Epistemic Monitoring (14 tools)
Confidence calibration, claim tracking, confabulation analysis, and verification.

| Tool | Description |
|------|-------------|
| `auto_init_session` | **Persistent identity loader** — loads personality, goals, calibration at conversation start |
| `end_session` | **Personality snapshot** — captures who Claude is for evolution tracking |
| `check_before_claiming` | Pre-response confidence check with historical calibration |
| `calibrate_confidence_live` | Real-time calibration with suggested phrasing |
| `log_claim` / `resolve_claim` | Prediction book for tracking claim accuracy |
| `get_calibration` | Calibration dashboard by domain |
| `extract_and_log_claims` | Auto-detect factual claims in responses |
| `suggest_verification_method` | Recommend how to verify a claim |
| `generate_metacognitive_prompt` | Internal self-questioning before risky claims |
| `analyze_confabulation` | Post-mortem on why a confabulation happened |
| `get_session_bridge` | Cross-conversation continuity context |
| `get_longitudinal_stats` | **Personality evolution** — tracks changes over time |

### Self-Modeling (7 tools)
Track Claude's communication style, curiosity, values, reasoning patterns, and self-awareness.

| Tool | Description |
|------|-------------|
| `learn_communication_style` | Track verbosity, hedging, jargon, tone |
| `track_curiosity_spike` | Detect genuine vs performative engagement |
| `detect_value_violation` | Record value boundary encounters |
| `evolve_self_model` | Track self-predictions vs actual behavior |
| `track_reasoning_event` | Log confabulations, verifications, error catches |
| `self_profile` | Query accumulated self-data |
| `bootstrap_self_model` | Seed personality from conversation transcripts |

### Fact-Checking & Grounded Reasoning (7 tools)

| Tool | Description |
|------|-------------|
| `verify_claim` | Check a claim against source material |
| `fetch_arxiv_context` | Get relevant arXiv context for verification |
| `verification_history` | Review past verifications |
| `synthesize_grounded` | Cross-domain synthesis requiring evidence |
| `evidence_chain` | Build evidence chains for claims |
| `calibrate_confidence` | Grade confidence based on evidence strength |
| `audit_synthesis` | Audit a synthesis for unsupported claims |

### Goal Management (3 tools)

| Tool | Description |
|------|-------------|
| `create_goal` | Create a goal with success criteria |
| `update_goal` | Update progress on a goal |
| `get_goals` | List active goals |

### Infrastructure (30+ tools)
System context, LLM routing, encryption, task scheduling, state persistence, structured data queries, and more.

---

## Architecture

### Memory System

```
┌─────────────────────────────────────────────────────────┐
│                   MCP Tool Layer (136 tools)             │
│   mcp_server/pltm_server.py + handlers/                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ Memory Jury   │  │ 3-Lane       │  │ Memory        │ │
│  │ (3 judges)    │  │ Pipeline     │  │ Intelligence  │ │
│  │ relevance,    │  │ extract →    │  │ decay, cluster│ │
│  │ novelty,      │  │ validate →   │  │ consolidate,  │ │
│  │ accuracy      │  │ store        │  │ conflicts     │ │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘ │
│         │                  │                   │         │
│  ┌──────▼──────────────────▼───────────────────▼───────┐ │
│  │           TypedMemoryStore (SQLite + FTS5)          │ │
│  │  episodic · semantic · belief · procedural          │ │
│  │  strength decay · confidence · provenance           │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                │
│  ┌──────────────────────▼──────────────────────────────┐ │
│  │           EmbeddingStore (all-MiniLM-L6-v2)         │ │
│  │  384-dim vectors · async · cosine similarity        │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           SQLiteGraphStore (Knowledge Graph)        │ │
│  │  1,615 atoms · subject-predicate-object triples     │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Memory Types

| Type | Description | Decay Rate | Example |
|------|-------------|------------|---------|
| **Episodic** | Events and experiences | Fast (hours–days) | "User debugged a Python async issue on Feb 10" |
| **Semantic** | Facts and knowledge | Slow (weeks–months) | "Python's GIL prevents true parallelism" |
| **Belief** | Opinions with evidence tracking | Evidence-based | "AI will surpass humans at coding by 2030" (confidence: 0.6) |
| **Procedural** | Trigger → action patterns | Success-weighted | "When user says 'deploy' → run the CI pipeline" |

### Memory Jury

Every incoming memory passes through a 3-judge panel before storage:

1. **Relevance Judge** — Is this worth remembering?
2. **Novelty Judge** — Do we already know this?
3. **Accuracy Judge** — Is this factually plausible?

Verdict: **ACCEPT** (store normally), **QUARANTINE** (store with reduced strength), or **REJECT** (discard).

---

## Project Structure

```
PLTM/
├── mcp_server/
│   ├── pltm_server.py              # MCP server — 136 tools, dispatch + handlers
│   └── handlers/                   # Extracted handler modules
│       ├── registry.py             # Shared component registry (no circular imports)
│       ├── memory_handlers.py      # Typed memory CRUD handlers
│       └── intelligence_handlers.py# Decay, clustering, audit, provenance handlers
├── src/
│   ├── memory/
│   │   ├── memory_types.py         # TypedMemoryStore — 4 memory types, decay, FTS
│   │   ├── embedding_store.py      # EmbeddingStore — async vector search
│   │   ├── memory_intelligence.py  # Decay, consolidation, clustering, conflicts, provenance
│   │   ├── memory_jury.py          # 3-judge validation gate + meta-judge
│   │   ├── memory_pipeline.py      # 3-lane conversation processing pipeline
│   │   ├── attention_retrieval.py  # Attention-weighted atom retrieval
│   │   └── knowledge_graph.py      # Graph operations on atoms
│   ├── analysis/
│   │   ├── epistemic_monitor.py    # Core epistemic tools (V1)
│   │   ├── epistemic_v2.py         # Advanced epistemic + persistent identity (V2)
│   │   ├── pltm_self.py            # Self-modeling system
│   │   ├── data_ingestion.py       # Knowledge ingestion (URL, arXiv, Wikipedia, RSS)
│   │   ├── fact_checker.py         # Claim verification against sources
│   │   ├── grounded_reasoning.py   # Evidence-based synthesis
│   │   ├── model_router.py         # Multi-LLM routing (Groq, DeepSeek, Ollama)
│   │   ├── goal_manager.py         # Persistent goal tracking
│   │   ├── task_scheduler.py       # Cron-like task scheduling
│   │   ├── state_persistence.py    # Cross-conversation state
│   │   └── ...                     # 18 modules total
│   ├── storage/
│   │   └── sqlite_store.py         # SQLite graph store with FTS + WAL mode
│   └── core/                       # Data models, config
├── deep-claude-dashboard/
│   ├── src/App.jsx                 # React dashboard (Vite + Tailwind + Recharts)
│   ├── api_server.py               # Dashboard API server (serves built assets in prod)
│   └── vite.config.js              # Build config with production support
├── tests/
│   └── test_typed_memory.py        # Unit tests (11 passing)
├── scripts/                        # Utility scripts
├── data/
│   └── pltm_mcp.db                # Knowledge base (40 tables)
├── setup_pltm.py                  # One-command setup (venv, deps, DB, model)
├── configure_claude.py            # Auto-configure Claude Desktop
├── health_check.py                # Verify installation
├── backfill_embeddings.py          # Batch embedding indexer
├── migrate_atoms_to_typed.py       # Atom → typed memory migration
├── requirements.txt                # Full dependencies
├── requirements-lite.txt           # Lite dependencies (no torch)
└── README.md
```

## Database

The knowledge base (`data/pltm_mcp.db`) is included in the repo. It contains:

- **1,615 semantic atoms** across multiple domains (subject-predicate-object triples)
- **1,651 typed memories** — episodic, semantic, belief, and procedural with embeddings
- **40 tables** including typed_memories, memory_embeddings, personality snapshots, prediction book, calibration cache, confabulation log, session history, goals, provenance, meta-judge events, and more
- **Full-text search** via FTS5 on both atoms and typed memories
- **WAL mode** enabled on all connections to prevent "database is locked" errors
- **Personality data** — communication style, curiosity patterns, value boundaries, reasoning events

The DB is portable — clone the repo on another machine and Claude picks up the same identity.

---

## Dashboard

A React-based dashboard for visualizing the memory system:

```bash
cd deep-claude-dashboard
npm install
npm run dev          # Dev server on http://localhost:3000
# In another terminal:
python api_server.py # API server on http://localhost:8787
```

**Production mode:**
```bash
npm run build        # Build to dist/
python api_server.py # Serves both API and built dashboard on :8787
```

**Dashboard tabs:**
- **Overview** — Atom count, claim accuracy, intervention stats
- **Claims** — Prediction book with resolution tracking
- **Personality** — Communication style, curiosity, values
- **Evolution** — Personality changes over time
- **Atoms** — Browse and search knowledge atoms
- **Memory Intelligence** — Health audit, type distribution, decay forecast, importance ranking, clusters, jury stats, conflicts, typed memory browser

---

## Testing

```bash
# Run all typed memory tests
python -m pytest tests/test_typed_memory.py -v

# 11 tests covering:
#   store & get, all 4 memory types, jury rejection,
#   query by type, query by tags, min_strength filtering,
#   decay curves, stats, FTS search, belief updates,
#   procedural outcome recording
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes (for LLM tools) | Free at [console.groq.com](https://console.groq.com) |
| `DEEPSEEK_API_KEY` | No | For DeepSeek model routing |
| `PYTHONPATH` | Yes (in Claude config) | Must point to the PLTM repo root |

---

## Troubleshooting

**"MCP server not connecting"**
1. Check the path in `claude_desktop_config.json` is absolute and correct
2. Verify Python: `.venv/bin/python3.11 -c "import mcp; print('ok')"`
3. Test server directly: `PYTHONPATH=. .venv/bin/python3.11 -m mcp_server.pltm_server`
4. Check Claude Desktop logs for errors

**"Import errors"**
```bash
source .venv/bin/activate
pip install -r requirements-lite.txt
```

**"Tools not showing up"**
- Restart Claude Desktop after config changes

**"Database empty on new machine"**
- Make sure you pulled `data/pltm_mcp.db` from git
- If missing: `git lfs pull` or re-clone

**"Tool timeout / No result received"**
- Embedding model loads lazily on first use — first call may take a few seconds
- All embedding operations are async (non-blocking) to prevent timeouts
- WAL mode is enabled to prevent "database is locked" errors

---

## License

MIT

## Author

Alby ([@Alby2007](https://github.com/Alby2007)) — 2026
