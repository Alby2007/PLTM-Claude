# Improvement Areas for PLTM

## 1. Meta-Optimization Loop for Tool Usage ✅ IMPLEMENTED
- **Implementation:** `src/analysis/tool_analytics.py` — `ToolAnalytics` class
- **Tables:** `tool_invocations`, `tool_sequences`
- **MCP Tools:** `tool_usage_stats`, `tool_redundancy_report`, `tool_consolidation_proposals`
- **Instrumentation:** Every `call_tool` invocation is logged with name, duration, success/error, args hash, result size.
- Capabilities delivered:
  - Log tool invocation sequences, outcomes, and redundant/bypassed calls.
  - Propose consolidation or pruning opportunities (deprecate unused, combine sequential, fix error-prone, optimize slow).
  - Score proposals on entropy/redundancy reduction while maintaining epistemic safeguards.
- **Dashboard:** `/api/tool-usage` endpoint for visualization.

## 2. Snapshotting and Evaluating Architecture Changes ✅ IMPLEMENTED
- **Implementation:** `src/analysis/architecture_snapshots.py` — `ArchitectureSnapshotter` class
- **Table:** `architecture_snapshots`
- **MCP Tools:** `snapshot_architecture`, `list_architecture_snapshots`, `compare_architectures`
- Capabilities delivered:
  - Version and snapshot architectures (tool list, counts, key metrics) before changes.
  - Compare quantitative metrics (accuracy, confabulation rate, Φ stats, personality) between snapshots.
  - Structured diff output: tool additions/removals, metric deltas, count changes.
- **Dashboard:** `/api/snapshots` endpoint for visualization.

## 4. Code Quality Bugfix Sweep ✅ FIXED
Bugs found and fixed in `mcp_server/pltm_server.py` and handler files:

### Critical
- **Duplicate tool names:** `resolve_conflict` and `consolidate_memories` were each defined twice with different schemas. The second handler was dead code (never reachable). Fixed by renaming: `resolve_memory_conflict` (memory intelligence) and `consolidate_typed_memories` (typed memory system).
- **Systemic type coercion:** ~30+ handlers passed MCP string args directly to functions expecting `int`/`float`, causing `'<' not supported between instances of 'int' and 'str'` crashes. Added `int()`/`float()` casts to all numeric `args.get()` calls across `pltm_server.py`, `memory_handlers.py`, `intelligence_handlers.py`, `phi_rms_handlers.py`.
- **Empty session_id in tool analytics:** Every tool invocation was logged with `session_id=""`, making sequence analysis useless. Fixed: `auto_init_session` now generates a session ID, `end_session` flushes the tool sequence, and `call_tool` passes the active session ID.

### Cleanup
- **Duplicate `compact_json`:** Was defined in both `pltm_server.py` and `registry.py`. Removed the local copy; `pltm_server.py` now imports from `registry.py`.

### Architecture Refactor ✅ DONE
- **Dictionary-based dispatch:** Replaced 300-line elif chain with `_TOOL_DISPATCH` dict (146 entries). O(1) lookup, duplicates impossible by definition. Built lazily on first call via `_build_dispatch_table()`.
- **Handler consolidation:** Dispatch now routes Memory Intelligence and ΦRMS tools to extracted handler files (`intelligence_handlers.py`, `phi_rms_handlers.py`). Removed 168 lines of dead inline duplicates. `pltm_server.py` reduced from 5,522 → 5,270 lines.

### Remaining (documented, not yet fixed)
- **No handler tests:** The entire dispatch + handler layer is untested. No tests for `epistemic_v2.py`, `goal_manager.py`, or `state_persistence.py`.
- **No input validation middleware:** Each handler does ad-hoc validation. A shared coercion/validation layer would prevent the entire class of type bugs.
- **~100 inline handlers remain:** Personality, learning, PLTM 2.0, entropy, epistemic, session handlers are still inline. Can be extracted incrementally.

## 5. Session Continuity System ✅ IMPLEMENTED
- **Implementation:** `src/memory/session_continuity.py` — 3 classes, ~550 lines
- **Tables:** `working_memory_snapshots`, `trajectory_snapshots`
- **MCP Tools:** `session_handoff`, `capture_session_state`, `get_working_memory`
- Components:
  - **WorkingMemoryCompressor** — Snapshots active working set (recently accessed memories, tools used, goals, confab patterns, state, open questions). Greedy-packs into token budget.
  - **TrajectoryEncoder** — Encodes conversation direction from tool sequences via TOOL_TOPICS mapping. Infers topic flow, decision points, open threads, and momentum.
  - **HandoffProtocol** — Orchestrates identity + working memory + trajectory + goals + warnings + Φ-context into one token-budgeted XML handoff block. Proportional budget allocation with redistribution from empty sections.
- Integration:
  - `end_session` auto-captures working memory + trajectory before ending.
  - `session_handoff` replaces calling `auto_init_session` + `get_session_bridge` + `phi_build_context` separately — one tool call, one compact block.
- **Tests:** 20/20 in `tests/test_session_continuity.py`

## 3. Proprietary Algorithms Worth Binary Shielding — Full Audit (Feb 13, 2026)

### TIER 1: SHIELD (Core differentiating IP — compile to binary)

These are the algorithms that make PLTM fundamentally different from any other memory system. A competitor reading this code could replicate the core value proposition.

1. **ΦRMS Scoring Formula** — `src/memory/phi_rms.py:77-260`
   - `PhiMemoryScorer` computes Φ-density per memory: weighted combination of graph_contribution (0.30), domain_bridging (0.25), semantic_uniqueness (0.25), consolidation_potential (0.20), normalized by token cost.
   - Sub-score algorithms: tag-overlap graph centrality, cross-domain bridging via DOMAIN_CATEGORIES taxonomy, embedding-distance uniqueness, type-specific consolidation potential.
   - **Why shield:** This is the core ranking function that determines which memories matter. The specific weights, normalization, and sub-score formulas are the "secret sauce" for memory optimization.

2. **Criticality-Aware Pruner** — `src/memory/phi_rms.py:290-423`
   - `CriticalityPruner` iteratively removes lowest-Φ memories while maintaining self-organized criticality constraints (ratio 0.8-1.0, integration drop <5%).
   - Guard rails: never prune successful procedures (success_count>3), never prune evidenced beliefs (confidence>0.7, evidence>2), periodic criticality re-check every 5 removals.
   - **Why shield:** The pruning guards and criticality constraints are the key innovation — pruning that preserves system-level coherence, not just individual memory value.

3. **Φ-Aware Consolidation** — `src/memory/phi_rms.py:430-560`
   - `PhiConsolidator` promotes episodic→semantic only if Φ is preserved (phi_after >= 0.9 × phi_before). Embedding-based cluster building + Φ preservation check before committing.
   - **Why shield:** The Φ preservation constraint on consolidation is novel — standard systems consolidate blindly.

4. **3-Lane Memory Pipeline** — `src/memory/memory_pipeline.py:1-595`
   - Lane 0 (Fast): `TypedMemoryExtractor` — 17 regex patterns classifying into 4 memory types with confidence scores, span-overlap dedup, procedural trigger/action extraction, sentiment detection.
   - Lane 2 (Write): `TypedMemoryReconciler` — embedding-similarity thresholds (>0.95 reject, >0.75 same-type supersede/merge, >0.7 cross-type contextualize), exclusive pattern detection, opposing sentiment detection, FTS fallback.
   - Orchestrator: `TypedMemoryPipeline` — Extract→Jury→Reconcile→Store with progressive updates.
   - **Why shield:** The specific similarity thresholds, reconciliation decision tree, and extraction patterns encode years of tuning. The 3-lane architecture itself is novel.

5. **Memory Jury + MetaJudge** — `src/memory/memory_jury.py:1-1008`
   - 4-judge system: SafetyJudge (PII/harmful veto), QualityJudge (dedup, internal contradiction, near-duplicate 85% overlap), TemporalJudge (rapid-fire detection, timestamp plausibility), ConsensusJudge (weighted voting with adaptive weights, safety veto, 2× reject weight).
   - `MetaJudge` (7 capabilities): persistent SQLite stats, ground truth feedback loop (false_positive/false_negative tracking), adaptive judge weighting (accuracy→weight, calibration penalty), per-memory-type accuracy, Expected Calibration Error (ECE) with 10-bin calibration curves, drift detection (50-window, 15% threshold), threshold tuning recommendations.
   - **Why shield:** The MetaJudge's self-improving feedback loop and adaptive weighting is the most novel part — judges that learn from their mistakes and adjust consensus weights automatically.

6. **Epistemic V2: Persistent Identity Loader** — `src/analysis/epistemic_v2.py:89-320`
   - `auto_init_session` synthesizes Claude's identity from 5 DB tables: communication style (verbosity/jargon/hedging by context), curiosity patterns (genuine engagement rate), value boundaries (pushback rate, violation types), reasoning honesty (confabulation/verification/error-catch rates → composite honesty score), self-awareness (prediction accuracy, surprise levels).
   - Personality evolution tracking: compares current session vs previous snapshot, generates evolution narrative.
   - `end_session` captures personality_snapshot to DB for longitudinal tracking.
   - **Why shield:** The identity synthesis algorithm — how raw behavioral data is aggregated into a coherent personality profile — is the core of bidirectional PLTM.

7. **PLTM-Self: Self-Modeling System** — `src/analysis/pltm_self.py:1-721`
   - `learn_communication_style`: NLP analysis of Claude's own responses (verbosity, jargon density, hedging rate, sentence structure, emotional tone).
   - `track_curiosity_spike`: genuine vs performative engagement scoring (deeper-than-required, autonomous research, followup questions, excitement markers).
   - `detect_value_violation`: boundary detection with intensity scoring and pushback tracking.
   - `evolve_self_model`: prediction vs actual behavior tracking, self-awareness accuracy measurement.
   - `bootstrap_from_text`: LLM-powered transcript analysis to populate self-model from conversation history.
   - **Why shield:** This is the only system that gives an LLM quantified self-awareness. The specific metrics and scoring algorithms are novel.

8. **Session Continuity: HandoffProtocol** — `src/memory/session_continuity.py:1-550`
   - `WorkingMemoryCompressor`: captures active working set (recently accessed memories, tools used, goals, confab patterns, state, open questions) with greedy token-budget packing.
   - `TrajectoryEncoder`: tool-sequence→topic-flow inference via TOOL_TOPICS mapping, momentum detection from last-N tools, decision point extraction from goal_log.
   - `HandoffProtocol`: proportional budget allocation across 6 sections with redistribution from empty sections, adaptive trimming to budget.
   - **Why shield:** The working memory concept (RAM snapshot vs disk) and trajectory encoding (conversation direction from tool sequences) are novel approaches to LLM session continuity.

### TIER 2: CONSIDER SHIELDING (Significant but less unique)

9. **Grounded Reasoning Engine** — `src/analysis/grounded_reasoning.py:1-524`
   - Every claim must cite specific atoms/papers. Cross-domain links explicitly flagged as NOVEL. Evidence chains with per-step sourcing. Confidence calibrated by independent source count.
   - **Why consider:** The anti-confabulation architecture is valuable but the approach (cite sources, flag novel claims) is more of a methodology than a secret algorithm.

10. **Memory Intelligence: Decay + Consolidation** — `src/memory/memory_intelligence.py:1-1066`
    - `DecayEngine`: Ebbinghaus forgetting curve with type-specific half-lives (episodic: 168h, semantic: 720h, belief: 336h, procedural: 2160h). Decay forecast algorithm.
    - `ConsolidationEngine`: embedding-based episodic clustering → semantic promotion with reinforcement of existing semantics.
    - `ImportanceScorer`: type×frequency×recency×confirmation weighted scoring.
    - `ConfidenceDecayEngine`: gradual confidence reduction on contradicting evidence.
    - **Why consider:** Ebbinghaus decay is well-known, but the type-specific profiles and the consolidation-via-embedding-clustering are tuned specifically for PLTM.

11. **Ontology-Driven Validation** — `src/core/ontology.py:1-386`
    - Type-specific rules: decay rates, exclusivity flags, temporal tracking, opposite-pair detection, contextual preferences.
    - Feeds into conflict detection, memory strength adjustments, and progression logic.
    - **Why consider:** The specific ontology rules encode domain knowledge about how personal facts behave.

12. **Attention-Weighted Retrieval** — `src/memory/attention_retrieval.py:1-628`
    - Transformer-inspired Q/K/V attention over memory atoms. Softmax normalization with temperature. LRU cache with TTL. Multi-head attention variant.
    - **Why consider:** The approach is inspired by well-known transformer attention, but the application to memory retrieval with recency decay and domain boosting is PLTM-specific.

13. **Cross-Domain Synthesis** — `src/learning/cross_domain_synthesis.py:1-609`
    - `MetaPattern` discovery across domains. `TransferOpportunity` detection. Novelty scoring for cross-domain insights.
    - **Why consider:** The meta-pattern discovery algorithm and transfer opportunity scoring are novel but depend heavily on LLM calls.

### TIER 3: SAFE TO OPEN-SOURCE (Infrastructure, plumbing, standard patterns)

These are valuable but not differentiating — they're standard engineering patterns:

- `src/storage/sqlite_store.py` — Standard SQLite CRUD
- `src/core/models.py` — Data models (MemoryAtom, AtomType, etc.)
- `src/core/config.py`, `src/core/vector_config.py` — Configuration
- `src/analysis/tool_analytics.py` — Standard usage logging
- `src/analysis/state_persistence.py` — Key-value state store
- `src/analysis/goal_manager.py` — CRUD for goals
- `src/analysis/task_scheduler.py` — Cron-like scheduling
- `src/analysis/api_client.py` — HTTP client wrapper
- `src/analysis/model_router.py` — LLM routing (standard pattern)
- `src/analysis/system_context.py` — System info gathering
- `src/analysis/crypto_ops.py` — Standard Fernet encryption
- `src/analysis/data_ingestion.py` — URL/text/file ingestion
- `src/analysis/structured_data.py` — API data queries
- `src/analysis/architecture_snapshots.py` — Schema snapshots
- `src/extraction/` — Rule-based and hybrid extractors (standard NLP)
- `src/pipeline/` — Pipeline orchestration (standard pattern)
- `src/agents/` — Agent wrappers (standard RAG/agent patterns)
- `src/personality/mood_tracker.py`, `mood_patterns.py` — Standard sentiment tracking
- `src/personality/personality_extractor.py` — Standard trait extraction
- `mcp_server/` — MCP protocol plumbing, dispatch, handlers
- `tests/` — All tests

### Implementation Strategy

**Recommended approach:** Cython compilation of Tier 1 files into `.so` shared libraries.

```
# Files to compile (Tier 1):
src/memory/phi_rms.py          → phi_rms.cpython-311-darwin.so
src/memory/memory_pipeline.py  → memory_pipeline.cpython-311-darwin.so
src/memory/memory_jury.py      → memory_jury.cpython-311-darwin.so
src/memory/session_continuity.py → session_continuity.cpython-311-darwin.so
src/analysis/epistemic_v2.py   → epistemic_v2.cpython-311-darwin.so
src/analysis/pltm_self.py      → pltm_self.cpython-311-darwin.so
```

**Steps:**
1. Add `setup.py` with Cython build config
2. Compile Tier 1 files to `.so` binaries
3. Replace `.py` files with `.so` in distribution
4. Add `.py` files to `.gitignore` for Tier 1
5. Keep Tier 3 files as readable Python for transparency
6. Tier 2 files: decide per-file based on competitive landscape
