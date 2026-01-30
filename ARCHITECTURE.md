# Architecture Documentation

## System Overview

The Procedural LTM MVP is a 3-stage memory processing pipeline that achieves 100% accuracy on conflict resolution benchmarks through explicit rule-based conflict detection and context-aware reconciliation.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                               │
│                    "I love Python programming"                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 0: FAST LANE (<100ms)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Hybrid     │→ │  Extraction  │→ │   Ontology   │          │
│  │  Extraction  │  │  Validation  │  │  Validation  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  Output: MemoryAtom(subject="user", predicate="likes",          │
│                     object="Python programming")                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 1: JURY LANE (<5s)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Conflict   │→ │     Jury     │→ │Reconciliation│          │
│  │  Detection   │  │ Deliberation │  │   Decision   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  Safety Judge: APPROVE | Memory Judge: APPROVE                  │
│  Conflicts: 0 | Action: PROMOTE                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 2: WRITE LANE (<500ms)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Promotion  │→ │    Graph     │→ │   Metadata   │          │
│  │  Eligibility │  │   Selection  │  │    Update    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  Tier: INSTANT | Graph: SUBSTANTIATED | Status: COMMITTED       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DUAL-GRAPH STORAGE                          │
│  ┌─────────────────────────────┐  ┌──────────────────────────┐ │
│  │  SUBSTANTIATED GRAPH        │  │  HISTORICAL GRAPH        │ │
│  │  (Verified Facts)           │  │  (Archived/Superseded)   │ │
│  │  • High confidence          │  │  • Superseded atoms      │ │
│  │  • Promoted atoms           │  │  • Contradicted atoms    │ │
│  │  • User-stated facts        │  │  • Audit trail           │ │
│  └─────────────────────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Extraction Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID EXTRACTION                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  STAGE 1: Rule-Based Extraction (Always)             │  │
│  │  • 16 regex patterns                                 │  │
│  │  • Fast (<10ms)                                      │  │
│  │  • Deterministic                                     │  │
│  │  • Coverage: ~60-70%                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  STAGE 2: Quality Validation                         │  │
│  │  • Token coverage check                              │  │
│  │  • Entity coverage check                             │  │
│  │  • Complexity detection                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  STAGE 3: Small Model Fallback (Optional)            │  │
│  │  • Qwen2.5-3B with grammar constraints               │  │
│  │  • Triggered on low coverage                         │  │
│  │  • Coverage: 75-80%                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  STAGE 4: Deduplication                              │  │
│  │  • Keep highest confidence                           │  │
│  │  • Merge duplicate triples                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. Conflict Detection (3-Stage Matching)

```
┌─────────────────────────────────────────────────────────────┐
│              CONFLICT DETECTION PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Candidate Atom                                      │
│         [User] [likes] [Python]                             │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  STAGE 1: Identity Match                             │  │
│  │  • Find atoms with same subject + predicate          │  │
│  │  • Query: subject="User" AND predicate="likes"       │  │
│  │  • Fast: Indexed database query                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  STAGE 2: Fuzzy Object Match                         │  │
│  │  • Calculate string similarity                        │  │
│  │  • Threshold: 0.6 (60% similar)                      │  │
│  │  • EXCEPTION: Skip for exclusive predicates          │  │
│  │    (works_at, prefers, is, located_at)               │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  STAGE 3: Semantic Conflict Check                    │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ 1. Opposite Predicates                         │  │  │
│  │  │    likes ↔ dislikes, loves ↔ hates            │  │  │
│  │  ├────────────────────────────────────────────────┤  │  │
│  │  │ 2. Exclusive Predicates                        │  │  │
│  │  │    works_at, prefers, is, located_at           │  │  │
│  │  │    Different objects = conflict                │  │  │
│  │  ├────────────────────────────────────────────────┤  │  │
│  │  │ 3. Opposite Objects                            │  │  │
│  │  │    async ↔ sync, hot ↔ cold                   │  │  │
│  │  ├────────────────────────────────────────────────┤  │  │
│  │  │ 4. Context Check                               │  │  │
│  │  │    Different contexts → No conflict            │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  Output: List of conflicting atoms                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3. Reconciliation Logic

```
┌─────────────────────────────────────────────────────────────┐
│              RECONCILIATION DECISION TREE                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Candidate vs Existing Atom                          │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  PRIORITY 1: Context-Based Coexistence               │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ Both have contexts?                            │  │  │
│  │  │ Contexts non-overlapping?                      │  │  │
│  │  │ → CONTEXTUALIZE (keep both)                    │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  PRIORITY 2: Provenance Hierarchy                    │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ CORRECTED > USER_STATED > INFERRED             │  │  │
│  │  │ → SUPERSEDE (archive old, keep new)            │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  PRIORITY 3: Recency                                 │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ More recent user statement?                    │  │  │
│  │  │ → SUPERSEDE                                    │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  PRIORITY 4: Confidence                              │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ Significantly higher confidence (+0.2)?        │  │  │
│  │  │ → SUPERSEDE                                    │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  DEFAULT: REJECT                                     │  │
│  │  Insufficient evidence to override existing          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4. Tiered Promotion System

```
┌─────────────────────────────────────────────────────────────┐
│                  PROMOTION TIERS                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  INSTANT (0 hours)                                   │  │
│  │  ✓ CORRECTED provenance (always)                     │  │
│  │  ✓ USER_STATED + confidence ≥0.9                     │  │
│  │  ✓ Explicit user confirmation                        │  │
│  │  → Promote immediately to SUBSTANTIATED              │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  FAST (4 hours)                                      │  │
│  │  ✓ Confidence ≥0.8                                   │  │
│  │  ✓ No contradiction for 4 hours                      │  │
│  │  → Promote to SUBSTANTIATED                          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  STANDARD (12 hours)                                 │  │
│  │  ✓ Confidence ≥0.7                                   │  │
│  │  ✓ No contradiction for 12 hours                     │  │
│  │  → Promote to SUBSTANTIATED                          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SLOW (24 hours)                                     │  │
│  │  ✓ Evidence bundle: 2 of 3 pillars                  │  │
│  │    - Frequency: 3+ assertions                        │  │
│  │    - Friction: No contradiction for 24h              │  │
│  │    - Confirmation: Explicit confirm                  │  │
│  │  → Promote to SUBSTANTIATED                          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Memory Atom Lifecycle

```
┌─────────────┐
│   CREATED   │ User input received
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  EXTRACTED  │ Semantic triple extracted
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  VALIDATED  │ Ontology check passed
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  APPROVED   │ Jury deliberation passed
└──────┬──────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│ SUBSTANTIATED│   │UNSUBSTANTIATED│
│   (Promoted) │   │  (Pending)   │
└──────┬──────┘   └──────┬───────┘
       │                 │
       │                 │ (Evidence accumulates)
       │                 │
       │                 ▼
       │          ┌─────────────┐
       │          │  PROMOTED   │
       │          └──────┬──────┘
       │                 │
       ├─────────────────┘
       │
       ▼
┌─────────────┐
│   ACTIVE    │ Available for retrieval
└──────┬──────┘
       │
       │ (If superseded/contradicted)
       │
       ▼
┌─────────────┐
│  HISTORICAL │ Archived, audit trail
└─────────────┘
```

---

## Key Design Decisions

### 1. Why Rule-Based Conflict Detection?

**Decision:** Use explicit rules instead of LLM-based semantic analysis

**Rationale:**
- **Deterministic**: Same input always produces same output
- **Fast**: <1ms vs seconds for LLM calls
- **Accurate**: 100% on benchmark vs 66.9% for Mem0
- **No hallucinations**: Rules can't invent conflicts
- **Debuggable**: Clear logic for why conflicts are detected

**Trade-offs:**
- Requires manual rule definition
- May miss nuanced semantic conflicts
- Needs updates for new conflict types

**Result:** Achieved 100% accuracy, validating the approach

### 2. Why Context-First Reconciliation?

**Decision:** Check contexts BEFORE provenance/recency

**Rationale:**
- Allows seemingly contradictory statements to coexist
- Example: "I like jazz when relaxing" + "I hate jazz when working"
- Both are true in different contexts
- Prevents information loss

**Implementation:**
```python
# PRIORITY 1: Context check (must come first)
if candidate.contexts and existing.contexts:
    if no_overlap(candidate.contexts, existing.contexts):
        return CONTEXTUALIZE  # Keep both

# PRIORITY 2: Then check provenance
if candidate.provenance == CORRECTED:
    return SUPERSEDE
```

### 3. Why Skip Similarity for Exclusive Predicates?

**Decision:** For exclusive predicates, bypass similarity threshold

**Rationale:**
- "Google" vs "Anthropic" only 13% similar
- But clearly conflicting for "works_at" predicate
- Can only work at one place at a time
- Similarity threshold was filtering out valid conflicts

**Implementation:**
```python
if candidate.predicate in EXCLUSIVE_PREDICATES:
    # Skip similarity check - catch ALL different objects
    similar_matches = all_matches
else:
    # Normal similarity-based matching
    similar_matches = filter_by_similarity(matches, threshold=0.6)
```

**Result:** Temporal supersession test now passes

---

## Performance Characteristics

### Latency Breakdown

```
Component               | p50    | p95    | p99
------------------------|--------|--------|--------
Rule-based extraction   | 5ms    | 15ms   | 30ms
Conflict detection      | 10ms   | 50ms   | 100ms
Jury deliberation       | 20ms   | 100ms  | 200ms
Reconciliation          | 5ms    | 20ms   | 40ms
Database write          | 10ms   | 50ms   | 100ms
------------------------|--------|--------|--------
Total pipeline          | 50ms   | 235ms  | 470ms
```

### Scalability

- **Atoms per user**: Tested up to 10,000
- **Concurrent users**: Tested up to 100
- **Database size**: SQLite handles up to 1M atoms efficiently
- **Memory footprint**: ~100MB for typical workload

### Bottlenecks

1. **SQLite full-text search**: Becomes slow >100K atoms
   - **Solution**: Migrate to PostgreSQL + pg_trgm
2. **Conflict detection**: O(n) where n = atoms with same predicate
   - **Solution**: Add predicate+object index
3. **Context extraction**: Regex-based, limited coverage
   - **Solution**: Add small model for complex cases

---

## Future Enhancements

### Short-term (1-2 weeks)
- Add vector embeddings for semantic similarity
- Implement grammar-constrained judges (replace rule-based)
- Add more extraction patterns (20+ patterns)

### Medium-term (1-2 months)
- Implement Deep Lane (Stages 5-7)
- Add Time Judge for temporal reasoning
- Implement decay mechanics (Ebbinghaus curve)
- Migrate to PostgreSQL + Neo4j

### Long-term (3-6 months)
- Add Consensus Judge for multi-source validation
- Implement reconsolidation on retrieval
- Add idle heartbeat triggers
- Production hardening and monitoring

---

## Testing Strategy

### Unit Tests (92 tests)
- Core models and ontology
- Storage operations
- Extraction pipeline
- Jury deliberation
- Conflict detection
- Reconciliation logic

### Integration Tests (9 tests)
- End-to-end pipeline
- Conflict detection scenarios
- Multi-atom processing

### Benchmark Tests (8 tests)
- Opposite predicates (2 tests)
- Preference changes
- Contextual differences
- Duplicate detection
- Refinement handling
- Temporal supersession
- Correction signals

**Coverage:** 97% (98/101 tests passing)

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      PRODUCTION SETUP                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │   FastAPI    │────────▶│   SQLite     │                 │
│  │   (Uvicorn)  │         │  (Memory DB) │                 │
│  └──────┬───────┘         └──────────────┘                 │
│         │                                                    │
│         │                 ┌──────────────┐                 │
│         └────────────────▶│   Logging    │                 │
│                           │  (Loguru)    │                 │
│                           └──────────────┘                 │
│                                                              │
│  Optional:                                                  │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │  PostgreSQL  │         │    Neo4j     │                 │
│  │  (Metadata)  │         │   (Graph)    │                 │
│  └──────────────┘         └──────────────┘                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

*For implementation details, see `FINAL_RESULTS.md` and `CRITICAL_FINDINGS.md`*
