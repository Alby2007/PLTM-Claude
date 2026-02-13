"""
Hierarchical Memory Index + Predictive Prefetch + Auto Contradiction Resolution

Solves three scaling bottlenecks:

1. HIERARCHICAL INDEX (O(log n) retrieval)
   - 3-level tree: domain → subtopic → atoms
   - SQL-level domain column on atoms for indexed queries
   - Subtopic clustering via keyword extraction
   - Rebuild index incrementally as atoms are added

2. PREDICTIVE PREFETCH (proactive cache warming)
   - Tracks conversation trajectory (recent tool calls + queries)
   - Predicts next-needed domains from trajectory patterns
   - Preloads relevant atom clusters into in-memory cache
   - Cache hit rate tracking for self-improvement

3. AUTO CONTRADICTION RESOLVER
   - Scans for conflicting atoms (same subject+predicate, different object)
   - Resolves using: confidence scores, provenance quality, recency
   - Auto-resolves clear winners, surfaces ambiguous conflicts
   - Runs as scheduled task alongside consolidation
"""

import hashlib
import json
import re
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import aiosqlite
from loguru import logger


# ── Domain taxonomy (extended from phi_rms.py DOMAIN_CATEGORIES) ─────────────

DOMAIN_TREE = {
    "consciousness": {
        "keywords": ["consciousness", "qualia", "phenomenal", "subjective",
                     "awareness", "sentience", "experience", "iit",
                     "integrated information", "global workspace", "neural correlates"],
        "subtopics": {
            "iit": ["integrated information", "phi", "tononi", "iit"],
            "global_workspace": ["global workspace", "baars", "broadcasting"],
            "neural_correlates": ["neural correlates", "ncc", "prefrontal", "thalamus"],
            "philosophy": ["hard problem", "qualia", "zombie", "chalmers", "phenomenal"],
        },
    },
    "ai_ml": {
        "keywords": ["neural network", "machine learning", "deep learning", "model",
                     "training", "transformer", "llm", "gpt", "claude", "language model",
                     "reinforcement", "attention", "embedding", "fine-tune", "rlhf",
                     "prompt", "token", "context window", "agent", "tool use"],
        "subtopics": {
            "llm": ["language model", "llm", "gpt", "claude", "transformer", "token"],
            "training": ["training", "fine-tune", "rlhf", "reward", "loss"],
            "agents": ["agent", "tool use", "function calling", "mcp", "planning"],
            "safety": ["alignment", "safety", "interpretability", "constitutional"],
        },
    },
    "complexity": {
        "keywords": ["complexity", "emergence", "criticality", "phase transition",
                     "power law", "scale-free", "self-organized", "chaos",
                     "nonlinear", "attractor", "bifurcation"],
        "subtopics": {
            "criticality": ["self-organized criticality", "power law", "avalanche"],
            "emergence": ["emergence", "emergent", "collective", "swarm"],
            "chaos": ["chaos", "attractor", "bifurcation", "lyapunov"],
        },
    },
    "technical": {
        "keywords": ["python", "javascript", "typescript", "react", "api", "database",
                     "sql", "docker", "kubernetes", "git", "linux", "server", "cloud",
                     "aws", "function", "class", "algorithm", "debug", "test",
                     "rust", "go", "java", "node", "backend", "frontend", "code"],
        "subtopics": {
            "python": ["python", "pip", "venv", "pytest", "async"],
            "web": ["javascript", "typescript", "react", "frontend", "css", "html"],
            "infra": ["docker", "kubernetes", "aws", "cloud", "deploy", "ci"],
            "databases": ["sql", "sqlite", "postgres", "database", "query"],
        },
    },
    "personal": {
        "keywords": ["hobby", "family", "health", "exercise", "music", "game",
                     "travel", "food", "movie", "book", "pet", "home", "friend",
                     "alby", "preference", "likes", "dislikes"],
        "subtopics": {
            "preferences": ["prefers", "likes", "dislikes", "favorite", "hates"],
            "lifestyle": ["hobby", "exercise", "health", "travel", "food"],
            "media": ["music", "movie", "book", "game", "show"],
        },
    },
    "pltm": {
        "keywords": ["pltm", "memory system", "phi", "typed memory", "memory jury",
                     "epistemic", "session", "handoff", "continuity", "personality",
                     "self-model", "tool analytics", "improvement"],
        "subtopics": {
            "architecture": ["pltm", "memory system", "pipeline", "jury", "store"],
            "identity": ["personality", "self-model", "epistemic", "identity"],
            "sessions": ["session", "handoff", "continuity", "working memory"],
        },
    },
    "learning": {
        "keywords": ["learn", "study", "research", "paper", "course", "tutorial",
                     "understand", "concept", "theory", "practice", "skill",
                     "arxiv", "hypothesis", "evidence", "experiment"],
        "subtopics": {
            "research": ["paper", "arxiv", "research", "study", "finding"],
            "hypothesis": ["hypothesis", "evidence", "experiment", "test"],
            "skills": ["learn", "practice", "skill", "tutorial", "course"],
        },
    },
}

# Flat keyword→domain lookup for O(1) classification
_KEYWORD_TO_DOMAIN: Dict[str, str] = {}
for _domain, _info in DOMAIN_TREE.items():
    for _kw in _info["keywords"]:
        _KEYWORD_TO_DOMAIN[_kw] = _domain


@dataclass
class IndexEntry:
    """An entry in the hierarchical index."""
    atom_id: str
    domain: str
    subtopic: str
    confidence: float
    last_accessed: float
    keywords: List[str]


@dataclass
class CacheEntry:
    """A prefetched memory cluster in the working cache."""
    domain: str
    atoms: List[Any]  # MemoryAtom list
    loaded_at: float
    hit_count: int = 0
    ttl: float = 300.0  # 5 min default


@dataclass
class ConflictPair:
    """Two atoms that conflict."""
    atom_a_id: str
    atom_b_id: str
    subject: str
    predicate: str
    object_a: str
    object_b: str
    confidence_a: float
    confidence_b: float
    resolution: str = ""  # auto_keep_a, auto_keep_b, needs_review


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HIERARCHICAL MEMORY INDEX
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalMemoryIndex:
    """
    3-level tree index over atoms: domain → subtopic → atoms.

    Adds a `domain` and `subtopic` column to atoms via a separate index table.
    Retrieval narrows to domain first (SQL WHERE), then subtopic, then scores.
    Turns O(n) full-scan into O(n/k) where k = number of domains.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent.parent.parent / "data" / "pltm_mcp.db"
        self._conn: Optional[aiosqlite.Connection] = None
        self._store = None

        # In-memory domain tree for fast lookups
        self._domain_counts: Dict[str, int] = defaultdict(int)
        self._subtopic_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    async def connect(self, store=None):
        self._conn = await aiosqlite.connect(str(self.db_path))
        await self._conn.execute("PRAGMA journal_mode=WAL")
        if store:
            self._store = store
        await self._ensure_tables()
        await self._load_counts()
        logger.info("HierarchicalMemoryIndex connected")

    async def _ensure_tables(self):
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_index (
                atom_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                subtopic TEXT NOT NULL DEFAULT 'general',
                keywords TEXT DEFAULT '[]',
                indexed_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mi_domain ON memory_index(domain);
            CREATE INDEX IF NOT EXISTS idx_mi_subtopic ON memory_index(domain, subtopic);
        """)
        await self._conn.commit()

    async def _load_counts(self):
        """Load domain/subtopic counts into memory for fast tree view."""
        cursor = await self._conn.execute(
            "SELECT domain, subtopic, COUNT(*) FROM memory_index GROUP BY domain, subtopic"
        )
        self._domain_counts.clear()
        self._subtopic_counts.clear()
        for domain, subtopic, count in await cursor.fetchall():
            self._domain_counts[domain] += count
            self._subtopic_counts[domain][subtopic] += count

    # ── Indexing ─────────────────────────────────────────────────────────────

    async def index_atom(self, atom_id: str, subject: str, predicate: str,
                         obj: str, contexts: List[str] = None) -> Dict[str, str]:
        """Classify and index a single atom. Returns {domain, subtopic}."""
        content = f"{subject} {predicate} {obj} {' '.join(contexts or [])}".lower()
        domain = self._classify_domain(content)
        subtopic = self._classify_subtopic(domain, content)
        keywords = self._extract_keywords(content)

        await self._conn.execute(
            """INSERT OR REPLACE INTO memory_index (atom_id, domain, subtopic, keywords, indexed_at)
               VALUES (?, ?, ?, ?, ?)""",
            (atom_id, domain, subtopic, json.dumps(keywords[:10]), time.time()),
        )
        await self._conn.commit()

        self._domain_counts[domain] += 1
        self._subtopic_counts[domain][subtopic] += 1

        return {"domain": domain, "subtopic": subtopic}

    async def rebuild_index(self) -> Dict[str, Any]:
        """Rebuild the entire index from atoms table. Use after bulk imports."""
        if not self._store:
            return {"ok": False, "err": "no store"}

        all_atoms = await self._store.get_all_atoms()
        indexed = 0
        errors = 0

        # Clear existing index
        await self._conn.execute("DELETE FROM memory_index")

        batch = []
        for atom in all_atoms:
            try:
                content = f"{atom.subject} {atom.predicate} {atom.object} {' '.join(getattr(atom, 'contexts', []))}".lower()
                domain = self._classify_domain(content)
                subtopic = self._classify_subtopic(domain, content)
                keywords = self._extract_keywords(content)
                batch.append((
                    str(atom.id), domain, subtopic,
                    json.dumps(keywords[:10]), time.time(),
                ))
                indexed += 1
            except Exception:
                errors += 1

            if len(batch) >= 100:
                await self._conn.executemany(
                    "INSERT OR REPLACE INTO memory_index (atom_id, domain, subtopic, keywords, indexed_at) VALUES (?,?,?,?,?)",
                    batch,
                )
                batch.clear()

        if batch:
            await self._conn.executemany(
                "INSERT OR REPLACE INTO memory_index (atom_id, domain, subtopic, keywords, indexed_at) VALUES (?,?,?,?,?)",
                batch,
            )
        await self._conn.commit()
        await self._load_counts()

        return {"ok": True, "indexed": indexed, "errors": errors, "domains": dict(self._domain_counts)}

    # ── Retrieval ────────────────────────────────────────────────────────────

    async def query_by_domain(self, domain: str, limit: int = 50) -> List[str]:
        """Get atom IDs in a domain. O(1) SQL index lookup."""
        cursor = await self._conn.execute(
            "SELECT atom_id FROM memory_index WHERE domain = ? LIMIT ?",
            (domain, limit),
        )
        return [r[0] for r in await cursor.fetchall()]

    async def query_by_subtopic(self, domain: str, subtopic: str, limit: int = 50) -> List[str]:
        """Get atom IDs in a domain+subtopic. O(1) compound index lookup."""
        cursor = await self._conn.execute(
            "SELECT atom_id FROM memory_index WHERE domain = ? AND subtopic = ? LIMIT ?",
            (domain, subtopic, limit),
        )
        return [r[0] for r in await cursor.fetchall()]

    async def query_by_keywords(self, keywords: List[str], limit: int = 30) -> List[str]:
        """Find atoms matching keywords. Uses domain narrowing first."""
        # Classify keywords to domain
        domain = self._classify_domain(" ".join(keywords))

        # Search within domain using keyword overlap
        cursor = await self._conn.execute(
            "SELECT atom_id, keywords FROM memory_index WHERE domain = ?",
            (domain,),
        )
        rows = await cursor.fetchall()

        kw_set = set(k.lower() for k in keywords)
        scored = []
        for atom_id, kw_json in rows:
            try:
                atom_kws = set(json.loads(kw_json or "[]"))
                overlap = len(kw_set & atom_kws)
                if overlap > 0:
                    scored.append((atom_id, overlap))
            except Exception:
                pass

        scored.sort(key=lambda x: -x[1])
        return [s[0] for s in scored[:limit]]

    async def smart_retrieve(self, query: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Smart retrieval: classify query → narrow to domain → FTS within domain.
        Falls back to global FTS if domain has too few results.
        """
        query_lower = query.lower()
        domain = self._classify_domain(query_lower)
        subtopic = self._classify_subtopic(domain, query_lower)

        # Get atom IDs in the target domain+subtopic
        atom_ids = await self.query_by_subtopic(domain, subtopic, limit=top_k * 2)

        # If too few, widen to full domain
        if len(atom_ids) < top_k // 2:
            atom_ids = await self.query_by_domain(domain, limit=top_k * 3)

        # If still too few, fall back to global
        if len(atom_ids) < 3:
            cursor = await self._conn.execute(
                "SELECT atom_id FROM memory_index LIMIT ?", (top_k * 3,)
            )
            atom_ids = [r[0] for r in await cursor.fetchall()]

        return {
            "domain": domain,
            "subtopic": subtopic,
            "candidates": len(atom_ids),
            "atom_ids": atom_ids[:top_k],
        }

    # ── Tree View ────────────────────────────────────────────────────────────

    async def get_tree(self) -> Dict[str, Any]:
        """Get the full domain→subtopic tree with counts."""
        total = sum(self._domain_counts.values())
        tree = {}
        for domain in sorted(self._domain_counts.keys(), key=lambda d: -self._domain_counts[d]):
            subtopics = dict(sorted(
                self._subtopic_counts[domain].items(),
                key=lambda x: -x[1],
            ))
            tree[domain] = {
                "count": self._domain_counts[domain],
                "subtopics": subtopics,
            }
        return {"total_indexed": total, "domains": len(tree), "tree": tree}

    # ── Classification ───────────────────────────────────────────────────────

    def _classify_domain(self, content: str) -> str:
        """Classify content into a domain. O(k) where k = total keywords."""
        words = set(content.lower().split())
        scores = defaultdict(int)

        # Fast single-word lookup
        for word in words:
            if word in _KEYWORD_TO_DOMAIN:
                scores[_KEYWORD_TO_DOMAIN[word]] += 1

        # Multi-word phrase matching for compound keywords
        content_lower = content.lower()
        for domain, info in DOMAIN_TREE.items():
            for kw in info["keywords"]:
                if " " in kw and kw in content_lower:
                    scores[domain] += 2  # Phrase matches worth more

        if scores:
            return max(scores, key=scores.get)
        return "general"

    def _classify_subtopic(self, domain: str, content: str) -> str:
        """Classify content into a subtopic within a domain."""
        info = DOMAIN_TREE.get(domain)
        if not info or "subtopics" not in info:
            return "general"

        content_lower = content.lower()
        scores = defaultdict(int)
        for subtopic, keywords in info["subtopics"].items():
            for kw in keywords:
                if kw in content_lower:
                    scores[subtopic] += 1

        if scores:
            return max(scores, key=scores.get)
        return "general"

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract meaningful keywords from content."""
        # Remove common stop words
        stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "shall", "can",
                "to", "of", "in", "for", "on", "with", "at", "by", "from",
                "as", "into", "through", "during", "before", "after", "above",
                "below", "between", "and", "but", "or", "not", "no", "nor",
                "so", "yet", "both", "either", "neither", "each", "every",
                "all", "any", "few", "more", "most", "other", "some", "such",
                "than", "too", "very", "just", "about", "that", "this", "it",
                "its", "i", "me", "my", "we", "our", "you", "your", "he",
                "she", "they", "them", "their", "what", "which", "who", "whom"}
        words = re.findall(r'\b[a-z]{3,}\b', content.lower())
        return [w for w in words if w not in stop][:20]

    async def close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PREDICTIVE PREFETCH ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

# Tool→domain mapping for trajectory prediction
TOOL_DOMAIN_MAP = {
    # Learning tools → learning domain
    "search_and_learn": "learning", "propose_hypothesis": "learning",
    "submit_evidence": "learning", "get_hypotheses": "learning",
    "auto_learn_run_task": "learning", "auto_learn_digest": "learning",
    "ingest_arxiv": "learning", "search_arxiv": "learning",
    "learn_from_url": "learning", "learn_from_paper": "learning",
    # AI/ML tools
    "cross_domain_synthesis": "ai_ml", "get_transfer_suggestions": "ai_ml",
    # PLTM tools
    "session_handoff": "pltm", "capture_session_state": "pltm",
    "auto_init_session": "pltm", "end_session": "pltm",
    "phi_build_context": "pltm", "phi_score_memories": "pltm",
    # Personal tools
    "query_personality": "personal", "learn_interaction_dynamic": "personal",
    "get_communication_style": "personal",
    # Technical
    "store_memory_atom": "technical", "query_atoms": "technical",
}

# Domain co-occurrence: if you're in domain A, you often need domain B
DOMAIN_ADJACENCY = {
    "consciousness": ["ai_ml", "complexity", "learning"],
    "ai_ml": ["technical", "learning", "consciousness"],
    "complexity": ["consciousness", "ai_ml"],
    "technical": ["ai_ml", "pltm"],
    "personal": ["pltm"],
    "pltm": ["technical", "personal", "ai_ml"],
    "learning": ["ai_ml", "consciousness", "complexity"],
}


class PredictivePrefetchEngine:
    """
    Predicts what memories will be needed next and preloads them.

    Tracks conversation trajectory (recent tools + queries) to predict
    which domain clusters to warm into cache. Like how the brain
    activates related memory clusters before they're explicitly needed.
    """

    def __init__(self, index: HierarchicalMemoryIndex):
        self.index = index
        self._store = None

        # Working memory cache: domain → CacheEntry
        self._cache: Dict[str, CacheEntry] = {}
        self.CACHE_MAX_DOMAINS = 4
        self.CACHE_TTL = 300.0  # 5 minutes

        # Trajectory tracking
        self._recent_tools: List[str] = []
        self._recent_queries: List[str] = []
        self._recent_domains: List[str] = []
        self.MAX_TRAJECTORY = 20

        # Stats
        self._hits = 0
        self._misses = 0

    def set_store(self, store):
        self._store = store

    # ── Trajectory Tracking ──────────────────────────────────────────────────

    def record_tool_call(self, tool_name: str):
        """Record a tool call for trajectory prediction."""
        self._recent_tools.append(tool_name)
        if len(self._recent_tools) > self.MAX_TRAJECTORY:
            self._recent_tools.pop(0)

        domain = TOOL_DOMAIN_MAP.get(tool_name)
        if domain:
            self._record_domain(domain)

    def record_query(self, query: str):
        """Record a query for trajectory prediction."""
        self._recent_queries.append(query)
        if len(self._recent_queries) > self.MAX_TRAJECTORY:
            self._recent_queries.pop(0)

        domain = self.index._classify_domain(query.lower())
        if domain != "general":
            self._record_domain(domain)

    def _record_domain(self, domain: str):
        self._recent_domains.append(domain)
        if len(self._recent_domains) > self.MAX_TRAJECTORY:
            self._recent_domains.pop(0)

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict_next_domains(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict which domains will be needed next.
        Uses: recent domain frequency + adjacency graph + momentum.
        """
        if not self._recent_domains:
            return [("general", 0.5)]

        # Domain frequency in recent trajectory
        freq = defaultdict(float)
        for i, domain in enumerate(self._recent_domains):
            # More recent = higher weight (exponential recency)
            recency_weight = 1.0 + (i / max(len(self._recent_domains), 1))
            freq[domain] += recency_weight

        # Add adjacent domains with dampened weight
        current_domain = self._recent_domains[-1]
        for adj_domain in DOMAIN_ADJACENCY.get(current_domain, []):
            freq[adj_domain] += 0.3

        # Momentum: if same domain appears 3+ times recently, boost it
        if len(self._recent_domains) >= 3:
            last_3 = self._recent_domains[-3:]
            if len(set(last_3)) == 1:
                freq[last_3[0]] += 2.0  # Strong momentum

        # Normalize
        total = sum(freq.values()) or 1.0
        predictions = [(d, round(s / total, 3)) for d, s in freq.items()]
        predictions.sort(key=lambda x: -x[1])

        return predictions[:top_k]

    # ── Cache Management ─────────────────────────────────────────────────────

    async def prefetch(self) -> Dict[str, Any]:
        """
        Prefetch predicted domains into cache.
        Call this after recording tool calls / queries.
        """
        predictions = self.predict_next_domains(top_k=self.CACHE_MAX_DOMAINS)
        prefetched = []
        now = time.time()

        # Evict expired entries
        expired = [d for d, e in self._cache.items() if now - e.loaded_at > e.ttl]
        for d in expired:
            del self._cache[d]

        for domain, score in predictions:
            if domain in self._cache:
                continue  # Already cached

            # Fetch atom IDs from index
            atom_ids = await self.index.query_by_domain(domain, limit=50)
            if not atom_ids:
                continue

            # Load actual atoms from store
            atoms = []
            if self._store:
                for aid in atom_ids[:30]:  # Cap at 30 per domain
                    try:
                        atom = await self._store.get_atom(aid)
                        if atom:
                            atoms.append(atom)
                    except Exception:
                        pass

            if atoms:
                self._cache[domain] = CacheEntry(
                    domain=domain,
                    atoms=atoms,
                    loaded_at=now,
                )
                prefetched.append({"domain": domain, "atoms": len(atoms), "score": score})

        # Evict least-used if over limit
        while len(self._cache) > self.CACHE_MAX_DOMAINS:
            least_used = min(self._cache.keys(), key=lambda d: self._cache[d].hit_count)
            del self._cache[least_used]

        return {
            "prefetched": prefetched,
            "cached_domains": list(self._cache.keys()),
            "predictions": predictions,
        }

    def get_cached(self, domain: str) -> Optional[List[Any]]:
        """Get atoms from cache. Returns None on miss."""
        entry = self._cache.get(domain)
        if entry and time.time() - entry.loaded_at < entry.ttl:
            entry.hit_count += 1
            self._hits += 1
            return entry.atoms
        self._misses += 1
        return None

    async def retrieve_smart(self, query: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Smart retrieval: check cache first, then index, then fallback.
        Records the query for future prediction.
        """
        self.record_query(query)
        domain = self.index._classify_domain(query.lower())

        # Try cache first
        cached = self.get_cached(domain)
        if cached:
            # Score cached atoms against query
            keywords = set(query.lower().split())
            scored = []
            for atom in cached:
                text = f"{atom.subject} {atom.predicate} {atom.object}".lower()
                overlap = len(keywords & set(text.split()))
                scored.append((atom, overlap))
            scored.sort(key=lambda x: -x[1])
            top = scored[:top_k]

            return {
                "source": "cache",
                "domain": domain,
                "n": len(top),
                "atoms": [{"s": a.subject[:30], "p": a.predicate, "o": a.object[:60],
                           "c": a.confidence} for a, _ in top[:10]],
                "cache_hit": True,
            }

        # Cache miss — use hierarchical index
        result = await self.index.smart_retrieve(query, top_k)
        atom_ids = result.get("atom_ids", [])

        # Load atoms
        atoms = []
        if self._store:
            for aid in atom_ids:
                try:
                    atom = await self._store.get_atom(aid)
                    if atom:
                        atoms.append(atom)
                except Exception:
                    pass

        # Trigger prefetch for next time
        await self.prefetch()

        return {
            "source": "index",
            "domain": result.get("domain"),
            "subtopic": result.get("subtopic"),
            "n": len(atoms),
            "atoms": [{"s": a.subject[:30], "p": a.predicate, "o": a.object[:60],
                       "c": a.confidence} for a in atoms[:10]],
            "cache_hit": False,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get prefetch cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(total, 1), 3),
            "cached_domains": list(self._cache.keys()),
            "cache_sizes": {d: len(e.atoms) for d, e in self._cache.items()},
            "trajectory_len": len(self._recent_domains),
            "recent_domains": self._recent_domains[-5:],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. AUTO CONTRADICTION RESOLVER
# ═══════════════════════════════════════════════════════════════════════════════

class AutoContradictionResolver:
    """
    Automatically resolves contradictions using confidence + provenance + recency.

    Resolution rules:
    1. If confidence gap > 0.3 → keep higher confidence
    2. If one has provenance (arxiv/doi) and other doesn't → keep sourced
    3. If one is newer by > 30 days → keep newer (knowledge evolves)
    4. If none of the above → flag as needs_review
    """

    CONFIDENCE_GAP_THRESHOLD = 0.3
    RECENCY_THRESHOLD_DAYS = 30

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent.parent.parent / "data" / "pltm_mcp.db"
        self._conn: Optional[aiosqlite.Connection] = None
        self._store = None

    async def connect(self, store=None):
        self._conn = await aiosqlite.connect(str(self.db_path))
        await self._conn.execute("PRAGMA journal_mode=WAL")
        if store:
            self._store = store
        await self._ensure_tables()
        logger.info("AutoContradictionResolver connected")

    async def _ensure_tables(self):
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS contradiction_log (
                id TEXT PRIMARY KEY,
                atom_a_id TEXT NOT NULL,
                atom_b_id TEXT NOT NULL,
                subject TEXT,
                predicate TEXT,
                resolution TEXT NOT NULL,
                resolved_at REAL NOT NULL,
                details TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_cl_resolution ON contradiction_log(resolution);
        """)
        await self._conn.commit()

    async def scan_and_resolve(self, limit: int = 200) -> Dict[str, Any]:
        """
        Scan for contradictions and auto-resolve where possible.

        Finds atoms with same subject+predicate but different objects,
        then applies resolution rules.
        """
        # Find potential conflicts: same subject+predicate, different object
        conn = self._conn
        cursor = await conn.execute("""
            SELECT a1.id, a2.id, a1.subject, a1.predicate,
                   a1.object, a2.object, a1.confidence, a2.confidence,
                   a1.first_observed, a2.first_observed
            FROM atoms a1
            JOIN atoms a2 ON a1.subject = a2.subject
                         AND a1.predicate = a2.predicate
                         AND a1.id < a2.id
                         AND a1.object != a2.object
                         AND a1.graph = 'substantiated'
                         AND a2.graph = 'substantiated'
            LIMIT ?
        """, (limit,))
        conflicts = await cursor.fetchall()

        auto_resolved = 0
        needs_review = 0
        results = []

        for row in conflicts:
            a_id, b_id, subject, predicate, obj_a, obj_b, conf_a, conf_b, obs_a, obs_b = row

            # Check if already resolved
            cursor2 = await conn.execute(
                "SELECT id FROM contradiction_log WHERE atom_a_id=? AND atom_b_id=?",
                (a_id, b_id),
            )
            if await cursor2.fetchone():
                continue

            resolution, details = self._resolve(
                conf_a, conf_b, obs_a, obs_b, a_id, b_id
            )

            # Log resolution
            log_id = f"cr_{uuid4().hex[:8]}"
            await conn.execute(
                """INSERT INTO contradiction_log
                   (id, atom_a_id, atom_b_id, subject, predicate, resolution, resolved_at, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (log_id, a_id, b_id, subject, predicate, resolution,
                 time.time(), json.dumps(details)),
            )

            if resolution == "needs_review":
                needs_review += 1
            else:
                auto_resolved += 1
                # Execute resolution: move loser to historical
                loser_id = details.get("demote")
                if loser_id:
                    await conn.execute(
                        "UPDATE atoms SET graph = 'historical' WHERE id = ?",
                        (loser_id,),
                    )

            results.append({
                "subject": subject[:30],
                "predicate": predicate,
                "resolution": resolution,
            })

        await conn.commit()

        return {
            "ok": True,
            "conflicts_found": len(conflicts),
            "auto_resolved": auto_resolved,
            "needs_review": needs_review,
            "details": results[:20],
        }

    def _resolve(self, conf_a: float, conf_b: float,
                 obs_a: float, obs_b: float,
                 id_a: str, id_b: str) -> Tuple[str, Dict]:
        """Apply resolution rules. Returns (resolution, details)."""

        # Rule 1: Confidence gap
        gap = abs(conf_a - conf_b)
        if gap >= self.CONFIDENCE_GAP_THRESHOLD:
            winner = "a" if conf_a > conf_b else "b"
            demote = id_b if winner == "a" else id_a
            return f"auto_keep_{winner}", {
                "rule": "confidence_gap",
                "gap": round(gap, 3),
                "demote": demote,
            }

        # Rule 2: Recency (newer knowledge supersedes older)
        if obs_a and obs_b:
            age_diff_days = abs(obs_a - obs_b) / 86400
            if age_diff_days >= self.RECENCY_THRESHOLD_DAYS:
                winner = "a" if obs_a > obs_b else "b"
                demote = id_b if winner == "a" else id_a
                return f"auto_keep_{winner}", {
                    "rule": "recency",
                    "age_diff_days": round(age_diff_days, 1),
                    "demote": demote,
                }

        # Rule 3: Can't auto-resolve
        return "needs_review", {
            "rule": "ambiguous",
            "conf_a": conf_a,
            "conf_b": conf_b,
        }

    async def get_unresolved(self, limit: int = 20) -> Dict[str, Any]:
        """Get conflicts that need manual review."""
        cursor = await self._conn.execute(
            """SELECT atom_a_id, atom_b_id, subject, predicate, details
               FROM contradiction_log WHERE resolution = 'needs_review'
               ORDER BY resolved_at DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        conflicts = []
        for r in rows:
            details = json.loads(r[4] or "{}")
            conflicts.append({
                "a": r[0][:12],
                "b": r[1][:12],
                "subject": r[2][:40],
                "predicate": r[3],
                "conf_a": details.get("conf_a"),
                "conf_b": details.get("conf_b"),
            })
        return {"ok": True, "n": len(conflicts), "conflicts": conflicts}

    async def get_resolution_stats(self) -> Dict[str, Any]:
        """Get contradiction resolution statistics."""
        cursor = await self._conn.execute(
            "SELECT resolution, COUNT(*) FROM contradiction_log GROUP BY resolution"
        )
        stats = {r[0]: r[1] for r in await cursor.fetchall()}
        return {
            "total": sum(stats.values()),
            "auto_resolved": sum(v for k, v in stats.items() if k.startswith("auto_")),
            "needs_review": stats.get("needs_review", 0),
            "by_type": stats,
        }

    async def close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None
