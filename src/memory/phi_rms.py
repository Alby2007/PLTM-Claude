"""
Φ-Optimized Resource Management System (ΦRMS)

Maximizes integrated information (Φ) under token constraints by:
1. PhiMemoryScorer  — scores each TypedMemory by Φ-density
2. CriticalityPruner — removes low-Φ memories while preserving criticality
3. PhiConsolidator  — Φ-aware episodic→semantic consolidation
4. PhiContextBuilder — knapsack-based context selection for conversations

All classes operate on the existing TypedMemoryStore + EmbeddingStore.
"""

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import aiosqlite
from loguru import logger

from src.memory.memory_types import TypedMemory, TypedMemoryStore, MemoryType
from src.memory.memory_intelligence import ImportanceScorer


# Domain taxonomy for bridging score (mirrors TypedMemoryStore.TAXONOMY)
DOMAIN_CATEGORIES = {
    "work": ["project", "deploy", "code", "build", "ship", "release", "sprint",
             "meeting", "deadline", "task", "bug", "feature", "pr", "review",
             "production", "staging", "client", "team", "standup"],
    "technical": ["python", "javascript", "typescript", "react", "api", "database",
                  "sql", "docker", "kubernetes", "git", "linux", "server", "cloud",
                  "aws", "function", "class", "algorithm", "debug", "test", "ci",
                  "rust", "go", "java", "c++", "node", "backend", "frontend"],
    "communication": ["concise", "verbose", "detailed", "brief", "formal", "casual",
                      "direct", "indirect", "tone", "style", "response", "explain"],
    "preferences": ["prefers", "likes", "dislikes", "wants", "avoids", "favorite",
                    "hates", "loves", "enjoys", "comfortable", "uncomfortable"],
    "personality": ["patient", "impatient", "curious", "creative", "analytical",
                    "introvert", "extrovert", "thorough", "quick", "careful"],
    "learning": ["learn", "study", "research", "paper", "course", "tutorial",
                 "understand", "concept", "theory", "practice", "skill"],
    "personal": ["hobby", "family", "health", "exercise", "music", "game",
                 "travel", "food", "movie", "book", "pet", "home"],
    "ai_interaction": ["claude", "gpt", "llm", "prompt", "model", "token",
                       "context", "memory", "personality", "pltm", "mcp"],
}


def _classify_domains(content: str, tags: List[str]) -> List[str]:
    """Map content + tags to domain categories."""
    domains = set()
    combined = (content + " " + " ".join(tags)).lower()
    words = set(combined.split())

    for domain, keywords in DOMAIN_CATEGORIES.items():
        matches = sum(1 for kw in keywords if kw in words or kw in combined)
        if matches >= 1:
            domains.add(domain)

    return list(domains)


def _estimate_tokens(mem: TypedMemory) -> int:
    """Estimate token cost of a memory (chars / 4 heuristic)."""
    text_len = len(mem.content) + len(mem.context)
    if mem.memory_type == MemoryType.PROCEDURAL:
        text_len += len(mem.trigger) + len(mem.action)
    return max(1, text_len // 4)


# ============================================================
# 1. PHI MEMORY SCORER
# ============================================================

class PhiMemoryScorer:
    """
    Computes Φ-density per TypedMemory.

    Φ-density = raw_score / normalized_token_cost

    raw_score = weighted combination of:
      - graph_contribution (tag overlap with other memories)
      - domain_bridging (cross-domain connectivity)
      - semantic_uniqueness (embedding distance from neighbors)
      - consolidation_potential (cluster size / reinforcement count)
    """

    def __init__(self, store: TypedMemoryStore, embedding_store=None, db_path: str = ""):
        self.store = store
        self.embeddings = embedding_store
        self.db_path = db_path or store.db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self):
        """Open DB connection and create schema."""
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA busy_timeout=10000")
        await self._ensure_table()

    async def close(self):
        if self._conn:
            await self._conn.close()

    async def _ensure_table(self):
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS phi_memory_scores (
                memory_id TEXT PRIMARY KEY,
                phi_score REAL NOT NULL,
                graph_contribution REAL DEFAULT 0,
                domain_bridging REAL DEFAULT 0,
                semantic_uniqueness REAL DEFAULT 0,
                consolidation_potential REAL DEFAULT 0,
                token_cost INTEGER DEFAULT 1,
                scored_at REAL NOT NULL
            )
        """)
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_phi_score ON phi_memory_scores(phi_score DESC)"
        )
        await self._conn.commit()

    # ---------- scoring ----------

    async def score_memory(self, mem: TypedMemory, all_mems: Optional[List[TypedMemory]] = None) -> Dict[str, Any]:
        """Compute and persist Φ-density for a single memory."""
        if all_mems is None:
            all_mems = await self.store.query(mem.user_id, min_strength=0.0, limit=10000)

        graph_c = self._graph_contribution(mem, all_mems)
        domain_b = self._domain_bridging(mem)
        semantic_u = await self._semantic_uniqueness(mem)
        consol_p = self._consolidation_potential(mem, all_mems)
        token_cost = _estimate_tokens(mem)

        raw = (0.30 * graph_c
               + 0.25 * domain_b
               + 0.25 * semantic_u
               + 0.20 * consol_p)

        phi_density = raw / max(1.0, token_cost / 100.0)
        phi_density = round(min(1.0, max(0.0, phi_density)), 4)

        record = {
            "memory_id": mem.id,
            "phi_score": phi_density,
            "graph_contribution": round(graph_c, 4),
            "domain_bridging": round(domain_b, 4),
            "semantic_uniqueness": round(semantic_u, 4),
            "consolidation_potential": round(consol_p, 4),
            "token_cost": token_cost,
            "scored_at": time.time(),
        }

        await self._persist(record)
        return record

    async def score_all(self, user_id: str) -> Dict[str, Any]:
        """Batch-score all memories for a user. Returns summary stats."""
        all_mems = await self.store.query(user_id, min_strength=0.0, limit=10000)
        if not all_mems:
            return {"total": 0, "scored": 0, "avg_phi": 0, "min_phi": 0, "max_phi": 0}

        scores = []
        for mem in all_mems:
            rec = await self.score_memory(mem, all_mems=all_mems)
            scores.append(rec["phi_score"])

        return {
            "total": len(all_mems),
            "scored": len(scores),
            "avg_phi": round(sum(scores) / len(scores), 4),
            "min_phi": round(min(scores), 4),
            "max_phi": round(max(scores), 4),
            "by_type": self._stats_by_type(all_mems, scores),
        }

    async def get_scores(self, user_id: str, min_phi: float = 0.0, limit: int = 100) -> List[Dict[str, Any]]:
        """Read persisted scores, joined with memory content."""
        cursor = await self._conn.execute("""
            SELECT p.memory_id, p.phi_score, p.graph_contribution, p.domain_bridging,
                   p.semantic_uniqueness, p.consolidation_potential, p.token_cost, p.scored_at,
                   t.content, t.memory_type, t.strength
            FROM phi_memory_scores p
            JOIN typed_memories t ON p.memory_id = t.id
            WHERE t.user_id = ? AND p.phi_score >= ?
            ORDER BY p.phi_score DESC
            LIMIT ?
        """, (user_id, min_phi, limit))
        rows = await cursor.fetchall()
        return [{
            "memory_id": r[0], "phi_score": r[1],
            "graph_contribution": r[2], "domain_bridging": r[3],
            "semantic_uniqueness": r[4], "consolidation_potential": r[5],
            "token_cost": r[6], "scored_at": r[7],
            "content": r[8][:100] if r[8] else "", "memory_type": r[9],
            "strength": r[10],
        } for r in rows]

    async def get_score(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Single score lookup."""
        cursor = await self._conn.execute(
            "SELECT * FROM phi_memory_scores WHERE memory_id = ?", (memory_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "memory_id": row[0], "phi_score": row[1],
            "graph_contribution": row[2], "domain_bridging": row[3],
            "semantic_uniqueness": row[4], "consolidation_potential": row[5],
            "token_cost": row[6], "scored_at": row[7],
        }

    # ---------- sub-scores ----------

    def _graph_contribution(self, mem: TypedMemory, all_mems: List[TypedMemory]) -> float:
        """Tag-overlap proxy for graph centrality."""
        if not mem.tags or len(all_mems) < 2:
            return 0.0
        my_tags = set(mem.tags)
        connected = sum(1 for other in all_mems
                        if other.id != mem.id and set(other.tags) & my_tags)
        return min(1.0, connected / max(1, len(all_mems) - 1))

    def _domain_bridging(self, mem: TypedMemory) -> float:
        """How many domain categories does this memory span?"""
        domains = _classify_domains(mem.content, mem.tags)
        return min(1.0, len(domains) / 3.0)

    async def _semantic_uniqueness(self, mem: TypedMemory) -> float:
        """1 - avg similarity to nearest neighbors. Unique memories score high."""
        if not self.embeddings:
            return 0.5
        try:
            similar = await self.embeddings.find_similar(
                mem.id, limit=5, min_similarity=0.3)
            if not similar:
                return 1.0  # No neighbors → maximally unique
            avg_sim = sum(h["similarity"] for h in similar) / len(similar)
            return round(1.0 - avg_sim, 4)
        except Exception:
            return 0.5

    def _consolidation_potential(self, mem: TypedMemory, all_mems: List[TypedMemory]) -> float:
        """How much consolidation value does this memory carry?"""
        if mem.memory_type == MemoryType.EPISODIC:
            # Count same-tag episodic siblings
            my_tags = set(mem.tags)
            siblings = sum(1 for other in all_mems
                           if other.id != mem.id
                           and other.memory_type == MemoryType.EPISODIC
                           and set(other.tags) & my_tags)
            return min(1.0, siblings / 5.0)
        elif mem.memory_type in (MemoryType.SEMANTIC, MemoryType.PROCEDURAL):
            return min(1.0, mem.consolidation_count / 10.0)
        elif mem.memory_type == MemoryType.BELIEF:
            return min(1.0, len(mem.evidence_for) / 5.0)
        return 0.0

    # ---------- helpers ----------

    def _stats_by_type(self, mems: List[TypedMemory], scores: List[float]) -> Dict[str, Dict]:
        by_type = defaultdict(list)
        for mem, score in zip(mems, scores):
            by_type[mem.memory_type.value].append(score)
        return {
            mt: {"count": len(s), "avg": round(sum(s) / len(s), 4)}
            for mt, s in by_type.items()
        }

    async def _persist(self, record: Dict[str, Any]):
        await self._conn.execute("""
            INSERT OR REPLACE INTO phi_memory_scores
            (memory_id, phi_score, graph_contribution, domain_bridging,
             semantic_uniqueness, consolidation_potential, token_cost, scored_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (record["memory_id"], record["phi_score"],
              record["graph_contribution"], record["domain_bridging"],
              record["semantic_uniqueness"], record["consolidation_potential"],
              record["token_cost"], record["scored_at"]))
        await self._conn.commit()


# ============================================================
# 2. CRITICALITY-AWARE PRUNER
# ============================================================

class CriticalityPruner:
    """
    Iteratively removes lowest-Φ memories while maintaining
    self-organized criticality constraints.
    """

    def __init__(self, scorer: PhiMemoryScorer, criticality, store: TypedMemoryStore):
        """
        Args:
            scorer: PhiMemoryScorer instance (must be connected)
            criticality: SelfOrganizedCriticality instance
            store: TypedMemoryStore instance
        """
        self.scorer = scorer
        self.criticality = criticality
        self.store = store

    async def prune(
        self, user_id: str,
        target_token_savings: int = 5000,
        max_removals: int = 20,
    ) -> Dict[str, Any]:
        """
        Remove low-Φ memories until target_token_savings reached or max_removals hit.
        Respects criticality constraints.
        """
        return await self._run_prune(user_id, target_token_savings, max_removals, dry_run=False)

    async def simulate_prune(self, user_id: str) -> Dict[str, Any]:
        """Dry-run: show what would be pruned without deleting."""
        return await self._run_prune(user_id, target_token_savings=999999, max_removals=50, dry_run=True)

    async def _run_prune(
        self, user_id: str,
        target_token_savings: int,
        max_removals: int,
        dry_run: bool,
    ) -> Dict[str, Any]:
        # Ensure scores are fresh
        await self.scorer.score_all(user_id)

        # Get scores sorted ascending (lowest Φ first = prune candidates)
        all_scores = await self.scorer.get_scores(user_id, min_phi=0.0, limit=10000)
        all_scores.sort(key=lambda x: x["phi_score"])

        # Snapshot criticality baseline
        baseline = await self.criticality.get_criticality_state()
        baseline_integration = baseline.get("i", 0.5)

        pruned = []
        protected = []
        tokens_freed = 0

        for rec in all_scores:
            if len(pruned) >= max_removals or tokens_freed >= target_token_savings:
                break

            mem = await self.store.get(rec["memory_id"])
            if not mem:
                continue

            # Guard: never prune successful procedures
            if (mem.memory_type == MemoryType.PROCEDURAL
                    and mem.success_count > 3):
                protected.append({
                    "id": mem.id, "reason": "successful_procedure",
                    "phi": rec["phi_score"],
                })
                continue

            # Guard: never prune high-confidence beliefs with evidence
            if (mem.memory_type == MemoryType.BELIEF
                    and mem.confidence > 0.7
                    and len(mem.evidence_for) > 2):
                protected.append({
                    "id": mem.id, "reason": "evidenced_belief",
                    "phi": rec["phi_score"],
                })
                continue

            # Guard: check criticality after removal
            # Use a lightweight check — if we've already pruned several,
            # re-check criticality state
            if len(pruned) > 0 and len(pruned) % 5 == 0:
                current = await self.criticality.get_criticality_state()
                ratio = current.get("r", 1.0)
                integration = current.get("i", 0.5)

                if ratio < 0.8 or ratio > 1.0:
                    protected.append({
                        "id": mem.id, "reason": "criticality_ratio_breach",
                        "phi": rec["phi_score"],
                    })
                    break  # Stop pruning entirely

                if baseline_integration > 0 and integration < baseline_integration * 0.95:
                    protected.append({
                        "id": mem.id, "reason": "integration_drop",
                        "phi": rec["phi_score"],
                    })
                    break

            # Safe to prune
            if not dry_run:
                await self.store.forget_memory(
                    rec["memory_id"], reason="phi_pruning")
                # Clean up score record
                await self.scorer._conn.execute(
                    "DELETE FROM phi_memory_scores WHERE memory_id = ?",
                    (rec["memory_id"],))
                await self.scorer._conn.commit()

            pruned.append({
                "id": rec["memory_id"],
                "content": rec.get("content", "")[:80],
                "phi": rec["phi_score"],
                "tokens": rec["token_cost"],
                "type": rec.get("memory_type", ""),
            })
            tokens_freed += rec["token_cost"]

        # Final criticality snapshot
        after = await self.criticality.get_criticality_state()

        return {
            "dry_run": dry_run,
            "pruned": len(pruned),
            "protected": len(protected),
            "tokens_freed": tokens_freed,
            "criticality_before": baseline,
            "criticality_after": after,
            "pruned_details": pruned[:20],
            "protected_details": protected[:10],
        }


# ============================================================
# 3. PHI-AWARE CONSOLIDATOR
# ============================================================

class PhiConsolidator:
    """
    Φ-aware episodic→semantic consolidation.
    Wraps the existing ConsolidationEngine pattern with Φ preservation checks.
    """

    def __init__(self, store: TypedMemoryStore, embedding_store=None, scorer: PhiMemoryScorer = None):
        self.store = store
        self.embeddings = embedding_store
        self.scorer = scorer

    async def consolidate(
        self, user_id: str,
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.55,
    ) -> Dict[str, Any]:
        """
        Find clusters of similar episodic memories and promote to semantic,
        but only if Φ is preserved (phi_after >= 0.9 * phi_before).
        """
        episodes = await self.store.query(
            user_id, MemoryType.EPISODIC, min_strength=0.05, limit=200)

        if len(episodes) < min_cluster_size:
            return {"clusters_found": 0, "promoted": 0, "reinforced": 0,
                    "phi_preserved": True, "tokens_saved": 0}

        # Build clusters
        clusters = await self._build_clusters(episodes, min_cluster_size, similarity_threshold)

        promoted = 0
        reinforced = 0
        tokens_saved = 0
        phi_preserved = True

        for cluster in clusters:
            # Compute cluster Φ before consolidation
            phi_before = 0.0
            cluster_tokens = 0
            if self.scorer:
                for ep in cluster:
                    score = await self.scorer.get_score(ep.id)
                    if score:
                        phi_before += score["phi_score"]
                    cluster_tokens += _estimate_tokens(ep)

            # Check if semantic memory already exists for this pattern
            existing = await self._find_existing_semantic(user_id, cluster)
            if existing:
                existing.consolidation_count += 1
                existing.strength = min(1.0, existing.strength + 0.1)
                existing.consolidated_from = list(set(
                    existing.consolidated_from + [e.id for e in cluster]))
                await self.store.store(existing, bypass_jury=True)
                reinforced += 1
                continue

            # Create new semantic memory
            contents = [ep.content for ep in cluster[:5]]
            all_tags = set()
            for ep in cluster:
                all_tags.update(ep.tags)

            semantic = TypedMemory(
                id=str(uuid4()),
                memory_type=MemoryType.SEMANTIC,
                user_id=user_id,
                content=f"Recurring pattern ({len(cluster)} episodes): {'; '.join(contents[:3])}",
                source="phi_consolidation",
                confidence=min(0.9, 0.5 + len(cluster) * 0.1),
                consolidated_from=[ep.id for ep in cluster],
                consolidation_count=1,
                tags=list(all_tags | {"consolidated", "phi_consolidated"}),
            )

            # Φ preservation check
            if self.scorer:
                # Score the proposed semantic memory
                new_score = await self.scorer.score_memory(semantic)
                new_phi = new_score["phi_score"]
                new_tokens = _estimate_tokens(semantic)

                if phi_before > 0 and new_phi < 0.9 * phi_before:
                    phi_preserved = False
                    logger.info(
                        f"PhiConsolidator: skipping cluster (phi_before={phi_before:.3f}, "
                        f"phi_after={new_phi:.3f})")
                    continue

                tokens_saved += cluster_tokens - new_tokens

            await self.store.store(semantic, bypass_jury=True)
            promoted += 1

        return {
            "clusters_found": len(clusters),
            "promoted": promoted,
            "reinforced": reinforced,
            "phi_preserved": phi_preserved,
            "tokens_saved": max(0, tokens_saved),
            "episodes_analyzed": len(episodes),
        }

    async def _build_clusters(
        self, episodes: List[TypedMemory],
        min_cluster_size: int, similarity_threshold: float,
    ) -> List[List[TypedMemory]]:
        """Build clusters using embeddings or tag fallback."""
        clusters = []
        used = set()

        if self.embeddings and len(episodes) <= 50:
            for ep in episodes:
                if ep.id in used:
                    continue
                try:
                    similar = await self.embeddings.find_similar(
                        ep.id, limit=20, min_similarity=similarity_threshold)
                except Exception:
                    continue

                cluster = [ep]
                for hit in similar:
                    if hit["memory_id"] in used:
                        continue
                    other = await self.store.get(hit["memory_id"])
                    if (other and other.user_id == ep.user_id
                            and other.memory_type == MemoryType.EPISODIC):
                        cluster.append(other)
                        used.add(hit["memory_id"])

                if len(cluster) >= min_cluster_size:
                    used.add(ep.id)
                    clusters.append(cluster)
        else:
            # Tag-based fallback
            tag_groups: Dict[str, List[TypedMemory]] = {}
            for ep in episodes:
                for tag in ep.tags:
                    tag_groups.setdefault(tag, []).append(ep)
            for tag, eps in tag_groups.items():
                if len(eps) >= min_cluster_size:
                    clusters.append(eps)

        return clusters

    async def _find_existing_semantic(
        self, user_id: str, cluster: List[TypedMemory],
    ) -> Optional[TypedMemory]:
        """Check if a semantic memory already covers this cluster."""
        if not self.embeddings:
            return None
        try:
            hits = await self.embeddings.search(
                cluster[0].content, limit=5, min_similarity=0.6)
            for hit in hits:
                existing = await self.store.get(hit["memory_id"])
                if (existing and existing.user_id == user_id
                        and existing.memory_type == MemoryType.SEMANTIC):
                    return existing
        except Exception:
            pass
        return None


# ============================================================
# 4. PHI CONTEXT BUILDER
# ============================================================

class PhiContextBuilder:
    """
    Knapsack-based context selection that combines relevance,
    Φ-density, and importance to maximize information value
    within a token budget.
    """

    def __init__(self, scorer: PhiMemoryScorer, store: TypedMemoryStore, embedding_store=None):
        self.scorer = scorer
        self.store = store
        self.embeddings = embedding_store

    async def build_context(
        self, user_id: str,
        messages: List[str],
        token_budget: int = 2000,
    ) -> Dict[str, Any]:
        """
        Build an optimal context block for conversation injection.

        Algorithm:
        1. Retrieve candidates via embedding search
        2. Score each by combined_score = relevance*0.4 + phi*0.3 + importance*0.3
        3. Greedy knapsack by score/token_cost ratio
        4. Criticality check: ensure integration-critical memories are included
        """
        combined_query = " ".join(messages[-5:])[:500]

        # Step 1: Retrieve candidates
        candidates = await self._get_candidates(user_id, combined_query)

        if not candidates:
            return {
                "prompt_block": "",
                "memories_selected": 0,
                "total_tokens": 0,
                "avg_phi": 0,
                "criticality_maintained": True,
            }

        # Step 2: Score candidates
        scored = []
        for mem, similarity in candidates:
            phi_rec = await self.scorer.get_score(mem.id)
            phi_density = phi_rec["phi_score"] if phi_rec else 0.0
            graph_c = phi_rec["graph_contribution"] if phi_rec else 0.0
            importance = ImportanceScorer.compute_importance(mem)
            token_cost = _estimate_tokens(mem)

            combined = (similarity * 0.4
                        + phi_density * 0.3
                        + importance * 0.3)

            scored.append({
                "mem": mem,
                "combined": combined,
                "phi": phi_density,
                "graph_c": graph_c,
                "importance": importance,
                "similarity": similarity,
                "token_cost": token_cost,
                "efficiency": combined / max(1, token_cost),
            })

        # Step 3: Greedy knapsack — sort by efficiency
        scored.sort(key=lambda x: x["efficiency"], reverse=True)

        selected = []
        total_tokens = 0

        for item in scored:
            if total_tokens + item["token_cost"] > token_budget:
                continue
            selected.append(item)
            total_tokens += item["token_cost"]

        # Step 4: Criticality check — ensure high graph_contribution memories are included
        critical_missing = [
            item for item in scored
            if item["graph_c"] > 0.7
            and item not in selected
        ]
        for item in critical_missing[:3]:  # Add up to 3 critical memories
            if item not in selected:
                selected.append(item)
                total_tokens += item["token_cost"]

        # Build prompt block
        prompt_lines = []
        entries = []
        for item in selected:
            mem = item["mem"]
            line = self._format_line(mem)
            prompt_lines.append(line)
            entries.append({
                "id": mem.id,
                "type": mem.memory_type.value,
                "content": mem.content[:100],
                "phi": round(item["phi"], 3),
                "combined_score": round(item["combined"], 3),
                "tokens": item["token_cost"],
            })

        prompt_block = ""
        if prompt_lines:
            prompt_block = (
                "<phi_context>\n"
                + "\n".join(prompt_lines)
                + "\n</phi_context>"
            )

        avg_phi = (sum(i["phi"] for i in selected) / len(selected)) if selected else 0

        return {
            "prompt_block": prompt_block,
            "memories_selected": len(selected),
            "total_tokens": total_tokens,
            "avg_phi": round(avg_phi, 4),
            "criticality_maintained": len(critical_missing) == 0,
            "entries": entries,
        }

    async def _get_candidates(
        self, user_id: str, query: str,
    ) -> List[Tuple[TypedMemory, float]]:
        """Retrieve candidate memories via embedding search + type-based fallbacks."""
        candidates = []
        seen = set()

        # Embedding search
        if self.embeddings and query.strip():
            try:
                hits = await self.embeddings.search(query, limit=50, min_similarity=0.25)
                for hit in hits:
                    mem = await self.store.get(hit["memory_id"])
                    if mem and mem.user_id == user_id and mem.id not in seen:
                        seen.add(mem.id)
                        candidates.append((mem, hit["similarity"]))
            except Exception:
                pass

        # Always include strong procedural memories
        procs = await self.store.query(
            user_id, MemoryType.PROCEDURAL, min_strength=0.3, limit=5)
        for mem in procs:
            if mem.id not in seen:
                seen.add(mem.id)
                candidates.append((mem, 0.3))

        # Always include high-confidence beliefs
        beliefs = await self.store.query(
            user_id, MemoryType.BELIEF, min_strength=0.3, limit=5)
        for mem in beliefs:
            if mem.id not in seen and mem.confidence > 0.6:
                seen.add(mem.id)
                candidates.append((mem, 0.3))

        return candidates

    def _format_line(self, mem: TypedMemory) -> str:
        """Format a memory as a context line."""
        prefix = {
            MemoryType.SEMANTIC: "[FACT]",
            MemoryType.BELIEF: f"[BELIEF conf={mem.confidence:.0%}]",
            MemoryType.EPISODIC: "[EVENT]",
            MemoryType.PROCEDURAL: f"[RULE] When: {mem.trigger} →",
        }.get(mem.memory_type, "[MEM]")

        content = mem.action if mem.memory_type == MemoryType.PROCEDURAL else mem.content
        return f"  {prefix} {content}"
