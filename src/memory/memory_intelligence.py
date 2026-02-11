"""
Memory Intelligence Module — 11 advanced capabilities for TypedMemory.

1.  Memory Decay & Forgetting Curve — apply Ebbinghaus decay, auto-archive stale
2.  Memory Consolidation — episodic→semantic promotion via embedding clustering
3.  Contextual Retrieval (RAG) — auto-inject top-K memories into conversation
4.  Memory Importance Scoring — weight by type, frequency, recency, confirmation
5.  Contradiction Resolution with User Confirmation — surface conflicts
6.  Memory Clustering / Topic Graph — group by embedding similarity
7.  Multi-User Shared Memories — shared context with access control
8.  Export / Import — JSON memory profile portability
9.  Memory Provenance Chain — track source conversation/message/pattern
10. Confidence Decay on Contradicting Evidence — gradual reduction
11. Periodic Self-Audit — health report generation

All features operate on the existing TypedMemoryStore + EmbeddingStore.
"""

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from loguru import logger

from src.memory.memory_types import (
    TypedMemory, TypedMemoryStore, MemoryType, DECAY_PROFILES,
)


# ============================================================
# 1. MEMORY DECAY ENGINE — Ebbinghaus forgetting curve
# ============================================================

class DecayEngine:
    """
    Applies Ebbinghaus-style exponential decay across all memories.
    Memories lose strength over time unless reinforced (accessed/confirmed).
    Stale memories are auto-archived (soft-delete via strength floor).
    """

    def __init__(self, store: TypedMemoryStore):
        self.store = store

    async def apply_decay(self, user_id: str) -> Dict[str, Any]:
        """
        Recalculate strength for all memories using type-specific decay curves.
        Persists updated strengths. Returns decay report.
        """
        all_mems = await self.store.query(user_id, min_strength=0.0, limit=10000)
        updated = 0
        archived = 0
        report_by_type: Dict[str, Dict] = {}

        for mem in all_mems:
            old_strength = mem.strength
            new_strength = mem.current_strength()

            if abs(new_strength - old_strength) > 0.001:
                mem.strength = new_strength
                # Persist without re-running jury
                await self.store.store(mem, bypass_jury=True)
                updated += 1

            mt = mem.memory_type.value
            if mt not in report_by_type:
                report_by_type[mt] = {"count": 0, "avg_strength": 0, "archived": 0}
            report_by_type[mt]["count"] += 1
            report_by_type[mt]["avg_strength"] += new_strength

            params = DECAY_PROFILES[mem.memory_type]
            if new_strength <= params.min_strength * 1.1:
                archived += 1
                report_by_type[mt]["archived"] += 1

        for mt, data in report_by_type.items():
            if data["count"]:
                data["avg_strength"] = round(data["avg_strength"] / data["count"], 3)

        return {
            "total_memories": len(all_mems),
            "updated": updated,
            "at_floor": archived,
            "by_type": report_by_type,
        }

    async def get_decay_forecast(self, user_id: str, hours_ahead: int = 168) -> List[Dict]:
        """
        Forecast which memories will decay below threshold in next N hours.
        """
        all_mems = await self.store.query(user_id, min_strength=0.1, limit=500)
        forecasts = []

        for mem in all_mems:
            params = DECAY_PROFILES[mem.memory_type]
            current = mem.current_strength()
            # Simulate future decay
            future_hours = (time.time() + hours_ahead * 3600 - mem.last_accessed) / 3600
            future_strength = mem.strength * math.pow(2, -future_hours / params.half_life_hours)
            future_strength = max(params.min_strength, future_strength)

            if future_strength < 0.2 and current >= 0.2:
                # Will cross threshold
                hours_to_threshold = params.half_life_hours * math.log2(
                    mem.strength / 0.2) - (time.time() - mem.last_accessed) / 3600
                forecasts.append({
                    "id": mem.id,
                    "content": mem.content[:80],
                    "type": mem.memory_type.value,
                    "current_strength": round(current, 3),
                    "forecast_strength": round(future_strength, 3),
                    "hours_until_weak": round(max(0, hours_to_threshold), 1),
                    "recommendation": "rehearse" if mem.access_count < 3 else "let_decay",
                })

        forecasts.sort(key=lambda x: x["hours_until_weak"])
        return forecasts


# ============================================================
# 2. MEMORY CONSOLIDATION ENGINE — episodic→semantic promotion
# ============================================================

class ConsolidationEngine:
    """
    Promotes repeated episodic patterns into semantic memories using
    embedding-based clustering (not just tag matching).
    """

    def __init__(self, store: TypedMemoryStore, embedding_store=None):
        self.store = store
        self.embeddings = embedding_store

    async def consolidate(
        self, user_id: str, min_cluster_size: int = 3,
        similarity_threshold: float = 0.55
    ) -> Dict[str, Any]:
        """
        Find clusters of similar episodic memories and promote to semantic.
        Uses embedding similarity for clustering.
        """
        episodes = await self.store.query(
            user_id, MemoryType.EPISODIC, min_strength=0.05, limit=200)

        if len(episodes) < min_cluster_size:
            return {"clusters_found": 0, "promoted": 0, "reinforced": 0}

        clusters = []
        used = set()

        if self.embeddings:
            # Embedding-based clustering
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
                    if (other and other.user_id == user_id
                            and other.memory_type == MemoryType.EPISODIC):
                        cluster.append(other)
                        used.add(hit["memory_id"])

                if len(cluster) >= min_cluster_size:
                    used.add(ep.id)
                    clusters.append(cluster)
        else:
            # Fallback: tag-based clustering (existing behavior)
            tag_groups: Dict[str, List[TypedMemory]] = {}
            for ep in episodes:
                for tag in ep.tags:
                    tag_groups.setdefault(tag, []).append(ep)
            for tag, eps in tag_groups.items():
                if len(eps) >= min_cluster_size:
                    clusters.append(eps)

        promoted = 0
        reinforced = 0

        for cluster in clusters:
            # Check if semantic memory already exists for this pattern
            if self.embeddings:
                try:
                    existing_hits = await self.embeddings.search(
                        cluster[0].content, limit=5, min_similarity=0.6)
                    found_existing = False
                    for hit in existing_hits:
                        existing = await self.store.get(hit["memory_id"])
                        if (existing and existing.user_id == user_id
                                and existing.memory_type == MemoryType.SEMANTIC):
                            # Reinforce existing
                            existing.consolidation_count += 1
                            existing.strength = min(1.0, existing.strength + 0.1)
                            existing.consolidated_from = list(set(
                                existing.consolidated_from + [e.id for e in cluster]))
                            await self.store.store(existing, bypass_jury=True)
                            reinforced += 1
                            found_existing = True
                            break
                    if found_existing:
                        continue
                except Exception:
                    pass

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
                source="consolidation",
                confidence=min(0.9, 0.5 + len(cluster) * 0.1),
                consolidated_from=[ep.id for ep in cluster],
                consolidation_count=1,
                tags=list(all_tags | {"consolidated"}),
            )
            await self.store.store(semantic, bypass_jury=True)
            promoted += 1

        return {
            "clusters_found": len(clusters),
            "promoted": promoted,
            "reinforced": reinforced,
            "episodes_analyzed": len(episodes),
        }


# ============================================================
# 3. CONTEXTUAL RETRIEVAL (RAG-style)
# ============================================================

class ContextualRetriever:
    """
    Auto-injects top-K most relevant memories into conversation context.
    Builds a structured prompt section from retrieved memories.
    """

    def __init__(self, store: TypedMemoryStore, embedding_store=None):
        self.store = store
        self.embeddings = embedding_store

    async def retrieve_for_conversation(
        self, user_id: str, messages: List[str],
        max_memories: int = 12, max_tokens: int = 1500
    ) -> Dict[str, Any]:
        """
        Given recent conversation messages, retrieve the most relevant memories.
        Returns structured context block ready for system prompt injection.
        """
        # Combine recent messages into a query
        combined = " ".join(messages[-5:])[:500]

        scored_memories: List[Tuple[float, TypedMemory, str]] = []
        seen = set()

        # Embedding search
        if self.embeddings and combined.strip():
            try:
                hits = await self.embeddings.search(
                    combined, limit=max_memories * 2, min_similarity=0.25)
                for hit in hits:
                    mem = await self.store.get(hit["memory_id"])
                    if mem and mem.user_id == user_id and mem.id not in seen:
                        seen.add(mem.id)
                        score = self._score_memory(mem, hit["similarity"])
                        scored_memories.append((score, mem, "semantic_match"))
            except Exception:
                pass

        # Always include strongest procedural memories
        procs = await self.store.query(
            user_id, MemoryType.PROCEDURAL, min_strength=0.3, limit=5)
        for mem in procs:
            if mem.id not in seen:
                seen.add(mem.id)
                scored_memories.append((
                    self._score_memory(mem, 0.3), mem, "active_procedure"))

        # Always include high-confidence beliefs
        beliefs = await self.store.query(
            user_id, MemoryType.BELIEF, min_strength=0.3, limit=5)
        for mem in beliefs:
            if mem.id not in seen and mem.confidence > 0.6:
                seen.add(mem.id)
                scored_memories.append((
                    self._score_memory(mem, 0.3), mem, "active_belief"))

        # Sort by score, take top-K
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        selected = scored_memories[:max_memories]

        # Build prompt block
        prompt_lines = []
        total_chars = 0
        entries = []

        for score, mem, reason in selected:
            line = self._format_memory_line(mem)
            if total_chars + len(line) > max_tokens * 4:  # rough char limit
                break
            prompt_lines.append(line)
            total_chars += len(line)
            entries.append({
                "id": mem.id, "type": mem.memory_type.value,
                "content": mem.content[:100], "score": round(score, 3),
                "reason": reason,
            })

        prompt_block = ""
        if prompt_lines:
            prompt_block = (
                "<user_memory_context>\n"
                + "\n".join(prompt_lines)
                + "\n</user_memory_context>"
            )

        return {
            "prompt_block": prompt_block,
            "memories_injected": len(entries),
            "entries": entries,
        }

    def _score_memory(self, mem: TypedMemory, similarity: float) -> float:
        """Score a memory for retrieval relevance."""
        importance = ImportanceScorer.compute_importance(mem)
        strength = mem.current_strength()
        recency = 1.0 / (1.0 + (time.time() - mem.last_accessed) / 86400)
        return (similarity * 0.4 + importance * 0.3
                + strength * 0.2 + recency * 0.1)

    def _format_memory_line(self, mem: TypedMemory) -> str:
        """Format a memory as a single context line."""
        prefix = {
            MemoryType.SEMANTIC: "[FACT]",
            MemoryType.BELIEF: f"[BELIEF conf={mem.confidence:.0%}]",
            MemoryType.EPISODIC: "[EVENT]",
            MemoryType.PROCEDURAL: f"[RULE] When: {mem.trigger} →",
        }.get(mem.memory_type, "[MEM]")

        content = mem.action if mem.memory_type == MemoryType.PROCEDURAL else mem.content
        return f"  {prefix} {content}"


# ============================================================
# 4. MEMORY IMPORTANCE SCORING
# ============================================================

class ImportanceScorer:
    """
    Scores memory importance based on type, frequency, recency, confirmation.
    """

    # Type base weights
    TYPE_WEIGHTS = {
        MemoryType.PROCEDURAL: 0.9,   # Rules are critical
        MemoryType.SEMANTIC: 0.7,     # Facts are important
        MemoryType.BELIEF: 0.5,       # Beliefs are uncertain
        MemoryType.EPISODIC: 0.3,     # Episodes fade
    }

    @classmethod
    def compute_importance(cls, mem: TypedMemory) -> float:
        """
        Compute importance score [0-1] for a memory.
        Factors: type weight, access frequency, recency, strength, confidence.
        """
        type_w = cls.TYPE_WEIGHTS.get(mem.memory_type, 0.5)

        # Frequency factor: log scale of access count
        freq = min(1.0, math.log1p(mem.access_count) / 5.0)

        # Recency factor: exponential decay from last access
        hours_since = (time.time() - mem.last_accessed) / 3600
        recency = math.exp(-hours_since / 168)  # 7-day half-life

        # Strength and confidence
        strength = mem.current_strength()
        confidence = mem.confidence

        # Consolidation bonus
        consol_bonus = min(0.2, mem.consolidation_count * 0.05)

        # Weighted combination
        score = (type_w * 0.3 + freq * 0.15 + recency * 0.15
                 + strength * 0.2 + confidence * 0.15 + consol_bonus * 0.05)

        return min(1.0, max(0.0, score))

    @classmethod
    async def rank_memories(
        cls, store: TypedMemoryStore, user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Rank all memories by importance."""
        all_mems = await store.query(user_id, min_strength=0.05, limit=500)
        ranked = []
        for mem in all_mems:
            score = cls.compute_importance(mem)
            ranked.append({
                "id": mem.id,
                "type": mem.memory_type.value,
                "content": mem.content[:100],
                "importance": round(score, 3),
                "strength": round(mem.current_strength(), 3),
                "access_count": mem.access_count,
                "tags": mem.tags,
            })
        ranked.sort(key=lambda x: x["importance"], reverse=True)
        return ranked[:limit]


# ============================================================
# 5. CONTRADICTION RESOLUTION WITH USER CONFIRMATION
# ============================================================

class ConflictSurfacer:
    """
    Detects conflicts and surfaces them for user resolution
    instead of silently superseding.
    """

    def __init__(self, store: TypedMemoryStore, embedding_store=None):
        self.store = store
        self.embeddings = embedding_store
        self._pending_conflicts: List[Dict] = []

    async def detect_and_surface(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Find conflicts and return them as actionable items for the user.
        """
        contradictions = await self.store.detect_contradictions(user_id)
        surfaced = []

        for c in contradictions:
            conflict = {
                "conflict_id": str(uuid4())[:8],
                "memory_a": {
                    "id": c["memory_a"]["id"],
                    "content": c["memory_a"]["content"],
                    "type": c["memory_a"]["type"],
                    "strength": c["memory_a"]["strength"],
                },
                "memory_b": {
                    "id": c["memory_b"]["id"],
                    "content": c["memory_b"]["content"],
                    "type": c["memory_b"]["type"],
                    "strength": c["memory_b"]["strength"],
                },
                "detection_method": c.get("detection", "unknown"),
                "actions": [
                    {"action": "keep_a", "description": f"Keep: {c['memory_a']['content'][:60]}"},
                    {"action": "keep_b", "description": f"Keep: {c['memory_b']['content'][:60]}"},
                    {"action": "keep_both", "description": "Both are valid in different contexts"},
                    {"action": "delete_both", "description": "Neither is correct anymore"},
                ],
            }
            surfaced.append(conflict)
            self._pending_conflicts.append(conflict)

        return surfaced

    async def resolve_conflict(
        self, conflict_id: str, action: str, user_id: str
    ) -> Dict[str, Any]:
        """
        Resolve a surfaced conflict based on user choice.
        """
        conflict = None
        for c in self._pending_conflicts:
            if c["conflict_id"] == conflict_id:
                conflict = c
                break

        if not conflict:
            return {"error": f"Conflict {conflict_id} not found"}

        a_id = conflict["memory_a"]["id"]
        b_id = conflict["memory_b"]["id"]

        if action == "keep_a":
            await self.store.forget_memory(b_id, reason=f"User resolved conflict {conflict_id}")
            result = {"kept": a_id, "deleted": b_id}
        elif action == "keep_b":
            await self.store.forget_memory(a_id, reason=f"User resolved conflict {conflict_id}")
            result = {"kept": b_id, "deleted": a_id}
        elif action == "keep_both":
            # Add context to both
            mem_a = await self.store.get(a_id)
            mem_b = await self.store.get(b_id)
            if mem_a:
                mem_a.context += f" [User confirmed valid alongside {b_id[:8]}]"
                await self.store.store(mem_a, bypass_jury=True)
            if mem_b:
                mem_b.context += f" [User confirmed valid alongside {a_id[:8]}]"
                await self.store.store(mem_b, bypass_jury=True)
            result = {"kept": [a_id, b_id], "action": "both_confirmed"}
        elif action == "delete_both":
            await self.store.forget_memory(a_id, reason=f"User resolved conflict {conflict_id}")
            await self.store.forget_memory(b_id, reason=f"User resolved conflict {conflict_id}")
            result = {"deleted": [a_id, b_id]}
        else:
            return {"error": f"Unknown action: {action}"}

        self._pending_conflicts = [
            c for c in self._pending_conflicts if c["conflict_id"] != conflict_id]
        result["conflict_id"] = conflict_id
        result["action"] = action
        return result


# ============================================================
# 6. MEMORY CLUSTERING / TOPIC GRAPH
# ============================================================

class MemoryClusterer:
    """
    Groups related memories into clusters using embedding similarity.
    Enables "tell me everything about X" queries.
    """

    def __init__(self, store: TypedMemoryStore, embedding_store=None):
        self.store = store
        self.embeddings = embedding_store

    async def build_clusters(
        self, user_id: str, similarity_threshold: float = 0.5,
        min_cluster_size: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Build topic clusters from all memories using greedy clustering.
        """
        all_mems = await self.store.query(user_id, min_strength=0.05, limit=500)
        if not all_mems or not self.embeddings:
            # Fallback: tag-based clustering
            return self._tag_based_clusters(all_mems)

        clusters = []
        used = set()

        for mem in all_mems:
            if mem.id in used:
                continue
            try:
                similar = await self.embeddings.find_similar(
                    mem.id, limit=30, min_similarity=similarity_threshold)
            except Exception:
                continue

            cluster_mems = [mem]
            used.add(mem.id)

            for hit in similar:
                if hit["memory_id"] in used:
                    continue
                other = await self.store.get(hit["memory_id"])
                if other and other.user_id == user_id:
                    cluster_mems.append(other)
                    used.add(hit["memory_id"])

            if len(cluster_mems) >= min_cluster_size:
                # Determine cluster topic from most common tags
                tag_counts: Dict[str, int] = {}
                for m in cluster_mems:
                    for t in m.tags:
                        tag_counts[t] = tag_counts.get(t, 0) + 1
                top_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)[:3]

                clusters.append({
                    "cluster_id": str(uuid4())[:8],
                    "topic": ", ".join(top_tags) if top_tags else "misc",
                    "size": len(cluster_mems),
                    "types": dict(defaultdict(int, {
                        m.memory_type.value: 1 for m in cluster_mems})),
                    "avg_strength": round(
                        sum(m.current_strength() for m in cluster_mems) / len(cluster_mems), 3),
                    "memories": [{
                        "id": m.id, "type": m.memory_type.value,
                        "content": m.content[:80],
                        "importance": round(ImportanceScorer.compute_importance(m), 3),
                    } for m in cluster_mems[:10]],  # Cap at 10 per cluster for output
                })

        clusters.sort(key=lambda c: c["size"], reverse=True)
        return clusters

    def _tag_based_clusters(self, memories: List[TypedMemory]) -> List[Dict]:
        """Fallback clustering by shared tags."""
        tag_groups: Dict[str, List[TypedMemory]] = {}
        for m in memories:
            for t in m.tags:
                tag_groups.setdefault(t, []).append(m)

        clusters = []
        for tag, mems in tag_groups.items():
            if len(mems) >= 2:
                clusters.append({
                    "cluster_id": str(uuid4())[:8],
                    "topic": tag,
                    "size": len(mems),
                    "memories": [{"id": m.id, "content": m.content[:80]}
                                 for m in mems[:10]],
                })
        return clusters


# ============================================================
# 7. MULTI-USER SHARED MEMORIES
# ============================================================

class SharedMemoryManager:
    """
    Manages shared memories across users with access control.
    Uses a separate table for shared memory links.
    """

    def __init__(self, store: TypedMemoryStore):
        self.store = store
        self._initialized = False

    async def _ensure_schema(self):
        if self._initialized:
            return
        await self.store._conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_memories (
                memory_id TEXT NOT NULL,
                owner_user_id TEXT NOT NULL,
                shared_with_user_id TEXT NOT NULL,
                permission TEXT DEFAULT 'read',
                shared_at REAL NOT NULL,
                PRIMARY KEY (memory_id, shared_with_user_id)
            )
        """)
        await self.store._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shared_mem_user
            ON shared_memories(shared_with_user_id)
        """)
        await self.store._conn.commit()
        self._initialized = True

    async def share_memory(
        self, memory_id: str, owner_id: str,
        target_user_id: str, permission: str = "read"
    ) -> Dict[str, Any]:
        """Share a memory with another user."""
        await self._ensure_schema()
        mem = await self.store.get(memory_id)
        if not mem or mem.user_id != owner_id:
            return {"error": "Memory not found or not owned by you"}

        await self.store._conn.execute("""
            INSERT OR REPLACE INTO shared_memories
            (memory_id, owner_user_id, shared_with_user_id, permission, shared_at)
            VALUES (?,?,?,?,?)
        """, (memory_id, owner_id, target_user_id, permission, time.time()))
        await self.store._conn.commit()

        return {"shared": memory_id, "with": target_user_id, "permission": permission}

    async def get_shared_with_me(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all memories shared with a user."""
        await self._ensure_schema()
        cursor = await self.store._conn.execute("""
            SELECT sm.memory_id, sm.owner_user_id, sm.permission, sm.shared_at,
                   tm.content, tm.memory_type, tm.confidence
            FROM shared_memories sm
            JOIN typed_memories tm ON sm.memory_id = tm.id
            WHERE sm.shared_with_user_id = ?
            ORDER BY sm.shared_at DESC
        """, (user_id,))
        rows = await cursor.fetchall()
        return [{
            "memory_id": r[0], "owner": r[1], "permission": r[2],
            "shared_at": r[3], "content": r[4], "type": r[5],
            "confidence": r[6],
        } for r in rows]

    async def unshare_memory(
        self, memory_id: str, owner_id: str, target_user_id: str
    ) -> bool:
        """Revoke sharing."""
        await self._ensure_schema()
        await self.store._conn.execute("""
            DELETE FROM shared_memories
            WHERE memory_id = ? AND owner_user_id = ? AND shared_with_user_id = ?
        """, (memory_id, owner_id, target_user_id))
        await self.store._conn.commit()
        return True


# ============================================================
# 8. EXPORT / IMPORT
# ============================================================

class MemoryPortability:
    """Export/import memory profiles as JSON."""

    def __init__(self, store: TypedMemoryStore):
        self.store = store

    async def export_profile(self, user_id: str) -> Dict[str, Any]:
        """Export all memories for a user as a portable JSON structure."""
        all_mems = await self.store.query(user_id, min_strength=0.0, limit=10000)

        memories = []
        for mem in all_mems:
            memories.append({
                "id": mem.id,
                "memory_type": mem.memory_type.value,
                "content": mem.content,
                "context": mem.context,
                "source": mem.source,
                "strength": mem.strength,
                "created_at": mem.created_at,
                "last_accessed": mem.last_accessed,
                "access_count": mem.access_count,
                "confidence": mem.confidence,
                "evidence_for": mem.evidence_for,
                "evidence_against": mem.evidence_against,
                "episode_timestamp": mem.episode_timestamp,
                "participants": mem.participants,
                "emotional_valence": mem.emotional_valence,
                "trigger": mem.trigger,
                "action": mem.action,
                "success_count": mem.success_count,
                "failure_count": mem.failure_count,
                "consolidated_from": mem.consolidated_from,
                "consolidation_count": mem.consolidation_count,
                "tags": mem.tags,
            })

        return {
            "version": "1.0",
            "exported_at": time.time(),
            "exported_at_human": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id,
            "memory_count": len(memories),
            "memories": memories,
        }

    async def import_profile(
        self, profile: Dict[str, Any], target_user_id: Optional[str] = None,
        merge: bool = True
    ) -> Dict[str, Any]:
        """
        Import a memory profile. If merge=True, keeps existing memories.
        If merge=False, replaces all memories for the user.
        """
        user_id = target_user_id or profile.get("user_id", "imported")
        memories = profile.get("memories", [])

        if not merge:
            # Delete existing
            existing = await self.store.query(user_id, min_strength=0.0, limit=10000)
            for mem in existing:
                await self.store.forget_memory(mem.id, reason="import_replace")

        imported = 0
        skipped = 0

        for m in memories:
            try:
                mem = TypedMemory(
                    id=m.get("id", str(uuid4())),
                    memory_type=MemoryType(m["memory_type"]),
                    user_id=user_id,
                    content=m["content"],
                    context=m.get("context", ""),
                    source=m.get("source", "imported"),
                    strength=m.get("strength", 0.8),
                    created_at=m.get("created_at", time.time()),
                    last_accessed=m.get("last_accessed", time.time()),
                    access_count=m.get("access_count", 0),
                    confidence=m.get("confidence", 0.5),
                    evidence_for=m.get("evidence_for", []),
                    evidence_against=m.get("evidence_against", []),
                    episode_timestamp=m.get("episode_timestamp", 0),
                    participants=m.get("participants", []),
                    emotional_valence=m.get("emotional_valence", 0),
                    trigger=m.get("trigger", ""),
                    action=m.get("action", ""),
                    success_count=m.get("success_count", 0),
                    failure_count=m.get("failure_count", 0),
                    consolidated_from=m.get("consolidated_from", []),
                    consolidation_count=m.get("consolidation_count", 0),
                    tags=m.get("tags", []),
                )
                await self.store.store(mem, bypass_jury=True)
                imported += 1
            except Exception as e:
                logger.warning(f"Import skip: {e}")
                skipped += 1

        return {
            "user_id": user_id,
            "imported": imported,
            "skipped": skipped,
            "merge": merge,
        }


# ============================================================
# 9. MEMORY PROVENANCE CHAIN
# ============================================================

class ProvenanceTracker:
    """
    Tracks where each memory came from: conversation, message, extraction pattern.
    """

    def __init__(self, store: TypedMemoryStore):
        self.store = store
        self._initialized = False

    async def _ensure_schema(self):
        if self._initialized:
            return
        await self.store._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_provenance (
                memory_id TEXT PRIMARY KEY,
                conversation_id TEXT DEFAULT '',
                message_index INTEGER DEFAULT -1,
                message_text TEXT DEFAULT '',
                extraction_pattern TEXT DEFAULT '',
                extraction_confidence REAL DEFAULT 0,
                pipeline_stage TEXT DEFAULT '',
                jury_verdict TEXT DEFAULT '',
                jury_explanation TEXT DEFAULT '',
                created_at REAL NOT NULL
            )
        """)
        await self.store._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prov_conv
            ON memory_provenance(conversation_id)
        """)
        await self.store._conn.commit()
        self._initialized = True

    async def record_provenance(
        self, memory_id: str, conversation_id: str = "",
        message_index: int = -1, message_text: str = "",
        extraction_pattern: str = "", extraction_confidence: float = 0,
        pipeline_stage: str = "", jury_verdict: str = "",
        jury_explanation: str = ""
    ):
        """Record provenance for a memory."""
        await self._ensure_schema()
        await self.store._conn.execute("""
            INSERT OR REPLACE INTO memory_provenance
            (memory_id, conversation_id, message_index, message_text,
             extraction_pattern, extraction_confidence, pipeline_stage,
             jury_verdict, jury_explanation, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (memory_id, conversation_id, message_index, message_text[:500],
              extraction_pattern, extraction_confidence, pipeline_stage,
              jury_verdict, jury_explanation, time.time()))
        await self.store._conn.commit()

    async def get_provenance(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get provenance for a memory."""
        await self._ensure_schema()
        cursor = await self.store._conn.execute(
            "SELECT * FROM memory_provenance WHERE memory_id = ?", (memory_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return {
            "memory_id": row[0], "conversation_id": row[1],
            "message_index": row[2], "message_text": row[3],
            "extraction_pattern": row[4], "extraction_confidence": row[5],
            "pipeline_stage": row[6], "jury_verdict": row[7],
            "jury_explanation": row[8], "created_at": row[9],
        }

    async def get_conversation_memories(self, conversation_id: str) -> List[Dict]:
        """Get all memories from a specific conversation."""
        await self._ensure_schema()
        cursor = await self.store._conn.execute("""
            SELECT mp.*, tm.content, tm.memory_type
            FROM memory_provenance mp
            JOIN typed_memories tm ON mp.memory_id = tm.id
            WHERE mp.conversation_id = ?
            ORDER BY mp.message_index
        """, (conversation_id,))
        rows = await cursor.fetchall()
        return [{
            "memory_id": r[0], "message_index": r[2],
            "extraction_pattern": r[4], "content": r[10],
            "type": r[11],
        } for r in rows]


# ============================================================
# 10. CONFIDENCE DECAY ON CONTRADICTING EVIDENCE
# ============================================================

class ConfidenceDecayEngine:
    """
    Gradually reduces confidence when contradicting evidence appears,
    rather than binary supersede/keep.
    """

    def __init__(self, store: TypedMemoryStore, embedding_store=None):
        self.store = store
        self.embeddings = embedding_store

    async def apply_evidence_decay(self, user_id: str) -> List[Dict[str, Any]]:
        """
        For each belief/semantic memory, check for contradicting evidence
        and gradually reduce confidence.
        """
        beliefs = await self.store.query(
            user_id, MemoryType.BELIEF, min_strength=0.1, limit=100)
        semantics = await self.store.query(
            user_id, MemoryType.SEMANTIC, min_strength=0.1, limit=200)

        adjustments = []

        for belief in beliefs:
            contra_count = len(belief.evidence_against)
            support_count = len(belief.evidence_for)

            if contra_count == 0:
                continue

            # Gradual decay: each contradicting piece reduces confidence by diminishing amount
            # First contradiction: -0.08, second: -0.06, third: -0.04, etc.
            total_penalty = 0
            for i in range(contra_count):
                total_penalty += max(0.02, 0.08 - i * 0.02)

            # Support counteracts
            total_support = 0
            for i in range(support_count):
                total_support += max(0.01, 0.05 - i * 0.01)

            net_change = total_support - total_penalty
            old_conf = belief.confidence
            new_conf = max(0.05, min(0.95, belief.confidence + net_change))

            if abs(new_conf - old_conf) > 0.01:
                belief.confidence = new_conf
                await self.store.store(belief, bypass_jury=True)
                adjustments.append({
                    "id": belief.id,
                    "content": belief.content[:80],
                    "old_confidence": round(old_conf, 3),
                    "new_confidence": round(new_conf, 3),
                    "evidence_for": support_count,
                    "evidence_against": contra_count,
                    "net_change": round(net_change, 3),
                })

        return adjustments


# ============================================================
# 11. PERIODIC SELF-AUDIT
# ============================================================

class MemoryAuditor:
    """
    Generates comprehensive health reports by running all checks.
    """

    def __init__(self, store: TypedMemoryStore, embedding_store=None, jury=None):
        self.store = store
        self.embeddings = embedding_store
        self.jury = jury
        self.decay_engine = DecayEngine(store)
        self.consolidation = ConsolidationEngine(store, embedding_store)
        self.confidence_decay = ConfidenceDecayEngine(store, embedding_store)
        self.clusterer = MemoryClusterer(store, embedding_store)

    async def full_audit(self, user_id: str) -> Dict[str, Any]:
        """
        Run all health checks and return comprehensive report.
        """
        t0 = time.time()
        report: Dict[str, Any] = {"user_id": user_id, "timestamp": t0}

        # 1. Memory stats
        report["stats"] = await self.store.get_stats(user_id)

        # 2. Decay status
        report["decay"] = await self.decay_engine.apply_decay(user_id)

        # 3. Decay forecast
        report["decay_forecast"] = await self.decay_engine.get_decay_forecast(user_id)

        # 4. Consolidation opportunities
        report["consolidation"] = await self.consolidation.consolidate(user_id)

        # 5. Belief auto-check
        report["belief_check"] = await self.store.belief_auto_check(user_id)

        # 6. Confidence decay
        report["confidence_adjustments"] = await self.confidence_decay.apply_evidence_decay(user_id)

        # 7. Contradiction detection
        report["contradictions"] = await self.store.detect_contradictions(user_id)

        # 8. Cluster analysis
        report["clusters"] = await self.clusterer.build_clusters(user_id)

        # 9. Auto-prune
        report["pruned"] = await self.store.auto_prune(user_id)

        # 10. Jury health (if available)
        if self.jury:
            jury_stats = self.jury.get_stats()
            report["jury_health"] = {
                "deliberations": jury_stats.get("deliberations", 0),
                "drift_alerts": jury_stats.get("drift_alerts", []),
                "tuning_recommendations": jury_stats.get("tuning_recommendations", []),
            }

        # 11. Importance ranking (top 10)
        report["top_memories"] = await ImportanceScorer.rank_memories(
            self.store, user_id, limit=10)

        elapsed = time.time() - t0
        report["audit_duration_ms"] = round(elapsed * 1000, 1)

        # Health score
        total = report["stats"].get("total", 0)
        if total > 0:
            avg_strength = sum(
                v.get("avg_strength", 0) for v in report["stats"].values()
                if isinstance(v, dict) and "avg_strength" in v
            ) / max(1, sum(1 for v in report["stats"].values()
                           if isinstance(v, dict) and "avg_strength" in v))
            contradictions = len(report.get("contradictions", []))
            pruned = report.get("pruned", {}).get("total_pruned", 0)

            health = max(0, min(100, int(
                100 * avg_strength
                - contradictions * 5
                - pruned * 2
            )))
        else:
            health = 100

        report["health_score"] = health
        report["health_label"] = (
            "excellent" if health >= 80 else
            "good" if health >= 60 else
            "needs_attention" if health >= 40 else "poor"
        )

        logger.info(f"Memory audit for {user_id}: health={health} ({report['health_label']}) "
                     f"[{elapsed*1000:.0f}ms]")
        return report
