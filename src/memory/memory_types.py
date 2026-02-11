"""
Memory Type System for PLTM

Implements cognitively-inspired memory types with distinct storage,
decay, consolidation, and retrieval behaviors:

  EPISODIC  — Specific events with temporal context. Fast decay unless rehearsed.
  SEMANTIC  — General facts and stable knowledge. Slow decay, reinforced by repetition.
  BELIEF    — Inferences/opinions that may be wrong. Confidence-tracked, revisable.
  PROCEDURAL — Learned patterns and how-to knowledge. Very stable once established.

Each type has its own:
  - Decay curve (how fast memories fade)
  - Consolidation rules (when episodic → semantic)
  - Retrieval strategy (recency vs relevance vs confidence)
  - Conflict resolution behavior
"""

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiosqlite
from loguru import logger


class MemoryType(str, Enum):
    EPISODIC = "episodic"       # "On Feb 10, user got frustrated with verbose answer"
    SEMANTIC = "semantic"       # "User prefers concise responses"
    BELIEF = "belief"           # "User likely has CS background"
    PROCEDURAL = "procedural"   # "When user says 'just do it', skip explanation"


@dataclass
class DecayParams:
    """Decay parameters per memory type."""
    half_life_hours: float      # Time for strength to drop to 50%
    min_strength: float         # Floor — never decays below this
    rehearsal_boost: float      # Strength boost per recall
    max_strength: float = 1.0   # Ceiling


# Cognitively-inspired decay rates
DECAY_PROFILES: Dict[MemoryType, DecayParams] = {
    MemoryType.EPISODIC: DecayParams(
        half_life_hours=48,       # Fades in ~2 days without rehearsal
        min_strength=0.05,        # Almost gone eventually
        rehearsal_boost=0.3,      # Recalling strengthens significantly
    ),
    MemoryType.SEMANTIC: DecayParams(
        half_life_hours=720,      # ~30 days — very stable
        min_strength=0.2,         # Never fully forgotten
        rehearsal_boost=0.1,      # Small boost (already stable)
    ),
    MemoryType.BELIEF: DecayParams(
        half_life_hours=168,      # ~7 days — moderate
        min_strength=0.1,         # Fades if not reinforced
        rehearsal_boost=0.15,     # Moderate reinforcement
    ),
    MemoryType.PROCEDURAL: DecayParams(
        half_life_hours=2160,     # ~90 days — very slow decay
        min_strength=0.4,         # Hard to forget once learned
        rehearsal_boost=0.05,     # Barely needs reinforcement
    ),
}


@dataclass
class TypedMemory:
    """A memory with explicit type and type-specific behavior."""
    id: str
    memory_type: MemoryType
    user_id: str
    content: str                    # The actual memory content
    context: str = ""               # When/where this was observed
    source: str = ""                # How we know this (user_stated, inferred, observed)
    
    # Strength & decay
    strength: float = 1.0           # Current strength [0-1]
    created_at: float = 0.0         # Unix timestamp
    last_accessed: float = 0.0      # Unix timestamp
    access_count: int = 0
    
    # Belief-specific
    confidence: float = 0.5         # How confident (beliefs especially)
    evidence_for: List[str] = field(default_factory=list)    # IDs of supporting memories
    evidence_against: List[str] = field(default_factory=list) # IDs of contradicting memories
    
    # Episodic-specific
    episode_timestamp: float = 0.0  # When the event happened
    participants: List[str] = field(default_factory=list)
    emotional_valence: float = 0.0  # -1 (negative) to +1 (positive)
    
    # Procedural-specific
    trigger: str = ""               # What activates this procedure
    action: str = ""                # What to do
    success_count: int = 0
    failure_count: int = 0
    
    # Consolidation
    consolidated_from: List[str] = field(default_factory=list)  # Episodic IDs that formed this semantic
    consolidation_count: int = 0    # How many times reinforced
    
    # Tags
    tags: List[str] = field(default_factory=list)
    
    def current_strength(self) -> float:
        """Calculate current strength with type-specific decay."""
        if self.created_at == 0:
            return self.strength
        
        params = DECAY_PROFILES[self.memory_type]
        hours_since_access = (time.time() - self.last_accessed) / 3600
        
        # Exponential decay: S(t) = S0 * 2^(-t/half_life)
        decay_factor = math.pow(2, -hours_since_access / params.half_life_hours)
        decayed = self.strength * decay_factor
        
        return max(params.min_strength, min(params.max_strength, decayed))
    
    def rehearse(self) -> float:
        """Access/recall this memory — boosts strength."""
        params = DECAY_PROFILES[self.memory_type]
        self.last_accessed = time.time()
        self.access_count += 1
        self.strength = min(params.max_strength, self.current_strength() + params.rehearsal_boost)
        return self.strength


class TypedMemoryStore:
    """
    Storage layer for typed memories.
    
    Adds a `typed_memories` table alongside the existing atoms table.
    The two systems coexist — typed memories are the new cognitive layer,
    atoms remain the knowledge graph layer.
    
    Optionally integrates with EmbeddingStore for semantic search.
    """
    
    def __init__(self, db_path: str, embedding_store=None, jury=None):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self.embeddings = embedding_store  # Optional EmbeddingStore
        self.jury = jury  # Optional MemoryJury validation gate
    
    async def connect(self):
        """Connect and create schema."""
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._setup_schema()
    
    async def close(self):
        if self._conn:
            await self._conn.close()
    
    async def _setup_schema(self):
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS typed_memories (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                context TEXT DEFAULT '',
                source TEXT DEFAULT '',
                
                strength REAL DEFAULT 1.0,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                
                confidence REAL DEFAULT 0.5,
                evidence_for TEXT DEFAULT '[]',
                evidence_against TEXT DEFAULT '[]',
                
                episode_timestamp REAL DEFAULT 0,
                participants TEXT DEFAULT '[]',
                emotional_valence REAL DEFAULT 0,
                
                trigger TEXT DEFAULT '',
                action TEXT DEFAULT '',
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                
                consolidated_from TEXT DEFAULT '[]',
                consolidation_count INTEGER DEFAULT 0,
                
                tags TEXT DEFAULT '[]'
            )
        """)
        
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_typed_mem_user_type
            ON typed_memories(user_id, memory_type)
        """)
        
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_typed_mem_strength
            ON typed_memories(strength DESC)
        """)
        
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_typed_mem_type
            ON typed_memories(memory_type)
        """)
        
        # FTS for content search
        await self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS typed_memories_fts
            USING fts5(id UNINDEXED, content, context, trigger, action,
                       content=typed_memories, content_rowid=rowid)
        """)
        
        await self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS typed_mem_fts_insert AFTER INSERT ON typed_memories BEGIN
                INSERT INTO typed_memories_fts(rowid, id, content, context, trigger, action)
                VALUES (new.rowid, new.id, new.content, new.context, new.trigger, new.action);
            END
        """)
        
        await self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS typed_mem_fts_delete AFTER DELETE ON typed_memories BEGIN
                DELETE FROM typed_memories_fts WHERE rowid = old.rowid;
            END
        """)
        
        await self._conn.commit()
        logger.info("TypedMemoryStore schema initialized")
    
    # ========== STORE ==========
    
    async def store(self, mem: TypedMemory, bypass_jury: bool = False) -> str:
        """Store a typed memory. Runs jury validation unless bypass_jury=True."""
        now = time.time()
        if mem.created_at == 0:
            mem.created_at = now
        if mem.last_accessed == 0:
            mem.last_accessed = now
        if not mem.id:
            mem.id = str(uuid4())
        
        # Jury validation gate
        if self.jury and not bypass_jury:
            from src.memory.memory_jury import Verdict
            # Gather existing contents for dedup check
            existing = []
            try:
                cursor = await self._conn.execute(
                    "SELECT content FROM typed_memories WHERE user_id = ? AND memory_type = ? LIMIT 50",
                    (mem.user_id, mem.memory_type.value)
                )
                rows = await cursor.fetchall()
                existing = [r[0] for r in rows]
            except Exception:
                pass
            
            decision = self.jury.deliberate(
                content=mem.content,
                memory_type=mem.memory_type.value,
                episode_timestamp=mem.episode_timestamp,
                existing_contents=existing,
            )
            
            if decision.verdict == Verdict.REJECT:
                logger.warning(f"Jury REJECTED memory: {decision.explanation} — content: {mem.content[:80]}")
                return ""  # Empty ID signals rejection
            elif decision.verdict == Verdict.QUARANTINE:
                logger.info(f"Jury QUARANTINED memory: {decision.explanation} — reducing strength")
                mem.strength = max(0.1, mem.strength * 0.5)  # Store with reduced strength
                mem.context = (mem.context + f" [QUARANTINED: {decision.explanation}]").strip()
        
        await self._conn.execute("""
            INSERT OR REPLACE INTO typed_memories
            (id, memory_type, user_id, content, context, source,
             strength, created_at, last_accessed, access_count,
             confidence, evidence_for, evidence_against,
             episode_timestamp, participants, emotional_valence,
             trigger, action, success_count, failure_count,
             consolidated_from, consolidation_count, tags)
            VALUES (?,?,?,?,?,?, ?,?,?,?, ?,?,?, ?,?,?, ?,?,?,?, ?,?,?)
        """, (
            mem.id, mem.memory_type.value, mem.user_id, mem.content,
            mem.context, mem.source,
            mem.strength, mem.created_at, mem.last_accessed, mem.access_count,
            mem.confidence,
            json.dumps(mem.evidence_for), json.dumps(mem.evidence_against),
            mem.episode_timestamp, json.dumps(mem.participants), mem.emotional_valence,
            mem.trigger, mem.action, mem.success_count, mem.failure_count,
            json.dumps(mem.consolidated_from), mem.consolidation_count,
            json.dumps(mem.tags),
        ))
        await self._conn.commit()
        
        # Auto-index embedding if store available (fire-and-forget to avoid timeout)
        if self.embeddings:
            async def _index_bg(mid, text):
                try:
                    await self.embeddings.index_memory(mid, text)
                except Exception as e:
                    logger.warning(f"Embedding index failed for {mid}: {e}")
            text = mem.content
            if mem.trigger:
                text += f" | trigger: {mem.trigger}"
            if mem.action:
                text += f" | action: {mem.action}"
            asyncio.create_task(_index_bg(mem.id, text))
        
        return mem.id
    
    # ========== RETRIEVE ==========
    
    async def get(self, memory_id: str) -> Optional[TypedMemory]:
        """Get a single memory by ID, applying decay."""
        cursor = await self._conn.execute(
            "SELECT * FROM typed_memories WHERE id = ?", (memory_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        mem = self._row_to_memory(row)
        # Update strength with decay
        mem.strength = mem.current_strength()
        return mem
    
    async def recall(self, memory_id: str) -> Optional[TypedMemory]:
        """Recall a memory — retrieves it AND strengthens it (rehearsal)."""
        mem = await self.get(memory_id)
        if not mem:
            return None
        mem.rehearse()
        await self.store(mem)  # Persist updated strength
        return mem
    
    async def query(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        min_strength: float = 0.0,
        limit: int = 50,
        tags: Optional[List[str]] = None,
    ) -> List[TypedMemory]:
        """Query memories with type-specific retrieval."""
        conditions = ["user_id = ?"]
        params: list = [user_id]
        
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type.value)
        
        if min_strength > 0:
            conditions.append("strength >= ?")
            params.append(min_strength)
        
        where = " AND ".join(conditions)
        
        # Type-specific ordering
        if memory_type == MemoryType.EPISODIC:
            order = "episode_timestamp DESC, last_accessed DESC"
        elif memory_type == MemoryType.BELIEF:
            order = "confidence DESC, strength DESC"
        elif memory_type == MemoryType.PROCEDURAL:
            order = "success_count DESC, strength DESC"
        else:
            order = "strength DESC, last_accessed DESC"
        
        cursor = await self._conn.execute(
            f"SELECT * FROM typed_memories WHERE {where} ORDER BY {order} LIMIT ?",
            params + [limit]
        )
        rows = await cursor.fetchall()
        
        memories = [self._row_to_memory(row) for row in rows]
        
        # Apply decay to returned memories
        for mem in memories:
            mem.strength = mem.current_strength()
        
        # Filter by tags if specified
        if tags:
            memories = [m for m in memories if any(t in m.tags for t in tags)]
        
        return memories
    
    async def search(self, user_id: str, query: str, limit: int = 20) -> List[TypedMemory]:
        """Full-text search across all memory types."""
        cursor = await self._conn.execute("""
            SELECT tm.* FROM typed_memories tm
            JOIN typed_memories_fts fts ON tm.id = fts.id
            WHERE typed_memories_fts MATCH ? AND tm.user_id = ?
            ORDER BY rank
            LIMIT ?
        """, (query, user_id, limit))
        rows = await cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]
    
    # ========== CONSOLIDATION ==========
    
    async def consolidate_episodes(self, user_id: str, min_episodes: int = 3) -> List[TypedMemory]:
        """
        Consolidation: repeated episodic patterns → semantic memory.
        
        If the same theme appears in 3+ episodes, extract a semantic fact.
        Returns newly created semantic memories.
        """
        episodes = await self.query(user_id, MemoryType.EPISODIC, min_strength=0.1)
        
        if len(episodes) < min_episodes:
            return []
        
        # Group episodes by tags
        tag_groups: Dict[str, List[TypedMemory]] = {}
        for ep in episodes:
            for tag in ep.tags:
                tag_groups.setdefault(tag, []).append(ep)
        
        new_semantics = []
        for tag, eps in tag_groups.items():
            if len(eps) >= min_episodes:
                # Check if semantic memory for this pattern already exists
                existing = await self.query(
                    user_id, MemoryType.SEMANTIC, tags=[tag]
                )
                
                if existing:
                    # Reinforce existing semantic memory
                    existing[0].consolidation_count += 1
                    existing[0].strength = min(1.0, existing[0].strength + 0.1)
                    await self.store(existing[0])
                else:
                    # Create new semantic memory from episodes
                    contents = [ep.content for ep in eps[:5]]
                    semantic = TypedMemory(
                        id=str(uuid4()),
                        memory_type=MemoryType.SEMANTIC,
                        user_id=user_id,
                        content=f"Pattern from {len(eps)} episodes [{tag}]: {'; '.join(contents[:3])}",
                        source="consolidation",
                        confidence=min(0.9, 0.5 + len(eps) * 0.1),
                        consolidated_from=[ep.id for ep in eps],
                        consolidation_count=1,
                        tags=[tag, "consolidated"],
                    )
                    await self.store(semantic)
                    new_semantics.append(semantic)
        
        return new_semantics
    
    # ========== BELIEF OPERATIONS ==========
    
    async def update_belief(
        self, belief_id: str, 
        evidence_type: str = "for",
        evidence_id: str = "",
        confidence_delta: float = 0.0
    ) -> Optional[TypedMemory]:
        """Update a belief with new evidence."""
        belief = await self.get(belief_id)
        if not belief or belief.memory_type != MemoryType.BELIEF:
            return None
        
        if evidence_type == "for" and evidence_id:
            belief.evidence_for.append(evidence_id)
            belief.confidence = min(0.99, belief.confidence + abs(confidence_delta))
        elif evidence_type == "against" and evidence_id:
            belief.evidence_against.append(evidence_id)
            belief.confidence = max(0.01, belief.confidence - abs(confidence_delta))
        else:
            belief.confidence = max(0.01, min(0.99, belief.confidence + confidence_delta))
        
        belief.strength = min(1.0, belief.strength + 0.1)  # Any update strengthens
        await self.store(belief)
        return belief
    
    # ========== PROCEDURAL OPERATIONS ==========
    
    async def record_procedure_outcome(
        self, procedure_id: str, success: bool
    ) -> Optional[TypedMemory]:
        """Record whether a procedure worked or not."""
        proc = await self.get(procedure_id)
        if not proc or proc.memory_type != MemoryType.PROCEDURAL:
            return None
        
        if success:
            proc.success_count += 1
            proc.strength = min(1.0, proc.strength + 0.05)
        else:
            proc.failure_count += 1
            proc.strength = max(0.1, proc.strength - 0.1)
        
        await self.store(proc)
        return proc
    
    # ========== STATS ==========
    
    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics by type."""
        stats = {}
        for mt in MemoryType:
            cursor = await self._conn.execute(
                "SELECT COUNT(*), AVG(strength), AVG(confidence) FROM typed_memories WHERE user_id = ? AND memory_type = ?",
                (user_id, mt.value)
            )
            row = await cursor.fetchone()
            stats[mt.value] = {
                "count": row[0] or 0,
                "avg_strength": round(row[1] or 0, 3),
                "avg_confidence": round(row[2] or 0, 3),
            }
        
        # Total
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM typed_memories WHERE user_id = ?", (user_id,)
        )
        stats["total"] = (await cursor.fetchone())[0]
        
        return stats
    
    # ========== BELIEF AUTO-CHECK ==========
    
    async def belief_auto_check(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Re-evaluate all beliefs against current semantic evidence.
        
        For each belief, searches for supporting and contradicting semantic
        memories using embeddings (if available) or FTS. Adjusts confidence
        and returns a report of changes.
        """
        beliefs = await self.query(user_id, MemoryType.BELIEF, min_strength=0.05, limit=100)
        semantics = await self.query(user_id, MemoryType.SEMANTIC, min_strength=0.1, limit=200)
        
        if not beliefs:
            return []
        
        report = []
        
        for belief in beliefs:
            supporting = []
            contradicting = []
            old_confidence = belief.confidence
            
            # Use embeddings for semantic similarity if available
            if self.embeddings:
                try:
                    hits = await self.embeddings.search(belief.content, limit=10, min_similarity=0.4)
                    for hit in hits:
                        if hit["memory_id"] == belief.id:
                            continue
                        mem = await self.get(hit["memory_id"])
                        if not mem or mem.user_id != user_id or mem.memory_type != MemoryType.SEMANTIC:
                            continue
                        # High similarity to a semantic fact = evidence for
                        if hit["similarity"] > 0.6:
                            supporting.append({"id": mem.id, "content": mem.content[:80], "sim": hit["similarity"]})
                        elif hit["similarity"] > 0.4:
                            # Moderate similarity — check for negation patterns
                            bl = belief.content.lower()
                            ml = mem.content.lower()
                            negation_words = ["not", "never", "doesn't", "don't", "isn't", "aren't", "won't", "can't", "no"]
                            b_neg = any(w in bl for w in negation_words)
                            m_neg = any(w in ml for w in negation_words)
                            if b_neg != m_neg:
                                contradicting.append({"id": mem.id, "content": mem.content[:80], "sim": hit["similarity"]})
                            else:
                                supporting.append({"id": mem.id, "content": mem.content[:80], "sim": hit["similarity"]})
                except Exception:
                    pass
            
            # Fallback: tag-based matching
            if not supporting and not contradicting:
                for sem in semantics:
                    shared_tags = set(belief.tags) & set(sem.tags)
                    if shared_tags:
                        supporting.append({"id": sem.id, "content": sem.content[:80]})
            
            # Adjust confidence based on evidence
            new_confidence = belief.confidence
            if supporting and not contradicting:
                boost = min(0.15, len(supporting) * 0.05)
                new_confidence = min(0.95, belief.confidence + boost)
            elif contradicting and not supporting:
                penalty = min(0.2, len(contradicting) * 0.07)
                new_confidence = max(0.05, belief.confidence - penalty)
            elif contradicting and supporting:
                # Mixed evidence — slight decrease (uncertainty)
                ratio = len(contradicting) / (len(supporting) + len(contradicting))
                new_confidence = max(0.1, belief.confidence - ratio * 0.1)
            
            changed = abs(new_confidence - old_confidence) > 0.01
            if changed:
                belief.confidence = new_confidence
                belief.evidence_for = list(set(belief.evidence_for + [s["id"] for s in supporting]))
                belief.evidence_against = list(set(belief.evidence_against + [c["id"] for c in contradicting]))
                await self.store(belief)
            
            report.append({
                "belief_id": belief.id,
                "content": belief.content[:100],
                "old_confidence": round(old_confidence, 3),
                "new_confidence": round(new_confidence, 3),
                "changed": changed,
                "supporting": len(supporting),
                "contradicting": len(contradicting),
                "direction": "up" if new_confidence > old_confidence else ("down" if new_confidence < old_confidence else "stable"),
            })
        
        return report
    
    # ========== CONTRADICTION DETECTION ==========
    
    async def detect_contradictions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Find contradicting memories.
        
        Strategy:
        1. Embedding-based: find high-similarity pairs with opposing sentiment/negation
        2. Keyword-based: compare memories with shared tags for opposing content patterns
        """
        # Get all active semantic + belief memories
        semantics = await self.query(user_id, MemoryType.SEMANTIC, min_strength=0.1, limit=200)
        beliefs = await self.query(user_id, MemoryType.BELIEF, min_strength=0.1, limit=200)
        all_mems = semantics + beliefs
        
        contradictions = []
        seen_pairs = set()
        
        # Opposing keyword pairs
        opposites = [
            ("concise", "verbose"), ("concise", "detailed"),
            ("direct", "indirect"), ("formal", "casual"), ("formal", "informal"),
            ("likes", "dislikes"), ("prefers", "avoids"),
            ("expert", "beginner"), ("advanced", "novice"),
            ("patient", "impatient"), ("fast", "slow"),
            ("simple", "complex"), ("technical", "non-technical"),
            ("positive", "negative"), ("yes", "no"),
            ("always", "never"), ("loves", "hates"),
        ]
        opposite_map = {}
        for a, b in opposites:
            opposite_map[a] = b
            opposite_map[b] = a
        
        for i, m1 in enumerate(all_mems):
            for m2 in all_mems[i + 1:]:
                pair_key = tuple(sorted([m1.id, m2.id]))
                if pair_key in seen_pairs:
                    continue
                
                # Check tag overlap (same topic = potential contradiction)
                shared_tags = set(m1.tags) & set(m2.tags)
                if not shared_tags:
                    continue
                
                # Check for opposing content
                c1 = m1.content.lower()
                c2 = m2.content.lower()
                
                conflict_words = []
                for word, opp in opposite_map.items():
                    if word in c1 and opp in c2:
                        conflict_words.append((word, opp))
                
                if conflict_words:
                    seen_pairs.add(pair_key)
                    contradictions.append({
                        "memory_a": {"id": m1.id, "type": m1.memory_type.value,
                                     "content": m1.content, "confidence": round(m1.confidence, 3),
                                     "strength": round(m1.current_strength(), 3)},
                        "memory_b": {"id": m2.id, "type": m2.memory_type.value,
                                     "content": m2.content, "confidence": round(m2.confidence, 3),
                                     "strength": round(m2.current_strength(), 3)},
                        "shared_tags": list(shared_tags),
                        "conflicts": [{"word_a": w, "word_b": o} for w, o in conflict_words],
                        "detection": "keyword",
                        "suggestion": "Keep the one with higher confidence/strength, or update both."
                    })
        
        # Embedding-based contradiction detection
        # Find semantically similar memories with opposing negation patterns
        if self.embeddings:
            negation_words = {"not", "never", "doesn't", "don't", "isn't", "aren't",
                              "won't", "can't", "no", "without", "lack", "unable"}
            
            for mem in all_mems:
                try:
                    similar = await self.embeddings.find_similar(
                        mem.id, limit=5, min_similarity=0.5
                    )
                except Exception:
                    continue
                
                for hit in similar:
                    pair_key = tuple(sorted([mem.id, hit["memory_id"]]))
                    if pair_key in seen_pairs:
                        continue
                    
                    other = await self.get(hit["memory_id"])
                    if not other or other.user_id != user_id:
                        continue
                    if other.memory_type not in (MemoryType.SEMANTIC, MemoryType.BELIEF):
                        continue
                    
                    # High similarity + different negation = likely contradiction
                    m1_neg = any(w in mem.content.lower().split() for w in negation_words)
                    m2_neg = any(w in other.content.lower().split() for w in negation_words)
                    
                    if m1_neg != m2_neg and hit["similarity"] > 0.5:
                        seen_pairs.add(pair_key)
                        contradictions.append({
                            "memory_a": {"id": mem.id, "type": mem.memory_type.value,
                                         "content": mem.content, "confidence": round(mem.confidence, 3),
                                         "strength": round(mem.current_strength(), 3)},
                            "memory_b": {"id": other.id, "type": other.memory_type.value,
                                         "content": other.content, "confidence": round(other.confidence, 3),
                                         "strength": round(other.current_strength(), 3)},
                            "similarity": hit["similarity"],
                            "detection": "embedding",
                            "suggestion": "These are semantically similar but one negates the other. Resolve the conflict."
                        })
        
        return contradictions
    
    # ========== CROSS-TYPE SYNTHESIZED RETRIEVAL ==========
    
    async def what_do_i_know_about(self, user_id: str, topic: str, limit: int = 30) -> Dict[str, Any]:
        """
        Synthesized retrieval across ALL memory types for a topic.
        
        Uses embedding search (semantic similarity) when available,
        falls back to FTS + tag matching. Returns organized results.
        """
        seen_ids = set()
        search_results = []
        similarity_scores: Dict[str, float] = {}  # memory_id -> similarity
        
        # 1. Embedding search (semantic similarity) — primary if available
        if self.embeddings:
            try:
                emb_hits = await self.embeddings.search(topic, limit=limit, min_similarity=0.25)
                for hit in emb_hits:
                    mem = await self.get(hit["memory_id"])
                    if mem and mem.user_id == user_id and mem.id not in seen_ids:
                        seen_ids.add(mem.id)
                        search_results.append(mem)
                        similarity_scores[mem.id] = hit["similarity"]
            except Exception as e:
                logger.warning(f"Embedding search failed, falling back to FTS: {e}")
        
        # 2. FTS search (keyword match) — supplement
        try:
            fts_results = await self.search(user_id, topic, limit=limit)
            for mem in fts_results:
                if mem.id not in seen_ids:
                    seen_ids.add(mem.id)
                    search_results.append(mem)
        except Exception:
            pass
        
        # 3. Tag matching
        all_mems = await self.query(user_id, limit=200)
        tag_matches = [m for m in all_mems if topic.lower() in " ".join(m.tags).lower()
                       and m.id not in seen_ids]
        
        combined = search_results + tag_matches
        
        # Organize by type
        result: Dict[str, Any] = {
            "topic": topic,
            "facts": [],       # semantic
            "beliefs": [],     # belief
            "episodes": [],    # episodic
            "procedures": [],  # procedural
            "total": len(combined),
        }
        
        for mem in combined:
            entry = {
                "id": mem.id, "content": mem.content,
                "strength": round(mem.current_strength(), 3),
                "confidence": round(mem.confidence, 3),
                "tags": mem.tags,
            }
            if mem.id in similarity_scores:
                entry["semantic_similarity"] = similarity_scores[mem.id]
            
            if mem.memory_type == MemoryType.SEMANTIC:
                entry["source"] = mem.source
                result["facts"].append(entry)
            elif mem.memory_type == MemoryType.BELIEF:
                entry["evidence_for"] = len(mem.evidence_for)
                entry["evidence_against"] = len(mem.evidence_against)
                result["beliefs"].append(entry)
            elif mem.memory_type == MemoryType.EPISODIC:
                entry["context"] = mem.context
                entry["emotional_valence"] = mem.emotional_valence
                entry["when"] = mem.episode_timestamp
                result["episodes"].append(entry)
            elif mem.memory_type == MemoryType.PROCEDURAL:
                entry["trigger"] = mem.trigger
                entry["action"] = mem.action
                total = mem.success_count + mem.failure_count
                entry["success_rate"] = round(mem.success_count / max(1, total), 2)
                result["procedures"].append(entry)
        
        # Summary line
        parts = []
        if result["facts"]:
            parts.append(f"{len(result['facts'])} facts")
        if result["beliefs"]:
            parts.append(f"{len(result['beliefs'])} beliefs")
        if result["episodes"]:
            parts.append(f"{len(result['episodes'])} episodes")
        if result["procedures"]:
            parts.append(f"{len(result['procedures'])} procedures")
        result["summary"] = f"Found {', '.join(parts)} about '{topic}'" if parts else f"No memories about '{topic}'"
        
        return result
    
    # ========== AUTO-TAGGING TAXONOMY ==========
    
    TAXONOMY = {
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
    
    @classmethod
    def auto_tag(cls, content: str, existing_tags: Optional[List[str]] = None) -> List[str]:
        """
        Auto-classify content into taxonomy domains.
        Returns list of domain tags to add.
        """
        tags = list(existing_tags or [])
        content_lower = content.lower()
        words = set(content_lower.split())
        
        for domain, keywords in cls.TAXONOMY.items():
            if domain in tags:
                continue
            # Check if any keyword appears in content
            matches = sum(1 for kw in keywords if kw in words or kw in content_lower)
            if matches >= 2:  # Need at least 2 keyword hits
                tags.append(domain)
            elif matches == 1 and len(content.split()) < 15:
                # For short content, 1 match is enough
                tags.append(domain)
        
        return tags
    
    async def auto_tag_memory(self, memory_id: str) -> Optional[TypedMemory]:
        """Auto-tag a single memory and persist."""
        mem = await self.get(memory_id)
        if not mem:
            return None
        
        new_tags = self.auto_tag(mem.content, mem.tags)
        if mem.trigger:
            new_tags = self.auto_tag(mem.trigger, new_tags)
        if mem.action:
            new_tags = self.auto_tag(mem.action, new_tags)
        
        if set(new_tags) != set(mem.tags):
            mem.tags = new_tags
            await self.store(mem)
        
        return mem
    
    async def auto_tag_all(self, user_id: str) -> Dict[str, int]:
        """Auto-tag all untagged or under-tagged memories for a user."""
        all_mems = await self.query(user_id, limit=1000)
        updated = 0
        for mem in all_mems:
            new_tags = self.auto_tag(mem.content, mem.tags)
            if mem.trigger:
                new_tags = self.auto_tag(mem.trigger, new_tags)
            if set(new_tags) != set(mem.tags):
                mem.tags = new_tags
                await self.store(mem)
                updated += 1
        
        # Count by domain
        domain_counts: Dict[str, int] = {}
        for mem in all_mems:
            for tag in mem.tags:
                domain_counts[tag] = domain_counts.get(tag, 0) + 1
        
        return {"updated": updated, "total": len(all_mems), "domains": domain_counts}
    
    # ========== MEMORY CORRECTION & FORGETTING ==========
    
    async def correct_memory(
        self, memory_id: str, new_content: str,
        reason: str = "", new_confidence: Optional[float] = None
    ) -> Optional[TypedMemory]:
        """
        Correct a memory's content. Keeps provenance trail.
        Old content is stored in context as correction history.
        """
        mem = await self.get(memory_id)
        if not mem:
            return None
        
        # Store correction history in context
        correction = f"[CORRECTED {time.strftime('%Y-%m-%d %H:%M')}] "
        correction += f"Was: '{mem.content[:200]}'"
        if reason:
            correction += f" | Reason: {reason}"
        
        if mem.context:
            mem.context = correction + " || " + mem.context
        else:
            mem.context = correction
        
        mem.content = new_content
        if new_confidence is not None:
            mem.confidence = max(0.01, min(0.99, new_confidence))
        
        mem.last_accessed = time.time()
        await self.store(mem)
        return mem
    
    async def forget_memory(self, memory_id: str, reason: str = "") -> bool:
        """
        Explicitly delete a memory. Returns True if deleted.
        Logs the deletion for audit trail.
        """
        mem = await self.get(memory_id)
        if not mem:
            return False
        
        # Log before deleting
        logger.info(f"Forgetting memory {memory_id}: '{mem.content[:80]}' reason='{reason}'")
        
        await self._conn.execute("DELETE FROM typed_memories WHERE id = ?", (memory_id,))
        await self._conn.commit()
        
        # Clean up embedding
        if self.embeddings:
            try:
                await self.embeddings.remove_embedding(memory_id)
            except Exception:
                pass
        
        return True
    
    async def auto_prune(self, user_id: str, strength_threshold: float = 0.05) -> Dict[str, int]:
        """
        Auto-prune memories that have decayed below threshold.
        Returns count of pruned memories by type.
        """
        all_mems = await self.query(user_id, min_strength=0.0, limit=10000)
        pruned: Dict[str, int] = {}
        
        pruned_ids = []
        for mem in all_mems:
            current = mem.current_strength()
            if current < strength_threshold:
                mt = mem.memory_type.value
                pruned[mt] = pruned.get(mt, 0) + 1
                pruned_ids.append(mem.id)
                await self._conn.execute(
                    "DELETE FROM typed_memories WHERE id = ?", (mem.id,)
                )
        
        if pruned_ids:
            await self._conn.commit()
            # Clean up embeddings
            if self.embeddings:
                for mid in pruned_ids:
                    try:
                        await self.embeddings.remove_embedding(mid)
                    except Exception:
                        pass
        
        total = sum(pruned.values())
        logger.info(f"Auto-pruned {total} memories for {user_id}: {pruned}")
        return {"pruned": pruned, "total_pruned": total, "threshold": strength_threshold}
    
    # ========== CONVERSATION-AWARE PRE-FETCH ==========
    
    async def get_relevant_context(
        self, user_id: str, conversation_topic: str, limit: int = 15
    ) -> Dict[str, Any]:
        """
        Pre-fetch memories relevant to the current conversation.
        
        Combines:
        - FTS search on topic
        - Recent episodic memories (last 48h)
        - Strongest semantic memories
        - Active beliefs
        - Relevant procedures
        
        Returns a ready-to-use context block.
        """
        results: Dict[str, Any] = {"topic": conversation_topic, "memories": []}
        seen_ids = set()
        
        # 1. Embedding search for semantic topic relevance (primary)
        if self.embeddings:
            try:
                emb_hits = await self.embeddings.search(conversation_topic, limit=limit, min_similarity=0.3)
                for hit in emb_hits:
                    mem = await self.get(hit["memory_id"])
                    if mem and mem.user_id == user_id and mem.id not in seen_ids:
                        seen_ids.add(mem.id)
                        entry = self._mem_to_context_entry(mem, "semantic_match")
                        entry["similarity"] = hit["similarity"]
                        results["memories"].append(entry)
            except Exception:
                pass
        
        # 1b. FTS search for keyword topic relevance (supplement)
        try:
            topic_matches = await self.search(user_id, conversation_topic, limit=limit)
            for mem in topic_matches:
                if mem.id not in seen_ids:
                    seen_ids.add(mem.id)
                    results["memories"].append(self._mem_to_context_entry(mem, "keyword_match"))
        except Exception:
            pass
        
        # 2. Recent episodes (last 48h)
        cutoff = time.time() - 48 * 3600
        cursor = await self._conn.execute(
            """SELECT * FROM typed_memories 
               WHERE user_id = ? AND memory_type = 'episodic' AND episode_timestamp > ?
               ORDER BY episode_timestamp DESC LIMIT 5""",
            (user_id, cutoff)
        )
        for row in await cursor.fetchall():
            mem = self._row_to_memory(row)
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                results["memories"].append(self._mem_to_context_entry(mem, "recent_episode"))
        
        # 3. Strongest semantic memories
        cursor = await self._conn.execute(
            """SELECT * FROM typed_memories 
               WHERE user_id = ? AND memory_type = 'semantic' AND strength > 0.3
               ORDER BY strength DESC, confidence DESC LIMIT 5""",
            (user_id,)
        )
        for row in await cursor.fetchall():
            mem = self._row_to_memory(row)
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                results["memories"].append(self._mem_to_context_entry(mem, "core_fact"))
        
        # 4. Active beliefs
        cursor = await self._conn.execute(
            """SELECT * FROM typed_memories 
               WHERE user_id = ? AND memory_type = 'belief' AND confidence > 0.5
               ORDER BY confidence DESC LIMIT 3""",
            (user_id,)
        )
        for row in await cursor.fetchall():
            mem = self._row_to_memory(row)
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                results["memories"].append(self._mem_to_context_entry(mem, "active_belief"))
        
        # 5. Relevant procedures
        cursor = await self._conn.execute(
            """SELECT * FROM typed_memories 
               WHERE user_id = ? AND memory_type = 'procedural' AND strength > 0.3
               ORDER BY success_count DESC LIMIT 3""",
            (user_id,)
        )
        for row in await cursor.fetchall():
            mem = self._row_to_memory(row)
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                results["memories"].append(self._mem_to_context_entry(mem, "procedure"))
        
        # Trim to limit
        results["memories"] = results["memories"][:limit]
        results["total"] = len(results["memories"])
        
        return results
    
    def _mem_to_context_entry(self, mem: TypedMemory, relevance_reason: str) -> Dict[str, Any]:
        """Convert a memory to a context entry."""
        entry = {
            "id": mem.id,
            "type": mem.memory_type.value,
            "content": mem.content,
            "strength": round(mem.current_strength(), 3),
            "confidence": round(mem.confidence, 3),
            "relevance": relevance_reason,
            "tags": mem.tags,
        }
        if mem.memory_type == MemoryType.PROCEDURAL:
            entry["trigger"] = mem.trigger
            entry["action"] = mem.action
        if mem.memory_type == MemoryType.EPISODIC:
            entry["emotional_valence"] = mem.emotional_valence
        return entry
    
    # ========== USER TIMELINE ==========
    
    async def user_timeline(
        self, user_id: str, limit: int = 20, offset: int = 0
    ) -> Dict[str, Any]:
        """
        Chronological view of all interactions/memories for a user.
        Most recent first.
        """
        cursor = await self._conn.execute(
            """SELECT * FROM typed_memories 
               WHERE user_id = ?
               ORDER BY created_at DESC
               LIMIT ? OFFSET ?""",
            (user_id, limit, offset)
        )
        rows = await cursor.fetchall()
        
        entries = []
        for row in rows:
            mem = self._row_to_memory(row)
            entry = {
                "id": mem.id,
                "type": mem.memory_type.value,
                "content": mem.content,
                "created_at": mem.created_at,
                "created_at_human": time.strftime("%Y-%m-%d %H:%M", time.localtime(mem.created_at)),
                "strength": round(mem.current_strength(), 3),
                "confidence": round(mem.confidence, 3),
                "tags": mem.tags,
            }
            if mem.memory_type == MemoryType.EPISODIC:
                entry["emotional_valence"] = mem.emotional_valence
                entry["context"] = mem.context
            elif mem.memory_type == MemoryType.PROCEDURAL:
                entry["trigger"] = mem.trigger
                entry["action"] = mem.action
            entries.append(entry)
        
        # Get total count
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM typed_memories WHERE user_id = ?", (user_id,)
        )
        total = (await cursor.fetchone())[0]
        
        return {
            "user_id": user_id,
            "entries": entries,
            "showing": len(entries),
            "total": total,
            "offset": offset,
            "has_more": offset + limit < total,
        }
    
    # ========== INTERNAL ==========
    
    def _row_to_memory(self, row) -> TypedMemory:
        return TypedMemory(
            id=row["id"],
            memory_type=MemoryType(row["memory_type"]),
            user_id=row["user_id"],
            content=row["content"],
            context=row["context"] or "",
            source=row["source"] or "",
            strength=row["strength"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            confidence=row["confidence"],
            evidence_for=json.loads(row["evidence_for"] or "[]"),
            evidence_against=json.loads(row["evidence_against"] or "[]"),
            episode_timestamp=row["episode_timestamp"],
            participants=json.loads(row["participants"] or "[]"),
            emotional_valence=row["emotional_valence"],
            trigger=row["trigger"] or "",
            action=row["action"] or "",
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            consolidated_from=json.loads(row["consolidated_from"] or "[]"),
            consolidation_count=row["consolidation_count"],
            tags=json.loads(row["tags"] or "[]"),
        )
