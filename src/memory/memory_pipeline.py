"""
3-Lane Memory Pipeline for TypedMemory

Lane 0: FAST LANE — TypedMemoryExtractor
  Extracts typed memories from conversation messages using rule-based patterns.
  Classifies into episodic/semantic/belief/procedural with confidence scores.
  Target: <100ms

Lane 1: JURY LANE — MemoryJury (from memory_jury.py)
  4-judge deliberation: Safety, Quality, Temporal, Consensus
  Verdicts: APPROVE / QUARANTINE / REJECT
  MetaJudge tracks judge accuracy over time.

Lane 2: WRITE LANE — TypedMemoryReconciler
  Embedding-based conflict detection + resolution.
  Actions: STORE / SUPERSEDE / CONTEXTUALIZE / REJECT / MERGE
  Handles dedup, temporal supersession, and conditional branching.

Orchestrator: TypedMemoryPipeline
  Extract → Jury → Reconcile → Store
  Returns progressive updates at each stage.
"""

import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from loguru import logger

from src.memory.memory_types import TypedMemory, MemoryType


# ============================================================
# LANE 0: FAST LANE — EXTRACTION
# ============================================================

@dataclass
class ExtractionResult:
    memory: TypedMemory
    pattern_name: str
    extraction_confidence: float


class TypedMemoryExtractor:
    """
    Extract typed memories from conversation messages.
    
    Rule-based patterns classify into 4 memory types:
    - EPISODIC: events, interactions, experiences
    - SEMANTIC: facts, preferences, skills, attributes
    - BELIEF: opinions, inferences, hypotheses
    - PROCEDURAL: behavioral patterns, triggers → actions
    """

    # (regex, memory_type, confidence, pattern_name)
    # Priority order: most specific first
    PATTERNS = [
        # === PROCEDURAL: trigger → action patterns ===
        (r"(?:when|if|whenever)\s+(?:I|the user|they)\s+(?:say|ask|want|need|type)s?\s+['\"]?(.+?)['\"]?,?\s+(?:then\s+)?(?:I should|you should|just|please)\s+(.+?)(?:\.|$)",
         MemoryType.PROCEDURAL, 0.85, "procedural_when_then"),
        (r"(?:don't|do not|never)\s+(.+?)(?:\s+unless\s+(.+?))?(?:\.|$)",
         MemoryType.PROCEDURAL, 0.75, "procedural_dont"),
        (r"(?:always|make sure to)\s+(.+?)(?:\.|$)",
         MemoryType.PROCEDURAL, 0.75, "procedural_always"),

        # === BELIEF: opinions, hypotheses, inferences ===
        (r"I (?:think|believe|suspect|guess|assume|feel like|bet)\s+(?:that\s+)?(.+?)(?:\.|$)",
         MemoryType.BELIEF, 0.7, "belief_think"),
        (r"(?:probably|maybe|perhaps|likely|possibly)\s+(.+?)(?:\.|$)",
         MemoryType.BELIEF, 0.5, "belief_hedge"),
        (r"I'm (?:pretty |fairly |quite )?(?:sure|certain|confident)\s+(?:that\s+)?(.+?)(?:\.|$)",
         MemoryType.BELIEF, 0.8, "belief_confident"),
        (r"(?:it seems|it looks) like\s+(.+?)(?:\.|$)",
         MemoryType.BELIEF, 0.5, "belief_seems"),

        # === EPISODIC: events, experiences, interactions ===
        (r"(?:today|yesterday|last (?:week|month|year|night|time)|this (?:morning|afternoon|evening)|just now|earlier|recently)\s*,?\s*(?:I|we)\s+(.+?)(?:\.|$)",
         MemoryType.EPISODIC, 0.85, "episodic_temporal"),
        (r"I (?:just|recently)\s+(.+?)(?:\.|$)",
         MemoryType.EPISODIC, 0.8, "episodic_recent"),
        (r"(?:we|I) (?:had|went|saw|met|did|tried|found|discovered|noticed|experienced|encountered)\s+(.+?)(?:\.|$)",
         MemoryType.EPISODIC, 0.8, "episodic_past_event"),
        (r"(?:I was|we were)\s+(.+?)(?:\.|$)",
         MemoryType.EPISODIC, 0.7, "episodic_was"),
        (r"(?:I got|I received|I heard)\s+(.+?)(?:\.|$)",
         MemoryType.EPISODIC, 0.75, "episodic_got"),

        # === SEMANTIC: facts, preferences, skills, attributes ===
        (r"I (?:work|am working) (?:at|in|for|on)\s+(.+?)(?:\.|,|$)",
         MemoryType.SEMANTIC, 0.9, "semantic_work"),
        (r"I (?:live|am living) in\s+(.+?)(?:\.|,|$)",
         MemoryType.SEMANTIC, 0.9, "semantic_live"),
        (r"my name is\s+(.+?)(?:\.|,|$)",
         MemoryType.SEMANTIC, 0.95, "semantic_name"),
        (r"I (?:really )?(?:love|like|enjoy|prefer|hate|dislike|can't stand)\s+(.+?)(?:\.|,|$)",
         MemoryType.SEMANTIC, 0.85, "semantic_preference"),
        (r"I'm (?:a |an )?(.+?)(?:\.|,|$)",
         MemoryType.SEMANTIC, 0.8, "semantic_identity"),
        (r"I (?:use|know|speak|study|practice)\s+(.+?)(?:\.|,|$)",
         MemoryType.SEMANTIC, 0.8, "semantic_skill"),
        (r"I (?:am|have)\s+(.+?)(?:\.|,|$)",
         MemoryType.SEMANTIC, 0.75, "semantic_attribute"),
        (r"I (?:need|want|would like)\s+(.+?)(?:\.|,|$)",
         MemoryType.SEMANTIC, 0.7, "semantic_want"),

        # === CATCH-ALL: default to episodic for first-person statements ===
        (r"I\s+(.{10,}?)(?:\.|$)",
         MemoryType.EPISODIC, 0.5, "catchall_first_person"),
    ]

    def __init__(self):
        self.extraction_count = 0
        logger.info(f"TypedMemoryExtractor initialized ({len(self.PATTERNS)} patterns)")

    def extract(self, message: str, user_id: str, context: str = "") -> List[ExtractionResult]:
        """
        Extract typed memories from a message.
        
        Returns list of ExtractionResult with memory + metadata.
        Multiple memories can be extracted from one message.
        """
        results = []
        used_spans = []  # Track matched spans to avoid overlaps

        for pattern, mem_type, confidence, name in self.PATTERNS:
            for match in re.finditer(pattern, message, re.IGNORECASE):
                span = match.span()
                # Skip if this span overlaps with an already-matched span
                if any(self._spans_overlap(span, used) for used in used_spans):
                    continue

                # Extract content
                content = match.group(1).strip() if match.lastindex >= 1 else match.group(0).strip()
                if len(content) < 3:
                    continue

                # For procedural: extract trigger + action
                trigger = ""
                action = ""
                if mem_type == MemoryType.PROCEDURAL:
                    if match.lastindex and match.lastindex >= 2:
                        trigger = match.group(1).strip()
                        action = match.group(2).strip()
                        content = f"When {trigger}, then {action}"
                    elif name == "procedural_dont":
                        trigger = "general"
                        action = f"Don't {content}"
                        content = action
                    elif name == "procedural_always":
                        trigger = "general"
                        action = f"Always {content}"
                        content = action

                # Detect emotional valence for episodic
                valence = self._detect_valence(message) if mem_type == MemoryType.EPISODIC else 0.0

                mem = TypedMemory(
                    id="",
                    memory_type=mem_type,
                    user_id=user_id,
                    content=content,
                    context=context,
                    source="pipeline_extraction",
                    confidence=confidence,
                    emotional_valence=valence,
                    episode_timestamp=time.time() if mem_type == MemoryType.EPISODIC else 0,
                    trigger=trigger,
                    action=action,
                    tags=[],
                )

                results.append(ExtractionResult(
                    memory=mem,
                    pattern_name=name,
                    extraction_confidence=confidence,
                ))
                used_spans.append(span)
                self.extraction_count += 1

        logger.debug(f"Extracted {len(results)} typed memories from message ({len(message)} chars)")
        return results

    def _spans_overlap(self, a: tuple, b: tuple) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    def _detect_valence(self, text: str) -> float:
        """Simple sentiment detection for emotional valence."""
        t = text.lower()
        pos = ["happy", "great", "awesome", "love", "excited", "glad", "wonderful", "fantastic", "enjoy", "fun"]
        neg = ["frustrated", "angry", "annoyed", "hate", "terrible", "awful", "sad", "upset", "disappointed", "stuck"]
        pos_count = sum(1 for w in pos if w in t)
        neg_count = sum(1 for w in neg if w in t)
        if pos_count + neg_count == 0:
            return 0.0
        return round((pos_count - neg_count) / (pos_count + neg_count), 2)

    def get_stats(self) -> Dict[str, Any]:
        return {"total_extractions": self.extraction_count, "pattern_count": len(self.PATTERNS)}


# ============================================================
# LANE 2: WRITE LANE — RECONCILIATION
# ============================================================

class ReconcileAction(str, Enum):
    STORE = "STORE"              # Store as new memory
    SUPERSEDE = "SUPERSEDE"      # Replace existing memory
    CONTEXTUALIZE = "CONTEXTUALIZE"  # Both coexist (different contexts)
    MERGE = "MERGE"              # Merge into existing
    REJECT = "REJECT"            # Don't store (existing is better)


@dataclass
class ReconcileDecision:
    action: ReconcileAction
    confidence: float
    explanation: str
    existing_id: Optional[str] = None  # ID of conflicting memory


class TypedMemoryReconciler:
    """
    Write Lane: conflict detection + resolution for typed memories.
    
    Uses embedding similarity to find conflicts, then applies rules:
    1. Exact duplicate → REJECT
    2. High similarity + same type → SUPERSEDE (newer wins) or MERGE
    3. High similarity + different type → CONTEXTUALIZE
    4. Moderate similarity + opposing sentiment → flag contradiction
    5. No conflict → STORE
    """

    # Predicates/patterns that are exclusive (only one can be true)
    EXCLUSIVE_PATTERNS = [
        "work at", "work for", "work in", "live in", "from",
        "name is", "my name",
    ]

    NEGATION_WORDS = {"not", "never", "doesn't", "don't", "isn't", "aren't",
                      "won't", "can't", "no", "without", "hate", "dislike"}

    def __init__(self, embedding_store=None):
        self.embeddings = embedding_store
        self.reconcile_count = 0
        self.supersessions = 0
        self.rejections = 0
        logger.info("TypedMemoryReconciler initialized")

    async def reconcile(
        self,
        candidate: TypedMemory,
        store,  # TypedMemoryStore
    ) -> ReconcileDecision:
        """
        Decide how to handle a candidate memory against existing memories.
        """
        self.reconcile_count += 1

        # 1. Try embedding-based conflict detection
        if self.embeddings:
            try:
                similar = await self.embeddings.search(candidate.content, limit=5, min_similarity=0.6)
                for hit in similar:
                    existing = await store.get(hit["memory_id"])
                    if not existing or existing.user_id != candidate.user_id:
                        continue

                    sim = hit["similarity"]

                    # Exact or near-exact duplicate
                    if sim > 0.95:
                        self.rejections += 1
                        return ReconcileDecision(
                            action=ReconcileAction.REJECT,
                            confidence=sim,
                            explanation=f"Near-duplicate (sim={sim:.2f}) of existing memory",
                            existing_id=existing.id,
                        )

                    # High similarity, same type → supersede or merge
                    if sim > 0.75 and existing.memory_type == candidate.memory_type:
                        # Check if exclusive pattern
                        if self._is_exclusive(candidate.content):
                            self.supersessions += 1
                            return ReconcileDecision(
                                action=ReconcileAction.SUPERSEDE,
                                confidence=sim,
                                explanation=f"Supersedes existing (exclusive pattern, sim={sim:.2f})",
                                existing_id=existing.id,
                            )

                        # Check for opposing sentiment (contradiction)
                        if self._has_opposing_sentiment(candidate.content, existing.content):
                            self.supersessions += 1
                            return ReconcileDecision(
                                action=ReconcileAction.SUPERSEDE,
                                confidence=sim,
                                explanation=f"Contradicts existing (opposing sentiment, sim={sim:.2f})",
                                existing_id=existing.id,
                            )

                        # Same type, high sim, not exclusive → merge
                        return ReconcileDecision(
                            action=ReconcileAction.MERGE,
                            confidence=sim,
                            explanation=f"Similar existing memory, merging (sim={sim:.2f})",
                            existing_id=existing.id,
                        )

                    # High similarity, different type → both valid in different contexts
                    if sim > 0.7 and existing.memory_type != candidate.memory_type:
                        return ReconcileDecision(
                            action=ReconcileAction.CONTEXTUALIZE,
                            confidence=sim,
                            explanation=f"Related but different type ({existing.memory_type.value} vs {candidate.memory_type.value})",
                            existing_id=existing.id,
                        )

            except Exception as e:
                logger.warning(f"Embedding reconciliation failed: {e}")

        # 2. Fallback: FTS-based duplicate check
        try:
            fts_results = await store.search(candidate.user_id, candidate.content[:50], limit=3)
            for existing in fts_results:
                if existing.id == candidate.id:
                    continue
                if existing.content.lower().strip() == candidate.content.lower().strip():
                    self.rejections += 1
                    return ReconcileDecision(
                        action=ReconcileAction.REJECT,
                        confidence=1.0,
                        explanation="Exact text duplicate",
                        existing_id=existing.id,
                    )
        except Exception:
            pass

        # 3. No conflicts found → store normally
        return ReconcileDecision(
            action=ReconcileAction.STORE,
            confidence=1.0,
            explanation="No conflicts detected",
        )

    async def execute(
        self,
        candidate: TypedMemory,
        decision: ReconcileDecision,
        store,  # TypedMemoryStore
    ) -> Optional[str]:
        """
        Execute a reconciliation decision. Returns memory ID or None.
        """
        if decision.action == ReconcileAction.REJECT:
            logger.info(f"Reconciler REJECTED: {decision.explanation}")
            return None

        if decision.action == ReconcileAction.SUPERSEDE and decision.existing_id:
            # Archive old, store new
            old = await store.get(decision.existing_id)
            if old:
                old.strength = max(0.05, old.strength * 0.1)
                old.context = (old.context + f" [SUPERSEDED by pipeline]").strip()
                await store.store(old, bypass_jury=True)
                logger.info(f"Superseded memory {decision.existing_id[:8]}")
            return await store.store(candidate, bypass_jury=True)

        if decision.action == ReconcileAction.MERGE and decision.existing_id:
            # Strengthen existing, add candidate context
            old = await store.get(decision.existing_id)
            if old:
                old.strength = min(1.0, old.strength + 0.1)
                old.access_count += 1
                if candidate.context and candidate.context not in old.context:
                    old.context = (old.context + " | " + candidate.context).strip(" | ")
                await store.store(old, bypass_jury=True)
                logger.info(f"Merged into existing memory {decision.existing_id[:8]}")
                return decision.existing_id
            # If old doesn't exist anymore, just store
            return await store.store(candidate, bypass_jury=True)

        if decision.action == ReconcileAction.CONTEXTUALIZE and decision.existing_id:
            # Store both — add cross-reference
            candidate.context = (candidate.context + f" [related: {decision.existing_id[:8]}]").strip()
            return await store.store(candidate, bypass_jury=True)

        # STORE — normal storage (already passed jury)
        return await store.store(candidate, bypass_jury=True)

    def _is_exclusive(self, content: str) -> bool:
        cl = content.lower()
        return any(p in cl for p in self.EXCLUSIVE_PATTERNS)

    def _has_opposing_sentiment(self, a: str, b: str) -> bool:
        a_neg = any(w in a.lower().split() for w in self.NEGATION_WORDS)
        b_neg = any(w in b.lower().split() for w in self.NEGATION_WORDS)
        return a_neg != b_neg

    def get_stats(self) -> Dict[str, Any]:
        return {
            "reconciliations": self.reconcile_count,
            "supersessions": self.supersessions,
            "rejections": self.rejections,
        }


# ============================================================
# ORCHESTRATOR: 3-LANE PIPELINE
# ============================================================

@dataclass
class PipelineUpdate:
    """Progressive update from pipeline stages."""
    stage: str
    memories_extracted: int = 0
    memories_approved: int = 0
    memories_quarantined: int = 0
    memories_rejected: int = 0
    memories_stored: int = 0
    memories_superseded: int = 0
    memories_merged: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)


class TypedMemoryPipeline:
    """
    3-Lane async pipeline for typed memory processing.
    
    Lane 0: Fast Lane — Extraction (<100ms)
    Lane 1: Jury Lane — Deliberation (<50ms per memory)
    Lane 2: Write Lane — Reconciliation + Persistence (<200ms)
    
    Usage:
        pipeline = TypedMemoryPipeline(store, embedding_store, jury)
        result = await pipeline.process_message(message, user_id)
    """

    def __init__(self, store, embedding_store=None, jury=None):
        self.store = store
        self.extractor = TypedMemoryExtractor()
        self.reconciler = TypedMemoryReconciler(embedding_store=embedding_store)
        self.jury = jury  # MemoryJury instance
        self.messages_processed = 0
        logger.info("TypedMemoryPipeline initialized (3-lane: Extract → Jury → Reconcile)")

    async def process_message(
        self,
        message: str,
        user_id: str,
        context: str = "",
        auto_tag: bool = True,
    ) -> PipelineUpdate:
        """
        Process a message through the full 3-lane pipeline.
        
        Returns a PipelineUpdate with stats from all stages.
        """
        self.messages_processed += 1
        t0 = time.time()
        result = PipelineUpdate(stage="complete")

        # ── LANE 0: FAST LANE — EXTRACTION ──
        extractions = self.extractor.extract(message, user_id, context)
        result.memories_extracted = len(extractions)

        if not extractions:
            logger.debug("No memories extracted from message")
            result.stage = "extraction"
            return result

        # ── LANE 1: JURY LANE — DELIBERATION ──
        approved = []
        from src.memory.memory_jury import Verdict

        for ext in extractions:
            mem = ext.memory

            if self.jury:
                # Gather existing contents for dedup
                existing_contents = []
                try:
                    cursor = await self.store._conn.execute(
                        "SELECT content FROM typed_memories WHERE user_id = ? AND memory_type = ? LIMIT 30",
                        (mem.user_id, mem.memory_type.value)
                    )
                    rows = await cursor.fetchall()
                    existing_contents = [r[0] for r in rows]
                except Exception:
                    pass

                decision = self.jury.deliberate(
                    content=mem.content,
                    memory_type=mem.memory_type.value,
                    episode_timestamp=mem.episode_timestamp,
                    existing_contents=existing_contents,
                )

                if decision.verdict == Verdict.REJECT:
                    result.memories_rejected += 1
                    result.details.append({
                        "content": mem.content[:80],
                        "stage": "jury",
                        "verdict": "REJECTED",
                        "reason": decision.explanation,
                    })
                    continue
                elif decision.verdict == Verdict.QUARANTINE:
                    result.memories_quarantined += 1
                    mem.strength = max(0.1, mem.strength * 0.5)
                    mem.context = (mem.context + f" [QUARANTINED: {decision.explanation}]").strip()

                result.memories_approved += 1
            else:
                result.memories_approved += 1

            approved.append(ext)

        # ── LANE 2: WRITE LANE — RECONCILIATION + PERSISTENCE ──
        for ext in approved:
            mem = ext.memory

            # Auto-tag
            if auto_tag:
                mem.tags = self.store.auto_tag(mem.content, mem.tags) if hasattr(self.store, 'auto_tag') else mem.tags

            # Reconcile against existing memories
            rec_decision = await self.reconciler.reconcile(mem, self.store)

            # Execute reconciliation
            mem_id = await self.reconciler.execute(mem, rec_decision, self.store)

            if mem_id:
                result.memories_stored += 1
                if rec_decision.action == ReconcileAction.SUPERSEDE:
                    result.memories_superseded += 1
                elif rec_decision.action == ReconcileAction.MERGE:
                    result.memories_merged += 1

                result.details.append({
                    "id": mem_id[:8] if mem_id else None,
                    "content": mem.content[:80],
                    "type": mem.memory_type.value,
                    "pattern": ext.pattern_name,
                    "stage": "stored",
                    "reconcile_action": rec_decision.action.value,
                })
            else:
                result.memories_rejected += 1
                result.details.append({
                    "content": mem.content[:80],
                    "stage": "reconciler",
                    "verdict": "REJECTED",
                    "reason": rec_decision.explanation,
                })

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            f"Pipeline #{self.messages_processed}: {result.memories_extracted} extracted, "
            f"{result.memories_approved} approved, {result.memories_stored} stored, "
            f"{result.memories_rejected} rejected [{elapsed_ms:.0f}ms]"
        )

        return result

    async def process_batch(
        self,
        messages: List[str],
        user_id: str,
        context: str = "",
    ) -> PipelineUpdate:
        """Process multiple messages through the pipeline."""
        combined = PipelineUpdate(stage="batch_complete")
        for msg in messages:
            update = await self.process_message(msg, user_id, context)
            combined.memories_extracted += update.memories_extracted
            combined.memories_approved += update.memories_approved
            combined.memories_quarantined += update.memories_quarantined
            combined.memories_rejected += update.memories_rejected
            combined.memories_stored += update.memories_stored
            combined.memories_superseded += update.memories_superseded
            combined.memories_merged += update.memories_merged
            combined.details.extend(update.details)
        return combined

    def get_stats(self) -> Dict[str, Any]:
        return {
            "messages_processed": self.messages_processed,
            "extractor": self.extractor.get_stats(),
            "reconciler": self.reconciler.get_stats(),
            "jury": self.jury.get_stats() if self.jury else None,
        }
