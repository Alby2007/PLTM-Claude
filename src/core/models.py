"""Core data models for the Procedural LTM system"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field


class AtomType(str, Enum):
    """Ontological categories for memory atoms"""

    ENTITY = "entity"
    RELATION = "relation"  # Legacy - being phased out
    STATE = "state"
    EVENT = "event"
    HYPOTHESIS = "hypothesis"
    INVARIANT = "invariant"
    
    # General knowledge types
    FACT = "fact"                # General factual knowledge
    ATTRIBUTE = "attribute"      # Properties/attributes of entities
    
    # New granular types (replacing broad RELATION)
    AFFILIATION = "affiliation"  # works_at, studies_at, member_of
    SOCIAL = "social"            # knows, reports_to, friends_with
    PREFERENCE = "preference"    # likes, dislikes, prefers
    BELIEF = "belief"            # thinks, believes, trusts
    SKILL = "skill"              # can_do, proficient_in, learning
    
    # Personality and mood tracking (Experiment 8)
    PERSONALITY_TRAIT = "personality_trait"        # Stable traits (humor, directness)
    COMMUNICATION_STYLE = "communication_style"    # Style preferences (concise, technical)
    INTERACTION_PATTERN = "interaction_pattern"    # Behavioral patterns (formal, casual)


class GraphType(str, Enum):
    """Graph storage types"""

    SUBSTANTIATED = "substantiated"
    UNSUBSTANTIATED = "unsubstantiated"
    HISTORICAL = "historical"


class Provenance(str, Enum):
    """Source of the memory atom"""

    USER_STATED = "user_stated"
    USER_CONFIRMED = "user_confirmed"
    INFERRED = "inferred"
    CORRECTED = "corrected"


class JudgeVerdict(str, Enum):
    """Jury judge verdicts (legacy)"""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    QUARANTINE = "QUARANTINE"
    ESCALATE = "ESCALATE"


class JuryVerdict(str, Enum):
    """Jury consensus verdicts (new 4-judge system)"""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    QUARANTINE = "QUARANTINE"


class SimpleJuryDecision(BaseModel):
    """Individual judge decision (new 4-judge system)"""
    
    verdict: JuryVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    judge_name: str


class ReconciliationAction(str, Enum):
    """Actions for conflict resolution"""

    SUPERSEDE = "SUPERSEDE"
    CONTEXTUALIZE = "CONTEXTUALIZE"
    REJECT = "REJECT"
    MERGE = "MERGE"


class JuryDecision(BaseModel):
    """Record of a single jury deliberation"""

    timestamp: datetime = Field(default_factory=datetime.now)
    stage: int

    # Individual judge verdicts
    memory_judge: Optional[JudgeVerdict] = None
    safety_judge: Optional[JudgeVerdict] = None
    consensus_judge: Optional[JudgeVerdict] = None
    time_judge: Optional[JudgeVerdict] = None

    # Final decision (safety judge can override all)
    final_verdict: JudgeVerdict
    confidence_adjustment: float = 0.0  # -0.3 to +0.3

    # Reasoning (for audit/debugging)
    reasoning: str = ""

    # For conflict resolution
    requires_user_clarification: bool = False
    suggested_contexts: list[str] = Field(default_factory=list)


class ReconciliationDecision(BaseModel):
    """Decision on how to resolve a conflict"""

    action: ReconciliationAction
    keep_existing: bool
    keep_candidate: bool
    archive_existing: bool
    explanation: str
    contexts_added: list[str] = Field(default_factory=list)


class MemoryAtom(BaseModel):
    """
    Fundamental unit of procedural long-term memory.
    Represents a semantic triple (Subject, Predicate, Object).
    
    Enhanced with explicit epistemic modeling to track WHO believes WHAT
    with WHAT certainty, avoiding ambiguity in belief attribution.
    
    Examples:
        # Direct belief: "I think Bob will get a job"
        MemoryAtom(
            subject="Bob",
            predicate="will_get",
            object="job",
            source_user="user_123",      # WHO stated this
            confidence=0.7,               # THEIR certainty
            belief_holder="user_123",     # WHO holds the belief
            epistemic_distance=0          # Direct belief
        )
        
        # Reported belief: "Mary thinks Bob will get a job"
        MemoryAtom(
            subject="Bob",
            predicate="will_get",
            object="job",
            source_user="user_123",       # User told us
            confidence=0.6,                # Lower (secondhand)
            belief_holder="Mary",          # Mary holds the belief
            epistemic_distance=1           # One level removed
        )
    """

    # ========== IDENTITY ==========
    id: UUID = Field(default_factory=uuid4)
    atom_type: AtomType
    graph: GraphType = GraphType.UNSUBSTANTIATED

    # ========== SEMANTIC TRIPLE ==========
    subject: str
    predicate: str
    object: str
    object_metadata: Optional[dict[str, Any]] = None

    # ========== CONTEXTUAL QUALIFIERS ==========
    contexts: list[str] = Field(default_factory=list)
    temporal_validity: Optional[dict[str, datetime]] = None

    # ========== EPISTEMIC MODELING ==========
    # Explicit tracking of WHO believes WHAT with WHAT certainty
    source_user: str = ""                           # WHO stated/observed this
    belief_holder: Optional[str] = None             # WHO holds the belief (if different from source)
    epistemic_distance: int = 0                     # How many levels removed (0=direct, 1=reported, etc.)

    # ========== EVIDENCE TRACKING ==========
    provenance: Provenance

    # The Three Pillars of Evidence
    assertion_count: int = 1
    explicit_confirms: list[datetime] = Field(default_factory=list)
    first_observed: datetime = Field(default_factory=datetime.now)
    last_contradicted: Optional[datetime] = None

    # ========== DECAY PARAMETERS ==========
    strength: float  # S in R = e^(-t/S), range [0.0, 1.0]
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = 0

    # ========== CONFLICT RESOLUTION ==========
    supersedes: Optional[UUID] = None
    superseded_by: Optional[UUID] = None
    related_atoms: list[UUID] = Field(default_factory=list)

    # ========== JURY METADATA ==========
    confidence: float = 0.5
    jury_history: list[JuryDecision] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)

    # ========== SESSION TRACKING ==========
    session_id: str = ""
    topic_cluster: Optional[str] = None

    @computed_field  # type: ignore[misc]
    @property
    def promotion_eligible(self) -> bool:
        """
        Check if atom meets criteria for promotion to substantiated graph.
        
        Tiered promotion system:
        - INSTANT: User-stated + high confidence (≥0.9) OR explicit confirmation OR corrected
        - FAST: High confidence (≥0.8) + 4 hours no contradiction
        - STANDARD: 3+ assertions + 12 hours no contradiction
        - SLOW: Evidence bundle (freq + friction) after 24 hours
        """
        from src.core.config import settings

        # ========================================================================
        # FAST TRACK 0: Corrected Information (HIGHEST PRIORITY)
        # ========================================================================
        # 
        # CRITICAL FIX: CORRECTED provenance must always promote instantly
        # 
        # Problem: Without this, correction signals like "Actually, I live in..."
        # would not be promoted, causing the correction test to fail (0 facts).
        # 
        # Why corrections need instant promotion:
        # 1. User explicitly correcting previous statement
        # 2. Should supersede ALL previous information immediately
        # 3. No waiting period needed - user is actively correcting
        # 
        # Example: "I live in Seattle" → "Actually, I live in San Francisco"
        # The correction must be promoted immediately to replace the old fact.
        # 
        # This fix enabled the correction test to pass (100% accuracy).
        # 
        if self.provenance == Provenance.CORRECTED:
            return True  # Instant promotion - no waiting period

        # FAST TRACK 1: User-stated with high confidence
        if self.provenance == Provenance.USER_STATED and self.confidence >= settings.INSTANT_CONFIDENCE:
            # Check flip-flop safety (no recent contradiction)
            if not self.last_contradicted or (
                datetime.now() - self.last_contradicted
            ).total_seconds() >= 3600:
                return True  # Instant promotion

        # FAST TRACK 2: User confirmation on any atom
        if len(self.explicit_confirms) >= 1:
            return True  # Instant promotion

        # STANDARD TRACK: Evidence Bundle with graduated friction
        freq_pass = self.assertion_count >= 3

        # Reduce friction window for high-confidence atoms
        hours_threshold = settings.SLOW_HOURS
        if self.confidence >= settings.FAST_CONFIDENCE:
            hours_threshold = settings.FAST_HOURS  # 4 hours instead of 24
        elif self.confidence >= settings.STANDARD_CONFIDENCE:
            hours_threshold = settings.STANDARD_HOURS  # 12 hours instead of 24

        hours_observed = (datetime.now() - self.first_observed).total_seconds() / 3600

        friction_pass = hours_observed >= hours_threshold and self.last_contradicted is None

        evidence_score = sum([freq_pass, friction_pass])

        # Need 2 of 2 pillars (frequency + friction) AND meet confidence threshold
        # OR just need high enough confidence with one pillar
        return (evidence_score >= 2 and self.confidence >= settings.PROMOTION_THRESHOLD) or \
               (evidence_score >= 1 and self.confidence >= 0.9)

    @computed_field  # type: ignore[misc]
    @property
    def promotion_tier(self) -> str:
        """Determine which tier would trigger promotion"""
        from src.core.config import settings

        # CORRECTED provenance always gets instant promotion
        if self.provenance == Provenance.CORRECTED:
            return "INSTANT"

        if self.provenance == Provenance.USER_STATED and self.confidence >= settings.INSTANT_CONFIDENCE:
            if not self.last_contradicted or (
                datetime.now() - self.last_contradicted
            ).total_seconds() >= 3600:
                return "INSTANT"

        if len(self.explicit_confirms) >= 1:
            return "INSTANT_CONFIRM"

        if self.confidence >= settings.FAST_CONFIDENCE:
            return "FAST"
        elif self.confidence >= settings.STANDARD_CONFIDENCE:
            return "STANDARD"
        else:
            return "SLOW"

    def model_dump_json_safe(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict for storage"""
        return {
            "id": str(self.id),
            "atom_type": self.atom_type.value,
            "graph": self.graph.value,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "object_metadata": self.object_metadata,
            "contexts": self.contexts,
            "temporal_validity": (
                {
                    k: v.isoformat() if isinstance(v, datetime) else v
                    for k, v in self.temporal_validity.items()
                }
                if self.temporal_validity
                else None
            ),
            "provenance": self.provenance.value,
            "assertion_count": self.assertion_count,
            "explicit_confirms": [dt.isoformat() for dt in self.explicit_confirms],
            "first_observed": self.first_observed.isoformat(),
            "last_contradicted": (
                self.last_contradicted.isoformat() if self.last_contradicted else None
            ),
            "strength": self.strength,
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "supersedes": str(self.supersedes) if self.supersedes else None,
            "superseded_by": str(self.superseded_by) if self.superseded_by else None,
            "related_atoms": [str(aid) for aid in self.related_atoms],
            "confidence": self.confidence,
            "jury_history": [j.model_dump() for j in self.jury_history],
            "flags": self.flags,
            "session_id": self.session_id,
            "topic_cluster": self.topic_cluster,
        }
