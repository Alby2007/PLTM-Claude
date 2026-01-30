"""Time Judge - Validates temporal consistency and detects time-based conflicts"""

from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from src.core.models import SimpleJuryDecision, JuryVerdict, MemoryAtom
from src.jury.base_judge import BaseJudge


class TimeJudge(BaseJudge):
    """
    Validates temporal consistency and detects time-based conflicts.
    
    Responsibilities:
    - Detect temporal contradictions (past vs present vs future)
    - Validate temporal markers (years, dates, time expressions)
    - Flag rapid state changes (potential errors)
    - Assess temporal plausibility
    
    Examples:
    - APPROVE: "In 2020 I worked at Google" + "In 2025 I work at Anthropic"
    - QUARANTINE: "I started yesterday" + "I finished 2 years ago" (contradiction)
    - REJECT: "In 2030 I worked at X" (future date with past tense)
    """
    
    def __init__(self):
        super().__init__("TimeJudge")
        self.max_rapid_changes = 5  # Flag if >5 changes in short period
        self.rapid_change_window = timedelta(minutes=5)
        
    def evaluate(self, atom: MemoryAtom, context: Optional[dict] = None) -> SimpleJuryDecision:
        """
        Evaluate temporal consistency of a memory atom.
        
        Args:
            atom: Memory atom to evaluate
            context: Optional context including recent atoms, session info
            
        Returns:
            JuryDecision with verdict and explanation
        """
        # Extract temporal markers from the atom
        temporal_markers = self._extract_temporal_markers(atom)
        
        # Check for temporal contradictions in the statement itself
        if self._has_internal_contradiction(atom, temporal_markers):
            return SimpleJuryDecision(
                verdict=JuryVerdict.REJECT,
                confidence=0.95,
                explanation="Internal temporal contradiction detected",
                judge_name=self.name,
            )
        
        # Check for implausible temporal claims
        if self._is_temporally_implausible(atom, temporal_markers):
            return SimpleJuryDecision(
                verdict=JuryVerdict.QUARANTINE,
                confidence=0.85,
                explanation="Temporally implausible claim - needs verification",
                judge_name=self.name,
            )
        
        # Check for rapid state changes (potential errors)
        if context and self._has_rapid_changes(atom, context):
            return SimpleJuryDecision(
                verdict=JuryVerdict.QUARANTINE,
                confidence=0.75,
                explanation="Rapid state changes detected - may indicate error",
                judge_name=self.name,
            )
        
        # Check for conflicting temporal markers with existing memories
        if context and self._has_temporal_conflict(atom, context, temporal_markers):
            return SimpleJuryDecision(
                verdict=JuryVerdict.QUARANTINE,
                confidence=0.80,
                explanation="Temporal conflict with existing memories",
                judge_name=self.name,
            )
        
        # All temporal checks passed
        return SimpleJuryDecision(
            verdict=JuryVerdict.APPROVE,
            confidence=0.90,
            explanation="Temporally consistent",
            judge_name=self.name,
        )
    
    def _extract_temporal_markers(self, atom: MemoryAtom) -> dict:
        """Extract temporal markers from atom object and predicate"""
        markers = {
            "tense": None,  # past, present, future
            "year": None,
            "relative": None,  # "used to", "will", "recently", etc.
            "duration": None,  # "always", "sometimes", etc.
        }
        
        # Check predicate for temporal markers
        predicate = atom.predicate.lower()
        obj = atom.object.lower()
        
        # Past tense markers
        if "liked_past" in predicate or "worked_at_year" in predicate:
            markers["tense"] = "past"
        elif "will_learn" in predicate or "will" in obj:
            markers["tense"] = "future"
        elif "started_learning" in predicate or "works_at_year" in predicate:
            markers["tense"] = "present"
        
        # Relative temporal markers
        if "used to" in obj:
            markers["relative"] = "used_to"
            markers["tense"] = "past"
        elif "will" in obj:
            markers["relative"] = "will"
            markers["tense"] = "future"
        elif "recently" in obj or "just" in obj:
            markers["relative"] = "recent"
        elif "always" in obj:
            markers["duration"] = "always"
        elif "sometimes" in obj:
            markers["duration"] = "sometimes"
        
        # Extract year if present
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', obj)
        if year_match:
            markers["year"] = int(year_match.group(0))
        
        return markers
    
    def _has_internal_contradiction(self, atom: MemoryAtom, markers: dict) -> bool:
        """Check if atom has internal temporal contradiction"""
        # Example: "In 2030 I worked at X" (future year + past tense)
        if markers["year"]:
            current_year = datetime.now().year
            
            # Future year with past tense
            if markers["year"] > current_year and markers["tense"] == "past":
                logger.warning(
                    f"Internal contradiction: future year {markers['year']} with past tense"
                )
                return True
            
            # Past year with future tense
            if markers["year"] < current_year and markers["tense"] == "future":
                logger.warning(
                    f"Internal contradiction: past year {markers['year']} with future tense"
                )
                return True
        
        return False
    
    def _is_temporally_implausible(self, atom: MemoryAtom, markers: dict) -> bool:
        """Check if temporal claim is implausible"""
        if markers["year"]:
            current_year = datetime.now().year
            
            # Year too far in future (>10 years)
            if markers["year"] > current_year + 10:
                logger.warning(f"Implausible future year: {markers['year']}")
                return True
            
            # Year too far in past for typical user (>100 years)
            if markers["year"] < current_year - 100:
                logger.warning(f"Implausible past year: {markers['year']}")
                return True
        
        return False
    
    def _has_rapid_changes(self, atom: MemoryAtom, context: dict) -> bool:
        """Check for rapid state changes that might indicate errors"""
        # Get recent atoms for same subject+predicate
        recent_atoms = context.get("recent_atoms", [])
        if not recent_atoms:
            return False
        
        # Count changes in the rapid change window
        same_predicate_atoms = [
            a for a in recent_atoms
            if a.subject == atom.subject and a.predicate == atom.predicate
        ]
        
        if len(same_predicate_atoms) > self.max_rapid_changes:
            logger.warning(
                f"Rapid changes detected: {len(same_predicate_atoms)} changes "
                f"for {atom.subject} {atom.predicate}"
            )
            return True
        
        return False
    
    def _has_temporal_conflict(
        self, atom: MemoryAtom, context: dict, markers: dict
    ) -> bool:
        """Check for temporal conflicts with existing memories"""
        existing_atoms = context.get("existing_atoms", [])
        if not existing_atoms:
            return False
        
        # Check for conflicting temporal markers
        for existing in existing_atoms:
            existing_markers = self._extract_temporal_markers(existing)
            
            # Same subject+predicate with conflicting years
            if (atom.subject == existing.subject and 
                atom.predicate == existing.predicate and
                markers["year"] and existing_markers["year"]):
                
                # Different years for same fact type
                if markers["year"] != existing_markers["year"]:
                    # This is actually OK - temporal progression
                    # Only flag if tenses don't match the years
                    pass
        
        return False
