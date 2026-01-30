"""Rule-based semantic triple extraction using regex patterns"""

import re
from typing import List

from loguru import logger

from src.core.config import settings
from src.core.models import AtomType, MemoryAtom, Provenance
from src.core.ontology import validate_atom
from src.extraction.context_extractor import ContextExtractor


class RuleBasedExtractor:
    """
    Fast, deterministic extraction for common patterns.
    
    Handles 70-80% of common cases with high confidence.
    No ML dependencies - pure regex matching.
    """

    # Pattern format: (regex_pattern, predicate, atom_type)
    # PRIORITY ORDER: Most specific patterns first, generic patterns last
    # First match wins to prevent pattern interference
    PATTERNS = [
        # ========== CORRECTION SIGNALS (HIGHEST PRIORITY) ==========
        (r"Actually,?\s+(.+)", "correction", AtomType.STATE),
        (r"No,?\s+I\s+meant\s+(.+)", "correction", AtomType.STATE),
        (r"To clarify,?\s+(.+)", "clarification", AtomType.STATE),
        
        # ========== SPECIFIC WORK/LOCATION ==========
        (r"I work (?:at|in|for) (.+?)(?:\.|,|$)", "works_at", AtomType.RELATION),
        (r"I'm (?:a |an )?(.+?) at (.+?)(?:\.|,|$)", "works_at", AtomType.RELATION),
        (r"I (?:live|am) in (.+?)(?:\.|,|$)", "located_at", AtomType.ENTITY),
        (r"I'm from (.+?)(?:\.|,|$)", "located_at", AtomType.ENTITY),
        
        # ========== STRONG PREFERENCES (SPECIFIC) ==========
        (r"I prefer (.+?)(?:\.|,|$)", "prefers", AtomType.RELATION),
        (r"I (?:really )?love (.+?)(?:\.|,|$)", "likes", AtomType.RELATION),
        (r"I (?:really )?hate (.+?)(?:\.|,|$)", "dislikes", AtomType.RELATION),
        (r"I (?:really )?like (.+?)(?:\.|,|$)", "likes", AtomType.RELATION),
        (r"I kinda like (.+?)(?:\.|,|$)", "likes", AtomType.RELATION),
        (r"I (?:really )?enjoy (.+?)(?:\.|,|$)", "likes", AtomType.RELATION),
        (r"I (?:really )?dislike (.+?)(?:\.|,|$)", "dislikes", AtomType.RELATION),
        (r"I (?:can't stand|don't like) (.+?)(?:\.|,|$)", "dislikes", AtomType.RELATION),
        (r"I don't dislike (.+?)(?:\.|,|$)", "likes", AtomType.RELATION),
        (r"I'm (?:passionate about|interested in|into) (.+?)(?:\.|,|$)", "likes", AtomType.RELATION),
        (r"I am obsessed with (.+?)(?:\.|,|$)", "likes", AtomType.RELATION),
        (r"I am neutral about (.+?)(?:\.|,|$)", "neutral", AtomType.RELATION),
        
        # Temporal patterns
        (r"I used to like (.+?)(?:\.|,|$)", "liked_past", AtomType.RELATION),
        (r"I liked (.+?)(?:\.|,|$)", "liked_past", AtomType.RELATION),
        (r"I loved (.+?)(?:\.|,|$)", "liked_past", AtomType.RELATION),
        (r"I will start learning (.+?)(?:\.|,|$)", "will_learn", AtomType.RELATION),
        (r"I started learning (.+?)(?:\.|,|$)", "started_learning", AtomType.RELATION),
        (r"I always (?:like|liked) (.+?)(?:\.|,|$)", "likes", AtomType.RELATION),
        (r"I recently started liking (.+?)(?:\.|,|$)", "likes", AtomType.RELATION),
        (r"I sometimes like (.+?)(?:\.|,|$)", "likes", AtomType.RELATION),
        (r"I definitely like (.+?)(?:\.|,|$)", "likes", AtomType.RELATION),
        
        # Temporal with year markers
        (r"In (\d{4}) I worked at (.+?)(?:\.|,|$)", "worked_at_year", AtomType.RELATION),
        (r"In (\d{4}) I work at (.+?)(?:\.|,|$)", "works_at_year", AtomType.RELATION),
        
        # ========== ACTIONS & BEHAVIORS ==========
        (r"I want (?:to )?(?:learn |study )?(.+?)(?:\.|,|$)", "wants", AtomType.RELATION),
        (r"I avoid (?:learning |studying )?(.+?)(?:\.|,|$)", "avoids", AtomType.RELATION),
        (r"I support (.+?)(?:\.|,|$)", "supports", AtomType.RELATION),
        (r"I oppose (.+?)(?:\.|,|$)", "opposes", AtomType.RELATION),
        (r"I accept (.+?)(?:\.|,|$)", "accepts", AtomType.RELATION),
        (r"I reject (.+?)(?:\.|,|$)", "rejects", AtomType.RELATION),
        (r"I trust (.+?)(?:\.|,|$)", "trusts", AtomType.RELATION),
        (r"I distrust (.+?)(?:\.|,|$)", "distrusts", AtomType.RELATION),
        (r"I agree with (.+?)(?:\.|,|$)", "agrees", AtomType.RELATION),
        (r"I disagree with (.+?)(?:\.|,|$)", "disagrees", AtomType.RELATION),
        
        # ========== SKILLS/TOOLS ==========
        (r"I (?:use|work with|know) (.+?)(?:\.|,|$)", "uses", AtomType.RELATION),
        (r"I'm (?:good at|skilled in|experienced with) (.+?)(?:\.|,|$)", "uses", AtomType.RELATION),
        (r"I drive (?:a |an )?(.+?)(?:\.|,|$)", "drives", AtomType.RELATION),
        (r"I do ((?:backend |frontend |web |mobile )?programming)", "does", AtomType.RELATION),
        (r"I studied (.+?)(?:\.|,|$)", "studied", AtomType.EVENT),
        
        # Quantifier patterns
        (r"I like all (.+?) features", "likes", AtomType.RELATION),
        (r"I like some (.+?) features", "likes", AtomType.RELATION),
        (r"I like (.+?) very much", "likes", AtomType.RELATION),
        (r"I like (.+?) a little bit", "likes", AtomType.RELATION),
        (r"I maybe like (.+?)(?:\.|,|$)", "likes", AtomType.RELATION),
        (r"I don'?t always like (.+?)(?:\.|,|$)", "likes", AtomType.RELATION),
        
        # Comparative/transitive patterns
        (r"I prefer ([A-Z]) over ([A-Z])", "prefers_over", AtomType.RELATION),
        
        # ========== EVENTS ==========
        (r"I (?:completed|finished) (.+?)(?:\.|,|$)", "completed", AtomType.EVENT),
        (r"I (?:started|began) (.+?)(?:\.|,|$)", "started", AtomType.EVENT),
        
        # ========== IDENTITY (LAST - most generic) ==========
        (r"^([A-Z]\w+) is not bad", "likes", AtomType.RELATION),
        (r"(?:I am|I'm) (?:a |an )?(.+?)(?:\.|,|$)", "is", AtomType.ENTITY),
        (r"my name is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "named", AtomType.ENTITY),
    ]

    def __init__(self) -> None:
        self.extraction_count = 0
        self.context_extractor = ContextExtractor()
        logger.info("RuleBasedExtractor initialized with {} patterns", len(self.PATTERNS))

    def extract(self, message: str, user_id: str, session_id: str = "") -> list[MemoryAtom]:
        """
        Extract semantic triples from user message using regex patterns.
        
        Args:
            message: User's message text
            user_id: Canonical user identifier
            session_id: Optional session identifier
            
        Returns:
            List of extracted MemoryAtoms
        """
        atoms: List[MemoryAtom] = []
        
        # Extract contexts from message
        contexts = self.context_extractor.extract(message)
        
        # Try patterns in priority order - FIRST MATCH WINS
        matched = False
        for pattern, predicate, atom_type in self.PATTERNS:
            if matched:
                break  # Already found a match, stop checking
            
            match = re.search(pattern, message, re.IGNORECASE)
            if not match:
                continue
            
            # Extract object (captured group)
            obj = match.group(1).strip()
            
            if not obj or len(obj) < 2:
                continue
            
            # Handle correction/clarification signals specially
            if predicate in ["correction", "clarification"]:
                # Re-extract from the corrected/clarified statement
                corrected_atoms = self.extract(obj, user_id, session_id)
                for atom in corrected_atoms:
                    if predicate == "correction":
                        atom.provenance = Provenance.CORRECTED
                        atom.flags.append("correction_signal")
                    else:  # clarification
                        atom.flags.append("clarification_signal")
                        # Clarifications are refinements, not corrections
                return corrected_atoms
            
            # Infer provenance
            provenance = self._infer_provenance(message)
            
            # Create atom
            atom = MemoryAtom(
                atom_type=atom_type,
                subject=user_id,
                predicate=predicate,
                object=obj,
                contexts=contexts,
                provenance=provenance,
                confidence=0.95 if provenance == Provenance.USER_STATED else 0.8,
                strength=0.8,
                session_id=session_id,
            )
            
            # Validate against ontology
            is_valid, reason = validate_atom(atom)
            if is_valid:
                atoms.append(atom)
                matched = True  # Don't check more patterns
                self.extraction_count += 1
                logger.debug(
                    "Extracted: [{subject}] [{predicate}] [{object}]",
                    subject=atom.subject,
                    predicate=atom.predicate,
                    object=atom.object,
                )
            else:
                logger.warning(
                    "Invalid atom rejected: {reason}",
                    reason=reason,
                )
        
        logger.info(
            "Rule-based extraction: {count} atoms from message",
            count=len(atoms),
        )
        return atoms

    def _infer_provenance(self, message: str) -> Provenance:
        """
        Determine if user stated directly or if it's inferred.
        
        Heuristics:
        - Correction signals (Actually, etc.) = CORRECTED
        - First-person indicators = USER_STATED
        - Otherwise = INFERRED
        """
        # Check for correction signals first
        correction_signals = [
            r"\bactually\b", r"\bno,?\s+I\s+meant\b", r"\bsorry,?\s+I\b",
            r"\bI meant\b", r"\bcorrection\b"
        ]
        
        # Check for clarification signals (different from corrections - should refine, not supersede)
        clarification_signals = [
            r"\bto clarify\b", r"\bmore specifically\b", r"\bto be precise\b"
        ]
        
        for signal in correction_signals:
            if re.search(signal, message, re.IGNORECASE):
                return Provenance.CORRECTED
        
        # Check for first-person indicators
        first_person_indicators = [
            r"\bI\b", r"\bI'm\b", r"\bI've\b", r"\bmy\b", r"\bme\b",
            r"\bmine\b", r"\bmyself\b"
        ]
        
        for indicator in first_person_indicators:
            if re.search(indicator, message, re.IGNORECASE):
                return Provenance.USER_STATED
        
        return Provenance.INFERRED

    def get_coverage_estimate(self, message: str) -> float:
        """
        Estimate what percentage of the message would be captured.
        
        Used by hybrid extractor to decide if fallback is needed.
        """
        total_words = len(message.split())
        if total_words == 0:
            return 0.0
        
        # Count words that would be captured
        captured_words = set()
        
        for pattern, _, _ in self.PATTERNS:
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                # Add words from captured groups
                for group in match.groups():
                    if group:
                        captured_words.update(group.lower().split())
        
        coverage = len(captured_words) / total_words
        return min(coverage, 1.0)

    def get_stats(self) -> dict[str, int]:
        """Get extraction statistics"""
        return {
            "total_extractions": self.extraction_count,
            "pattern_count": len(self.PATTERNS),
        }
