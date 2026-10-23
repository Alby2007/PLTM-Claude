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
        (r"I worked (?:at|in|for) (.+?)(?:[.,](?=\s|$)|$)", "worked_at_past", AtomType.EVENT),
        (r"I used to work (?:at|in|for) (.+?)(?:[.,](?=\s|$)|$)", "worked_at_past", AtomType.EVENT),
        (r"I work in (.+?)(?:[.,](?=\s|$)|$)", "works_in", AtomType.AFFILIATION),
        (r"I work (?:at|for) (.+?)(?:[.,](?=\s|$)|$)", "works_at", AtomType.AFFILIATION),
        (r"I'm (?:a |an )?(.+?) at (.+?)(?:[.,](?=\s|$)|$)", "works_at", AtomType.AFFILIATION),
        (r"I (?:live|am living) in (.+?)(?:[.,](?=\s|$)|$)", "located_at", AtomType.AFFILIATION),
        (r"I'm from (.+?)(?:[.,](?=\s|$)|$)", "located_at", AtomType.AFFILIATION),
        
        # ========== MEDICAL/ATTRIBUTES ==========
        (r"I am allergic to (.+?)(?:[.,](?=\s|$)|$)", "allergic_to", AtomType.ENTITY),
        (r"I'm allergic to (.+?)(?:[.,](?=\s|$)|$)", "allergic_to", AtomType.ENTITY),
        
        # ========== STRONG PREFERENCES (SPECIFIC) ==========
        # Quantifier/degree patterns MUST precede generic "I like" so the
        # qualifier is captured instead of swallowed into the object.
        (r"I like (all .+? features)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        (r"I like (some .+? features)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        (r"I like (.+? very much)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        (r"I like (.+? a little bit)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        (r"I maybe like (.+?)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        (r"I don'?t always like (.+?)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),

        (r"I prefer (.+?)(?:[.,](?=\s|$)|$)", "prefers", AtomType.PREFERENCE),
        (r"I (?:really )?love (.+?)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        (r"I (?:really )?hate (.+?)(?:[.,](?=\s|$)|$)", "dislikes", AtomType.PREFERENCE),
        (r"I (?:really )?like (.+?)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        (r"I kinda like (.+?)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        (r"I (?:really )?enjoy (.+?)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        (r"I (?:really )?dislike (.+?)(?:[.,](?=\s|$)|$)", "dislikes", AtomType.PREFERENCE),
        (r"I (?:can't stand|don't like|do not like) (.+?)(?:[.,](?=\s|$)|$)", "dislikes", AtomType.PREFERENCE),
        (r"I don't dislike (.+?)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        (r"I would (?:like|love) (?:to )?(?:learn |study |try )?(.+?)(?:[.,](?=\s|$)|$)", "wants", AtomType.PREFERENCE),
        (r"I'm (?:passionate about|interested in|into) (.+?)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        (r"I am obsessed with (.+?)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        (r"I am neutral about (.+?)(?:[.,](?=\s|$)|$)", "neutral", AtomType.PREFERENCE),
        
        # Temporal patterns
        (r"I used to like (.+?)(?:[.,](?=\s|$)|$)", "liked_past", AtomType.PREFERENCE),
        (r"I liked (.+?)(?:[.,](?=\s|$)|$)", "liked_past", AtomType.PREFERENCE),
        (r"I loved (.+?)(?:[.,](?=\s|$)|$)", "liked_past", AtomType.PREFERENCE),
        (r"I will start learning (.+?)(?:[.,](?=\s|$)|$)", "will_learn", AtomType.SKILL),
        (r"I started learning (.+?)(?:[.,](?=\s|$)|$)", "started_learning", AtomType.SKILL),
        (r"I always (?:like|liked) (.+?)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        (r"I recently started liking (.+?)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        (r"I sometimes like (.+?)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        (r"I definitely like (.+?)(?:[.,](?=\s|$)|$)", "likes", AtomType.PREFERENCE),
        
        # Temporal with year markers
        (r"In (\d{4}) I worked at (.+?)(?:[.,](?=\s|$)|$)", "worked_at_year", AtomType.EVENT),
        (r"In (\d{4}) I work at (.+?)(?:[.,](?=\s|$)|$)", "works_at_year", AtomType.EVENT),
        
        # ========== ACTIONS & BEHAVIORS ==========
        (r"I want (?:to )?(?:learn |study )?(.+?)(?:[.,](?=\s|$)|$)", "wants", AtomType.PREFERENCE),
        (r"I avoid (?:learning |studying )?(.+?)(?:[.,](?=\s|$)|$)", "avoids", AtomType.PREFERENCE),
        (r"I support (.+?)(?:[.,](?=\s|$)|$)", "supports", AtomType.BELIEF),
        (r"I oppose (.+?)(?:[.,](?=\s|$)|$)", "opposes", AtomType.BELIEF),
        (r"I accept (.+?)(?:[.,](?=\s|$)|$)", "accepts", AtomType.BELIEF),
        (r"I reject (.+?)(?:[.,](?=\s|$)|$)", "rejects", AtomType.BELIEF),
        (r"I trust (.+?)(?:[.,](?=\s|$)|$)", "trusts", AtomType.BELIEF),
        (r"I distrust (.+?)(?:[.,](?=\s|$)|$)", "distrusts", AtomType.BELIEF),
        (r"I agree with (.+?)(?:[.,](?=\s|$)|$)", "agrees", AtomType.BELIEF),
        (r"I disagree with (.+?)(?:[.,](?=\s|$)|$)", "disagrees", AtomType.BELIEF),
        
        # ========== SKILLS/TOOLS ==========
        (r"I (?:am|'m) (?:currently )?learning (.+?)(?:[.,](?=\s|$)|$)", "learning", AtomType.SKILL),
        (r"I (?:use|work with|know) (.+?)(?:[.,](?=\s|$)|$)", "uses", AtomType.SKILL),
        (r"I'm (?:good at|skilled in|experienced with) (.+?)(?:[.,](?=\s|$)|$)", "uses", AtomType.SKILL),
        (r"I drive (?:a |an )?(.+?)(?:[.,](?=\s|$)|$)", "drives", AtomType.SKILL),
        (r"I do ((?:backend |frontend |web |mobile )?programming)", "does", AtomType.SKILL),
        (r"I studied (.+?)(?:[.,](?=\s|$)|$)", "studied", AtomType.EVENT),
        
        # Comparative patterns: "X is better than Y" / "I prefer X over Y"
        (r"([A-Za-z][\w+#.-]*(?:\s+[\w+#.-]+){0,3}) is better than ([A-Za-z][\w+#.-]*(?:\s+[\w+#.-]+){0,3})(?:[.,](?=\s|$)|$)", "prefers_over", AtomType.PREFERENCE),
        (r"([A-Za-z][\w+#.-]*(?:\s+[\w+#.-]+){0,3}) is worse than ([A-Za-z][\w+#.-]*(?:\s+[\w+#.-]+){0,3})(?:[.,](?=\s|$)|$)", "dislikes_over", AtomType.PREFERENCE),
        (r"I prefer ([A-Z]) over ([A-Z])", "prefers_over", AtomType.PREFERENCE),
        
        # ========== EVENTS ==========
        (r"I (?:completed|finished) (.+?)(?:[.,](?=\s|$)|$)", "completed", AtomType.EVENT),
        (r"I (?:started|began) (.+?)(?:[.,](?=\s|$)|$)", "started", AtomType.EVENT),
        
        # ========== IDENTITY (LAST - most generic) ==========
        (r"^([A-Z]\w+) is not bad", "likes", AtomType.PREFERENCE),
        (r"(?:I am|I'm) (?:a |an )?(.+?)(?:[.,](?=\s|$)|$)", "is", AtomType.ENTITY),
        (r"my name is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "named", AtomType.ENTITY),
    ]

    # Clause boundary: sentence punctuation, commas, or a coordinating
    # conjunction opening a new first-person clause ("... and I ...").
    # Punctuation only splits when followed by whitespace/EOL so versions
    # ("3.11"), paths ("file.py") and numbers ("1,000") stay intact.
    _CLAUSE_SPLIT = re.compile(
        r"(?:[.!?;]+(?=\s|$)|,\s+|\s+(?:and|but|so|then|yet)\s+(?=i\b|i'm\b|i've\b|my\b))",
        re.IGNORECASE,
    )

    # Whole-message signals that must be checked before clause splitting.
    _SIGNAL_PREDICATES = {"correction", "clarification"}

    def __init__(self) -> None:
        self.extraction_count = 0
        self.context_extractor = ContextExtractor()
        logger.info("RuleBasedExtractor initialized with {} patterns", len(self.PATTERNS))

    @property
    def extractions(self) -> int:
        """Number of extract() calls (metrics alias)"""
        return self.extraction_count

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
        self.extraction_count += 1

        # Extract contexts from message
        contexts = self.context_extractor.extract(message)

        # Correction/clarification signals apply to the whole message and must
        # run before clause splitting (their payload can contain commas).
        for pattern, predicate, _ in self.PATTERNS:
            if predicate not in self._SIGNAL_PREDICATES:
                continue
            match = re.search(pattern, message, re.IGNORECASE)
            if not match:
                continue
            obj = match.group(1).strip()
            if len(obj) < 2:
                continue
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

        # Split into clauses so one message can yield multiple atoms.
        # Boundaries: sentence punctuation, commas, and a coordinating
        # conjunction that starts a new first-person clause ("... and I ...").
        seen: set[tuple[str, str]] = set()
        for clause in self._CLAUSE_SPLIT.split(message):
            clause = clause.strip()
            if len(clause) < 3:
                continue

            # Try patterns in priority order - FIRST MATCH WINS per clause
            for pattern, predicate, atom_type in self.PATTERNS:
                if predicate in self._SIGNAL_PREDICATES:
                    continue

                match = re.search(pattern, clause, re.IGNORECASE)
                if not match:
                    continue

                # Object is the last captured group (two-group patterns put
                # year/role first, object second). Comparative patterns build
                # a normalized "X over Y" ranking object.
                if predicate == "prefers_over" and (match.lastindex or 0) >= 2:
                    obj = f"{match.group(1)} over {match.group(2)}"
                elif predicate == "dislikes_over" and (match.lastindex or 0) >= 2:
                    # "A is worse than B" == prefer B over A
                    obj = f"{match.group(2)} over {match.group(1)}"
                    predicate = "prefers_over"
                else:
                    obj = match.group(match.lastindex or 1).strip()

                # Trim dangling conjunctions left at clause edges
                obj = re.sub(r"\s+(?:and|or|but|so)\s*$", "", obj).strip()
                # Trim trailing temporal markers ("in 2020", "now", "currently")
                obj = re.sub(
                    r"\s+(?:in \d{4}|now|currently|these days|anymore)$",
                    "",
                    obj,
                    flags=re.IGNORECASE,
                ).strip()

                if not obj or len(obj) < 2:
                    continue

                # Infer provenance
                provenance = self._infer_provenance(clause)

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
                    key = (atom.predicate, atom.object.lower())
                    if key in seen:
                        break
                    seen.add(key)
                    atoms.append(atom)
                    logger.debug(
                        "Extracted: [{subject}] [{predicate}] [{object}]",
                        subject=atom.subject,
                        predicate=atom.predicate,
                        object=atom.object,
                    )
                    break  # First match wins for this clause
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
