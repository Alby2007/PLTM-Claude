"""Memory judge for ontology compliance and semantic validity"""

from typing import Any, Dict

from src.core.models import JudgeVerdict, MemoryAtom
from src.core.ontology import validate_atom
from src.jury.base_judge import BaseJudge


class MemoryJudge(BaseJudge):
    """
    Evaluates ontology compliance and semantic validity.
    
    For MVP: Rule-based ontology validation.
    Future: Grammar-constrained LLM with Outlines.
    
    Checks:
    - Ontology compliance (predicate allowed for atom type)
    - Semantic validity (triple makes sense)
    - Completeness (all required fields present)
    """

    def __init__(self) -> None:
        super().__init__("MemoryJudge")

    def evaluate(self, atom: MemoryAtom) -> Dict[str, Any]:
        """
        Evaluate atom for ontology compliance.
        
        Returns:
            {
                "verdict": JudgeVerdict,
                "reason": str,
                "confidence": float (0.0-1.0)
            }
        """
        # Check ontology compliance
        is_valid, reason = validate_atom(atom)
        
        if not is_valid:
            verdict = JudgeVerdict.REJECT
            confidence = 1.0  # Confident in ontology violations
            self._log_decision(atom, verdict, reason)
            return {
                "verdict": verdict,
                "reason": reason,
                "confidence": confidence,
            }

        # Check for empty or trivial content
        if len(atom.object.strip()) < 2:
            verdict = JudgeVerdict.REJECT
            reason = "Object too short or empty"
            confidence = 1.0
            self._log_decision(atom, verdict, reason)
            return {
                "verdict": verdict,
                "reason": reason,
                "confidence": confidence,
            }

        # Check for nonsensical combinations
        nonsense_check = self._check_semantic_sense(atom)
        if not nonsense_check["makes_sense"]:
            verdict = JudgeVerdict.QUARANTINE
            reason = nonsense_check["reason"]
            confidence = 0.7  # Less confident on semantic checks
            self._log_decision(atom, verdict, reason)
            return {
                "verdict": verdict,
                "reason": reason,
                "confidence": confidence,
            }

        # All checks passed
        verdict = JudgeVerdict.APPROVE
        reason = "Ontology compliant and semantically valid"
        confidence = 0.9
        self._log_decision(atom, verdict, reason)
        return {
            "verdict": verdict,
            "reason": reason,
            "confidence": confidence,
        }

    def _check_semantic_sense(self, atom: MemoryAtom) -> Dict[str, Any]:
        """
        Check if the semantic triple makes sense.
        
        For MVP: Basic heuristics.
        Future: LLM-based semantic validation.
        """
        # Check for repeated words (often indicates extraction error)
        words = atom.object.lower().split()
        if len(words) > 1 and len(set(words)) == 1:
            return {
                "makes_sense": False,
                "reason": "Object contains only repeated words",
            }

        # Check for very generic objects that provide no information
        generic_objects = {"something", "anything", "nothing", "stuff", "things"}
        if atom.object.lower().strip() in generic_objects:
            return {
                "makes_sense": False,
                "reason": "Object is too generic to be useful",
            }

        # Check for contradiction in same triple (e.g., "likes" + "hate")
        contradictions = [
            (["like", "love", "enjoy"], ["hate", "dislike"]),
            (["good", "great", "excellent"], ["bad", "terrible", "awful"]),
            (["yes"], ["no"]),
        ]
        
        predicate_lower = atom.predicate.lower()
        object_lower = atom.object.lower()
        
        for positive_words, negative_words in contradictions:
            if any(p in predicate_lower for p in positive_words):
                if any(n in object_lower for n in negative_words):
                    return {
                        "makes_sense": False,
                        "reason": "Contradictory sentiment in predicate and object",
                    }

        # Passed all checks
        return {"makes_sense": True, "reason": ""}
