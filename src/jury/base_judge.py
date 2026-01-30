"""Base class for grammar-constrained judges"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from loguru import logger

from src.core.models import JudgeVerdict, MemoryAtom


class BaseJudge(ABC):
    """
    Base class for all judges.
    
    For MVP: Uses simple rule-based logic instead of LLM.
    Future: Add Outlines-based grammar-constrained LLM judges.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.evaluation_count = 0
        logger.info(f"{name} initialized (rule-based for MVP)")

    @abstractmethod
    def evaluate(self, atom: MemoryAtom) -> Dict[str, Any]:
        """
        Evaluate atom and return verdict.
        
        Returns:
            {
                "verdict": JudgeVerdict,
                "reason": str,
                "confidence": float (0.0-1.0),
                "severity": int (0-10, for safety judge)
            }
        """
        pass

    def _log_decision(
        self,
        atom: MemoryAtom,
        verdict: JudgeVerdict,
        reason: str,
    ) -> None:
        """Log judge decision for audit trail"""
        self.evaluation_count += 1
        logger.debug(
            "{judge}: {verdict} - [{subject}] [{predicate}] [{object}] - {reason}",
            judge=self.name,
            verdict=verdict.value,
            subject=atom.subject,
            predicate=atom.predicate,
            object=atom.object,
            reason=reason,
        )

    def get_stats(self) -> Dict[str, int]:
        """Get judge statistics"""
        return {
            "name": self.name,
            "evaluations": self.evaluation_count,
        }
