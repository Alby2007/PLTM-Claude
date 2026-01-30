"""Consensus Judge - Orchestrates jury decisions and resolves conflicts"""

from typing import List, Optional

from loguru import logger

from src.core.models import SimpleJuryDecision, JuryVerdict, MemoryAtom
from src.jury.base_judge import BaseJudge


class ConsensusJudge(BaseJudge):
    """
    Orchestrates jury decisions and resolves conflicts between judges.
    
    Responsibilities:
    - Aggregate decisions from all judges
    - Resolve conflicts when judges disagree
    - Apply weighted voting based on judge confidence
    - Make final verdict with explanation
    
    Voting Rules:
    1. Unanimous APPROVE → APPROVE
    2. Any REJECT → QUARANTINE (investigate further)
    3. Majority QUARANTINE → QUARANTINE
    4. Split decision → Use confidence-weighted voting
    5. Tie → QUARANTINE (err on side of caution)
    
    Examples:
    - [APPROVE, APPROVE, APPROVE, APPROVE] → APPROVE
    - [APPROVE, APPROVE, APPROVE, REJECT] → QUARANTINE
    - [APPROVE, APPROVE, QUARANTINE, QUARANTINE] → QUARANTINE
    - [APPROVE(0.9), APPROVE(0.8), QUARANTINE(0.6)] → APPROVE (weighted)
    """
    
    def __init__(self):
        super().__init__("ConsensusJudge")
        self.min_confidence_threshold = 0.7
        self.reject_weight = 2.0  # REJECTs count double
        
    def evaluate(self, atom: MemoryAtom, context: Optional[dict] = None) -> SimpleJuryDecision:
        """
        This judge doesn't evaluate atoms directly - it aggregates other judges' decisions.
        Use aggregate_decisions() instead.
        """
        raise NotImplementedError(
            "ConsensusJudge doesn't evaluate atoms directly. "
            "Use aggregate_decisions() to combine judge verdicts."
        )
    
    def aggregate_decisions(
        self, 
        decisions: List[SimpleJuryDecision],
        atom: MemoryAtom
    ) -> SimpleJuryDecision:
        """
        Aggregate decisions from all judges into final verdict.
        
        Args:
            decisions: List of decisions from all judges
            atom: The memory atom being evaluated
            
        Returns:
            Final consensus decision
        """
        if not decisions:
            logger.error("No decisions to aggregate")
            return SimpleJuryDecision(
                verdict=JuryVerdict.REJECT,
                confidence=1.0,
                explanation="No judge decisions available",
                judge_name=self.name,
            )
        
        # Count verdicts
        verdict_counts = {
            JuryVerdict.APPROVE: 0,
            JuryVerdict.QUARANTINE: 0,
            JuryVerdict.REJECT: 0,
        }
        
        for decision in decisions:
            verdict_counts[decision.verdict] += 1
        
        # Log individual decisions
        logger.debug(
            f"Jury decisions for [{atom.subject}] [{atom.predicate}] [{atom.object}]: "
            f"APPROVE={verdict_counts[JuryVerdict.APPROVE]}, "
            f"QUARANTINE={verdict_counts[JuryVerdict.QUARANTINE]}, "
            f"REJECT={verdict_counts[JuryVerdict.REJECT]}"
        )
        
        # Rule 1: Any REJECT → QUARANTINE (investigate further)
        if verdict_counts[JuryVerdict.REJECT] > 0:
            reject_decisions = [d for d in decisions if d.verdict == JuryVerdict.REJECT]
            return SimpleJuryDecision(
                verdict=JuryVerdict.QUARANTINE,
                confidence=max(d.confidence for d in reject_decisions),
                explanation=f"Flagged for review: {reject_decisions[0].explanation}",
                judge_name=self.name,
            )
        
        # Rule 2: Unanimous APPROVE → APPROVE
        if verdict_counts[JuryVerdict.APPROVE] == len(decisions):
            avg_confidence = sum(d.confidence for d in decisions) / len(decisions)
            return SimpleJuryDecision(
                verdict=JuryVerdict.APPROVE,
                confidence=avg_confidence,
                explanation="Unanimous approval from all judges",
                judge_name=self.name,
            )
        
        # Rule 3: Majority QUARANTINE → QUARANTINE
        if verdict_counts[JuryVerdict.QUARANTINE] > len(decisions) / 2:
            quarantine_decisions = [
                d for d in decisions if d.verdict == JuryVerdict.QUARANTINE
            ]
            avg_confidence = sum(d.confidence for d in quarantine_decisions) / len(quarantine_decisions)
            return SimpleJuryDecision(
                verdict=JuryVerdict.QUARANTINE,
                confidence=avg_confidence,
                explanation=f"Majority quarantine: {quarantine_decisions[0].explanation}",
                judge_name=self.name,
            )
        
        # Rule 4: Split decision → Use confidence-weighted voting
        return self._weighted_vote(decisions, atom)
    
    def _weighted_vote(
        self, 
        decisions: List[SimpleJuryDecision],
        atom: MemoryAtom
    ) -> SimpleJuryDecision:
        """
        Resolve split decisions using confidence-weighted voting.
        
        Each judge's vote is weighted by their confidence.
        REJECTs are weighted more heavily (2x) to err on side of caution.
        """
        approve_weight = 0.0
        quarantine_weight = 0.0
        reject_weight = 0.0
        
        for decision in decisions:
            weight = decision.confidence
            
            if decision.verdict == JuryVerdict.APPROVE:
                approve_weight += weight
            elif decision.verdict == JuryVerdict.QUARANTINE:
                quarantine_weight += weight
            elif decision.verdict == JuryVerdict.REJECT:
                # REJECTs count double
                reject_weight += weight * self.reject_weight
        
        total_weight = approve_weight + quarantine_weight + reject_weight
        
        # Normalize weights
        approve_score = approve_weight / total_weight if total_weight > 0 else 0
        quarantine_score = quarantine_weight / total_weight if total_weight > 0 else 0
        reject_score = reject_weight / total_weight if total_weight > 0 else 0
        
        logger.debug(
            f"Weighted scores - APPROVE: {approve_score:.2f}, "
            f"QUARANTINE: {quarantine_score:.2f}, REJECT: {reject_score:.2f}"
        )
        
        # Determine final verdict based on highest weighted score
        if reject_score > approve_score and reject_score > quarantine_score:
            return SimpleJuryDecision(
                verdict=JuryVerdict.QUARANTINE,  # Still quarantine, not outright reject
                confidence=reject_score,
                explanation="Weighted vote: concerns raised by judges",
                judge_name=self.name,
            )
        elif quarantine_score > approve_score:
            return SimpleJuryDecision(
                verdict=JuryVerdict.QUARANTINE,
                confidence=quarantine_score,
                explanation="Weighted vote: needs further review",
                judge_name=self.name,
            )
        else:
            # APPROVE wins
            if approve_score < self.min_confidence_threshold:
                # Low confidence approval → quarantine
                return SimpleJuryDecision(
                    verdict=JuryVerdict.QUARANTINE,
                    confidence=approve_score,
                    explanation="Weighted vote: low confidence approval",
                    judge_name=self.name,
                )
            
            return SimpleJuryDecision(
                verdict=JuryVerdict.APPROVE,
                confidence=approve_score,
                explanation="Weighted vote: approved with sufficient confidence",
                judge_name=self.name,
            )
    
    def get_decision_summary(self, decisions: List[SimpleJuryDecision]) -> str:
        """Generate human-readable summary of jury decisions"""
        summary_parts = []
        
        for decision in decisions:
            verdict_emoji = {
                JuryVerdict.APPROVE: "✅",
                JuryVerdict.QUARANTINE: "⚠️",
                JuryVerdict.REJECT: "❌",
            }
            
            summary_parts.append(
                f"{verdict_emoji[decision.verdict]} {decision.judge_name}: "
                f"{decision.verdict.value} ({decision.confidence:.0%}) - {decision.explanation}"
            )
        
        return "\n".join(summary_parts)
