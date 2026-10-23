"""Jury orchestrator for batch deliberation"""

from typing import Dict, List, Optional, Tuple

from loguru import logger

from datetime import datetime

from src.core.models import JudgeVerdict, JuryDecision, SimpleJuryDecision, JuryVerdict, MemoryAtom
from src.jury.consensus_judge import ConsensusJudge
from src.jury.memory_judge import MemoryJudge
from src.jury.safety_judge import SafetyJudge
from src.jury.time_judge import TimeJudge


_JURY_TO_JUDGE = {
    JuryVerdict.APPROVE: JudgeVerdict.APPROVE,
    JuryVerdict.QUARANTINE: JudgeVerdict.QUARANTINE,
    JuryVerdict.REJECT: JudgeVerdict.REJECT,
}

_JUDGE_TO_JURY = {
    JudgeVerdict.APPROVE: JuryVerdict.APPROVE,
    JudgeVerdict.QUARANTINE: JuryVerdict.QUARANTINE,
    JudgeVerdict.REJECT: JuryVerdict.REJECT,
    JudgeVerdict.ESCALATE: JuryVerdict.QUARANTINE,
}


def _to_judge_verdict(decision) -> JudgeVerdict:
    """Normalize a SimpleJuryDecision (JuryVerdict) to legacy JudgeVerdict."""
    verdict = decision.verdict
    if isinstance(verdict, JudgeVerdict):
        return verdict
    return _JURY_TO_JUDGE.get(verdict, JudgeVerdict.QUARANTINE)


def _confidence_adjustment(verdict: JudgeVerdict) -> float:
    if verdict == JudgeVerdict.APPROVE:
        return 0.1
    if verdict == JudgeVerdict.QUARANTINE:
        return -0.1
    return -0.3


def convert_to_legacy_decision(
    simple_decision: SimpleJuryDecision,
    stage: int = 1,
    judge_verdicts: Optional[Dict[str, JudgeVerdict]] = None,
    reasoning: Optional[str] = None,
) -> JuryDecision:
    """
    Convert new SimpleJuryDecision to legacy JuryDecision format for backward compatibility.
    
    This adapter allows the new 4-judge system to work with the existing pipeline.
    """
    final_verdict = _to_judge_verdict(simple_decision)
    judge_verdicts = judge_verdicts or {}

    return JuryDecision(
        timestamp=datetime.now(),
        stage=stage,
        final_verdict=final_verdict,
        confidence_adjustment=_confidence_adjustment(final_verdict),
        reasoning=reasoning if reasoning is not None else simple_decision.explanation,
        memory_judge=judge_verdicts.get("memory_judge"),
        safety_judge=judge_verdicts.get("safety_judge"),
        consensus_judge=judge_verdicts.get("consensus_judge"),
        time_judge=judge_verdicts.get("time_judge"),
    )


class JuryOrchestrator:
    """
    Orchestrates jury deliberation with 4 judges.
    
    Judges:
    1. Safety Judge - Detects harmful/inappropriate content (VETO authority)
    2. Memory Judge - Validates ontology and semantic correctness
    3. Time Judge - Validates temporal consistency
    4. Consensus Judge - Aggregates decisions and resolves conflicts
    
    Rules:
    - Safety judge has VETO authority (can override all others)
    - Any judge REJECT is binding (final verdict REJECT)
    - Consensus judge aggregates the remaining decisions
    - Weighted voting for split decisions
    """

    def __init__(
        self,
        safety_judge: Optional[SafetyJudge] = None,
        memory_judge: Optional[MemoryJudge] = None,
        time_judge: Optional[TimeJudge] = None,
        consensus_judge: Optional[ConsensusJudge] = None,
    ) -> None:
        self.safety_judge = safety_judge or SafetyJudge()
        self.memory_judge = memory_judge or MemoryJudge()
        self.time_judge = time_judge or TimeJudge()
        self.consensus_judge = consensus_judge or ConsensusJudge()
        self.deliberation_count = 0
        logger.info("JuryOrchestrator initialized (4-judge system: Safety + Memory + Time + Consensus)")

    @staticmethod
    def _normalize(judge_name: str, result, default_explanation: str) -> SimpleJuryDecision:
        """Convert legacy dict-returning judges to SimpleJuryDecision."""
        if isinstance(result, SimpleJuryDecision):
            return result
        if isinstance(result, dict):
            verdict = result.get("verdict", JudgeVerdict.APPROVE)
            jury_verdict = (
                verdict if isinstance(verdict, JuryVerdict)
                else _JUDGE_TO_JURY.get(verdict, JuryVerdict.QUARANTINE)
            )
            return SimpleJuryDecision(
                verdict=jury_verdict,
                confidence=float(result.get("confidence", 0.9)),
                explanation=result.get("reason", default_explanation),
                judge_name=judge_name,
            )
        raise TypeError(f"Unexpected decision type from {judge_name}: {type(result)}")

    async def deliberate(
        self,
        atom: MemoryAtom,
        stage: int = 1,
        context: Optional[dict] = None,
    ) -> JuryDecision:
        """
        Run jury deliberation on a single atom using 4-judge system.
        
        Args:
            atom: MemoryAtom to evaluate
            stage: Processing stage (1 for current system)
            context: Optional context (recent atoms, session info, etc.)
            
        Returns:
            JuryDecision with final verdict from consensus
        """
        self.deliberation_count += 1

        # Run all judges
        safety_decision = self._normalize(
            "SafetyJudge", self.safety_judge.evaluate(atom), "Safety check"
        )
        memory_decision = self._normalize(
            "MemoryJudge", self.memory_judge.evaluate(atom), "Memory validation"
        )
        time_decision = self._normalize(
            "TimeJudge", self.time_judge.evaluate(atom, context), "Temporal check"
        )

        all_decisions = [safety_decision, memory_decision, time_decision]
        judge_verdicts = {
            "safety_judge": _to_judge_verdict(safety_decision),
            "memory_judge": _to_judge_verdict(memory_decision),
            "time_judge": _to_judge_verdict(time_decision),
        }

        def _reasoning(final: SimpleJuryDecision) -> str:
            parts = [
                f"{d.judge_name}: {d.verdict.value} ({d.explanation})"
                for d in all_decisions
            ]
            parts.append(f"ConsensusJudge: {final.verdict.value} ({final.explanation})")
            return "; ".join(parts)

        # Safety judge has VETO authority - if it rejects, override consensus
        if safety_decision.verdict == JuryVerdict.REJECT:
            logger.warning(
                "Jury REJECTED atom (safety veto): [{subject}] [{predicate}] [{object}]",
                subject=atom.subject,
                predicate=atom.predicate,
                object=atom.object,
            )
            final = SimpleJuryDecision(
                verdict=JuryVerdict.REJECT,
                confidence=safety_decision.confidence,
                explanation=f"Safety veto: {safety_decision.explanation}",
                judge_name="ConsensusJudge",
            )
            decision = convert_to_legacy_decision(
                final, stage, judge_verdicts, _reasoning(final)
            )
            decision.consensus_judge = JudgeVerdict.REJECT
            return decision

        # Any other judge REJECT is binding
        rejecters = [d for d in all_decisions if d.verdict == JuryVerdict.REJECT]
        if rejecters:
            final = SimpleJuryDecision(
                verdict=JuryVerdict.REJECT,
                confidence=max(d.confidence for d in rejecters),
                explanation=f"{rejecters[0].judge_name} rejected: {rejecters[0].explanation}",
                judge_name="ConsensusJudge",
            )
            decision = convert_to_legacy_decision(
                final, stage, judge_verdicts, _reasoning(final)
            )
            decision.consensus_judge = JudgeVerdict.REJECT
            return decision

        # Get consensus from all non-reject decisions
        consensus_decision = self.consensus_judge.aggregate_decisions(all_decisions, atom)
        judge_verdicts["consensus_judge"] = _to_judge_verdict(consensus_decision)

        # Log the decision
        verdict_emoji = {
            JuryVerdict.APPROVE: "✅",
            JuryVerdict.QUARANTINE: "⚠️",
            JuryVerdict.REJECT: "❌",
        }

        logger.debug(
            f"{verdict_emoji[consensus_decision.verdict]} Jury decision: "
            f"[{atom.subject}] [{atom.predicate}] [{atom.object}] - "
            f"{consensus_decision.verdict.value} ({consensus_decision.confidence:.0%})"
        )

        # Convert to legacy format for backward compatibility with pipeline
        return convert_to_legacy_decision(
            consensus_decision, stage, judge_verdicts, _reasoning(consensus_decision)
        )

    async def deliberate_batch(
        self,
        atoms: List[MemoryAtom],
        stage: int = 1,
        context: Optional[dict] = None,
    ) -> Tuple[List[MemoryAtom], List[MemoryAtom], List[MemoryAtom]]:
        """
        Run jury deliberation on a batch of atoms using 4-judge system.
        
        Args:
            atoms: List of MemoryAtoms to evaluate
            stage: Processing stage
            context: Optional context for judges
            
        Returns:
            Tuple of (approved, rejected, quarantined) atoms
        """
        approved = []
        rejected = []
        quarantined = []

        logger.info(f"Jury deliberating on {len(atoms)} atoms")

        for atom in atoms:
            decision = await self.deliberate(atom, stage, context)

            # Store decision in atom's history
            atom.jury_history.append(decision)

            if decision.final_verdict == JudgeVerdict.APPROVE:
                approved.append(atom)
            elif decision.final_verdict == JudgeVerdict.REJECT:
                rejected.append(atom)
            else:  # QUARANTINE
                quarantined.append(atom)

            self._log_decision(atom, decision)

        logger.info(
            f"Batch deliberation complete: {len(approved)} approved, "
            f"{len(rejected)} rejected, {len(quarantined)} quarantined"
        )

        return approved, rejected, quarantined

    def _log_decision(self, atom: MemoryAtom, decision: JuryDecision) -> None:
        """Log jury decision"""
        logger.debug(
            "Jury decision: {verdict} for [{subject}] [{predicate}] [{object}]",
            verdict=decision.final_verdict.value,
            subject=atom.subject,
            predicate=atom.predicate,
            object=atom.object,
        )

    def get_stats(self) -> dict:
        """Get jury statistics"""
        return {
            "deliberations": self.deliberation_count,
            "safety_judge": self.safety_judge.get_stats(),
            "memory_judge": self.memory_judge.get_stats(),
        }
