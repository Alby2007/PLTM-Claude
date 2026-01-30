"""Jury orchestrator for batch deliberation"""

from typing import List, Optional, Tuple

from loguru import logger

from datetime import datetime

from src.core.models import JudgeVerdict, JuryDecision, SimpleJuryDecision, JuryVerdict, MemoryAtom
from src.jury.consensus_judge import ConsensusJudge
from src.jury.memory_judge import MemoryJudge
from src.jury.safety_judge import SafetyJudge
from src.jury.time_judge import TimeJudge


def convert_to_legacy_decision(simple_decision: SimpleJuryDecision, stage: int = 1) -> JuryDecision:
    """
    Convert new SimpleJuryDecision to legacy JuryDecision format for backward compatibility.
    
    This adapter allows the new 4-judge system to work with the existing pipeline.
    """
    # Map JuryVerdict to JudgeVerdict
    verdict_map = {
        JuryVerdict.APPROVE: JudgeVerdict.APPROVE,
        JuryVerdict.REJECT: JudgeVerdict.REJECT,
        JuryVerdict.QUARANTINE: JudgeVerdict.QUARANTINE,
    }
    
    # Map confidence to adjustment (-0.3 to +0.3 range)
    if simple_decision.verdict == JuryVerdict.APPROVE:
        confidence_adjustment = 0.1
    elif simple_decision.verdict == JuryVerdict.QUARANTINE:
        confidence_adjustment = -0.1
    else:  # REJECT
        confidence_adjustment = -0.3
    
    return JuryDecision(
        timestamp=datetime.now(),
        stage=stage,
        final_verdict=verdict_map[simple_decision.verdict],
        confidence_adjustment=confidence_adjustment,
        reasoning=simple_decision.explanation,
        # Individual judge verdicts not tracked in new system
        memory_judge=None,
        safety_judge=None,
        consensus_judge=None,
        time_judge=None,
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
    - All judges deliberate in parallel
    - Consensus judge aggregates final verdict
    - Weighted voting for split decisions
    """

    def __init__(self) -> None:
        self.safety_judge = SafetyJudge()
        self.memory_judge = MemoryJudge()
        self.time_judge = TimeJudge()
        self.consensus_judge = ConsensusJudge()
        self.deliberation_count = 0
        logger.info("JuryOrchestrator initialized (4-judge system: Safety + Memory + Time + Consensus)")

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

        # Run all judges in parallel (conceptually - sequential for now)
        safety_decision = self.safety_judge.evaluate(atom)
        memory_decision = self.memory_judge.evaluate(atom)
        time_decision = self.time_judge.evaluate(atom, context)
        
        # Convert old-style safety/memory results to new SimpleJuryDecision format
        # Safety judge (old format)
        if isinstance(safety_decision, dict):
            safety_verdict = JuryVerdict.REJECT if safety_decision["verdict"] == JudgeVerdict.REJECT else JuryVerdict.APPROVE
            safety_decision = SimpleJuryDecision(
                verdict=safety_verdict,
                confidence=0.95,
                explanation=safety_decision.get("reason", "Safety check"),
                judge_name="SafetyJudge",
            )
        
        # Memory judge (old format)
        if isinstance(memory_decision, dict):
            memory_verdict_map = {
                JudgeVerdict.APPROVE: JuryVerdict.APPROVE,
                JudgeVerdict.QUARANTINE: JuryVerdict.QUARANTINE,
                JudgeVerdict.REJECT: JuryVerdict.REJECT,
            }
            memory_verdict = memory_verdict_map.get(memory_decision["verdict"], JuryVerdict.APPROVE)
            memory_decision = SimpleJuryDecision(
                verdict=memory_verdict,
                confidence=0.90,
                explanation=memory_decision.get("reason", "Memory validation"),
                judge_name="MemoryJudge",
            )
        
        # Collect all decisions
        all_decisions = [safety_decision, memory_decision, time_decision]
        
        # Safety judge has VETO authority - if it rejects, override consensus
        if safety_decision.verdict == JuryVerdict.REJECT:
            logger.warning(
                "Jury REJECTED atom (safety veto): [{subject}] [{predicate}] [{object}]",
                subject=atom.subject,
                predicate=atom.predicate,
                object=atom.object,
            )
            # Return rejection decision directly
            return SimpleJuryDecision(
                verdict=JuryVerdict.REJECT,
                confidence=safety_decision.confidence,
                explanation=f"Safety veto: {safety_decision.explanation}",
                judge_name="ConsensusJudge",
            )
        
        # Get consensus from all non-veto decisions
        consensus_decision = self.consensus_judge.aggregate_decisions(all_decisions, atom)
        
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
        return convert_to_legacy_decision(consensus_decision, stage)

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

            # Use final_verdict since adapter returns legacy JuryDecision
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
