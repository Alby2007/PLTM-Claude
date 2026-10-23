"""Write Lane - Stage 2: Evidence evaluation and persistence"""

from typing import List, Tuple

from loguru import logger

from src.core.models import GraphType, JudgeVerdict, JuryDecision, MemoryAtom, ReconciliationAction
from src.reconciliation.conflict_detector import ConflictDetector
from src.reconciliation.resolver import ReconciliationResolver
from src.storage.sqlite_store import SQLiteGraphStore


class WriteLane:
    """
    Stage 2: Write Lane
    
    Operations:
    - Apply jury verdicts and confidence adjustments
    - Check promotion eligibility (tiered promotion)
    - Handle conflict reconciliation
    - Write to appropriate graph (substantiated/unsubstantiated)
    - Update atom metadata
    - Log all decisions
    
    Latency target: <500ms for batch write
    """

    def __init__(self, store: SQLiteGraphStore) -> None:
        self.store = store
        self.conflict_detector = ConflictDetector(store)
        self.reconciliation_resolver = ReconciliationResolver()
        self.atoms_processed = 0
        self.atoms_promoted = 0
        self.conflicts_resolved = 0
        logger.info("WriteLane initialized")

    async def process_verdicts(
        self,
        atoms_with_verdicts: List[Tuple[MemoryAtom, JuryDecision]],
    ) -> dict:
        """
        Process atoms with jury verdicts.
        
        Args:
            atoms_with_verdicts: List of (atom, jury_decision) tuples
            
        Returns:
            Summary statistics
        """
        approved = 0
        rejected = 0
        promoted = 0
        conflicts = 0

        for atom, decision in atoms_with_verdicts:
            self.atoms_processed += 1

            # Skip rejected atoms
            if decision.final_verdict == JudgeVerdict.REJECT:
                rejected += 1
                logger.info(
                    "Atom rejected by jury: [{subject}] [{predicate}] [{object}]",
                    subject=atom.subject,
                    predicate=atom.predicate,
                    object=atom.object[:50],
                )
                continue

            # Apply confidence adjustment from jury
            atom.confidence += decision.confidence_adjustment
            atom.confidence = max(0.0, min(1.0, atom.confidence))

            # Store jury decision in history (dedupe — deliberate_batch may
            # already have recorded this exact decision)
            if not atom.jury_history or atom.jury_history[-1] is not decision:
                atom.jury_history.append(decision)

            # Check for conflicts (same predicate)
            conflict_atoms = await self.conflict_detector.find_conflicts(atom)
            
            # Also check for opposite predicates (likes vs dislikes)
            opposite_atoms = await self.conflict_detector.check_opposite_predicates(atom)
            
            # Combine all conflicts
            all_conflicts = conflict_atoms + opposite_atoms
            
            if all_conflicts:
                conflicts += len(all_conflicts)
                await self._handle_conflicts(atom, all_conflicts)
                self.conflicts_resolved += len(all_conflicts)

            # Reconciliation can reject the candidate outright — don't store it
            if "rejected_by_reconciliation" in atom.flags:
                rejected += 1
                continue

            # Preference rankings are singular per subject: a new "X over Y"
            # statement supersedes all prior ranking statements.
            if self._is_ranking_atom(atom):
                await self._supersede_prior_rankings(atom)

            # Check promotion eligibility (tiered promotion)
            if atom.promotion_eligible:
                await self._promote_to_substantiated(atom)
                promoted += 1
                self.atoms_promoted += 1
            else:
                await self._write_to_unsubstantiated(atom)

            approved += 1

        logger.info(
            "WriteLane batch complete: {approved} approved, {rejected} rejected, "
            "{promoted} promoted, {conflicts} conflicts resolved",
            approved=approved,
            rejected=rejected,
            promoted=promoted,
            conflicts=conflicts,
        )

        return {
            "approved": approved,
            "rejected": rejected,
            "promoted": promoted,
            "conflicts_resolved": conflicts,
        }

    async def _promote_to_substantiated(self, atom: MemoryAtom) -> None:
        """
        Promote atom to substantiated graph.
        
        Sets:
        - graph = SUBSTANTIATED
        - strength = 1.0 (no decay)
        Preserves the jury-calibrated confidence rather than forcing 1.0.
        """
        atom.graph = GraphType.SUBSTANTIATED
        atom.strength = 1.0

        await self.store.insert_atom(atom)

        logger.info(
            "Promoted to substantiated: [{subject}] [{predicate}] [{object}] (tier: {tier})",
            subject=atom.subject,
            predicate=atom.predicate,
            object=atom.object[:50],
            tier=atom.promotion_tier,
        )

    async def _write_to_unsubstantiated(self, atom: MemoryAtom) -> None:
        """Write to unsubstantiated graph (shadow buffer)"""
        atom.graph = GraphType.UNSUBSTANTIATED

        await self.store.insert_atom(atom)

        logger.debug(
            "Written to unsubstantiated: [{subject}] [{predicate}] [{object}] "
            "(confidence: {conf:.2f}, tier: {tier})",
            subject=atom.subject,
            predicate=atom.predicate,
            object=atom.object[:50],
            conf=atom.confidence,
            tier=atom.promotion_tier,
        )

    async def _handle_conflicts(
        self,
        candidate: MemoryAtom,
        conflicts: List[MemoryAtom],
    ) -> None:
        """
        Handle conflicts through reconciliation.
        
        Args:
            candidate: New atom
            conflicts: List of conflicting atoms
        """
        logger.info(
            "Handling {count} conflicts for [{subject}] [{predicate}] [{object}]",
            count=len(conflicts),
            subject=candidate.subject,
            predicate=candidate.predicate,
            object=candidate.object[:50],
        )

        # Get reconciliation decisions
        decisions = self.reconciliation_resolver.reconcile_batch(candidate, conflicts)

        # Execute reconciliation actions
        for existing, decision in zip(conflicts, decisions):
            await self._execute_reconciliation(candidate, existing, decision)

    async def _execute_reconciliation(
        self,
        candidate: MemoryAtom,
        existing: MemoryAtom,
        decision: ReconciliationAction,
    ) -> None:
        """
        Execute a reconciliation decision.
        
        Args:
            candidate: New atom
            existing: Existing conflicting atom
            decision: Reconciliation decision
        """
        if decision.action == ReconciliationAction.SUPERSEDE:
            # Link supersession chain in candidate
            candidate.supersedes = existing.id
            
            # Archive existing atom (this also updates its superseded_by field)
            if decision.archive_existing:
                existing.superseded_by = candidate.id
                existing.graph = GraphType.HISTORICAL
                await self.store.insert_atom(existing)  # Update to historical
                logger.info(
                    "Archived superseded atom: {id}",
                    id=existing.id,
                )

            logger.info(
                "Supersession: [{new}] replaces [{old}] - {reason}",
                new=candidate.object[:30],
                old=existing.object[:30],
                reason=decision.explanation,
            )

        elif decision.action == ReconciliationAction.CONTEXTUALIZE:
            # Add contexts if provided
            if decision.contexts_added:
                candidate.contexts.extend(decision.contexts_added)

            # Both atoms coexist
            logger.info(
                "Contextualization: Both atoms kept - {reason}",
                reason=decision.explanation,
            )

        elif decision.action == ReconciliationAction.REJECT:
            # Candidate is rejected, existing wins
            # Mark candidate as rejected (don't insert)
            candidate.flags.append("rejected_by_reconciliation")
            
            logger.info(
                "Reconciliation rejected candidate: {reason}",
                reason=decision.explanation,
            )

        elif decision.action == ReconciliationAction.MERGE:
            # Future: Merge information from both atoms
            logger.warning("MERGE action not implemented in MVP")

    @staticmethod
    def _is_ranking_atom(atom: MemoryAtom) -> bool:
        """Atoms expressing an ordered preference ("X over Y")"""
        return atom.predicate == "prefers_over" or (
            atom.predicate == "prefers" and " over " in atom.object.lower()
        )

    async def _supersede_prior_rankings(self, candidate: MemoryAtom) -> None:
        """Archive prior ranking atoms for the same subject"""
        for predicate in ("prefers_over", "prefers"):
            existing = await self.store.find_by_triple(
                candidate.subject, predicate, exclude_historical=True
            )
            for atom in existing:
                if atom.id == candidate.id or " over " not in atom.object.lower():
                    continue
                atom.graph = GraphType.HISTORICAL
                atom.superseded_by = candidate.id
                await self.store.insert_atom(atom)
                candidate.supersedes = atom.id
                logger.info(
                    "Ranking superseded: [{old}] replaced by [{new}]",
                    old=atom.object[:40],
                    new=candidate.object[:40],
                )

    def get_stats(self) -> dict:
        """Get write lane statistics"""
        return {
            "atoms_processed": self.atoms_processed,
            "atoms_promoted": self.atoms_promoted,
            "conflicts_resolved": self.conflicts_resolved,
            "promotion_rate": (
                self.atoms_promoted / self.atoms_processed
                if self.atoms_processed > 0
                else 0.0
            ),
        }
