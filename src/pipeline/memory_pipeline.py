"""Main memory pipeline orchestrating all stages"""

from typing import AsyncIterator, Dict, List

from loguru import logger

from src.core.models import MemoryAtom
from src.extraction.hybrid import HybridExtractor
from src.jury.orchestrator import JuryOrchestrator
from src.pipeline.write_lane import WriteLane
from src.storage.sqlite_store import SQLiteGraphStore


class MemoryUpdate:
    """Progressive update from pipeline stages"""

    def __init__(
        self,
        stage: str,
        atoms_extracted: int = 0,
        atoms_approved: int = 0,
        atoms_promoted: int = 0,
        conflicts_resolved: int = 0,
    ):
        self.stage = stage
        self.atoms_extracted = atoms_extracted
        self.atoms_approved = atoms_approved
        self.atoms_promoted = atoms_promoted
        self.conflicts_resolved = conflicts_resolved


class MemoryPipeline:
    """
    Async-first 3-stage memory pipeline.
    
    Stage 0: Fast Lane - Extraction (<100ms)
    Stage 1: Jury Lane - Deliberation (<5s)
    Stage 2: Write Lane - Persistence (<500ms)
    
    Progressive updates allow UI to show results as they complete.
    """

    def __init__(self, store: SQLiteGraphStore) -> None:
        self.store = store
        self.extractor = HybridExtractor()
        self.jury = JuryOrchestrator()
        self.write_lane = WriteLane(store)
        self.messages_processed = 0
        logger.info("MemoryPipeline initialized (3-stage async)")

    async def process_message(
        self,
        message: str,
        user_id: str,
        session_id: str = "",
    ) -> AsyncIterator[MemoryUpdate]:
        """
        Process message through all pipeline stages.
        
        Yields progressive updates as stages complete.
        
        Args:
            message: User's message text
            user_id: Canonical user identifier
            session_id: Optional session identifier
            
        Yields:
            MemoryUpdate objects as stages complete
        """
        self.messages_processed += 1

        logger.info(
            "Processing message for user={user}, session={session}",
            user=user_id,
            session=session_id or "none",
        )

        # STAGE 0: Fast Lane - Extraction
        logger.debug("Stage 0: Fast Lane (extraction)")
        atoms = await self.extractor.extract_atoms(message, user_id, session_id)

        yield MemoryUpdate(
            stage="extraction",
            atoms_extracted=len(atoms),
        )

        if not atoms:
            logger.info("No atoms extracted from message")
            return

        # STAGE 1: Jury Lane - Deliberation
        logger.debug("Stage 1: Jury Lane (deliberation)")
        approved_atoms, rejected_atoms, quarantined_atoms = await self.jury.deliberate_batch(atoms)
        
        approved_count = len(approved_atoms)
        rejected_count = len(rejected_atoms)
        quarantined_count = len(quarantined_atoms)

        yield MemoryUpdate(
            stage="deliberation",
            atoms_approved=approved_count,
        )

        # STAGE 2: Write Lane - Persistence
        logger.debug("Stage 2: Write Lane (persistence)")
        # Each approved atom should have a jury decision in its history
        atoms_with_verdicts = [(atom, atom.jury_history[-1]) for atom in approved_atoms if atom.jury_history]
        write_stats = await self.write_lane.process_verdicts(atoms_with_verdicts)

        yield MemoryUpdate(
            stage="persistence",
            atoms_promoted=write_stats["promoted"],
            conflicts_resolved=write_stats["conflicts_resolved"],
        )

        logger.info(
            "Pipeline complete: {extracted} extracted, {approved} approved, "
            "{promoted} promoted, {conflicts} conflicts",
            extracted=len(atoms),
            approved=approved_count,
            promoted=write_stats["promoted"],
            conflicts=write_stats["conflicts_resolved"],
        )

    async def get_memory(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[MemoryAtom]:
        """
        Retrieve substantiated memory for user.
        
        Args:
            user_id: User identifier
            limit: Maximum atoms to return
            
        Returns:
            List of substantiated MemoryAtoms
        """
        atoms = await self.store.get_substantiated_atoms(subject=user_id, limit=limit)
        
        logger.debug(
            "Retrieved {count} substantiated atoms for user={user}",
            count=len(atoms),
            user=user_id,
        )
        
        return atoms

    async def get_unsubstantiated_memory(
        self,
        user_id: str,
        min_confidence: float = 0.3,
    ) -> List[MemoryAtom]:
        """
        Retrieve unsubstantiated memory (shadow buffer).
        
        Args:
            user_id: User identifier
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of unsubstantiated MemoryAtoms
        """
        atoms = await self.store.get_unsubstantiated_atoms(
            subject=user_id,
            min_confidence=min_confidence,
        )
        
        logger.debug(
            "Retrieved {count} unsubstantiated atoms for user={user}",
            count=len(atoms),
            user=user_id,
        )
        
        return atoms

    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            "messages_processed": self.messages_processed,
            "extractor": self.extractor.get_stats(),
            "jury": self.jury.get_stats(),
            "write_lane": self.write_lane.get_stats(),
            "storage": self.store.get_stats() if hasattr(self.store, 'get_stats') else {},
        }
