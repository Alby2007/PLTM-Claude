"""Unit tests for conflict detection and reconciliation"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.core.models import (
    AtomType,
    GraphType,
    MemoryAtom,
    Provenance,
    ReconciliationAction,
)
from src.reconciliation.conflict_detector import ConflictDetector
from src.reconciliation.resolver import ReconciliationResolver
from src.storage.sqlite_store import SQLiteGraphStore


@pytest.fixture
async def store(tmp_path: Path) -> SQLiteGraphStore:
    """Create temporary database for testing"""
    db_path = tmp_path / "test_reconciliation.db"
    store = SQLiteGraphStore(db_path)
    await store.connect()
    yield store
    await store.close()


class TestConflictDetector:
    """Test three-stage conflict detection"""

    @pytest.mark.asyncio
    async def test_no_conflict_different_subjects(self, store: SQLiteGraphStore) -> None:
        """No conflict when subjects differ"""
        detector = ConflictDetector(store)
        
        existing = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        candidate = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_002",  # Different subject
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        await store.insert_atom(existing)
        
        conflicts = await detector.find_conflicts(candidate)
        
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_no_conflict_different_predicates(self, store: SQLiteGraphStore) -> None:
        """No conflict when predicates differ"""
        detector = ConflictDetector(store)
        
        existing = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        candidate = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="uses",  # Different predicate
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        await store.insert_atom(existing)
        
        conflicts = await detector.find_conflicts(candidate)
        
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_detect_different_objects(self, store: SQLiteGraphStore) -> None:
        """Detect conflict when objects differ"""
        detector = ConflictDetector(store)
        
        existing = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        candidate = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="rock",  # Different object
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        await store.insert_atom(existing)
        
        conflicts = await detector.find_conflicts(candidate)
        
        assert len(conflicts) == 1
        assert conflicts[0].object == "jazz"

    @pytest.mark.asyncio
    async def test_detect_opposite_predicates(self, store: SQLiteGraphStore) -> None:
        """Detect conflict with opposite predicates"""
        detector = ConflictDetector(store)
        
        existing = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz music",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        await store.insert_atom(existing)
        
        candidate = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="dislikes",  # Opposite
            object="jazz music",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        # Check for opposite predicate conflicts
        opposite_conflicts = await detector.check_opposite_predicates(candidate)
        
        assert len(opposite_conflicts) == 1
        assert opposite_conflicts[0].predicate == "likes"

    @pytest.mark.asyncio
    async def test_detect_opposite_sentiments(self, store: SQLiteGraphStore) -> None:
        """Detect conflict with opposite sentiments in objects"""
        detector = ConflictDetector(store)
        
        existing = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="feels",
            object="good about the project",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        candidate = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="feels",
            object="bad about the project",  # Opposite sentiment
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        await store.insert_atom(existing)
        
        conflicts = await detector.find_conflicts(candidate)
        
        assert len(conflicts) == 1

    @pytest.mark.asyncio
    async def test_no_conflict_refinement(self, store: SQLiteGraphStore) -> None:
        """No conflict when one object refines the other"""
        detector = ConflictDetector(store)
        
        existing = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="music",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        candidate = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz music",  # Refinement of "music"
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        await store.insert_atom(existing)
        
        conflicts = await detector.find_conflicts(candidate)
        
        # Should not detect as conflict (refinement)
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_similarity_threshold(self, store: SQLiteGraphStore) -> None:
        """Similarity threshold filters fuzzy matches"""
        detector = ConflictDetector(store)
        
        existing = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz music from the 1950s",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        candidate = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="completely different thing",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        await store.insert_atom(existing)
        
        # High threshold should not match
        conflicts = await detector.find_conflicts(candidate, similarity_threshold=0.9)
        assert len(conflicts) == 0
        
        # Low threshold might match
        conflicts = await detector.find_conflicts(candidate, similarity_threshold=0.1)
        # May or may not match depending on similarity


class TestReconciliationResolver:
    """Test reconciliation decision logic"""

    def test_user_stated_supersedes_inferred(self) -> None:
        """User statement supersedes inference"""
        resolver = ReconciliationResolver()
        
        existing = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.INFERRED,
            strength=0.3,
            confidence=0.5,
        )
        
        candidate = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="rock",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.9,
        )
        
        decision = resolver.reconcile(candidate, existing)
        
        assert decision.action == ReconciliationAction.SUPERSEDE
        assert decision.keep_candidate is True
        assert decision.keep_existing is False
        assert decision.archive_existing is True

    def test_recent_user_statement_supersedes_old(self) -> None:
        """More recent user statement supersedes older"""
        resolver = ReconciliationResolver()
        
        existing = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.9,
            first_observed=datetime.now() - timedelta(days=7),
        )
        
        candidate = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="rock",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.9,
            first_observed=datetime.now(),
        )
        
        decision = resolver.reconcile(candidate, existing)
        
        assert decision.action == ReconciliationAction.SUPERSEDE
        assert "recent" in decision.explanation.lower()

    def test_corrected_supersedes_everything(self) -> None:
        """Corrected information supersedes all"""
        resolver = ReconciliationResolver()
        
        existing = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.9,
        )
        
        candidate = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="rock",
            provenance=Provenance.CORRECTED,
            strength=0.85,
            confidence=0.95,
        )
        
        decision = resolver.reconcile(candidate, existing)
        
        assert decision.action == ReconciliationAction.SUPERSEDE
        assert "corrected" in decision.explanation.lower()

    def test_higher_confidence_wins(self) -> None:
        """Significantly higher confidence wins"""
        resolver = ReconciliationResolver()
        
        existing = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.INFERRED,
            strength=0.3,
            confidence=0.5,
        )
        
        candidate = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="rock",
            provenance=Provenance.INFERRED,
            strength=0.3,
            confidence=0.9,  # Much higher
        )
        
        decision = resolver.reconcile(candidate, existing)
        
        assert decision.action == ReconciliationAction.SUPERSEDE
        assert "confidence" in decision.explanation.lower()

    def test_contextualize_different_contexts(self) -> None:
        """Atoms with different contexts can coexist"""
        resolver = ReconciliationResolver()
        
        existing = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.9,
            contexts=["leisure", "evening"],
        )
        
        candidate = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="rock",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.9,
            contexts=["working", "morning"],
        )
        
        decision = resolver.reconcile(candidate, existing)
        
        assert decision.action == ReconciliationAction.CONTEXTUALIZE
        assert decision.keep_candidate is True
        assert decision.keep_existing is True

    def test_reject_insufficient_evidence(self) -> None:
        """Reject when insufficient evidence to override"""
        resolver = ReconciliationResolver()
        
        existing = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.9,
        )
        
        candidate = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="rock",
            provenance=Provenance.INFERRED,  # Weaker provenance
            strength=0.3,
            confidence=0.5,  # Lower confidence
        )
        
        decision = resolver.reconcile(candidate, existing)
        
        assert decision.action == ReconciliationAction.REJECT
        assert decision.keep_candidate is False
        assert decision.keep_existing is True

    def test_batch_reconciliation(self) -> None:
        """Batch reconciliation handles multiple conflicts"""
        resolver = ReconciliationResolver()
        
        candidate = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="rock",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.9,
        )
        
        conflicts = [
            MemoryAtom(
                atom_type=AtomType.RELATION,
                subject="user_001",
                predicate="likes",
                object="jazz",
                provenance=Provenance.INFERRED,
                strength=0.3,
                confidence=0.5,
            ),
            MemoryAtom(
                atom_type=AtomType.RELATION,
                subject="user_001",
                predicate="likes",
                object="classical",
                provenance=Provenance.INFERRED,
                strength=0.3,
                confidence=0.5,
            ),
        ]
        
        decisions = resolver.reconcile_batch(candidate, conflicts)
        
        assert len(decisions) == 2
        # Both should be superseded (user stated > inferred)
        assert all(d.action == ReconciliationAction.SUPERSEDE for d in decisions)

    def test_get_stats(self) -> None:
        """Statistics are tracked"""
        resolver = ReconciliationResolver()
        
        existing = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.9,
        )
        
        candidate = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="rock",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.9,
        )
        
        resolver.reconcile(candidate, existing)
        
        stats = resolver.get_stats()
        assert stats["reconciliations"] == 1
