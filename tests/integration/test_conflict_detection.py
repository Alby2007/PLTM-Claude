"""
Integration tests for conflict detection.

CRITICAL CHECKPOINT: These tests validate the core conflict detection
functionality that the entire system depends on.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.core.models import AtomType, GraphType, MemoryAtom, Provenance
from src.storage.sqlite_store import SQLiteGraphStore


@pytest.fixture
async def store(tmp_path: Path) -> SQLiteGraphStore:
    """Create temporary database for testing"""
    db_path = tmp_path / "test_conflicts.db"
    store = SQLiteGraphStore(db_path)
    await store.connect()
    yield store
    await store.close()


class TestConflictDetection:
    """
    CHECKPOINT: Conflict detection must work correctly before proceeding.
    
    These tests validate that we can:
    1. Find atoms with same subject+predicate (potential conflicts)
    2. Order results by confidence (for reconciliation)
    3. Exclude historical atoms (don't conflict with archived data)
    4. Handle edge cases (no conflicts, multiple conflicts)
    """

    @pytest.mark.asyncio
    async def test_detect_simple_conflict(self, store: SQLiteGraphStore) -> None:
        """Detect conflict: user likes jazz vs user likes rock"""
        atom1 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.9,
            first_observed=datetime.now() - timedelta(days=1),
        )
        
        atom2 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="rock",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.85,
            first_observed=datetime.now(),
        )
        
        await store.insert_atom(atom1)
        await store.insert_atom(atom2)
        
        # Find potential conflicts
        conflicts = await store.find_by_triple("user_001", "likes")
        
        assert len(conflicts) == 2
        # Higher confidence first
        assert conflicts[0].confidence == 0.9
        assert conflicts[0].object == "jazz"
        assert conflicts[1].confidence == 0.85
        assert conflicts[1].object == "rock"

    @pytest.mark.asyncio
    async def test_detect_opposite_predicates(self, store: SQLiteGraphStore) -> None:
        """Detect conflict: user likes jazz vs user dislikes jazz"""
        atom1 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.9,
        )
        
        atom2 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="dislikes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.95,
        )
        
        await store.insert_atom(atom1)
        await store.insert_atom(atom2)
        
        # These have different predicates, so won't be found by same query
        likes_atoms = await store.find_by_triple("user_001", "likes")
        dislikes_atoms = await store.find_by_triple("user_001", "dislikes")
        
        assert len(likes_atoms) == 1
        assert len(dislikes_atoms) == 1
        
        # Semantic conflict detection will need to check both
        # This is handled in the reconciliation layer

    @pytest.mark.asyncio
    async def test_no_conflict_different_subjects(
        self, 
        store: SQLiteGraphStore
    ) -> None:
        """No conflict when subjects differ"""
        atom1 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        atom2 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_002",  # Different user
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        await store.insert_atom(atom1)
        await store.insert_atom(atom2)
        
        # Each user has their own preference
        user1_likes = await store.find_by_triple("user_001", "likes")
        user2_likes = await store.find_by_triple("user_002", "likes")
        
        assert len(user1_likes) == 1
        assert len(user2_likes) == 1
        assert user1_likes[0].subject != user2_likes[0].subject

    @pytest.mark.asyncio
    async def test_conflict_with_archived_atom(
        self, 
        store: SQLiteGraphStore
    ) -> None:
        """Archived atoms don't conflict with active atoms"""
        # Old preference (archived)
        old_atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="rock",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            graph=GraphType.HISTORICAL,
        )
        
        # New preference (active)
        new_atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            graph=GraphType.SUBSTANTIATED,
        )
        
        await store.insert_atom(old_atom)
        await store.insert_atom(new_atom)
        
        # Should only find active atom
        active_conflicts = await store.find_by_triple("user_001", "likes")
        assert len(active_conflicts) == 1
        assert active_conflicts[0].object == "jazz"
        assert active_conflicts[0].graph == GraphType.SUBSTANTIATED

    @pytest.mark.asyncio
    async def test_multiple_conflicts_ordered_by_confidence(
        self, 
        store: SQLiteGraphStore
    ) -> None:
        """Multiple conflicts ordered by confidence for reconciliation"""
        atoms = [
            MemoryAtom(
                atom_type=AtomType.RELATION,
                subject="user_001",
                predicate="prefers",
                object=f"option_{i}",
                provenance=Provenance.INFERRED,
                strength=0.3,
                confidence=0.5 + (i * 0.1),  # Increasing confidence
            )
            for i in range(5)
        ]
        
        for atom in atoms:
            await store.insert_atom(atom)
        
        conflicts = await store.find_by_triple("user_001", "prefers")
        
        assert len(conflicts) == 5
        # Should be in descending confidence order
        for i in range(len(conflicts) - 1):
            assert conflicts[i].confidence >= conflicts[i + 1].confidence

    @pytest.mark.asyncio
    async def test_conflict_detection_performance(
        self, 
        store: SQLiteGraphStore
    ) -> None:
        """Conflict detection is fast even with many atoms"""
        import time
        
        # Insert 100 atoms for same user
        for i in range(100):
            atom = MemoryAtom(
                atom_type=AtomType.RELATION,
                subject="user_001",
                predicate="likes" if i % 2 == 0 else "dislikes",
                object=f"item_{i}",
                provenance=Provenance.INFERRED,
                strength=0.3,
                confidence=0.5,
            )
            await store.insert_atom(atom)
        
        # Time the conflict detection query
        start = time.time()
        conflicts = await store.find_by_triple("user_001", "likes")
        elapsed = time.time() - start
        
        assert len(conflicts) == 50  # Half are "likes"
        assert elapsed < 0.1  # Should be very fast (<100ms)

    @pytest.mark.asyncio
    async def test_supersession_chain(self, store: SQLiteGraphStore) -> None:
        """Track supersession relationships"""
        # Original atom
        atom1 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="rock",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            graph=GraphType.HISTORICAL,
        )
        
        # Superseding atom
        atom2 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            graph=GraphType.SUBSTANTIATED,
            supersedes=atom1.id,
        )
        
        atom1.superseded_by = atom2.id
        
        await store.insert_atom(atom1)
        await store.insert_atom(atom2)
        
        # Verify supersession chain
        current = await store.get_atom(atom2.id)
        old = await store.get_atom(atom1.id)
        
        assert current is not None
        assert old is not None
        assert current.supersedes == old.id
        assert old.superseded_by == current.id
