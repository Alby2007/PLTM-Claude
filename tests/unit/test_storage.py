"""Unit tests for SQLite graph store"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from src.core.models import AtomType, GraphType, MemoryAtom, Provenance
from src.storage.sqlite_store import SQLiteGraphStore


@pytest.fixture
async def store(tmp_path: Path) -> SQLiteGraphStore:
    """Create temporary database for testing"""
    db_path = tmp_path / "test_memory.db"
    store = SQLiteGraphStore(db_path)
    await store.connect()
    yield store
    await store.close()


@pytest.fixture
def sample_atom() -> MemoryAtom:
    """Create sample atom for testing"""
    return MemoryAtom(
        atom_type=AtomType.RELATION,
        subject="user_001",
        predicate="likes",
        object="jazz",
        provenance=Provenance.USER_STATED,
        strength=0.8,
        confidence=0.9,
    )


class TestSQLiteGraphStore:
    """Test SQLite graph store operations"""

    @pytest.mark.asyncio
    async def test_connect_creates_schema(self, tmp_path: Path) -> None:
        """Database connection creates schema"""
        db_path = tmp_path / "test.db"
        store = SQLiteGraphStore(db_path)
        
        await store.connect()
        
        # Verify tables exist
        assert store._conn is not None
        cursor = await store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in await cursor.fetchall()]
        
        assert "atoms" in tables
        assert "atoms_fts" in tables
        
        await store.close()

    @pytest.mark.asyncio
    async def test_insert_and_retrieve_atom(
        self, 
        store: SQLiteGraphStore, 
        sample_atom: MemoryAtom
    ) -> None:
        """Insert atom and retrieve it by ID"""
        await store.insert_atom(sample_atom)
        
        retrieved = await store.get_atom(sample_atom.id)
        
        assert retrieved is not None
        assert retrieved.id == sample_atom.id
        assert retrieved.subject == sample_atom.subject
        assert retrieved.predicate == sample_atom.predicate
        assert retrieved.object == sample_atom.object
        assert retrieved.confidence == sample_atom.confidence

    @pytest.mark.asyncio
    async def test_insert_updates_existing(
        self, 
        store: SQLiteGraphStore, 
        sample_atom: MemoryAtom
    ) -> None:
        """Inserting same atom updates it"""
        await store.insert_atom(sample_atom)
        
        # Update confidence
        sample_atom.confidence = 0.95
        await store.insert_atom(sample_atom)
        
        retrieved = await store.get_atom(sample_atom.id)
        assert retrieved is not None
        assert retrieved.confidence == 0.95

    @pytest.mark.asyncio
    async def test_find_by_triple_conflict_detection(
        self, 
        store: SQLiteGraphStore
    ) -> None:
        """
        CRITICAL CHECKPOINT: Conflict detection queries work correctly.
        
        This tests the core query used for finding conflicting atoms.
        """
        # Insert two atoms with same subject + predicate
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
            predicate="likes",
            object="classical",  # Different object, same subject+predicate
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.85,
        )
        
        await store.insert_atom(atom1)
        await store.insert_atom(atom2)
        
        # Find conflicts
        conflicts = await store.find_by_triple("user_001", "likes")
        
        assert len(conflicts) == 2
        # Should be ordered by confidence DESC
        assert conflicts[0].confidence >= conflicts[1].confidence
        assert conflicts[0].object == "jazz"  # Higher confidence
        assert conflicts[1].object == "classical"

    @pytest.mark.asyncio
    async def test_find_by_triple_excludes_historical(
        self, 
        store: SQLiteGraphStore
    ) -> None:
        """Conflict detection excludes historical atoms by default"""
        atom1 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            graph=GraphType.SUBSTANTIATED,
        )
        
        atom2 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="rock",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            graph=GraphType.HISTORICAL,  # Archived
        )
        
        await store.insert_atom(atom1)
        await store.insert_atom(atom2)
        
        # Should only find active atom
        conflicts = await store.find_by_triple("user_001", "likes")
        assert len(conflicts) == 1
        assert conflicts[0].object == "jazz"
        
        # With historical included
        all_atoms = await store.find_by_triple(
            "user_001", 
            "likes", 
            exclude_historical=False
        )
        assert len(all_atoms) == 2

    @pytest.mark.asyncio
    async def test_get_substantiated_atoms(
        self, 
        store: SQLiteGraphStore
    ) -> None:
        """Retrieve only substantiated knowledge"""
        atom1 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=1.0,
            graph=GraphType.SUBSTANTIATED,
        )
        
        atom2 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="might",
            object="like classical",
            provenance=Provenance.INFERRED,
            strength=0.3,
            graph=GraphType.UNSUBSTANTIATED,
        )
        
        await store.insert_atom(atom1)
        await store.insert_atom(atom2)
        
        substantiated = await store.get_substantiated_atoms()
        
        assert len(substantiated) == 1
        assert substantiated[0].graph == GraphType.SUBSTANTIATED
        assert substantiated[0].object == "jazz"

    @pytest.mark.asyncio
    async def test_get_substantiated_by_subject(
        self, 
        store: SQLiteGraphStore
    ) -> None:
        """Filter substantiated atoms by subject"""
        atom1 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=1.0,
            graph=GraphType.SUBSTANTIATED,
        )
        
        atom2 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_002",
            predicate="likes",
            object="rock",
            provenance=Provenance.USER_STATED,
            strength=1.0,
            graph=GraphType.SUBSTANTIATED,
        )
        
        await store.insert_atom(atom1)
        await store.insert_atom(atom2)
        
        user1_atoms = await store.get_substantiated_atoms(subject="user_001")
        
        assert len(user1_atoms) == 1
        assert user1_atoms[0].subject == "user_001"

    @pytest.mark.asyncio
    async def test_promote_atom(
        self, 
        store: SQLiteGraphStore, 
        sample_atom: MemoryAtom
    ) -> None:
        """Promote atom to substantiated graph"""
        await store.insert_atom(sample_atom)
        
        # Promote
        await store.promote_atom(sample_atom.id)
        
        # Verify promotion
        retrieved = await store.get_atom(sample_atom.id)
        assert retrieved is not None
        assert retrieved.graph == GraphType.SUBSTANTIATED
        assert retrieved.confidence == 1.0

    @pytest.mark.asyncio
    async def test_archive_atom(
        self, 
        store: SQLiteGraphStore, 
        sample_atom: MemoryAtom
    ) -> None:
        """Archive atom to historical graph"""
        await store.insert_atom(sample_atom)
        
        await store.archive_atom(sample_atom.id)
        
        retrieved = await store.get_atom(sample_atom.id)
        assert retrieved is not None
        assert retrieved.graph == GraphType.HISTORICAL

    @pytest.mark.asyncio
    async def test_link_atoms(self, store: SQLiteGraphStore) -> None:
        """Create bidirectional relationship between atoms"""
        atom1 = MemoryAtom(
            atom_type=AtomType.ENTITY,
            subject="user_001",
            predicate="is",
            object="engineer",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        atom2 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="works_at",
            object="Anthropic",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        await store.insert_atom(atom1)
        await store.insert_atom(atom2)
        
        # Link them
        await store.link_atoms(atom1.id, atom2.id)
        
        # Verify bidirectional link
        related_to_atom1 = await store.get_related_atoms(atom1.id)
        related_to_atom2 = await store.get_related_atoms(atom2.id)
        
        assert len(related_to_atom1) == 1
        assert related_to_atom1[0].id == atom2.id
        
        assert len(related_to_atom2) == 1
        assert related_to_atom2[0].id == atom1.id

    @pytest.mark.asyncio
    async def test_full_text_search(self, store: SQLiteGraphStore) -> None:
        """Full-text search on object field"""
        atom1 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz music from the 1950s",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        atom2 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="classical symphony",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        await store.insert_atom(atom1)
        await store.insert_atom(atom2)
        
        # Search for "jazz"
        results = await store.search_objects("jazz")
        
        assert len(results) == 1
        assert "jazz" in results[0].object.lower()

    @pytest.mark.asyncio
    async def test_get_stats(self, store: SQLiteGraphStore) -> None:
        """Get database statistics"""
        atom1 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=1.0,
            graph=GraphType.SUBSTANTIATED,
        )
        
        atom2 = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="might",
            object="like classical",
            provenance=Provenance.INFERRED,
            strength=0.3,
            graph=GraphType.UNSUBSTANTIATED,
        )
        
        await store.insert_atom(atom1)
        await store.insert_atom(atom2)
        
        stats = await store.get_stats()
        
        assert stats["total_atoms"] == 2
        assert stats["substantiated_count"] == 1
        assert stats["unsubstantiated_count"] == 1
        assert stats["historical_count"] == 0

    @pytest.mark.asyncio
    async def test_complex_metadata_serialization(
        self, 
        store: SQLiteGraphStore
    ) -> None:
        """Complex metadata fields serialize/deserialize correctly"""
        from src.core.models import JuryDecision, JudgeVerdict
        
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            confidence=0.9,
            explicit_confirms=[datetime.now(), datetime.now() - timedelta(hours=1)],
            contexts=["working", "evening"],
            flags=["high_confidence"],
            jury_history=[
                JuryDecision(
                    stage=1,
                    final_verdict=JudgeVerdict.APPROVE,
                    reasoning="Test decision"
                )
            ],
        )
        
        await store.insert_atom(atom)
        retrieved = await store.get_atom(atom.id)
        
        assert retrieved is not None
        assert len(retrieved.explicit_confirms) == 2
        assert len(retrieved.contexts) == 2
        assert "working" in retrieved.contexts
        assert len(retrieved.flags) == 1
        assert len(retrieved.jury_history) == 1
        assert retrieved.jury_history[0].final_verdict == JudgeVerdict.APPROVE
