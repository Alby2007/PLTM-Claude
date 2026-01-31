"""
Test suite for multi-hop reasoning capabilities.

Tests the InferenceEngine's ability to detect conflicts that require
chaining multiple facts together.
"""

import pytest
from src.reconciliation.inference_engine import InferenceEngine, ConflictChain
from src.reconciliation.conflict_detector import ConflictDetector
from src.storage.sqlite_store import SQLiteGraphStore
from src.core.models import MemoryAtom, AtomType, GraphType, Provenance
from src.core.ontology import Ontology
import uuid


def create_test_atom(subject, predicate, obj, atom_type=AtomType.STATE):
    """Helper to create test atoms with all required fields"""
    return MemoryAtom(
        id=str(uuid.uuid4()),
        subject=subject,
        predicate=predicate,
        object=obj,
        atom_type=atom_type,
        graph=GraphType.SUBSTANTIATED,
        confidence=0.9,
        strength=0.9,
        provenance=Provenance.USER_STATED,
        contexts=[]
    )


@pytest.fixture
async def setup():
    """Setup test environment"""
    store = SQLiteGraphStore(":memory:")
    await store.connect()
    ontology = Ontology()
    engine = InferenceEngine(store, ontology)
    detector = ConflictDetector(store, ontology, enable_multihop=True)
    
    yield store, engine, detector
    
    await store.close()


@pytest.mark.asyncio
async def test_dietary_conflict_vegetarian_eats_meat(setup):
    """Test: Vegetarian eating meat should be detected as conflict"""
    store, engine, detector = setup
    
    # Add: Alice is vegetarian
    atom1 = create_test_atom("alice", "is", "vegetarian", AtomType.STATE)
    await store.insert_atom(atom1)
    
    # Try to add: Alice eats steak
    atom2 = create_test_atom("alice", "eats", "steak", AtomType.EVENT)
    
    # Check for conflicts
    conflicts = await engine.find_transitive_conflicts(atom2, max_hops=2)
    
    assert len(conflicts) > 0, "Should detect vegetarian eating meat conflict"
    assert conflicts[0].conflict_type == "2hop_dietary"
    assert "vegetarian" in conflicts[0].explanation.lower()


@pytest.mark.asyncio
async def test_dietary_conflict_vegan_eats_dairy(setup):
    """Test: Vegan eating dairy should be detected"""
    store, engine, detector = setup
    
    # Add: Bob is vegan
    atom1 = create_test_atom("bob", "is", "vegan", AtomType.STATE)
    await store.insert_atom(atom1)
    
    # Try to add: Bob eats cheese
    atom2 = create_test_atom("bob", "eats", "cheese", AtomType.EVENT)
    
    conflicts = await engine.find_transitive_conflicts(atom2, max_hops=2)
    
    assert len(conflicts) > 0, "Should detect vegan eating dairy conflict"


@pytest.mark.asyncio
async def test_allergy_conflict(setup):
    """Test: Eating food you're allergic to should be detected"""
    store, engine, detector = setup
    
    # Add: Charlie is allergic to peanuts
    atom1 = create_test_atom("charlie", "allergic_to", "peanuts", AtomType.STATE)
    await store.insert_atom(atom1)
    
    # Try to add: Charlie eats peanuts
    atom2 = create_test_atom("charlie", "eats", "peanuts", AtomType.EVENT)
    
    conflicts = await engine.find_transitive_conflicts(atom2, max_hops=2)
    
    assert len(conflicts) > 0, "Should detect allergy conflict"


@pytest.mark.asyncio
async def test_no_conflict_vegetarian_eats_vegetables(setup):
    """Test: Vegetarian eating vegetables should NOT conflict"""
    store, engine, detector = setup
    
    # Add: Alice is vegetarian
    atom1 = create_test_atom("alice", "is", "vegetarian", AtomType.STATE)
    await store.insert_atom(atom1)
    
    # Try to add: Alice eats salad
    atom2 = create_test_atom("alice", "eats", "salad", AtomType.EVENT)
    
    conflicts = await engine.find_transitive_conflicts(atom2, max_hops=2)
    
    assert len(conflicts) == 0, "Vegetarian eating vegetables should not conflict"


@pytest.mark.asyncio
async def test_integrated_multihop_detection(setup):
    """Test: ConflictDetector with multihop enabled should catch 2-hop conflicts"""
    store, engine, detector = setup
    
    # Add: Dave is vegetarian
    atom1 = create_test_atom("dave", "is", "vegetarian", AtomType.STATE)
    await store.insert_atom(atom1)
    
    # Try to add: Dave eats chicken
    atom2 = create_test_atom("dave", "eats", "chicken", AtomType.EVENT)
    
    # Use integrated conflict detector
    conflicts = await detector.find_conflicts(atom2)
    
    assert len(conflicts) > 0, "Integrated detector should catch multi-hop conflicts"


@pytest.mark.asyncio
async def test_preference_conflict_loves_vs_hates(setup):
    """Test: Loving and hating same thing should conflict"""
    store, engine, detector = setup
    
    # Add: Eve loves Python
    atom1 = create_test_atom("eve", "loves", "Python", AtomType.PREFERENCE)
    await store.insert_atom(atom1)
    
    # Try to add: Eve hates Python
    atom2 = create_test_atom("eve", "hates", "Python", AtomType.PREFERENCE)
    
    conflicts = await engine.find_transitive_conflicts(atom2, max_hops=2)
    
    assert len(conflicts) > 0, "Should detect love/hate conflict"
    assert conflicts[0].conflict_type == "2hop_preference"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
