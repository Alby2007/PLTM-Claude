"""Manual test for conflict detection"""
import asyncio
from pathlib import Path
from src.core.models import AtomType, MemoryAtom, Provenance
from src.storage.sqlite_store import SQLiteGraphStore
from src.reconciliation.conflict_detector import ConflictDetector

async def test_opposite_predicates():
    # Setup
    db_path = Path("data/test_conflict.db")
    db_path.unlink(missing_ok=True)
    
    store = SQLiteGraphStore(db_path)
    await store.connect()
    
    detector = ConflictDetector(store)
    
    # Insert "likes jazz"
    atom1 = MemoryAtom(
        atom_type=AtomType.RELATION,
        subject="user_002",
        predicate="likes",
        object="jazz music",
        provenance=Provenance.USER_STATED,
        strength=0.8,
        confidence=0.9,
    )
    await store.insert_atom(atom1)
    print(f"✓ Inserted: {atom1.subject} {atom1.predicate} {atom1.object}")
    
    # Create "dislikes jazz" (opposite predicate)
    atom2 = MemoryAtom(
        atom_type=AtomType.RELATION,
        subject="user_002",
        predicate="dislikes",
        object="jazz music",
        provenance=Provenance.USER_STATED,
        strength=0.8,
        confidence=0.9,
    )
    
    # Test opposite predicate detection
    print(f"\nChecking for opposite predicates of: {atom2.predicate}")
    opposite_conflicts = await detector.check_opposite_predicates(atom2)
    
    print(f"\n{'='*60}")
    print(f"RESULT: Found {len(opposite_conflicts)} opposite predicate conflicts")
    print(f"{'='*60}")
    
    if opposite_conflicts:
        for conflict in opposite_conflicts:
            print(f"  ✓ Conflict: {conflict.subject} {conflict.predicate} {conflict.object}")
        print("\n✅ TEST PASSED: Opposite predicate detection works!")
    else:
        print("\n❌ TEST FAILED: No conflicts detected!")
    
    await store.close()
    db_path.unlink(missing_ok=True)

if __name__ == "__main__":
    asyncio.run(test_opposite_predicates())
