"""Integration tests for full LTM pipeline"""

import pytest
import asyncio
from datetime import datetime
from typing import List

from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
from src.extraction.rule_based import RuleBasedExtractor
from src.jury.orchestrator import JuryOrchestrator
from src.jury.safety_judge import SafetyJudge
from src.jury.memory_judge import MemoryJudge
from src.jury.time_judge import TimeJudge
from src.jury.consensus_judge import ConsensusJudge
from src.storage.sqlite_store import SQLiteGraphStore
from src.reconciliation.conflict_detector import ConflictDetector
from src.pipeline.write_lane import WriteLane
from src.core.decay import DecayEngine
from src.core.retrieval import MemoryRetriever


@pytest.fixture
async def pipeline_components():
    """Initialize all pipeline components"""
    store = SQLiteGraphStore(":memory:")
    extractor = RuleBasedExtractor()
    
    # Initialize judges
    safety_judge = SafetyJudge()
    memory_judge = MemoryJudge(store)
    time_judge = TimeJudge()
    consensus_judge = ConsensusJudge()
    
    # Initialize orchestrator
    jury = JuryOrchestrator(
        safety_judge=safety_judge,
        memory_judge=memory_judge,
        time_judge=time_judge,
        consensus_judge=consensus_judge
    )
    
    # Initialize conflict detector
    conflict_detector = ConflictDetector(store)
    
    # Initialize decay components
    decay_engine = DecayEngine()
    retriever = MemoryRetriever(store, decay_engine)

    # Write lane (promotion + persistence)
    write_lane = WriteLane(store)

    yield {
        "store": store,
        "extractor": extractor,
        "jury": jury,
        "conflict_detector": conflict_detector,
        "decay_engine": decay_engine,
        "retriever": retriever,
        "write_lane": write_lane,
    }

    await store.close()


class TestFullPipeline:
    """Test complete pipeline from message to storage"""
    
    @pytest.mark.asyncio
    async def test_simple_extraction_and_storage(self, pipeline_components):
        """Test basic message processing"""
        store = pipeline_components["store"]
        extractor = pipeline_components["extractor"]
        jury = pipeline_components["jury"]
        
        # Extract atoms
        message = "I love Python programming"
        atoms = extractor.extract(message, "user_1")
        
        assert len(atoms) > 0
        assert atoms[0].subject == "user_1"
        assert atoms[0].predicate == "likes"
        assert "Python" in atoms[0].object
        
        # Jury deliberation
        approved, rejected, quarantined = await jury.deliberate_batch(atoms)
        
        assert len(approved) > 0
        
        # Store atoms
        for atom in approved:
            await store.insert_atom(atom)
        
        # Verify storage
        stored_atoms = await store.get_all_atoms(subject="user_1")
        assert len(stored_atoms) == len(approved)
    
    @pytest.mark.asyncio
    async def test_conflict_detection_pipeline(self, pipeline_components):
        """Test conflict detection in pipeline"""
        store = pipeline_components["store"]
        extractor = pipeline_components["extractor"]
        conflict_detector = pipeline_components["conflict_detector"]
        
        # First message
        atoms1 = extractor.extract("I like Python", "user_1")
        for atom in atoms1:
            await store.insert_atom(atom)
        
        # Conflicting message
        atoms2 = extractor.extract("I dislike Python", "user_1")
        
        # Detect conflicts
        for atom in atoms2:
            conflicts = await conflict_detector.find_conflicts(atom)
            assert len(conflicts) > 0  # Should detect opposite predicate
    
    @pytest.mark.asyncio
    async def test_decay_integration(self, pipeline_components):
        """Test decay mechanics integration"""
        store = pipeline_components["store"]
        decay_engine = pipeline_components["decay_engine"]
        retriever = pipeline_components["retriever"]
        
        # Create atom
        atom = MemoryAtom(
            id="test_atom_1",
            subject="user_1",
            predicate="likes",
            object="Python",
            atom_type=AtomType.PREFERENCE,
            provenance=Provenance.USER_STATED,
            confidence=0.9,
            graph=GraphType.SUBSTANTIATED
        )
        
        await store.insert_atom(atom)
        
        # Calculate stability
        stability = decay_engine.calculate_stability(atom)
        assert 0.0 <= stability <= 1.0
        
        # Reconsolidate
        decay_engine.reconsolidate(atom)
        assert atom.confidence > 0.9  # Should increase
        
        # Get with reconsolidation
        atoms = await retriever.get_atoms("user_1", reconsolidate=True)
        assert len(atoms) > 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, pipeline_components):
        """Test complete end-to-end workflow"""
        store = pipeline_components["store"]
        extractor = pipeline_components["extractor"]
        jury = pipeline_components["jury"]
        conflict_detector = pipeline_components["conflict_detector"]
        retriever = pipeline_components["retriever"]
        
        # Step 1: Extract atoms from message
        message = "I work at Anthropic and I love Python programming"
        atoms = extractor.extract(message, "user_1")
        
        assert len(atoms) >= 2  # Should extract multiple atoms
        
        # Step 2: Jury deliberation
        approved, rejected, quarantined = await jury.deliberate_batch(atoms)
        
        assert len(approved) > 0
        assert len(rejected) == 0  # No safety issues
        
        # Step 3: Conflict detection
        for atom in approved:
            conflicts = await conflict_detector.find_conflicts(atom)
            # First time, no conflicts expected
            assert len(conflicts) == 0
        
        # Step 4: Persist via write lane (handles promotion to substantiated)
        write_lane = pipeline_components["write_lane"]
        atoms_with_verdicts = [
            (atom, atom.jury_history[-1]) for atom in approved if atom.jury_history
        ]
        await write_lane.process_verdicts(atoms_with_verdicts)

        # Step 5: Retrieve and verify
        stored_atoms = await retriever.get_atoms("user_1", reconsolidate=True)
        
        assert len(stored_atoms) == len(approved)
        
        # Step 6: Process conflicting message
        conflict_message = "I dislike Python programming"
        conflict_atoms = extractor.extract(conflict_message, "user_1")
        
        # Should detect conflict
        for atom in conflict_atoms:
            conflicts = await conflict_detector.find_conflicts(atom)
            assert len(conflicts) > 0
    
    @pytest.mark.asyncio
    async def test_multi_user_isolation(self, pipeline_components):
        """Test that users' data is isolated"""
        store = pipeline_components["store"]
        extractor = pipeline_components["extractor"]
        
        # User 1
        atoms1 = extractor.extract("I like Python", "user_1")
        for atom in atoms1:
            await store.insert_atom(atom)
        
        # User 2
        atoms2 = extractor.extract("I like JavaScript", "user_2")
        for atom in atoms2:
            await store.insert_atom(atom)
        
        # Verify isolation
        user1_atoms = await store.get_all_atoms(subject="user_1")
        user2_atoms = await store.get_all_atoms(subject="user_2")
        
        assert len(user1_atoms) > 0
        assert len(user2_atoms) > 0
        assert all(a.subject == "user_1" for a in user1_atoms)
        assert all(a.subject == "user_2" for a in user2_atoms)
    
    @pytest.mark.asyncio
    async def test_graph_transitions(self, pipeline_components):
        """Test atom transitions between graphs"""
        store = pipeline_components["store"]
        
        # Create unsubstantiated atom
        atom = MemoryAtom(
            id="test_atom_2",
            subject="user_1",
            predicate="likes",
            object="Python",
            atom_type=AtomType.PREFERENCE,
            provenance=Provenance.INFERRED,
            confidence=0.7,
            graph=GraphType.UNSUBSTANTIATED
        )
        
        await store.insert_atom(atom)
        
        # Verify in unsubstantiated graph
        unsub_atoms = await store.get_atoms_by_graph(
            GraphType.UNSUBSTANTIATED,
            subject="user_1"
        )
        assert len(unsub_atoms) == 1
        
        # Promote to substantiated
        atom.graph = GraphType.SUBSTANTIATED
        atom.confidence = 0.9
        await store.update_atom(atom)
        
        # Verify transition
        sub_atoms = await store.get_substantiated_atoms(subject="user_1")
        assert len(sub_atoms) == 1
        assert sub_atoms[0].graph == GraphType.SUBSTANTIATED
    
    @pytest.mark.asyncio
    async def test_performance_batch_operations(self, pipeline_components):
        """Test performance with batch operations"""
        store = pipeline_components["store"]
        extractor = pipeline_components["extractor"]
        
        # Create multiple messages
        messages = [
            f"I like technology_{i}" for i in range(100)
        ]
        
        start_time = datetime.utcnow()
        
        # Process all messages
        for message in messages:
            atoms = extractor.extract(message, "user_1")
            for atom in atoms:
                await store.insert_atom(atom)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Should process 100 messages in reasonable time
        assert duration < 5.0  # Less than 5 seconds
        
        # Verify all stored
        stored_atoms = await store.get_all_atoms(subject="user_1")
        assert len(stored_atoms) >= 100


class TestPipelineEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_empty_message(self, pipeline_components):
        """Test handling of empty message"""
        extractor = pipeline_components["extractor"]
        
        atoms = extractor.extract("", "user_1")
        assert len(atoms) == 0
    
    @pytest.mark.asyncio
    async def test_invalid_user_id(self, pipeline_components):
        """Test handling of invalid user ID"""
        store = pipeline_components["store"]
        
        atoms = await store.get_all_atoms(subject="nonexistent_user")
        assert len(atoms) == 0
    
    @pytest.mark.asyncio
    async def test_duplicate_atoms(self, pipeline_components):
        """Test handling of duplicate atoms"""
        store = pipeline_components["store"]
        
        atom1 = MemoryAtom(
            id="test_atom_3",
            subject="user_1",
            predicate="likes",
            object="Python",
            atom_type=AtomType.PREFERENCE,
            provenance=Provenance.USER_STATED,
            confidence=0.9,
            graph=GraphType.SUBSTANTIATED
        )
        
        atom2 = MemoryAtom(
            id="test_atom_4",
            subject="user_1",
            predicate="likes",
            object="Python",
            atom_type=AtomType.PREFERENCE,
            provenance=Provenance.USER_STATED,
            confidence=0.9,
            graph=GraphType.SUBSTANTIATED
        )
        
        await store.insert_atom(atom1)
        await store.insert_atom(atom2)

        # Identical (subject, predicate, object, graph) triples are re-assertions:
        # the UNIQUE constraint + INSERT OR REPLACE consolidates them to one row.
        atoms = await store.get_all_atoms(subject="user_1")
        assert len(atoms) == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, pipeline_components):
        """Test concurrent operations on same user"""
        store = pipeline_components["store"]
        extractor = pipeline_components["extractor"]
        
        async def process_message(message: str, user_id: str):
            atoms = extractor.extract(message, user_id)
            for atom in atoms:
                await store.insert_atom(atom)
        
        # Process multiple messages concurrently
        messages = [
            "I like Python",
            "I work at Anthropic",
            "I enjoy programming"
        ]
        
        await asyncio.gather(*[
            process_message(msg, "user_1") for msg in messages
        ])
        
        # Verify all stored
        atoms = await store.get_all_atoms(subject="user_1")
        assert len(atoms) >= 3


class TestPipelineMetrics:
    """Test metrics collection during pipeline execution"""
    
    @pytest.mark.asyncio
    async def test_extraction_metrics(self, pipeline_components):
        """Test extraction metrics are collected"""
        extractor = pipeline_components["extractor"]
        
        initial_count = extractor.extractions
        
        atoms = extractor.extract("I like Python", "user_1")
        
        assert extractor.extractions > initial_count
        assert len(atoms) > 0
    
    @pytest.mark.asyncio
    async def test_decay_metrics(self, pipeline_components):
        """Test decay metrics are collected"""
        decay_engine = pipeline_components["decay_engine"]
        
        atom = MemoryAtom(
            id="test_atom_5",
            subject="user_1",
            predicate="likes",
            object="Python",
            atom_type=AtomType.PREFERENCE,
            provenance=Provenance.USER_STATED,
            confidence=0.9,
            graph=GraphType.SUBSTANTIATED
        )
        
        initial_checks = decay_engine.stability_checks
        
        decay_engine.calculate_stability(atom)
        
        assert decay_engine.stability_checks > initial_checks
    
    @pytest.mark.asyncio
    async def test_end_to_end_metrics(self, pipeline_components):
        """Test all metrics are collected in full pipeline"""
        store = pipeline_components["store"]
        extractor = pipeline_components["extractor"]
        jury = pipeline_components["jury"]
        decay_engine = pipeline_components["decay_engine"]
        
        # Process message
        atoms = extractor.extract("I love Python programming", "user_1")
        approved, rejected, quarantined = await jury.deliberate_batch(atoms)
        
        for atom in approved:
            await store.insert_atom(atom)
            decay_engine.calculate_stability(atom)
        
        # Verify metrics collected
        assert extractor.extractions > 0
        assert decay_engine.stability_checks > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
