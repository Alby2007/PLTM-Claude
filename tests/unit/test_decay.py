"""Unit tests for Ebbinghaus decay engine"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.core.decay import DecayEngine
from src.core.models import MemoryAtom, AtomType, Provenance, GraphType


@pytest.fixture
def decay_engine():
    """Create decay engine instance"""
    return DecayEngine()


@pytest.fixture
def create_atom():
    """Factory for creating test atoms"""
    def _create(
        atom_type: AtomType = AtomType.PREFERENCE,
        confidence: float = 0.9,
        graph: GraphType = GraphType.SUBSTANTIATED,
        hours_ago: int = 0
    ) -> MemoryAtom:
        atom = MemoryAtom(
            id=f"atom_{atom_type.value}",
            subject="user",
            predicate="test",
            object="test_object",
            atom_type=atom_type,
            provenance=Provenance.USER_STATED,
            confidence=confidence,
            graph=graph,
        )
        # Set last_accessed to specified time in past
        atom.last_accessed = datetime.now() - timedelta(hours=hours_ago)
        return atom
    return _create


class TestStabilityCalculation:
    """Test Ebbinghaus stability calculations"""
    
    def test_invariant_never_decays(self, decay_engine, create_atom):
        """INVARIANT atoms should have stability = 1.0 forever"""
        atom = create_atom(atom_type=AtomType.INVARIANT, hours_ago=8760)  # 1 year
        
        stability = decay_engine.calculate_stability(atom)
        
        assert stability == 1.0
    
    def test_entity_decays_very_slowly(self, decay_engine, create_atom):
        """ENTITY atoms should decay very slowly (rate=0.01)"""
        atom = create_atom(atom_type=AtomType.ENTITY, hours_ago=720)  # 1 month
        
        stability = decay_engine.calculate_stability(atom)
        
        # After 1 month, ENTITY should still be >0.9
        assert stability > 0.9
    
    def test_state_decays_quickly(self, decay_engine, create_atom):
        """STATE atoms should decay quickly (rate=0.50)"""
        atom = create_atom(atom_type=AtomType.STATE, hours_ago=24)  # 1 day
        
        stability = decay_engine.calculate_stability(atom)
        
        # After 1 day, STATE should be <0.7
        assert stability < 0.7
    
    def test_preference_decays_medium(self, decay_engine, create_atom):
        """PREFERENCE atoms should decay at medium rate (rate=0.08)"""
        atom = create_atom(atom_type=AtomType.PREFERENCE, hours_ago=168)  # 1 week
        
        stability = decay_engine.calculate_stability(atom)
        
        # After 1 week, PREFERENCE should be in middle range
        assert 0.5 < stability < 0.9
    
    def test_higher_confidence_decays_slower(self, decay_engine, create_atom):
        """Higher confidence atoms should decay slower"""
        atom_high = create_atom(confidence=0.9, hours_ago=24)
        atom_low = create_atom(confidence=0.5, hours_ago=24)
        
        stability_high = decay_engine.calculate_stability(atom_high)
        stability_low = decay_engine.calculate_stability(atom_low)
        
        assert stability_high > stability_low
    
    def test_stability_clamped_to_zero_one(self, decay_engine, create_atom):
        """Stability should always be between 0 and 1"""
        atom = create_atom(hours_ago=100000)  # Very old
        
        stability = decay_engine.calculate_stability(atom)
        
        assert 0.0 <= stability <= 1.0
    
    def test_fresh_atom_has_high_stability(self, decay_engine, create_atom):
        """Newly accessed atom should have stability ~1.0"""
        atom = create_atom(hours_ago=0)  # Just accessed
        
        stability = decay_engine.calculate_stability(atom)
        
        assert stability > 0.99


class TestDissolution:
    """Test atom dissolution logic"""
    
    def test_substantiated_never_dissolves(self, decay_engine, create_atom):
        """Substantiated atoms should never be dissolved"""
        atom = create_atom(
            graph=GraphType.SUBSTANTIATED,
            hours_ago=10000  # Very old
        )
        
        should_dissolve = decay_engine.should_dissolve(atom)
        
        assert should_dissolve == False
    
    def test_historical_never_dissolves(self, decay_engine, create_atom):
        """Historical atoms should never be dissolved"""
        atom = create_atom(
            graph=GraphType.HISTORICAL,
            hours_ago=10000
        )
        
        should_dissolve = decay_engine.should_dissolve(atom)
        
        assert should_dissolve == False
    
    def test_unsubstantiated_low_stability_dissolves(self, decay_engine, create_atom):
        """Unsubstantiated atoms with low stability should dissolve"""
        atom = create_atom(
            atom_type=AtomType.STATE,  # Fast decay
            graph=GraphType.UNSUBSTANTIATED,
            confidence=0.5,
            hours_ago=168  # 1 week
        )
        
        should_dissolve = decay_engine.should_dissolve(atom)
        
        # STATE atom with low confidence after 1 week should dissolve
        assert should_dissolve == True
    
    def test_unsubstantiated_high_stability_persists(self, decay_engine, create_atom):
        """Unsubstantiated atoms with high stability should persist"""
        atom = create_atom(
            atom_type=AtomType.ENTITY,  # Slow decay
            graph=GraphType.UNSUBSTANTIATED,
            confidence=0.9,
            hours_ago=24  # 1 day
        )
        
        should_dissolve = decay_engine.should_dissolve(atom)
        
        # ENTITY atom with high confidence after 1 day should persist
        assert should_dissolve == False
    
    def test_custom_dissolution_threshold(self, decay_engine, create_atom):
        """Should respect custom dissolution threshold"""
        atom = create_atom(
            graph=GraphType.UNSUBSTANTIATED,
            hours_ago=100
        )
        
        # With high threshold (0.5), more atoms dissolve
        should_dissolve_high = decay_engine.should_dissolve(atom, threshold=0.5)
        
        # With low threshold (0.01), fewer atoms dissolve
        should_dissolve_low = decay_engine.should_dissolve(atom, threshold=0.01)
        
        # High threshold should be more aggressive
        assert should_dissolve_high or not should_dissolve_low


class TestReconsolidation:
    """Test memory reconsolidation mechanics"""
    
    def test_reconsolidation_increases_confidence(self, decay_engine, create_atom):
        """Reconsolidation should increase confidence"""
        atom = create_atom(confidence=0.6)
        original_confidence = atom.confidence
        
        decay_engine.reconsolidate(atom)
        
        assert atom.confidence > original_confidence
    
    def test_reconsolidation_capped_at_one(self, decay_engine, create_atom):
        """Reconsolidation should not exceed confidence = 1.0"""
        atom = create_atom(confidence=0.95)
        
        decay_engine.reconsolidate(atom, boost_factor=2.0)
        
        assert atom.confidence == 1.0
    
    def test_reconsolidation_resets_timer(self, decay_engine, create_atom):
        """Reconsolidation should reset last_accessed"""
        atom = create_atom(hours_ago=24)
        old_time = atom.last_accessed
        
        decay_engine.reconsolidate(atom)
        
        assert atom.last_accessed > old_time
        # Should be very recent (within last second)
        assert (datetime.now() - atom.last_accessed).total_seconds() < 1
    
    def test_reconsolidation_increments_access_count(self, decay_engine, create_atom):
        """Reconsolidation should track access count"""
        atom = create_atom()
        
        decay_engine.reconsolidate(atom)
        assert atom.access_count == 1
        
        decay_engine.reconsolidate(atom)
        assert atom.access_count == 2
        
        decay_engine.reconsolidate(atom)
        assert atom.access_count == 3
    
    def test_custom_boost_factor(self, decay_engine, create_atom):
        """Should respect custom boost factor"""
        atom1 = create_atom(confidence=0.5)
        atom2 = create_atom(confidence=0.5)
        
        decay_engine.reconsolidate(atom1, boost_factor=1.2)  # 20% boost
        decay_engine.reconsolidate(atom2, boost_factor=2.0)  # 100% boost
        
        assert atom2.confidence > atom1.confidence


class TestDecaySchedule:
    """Test decay schedule predictions"""
    
    def test_invariant_never_decays_schedule(self, decay_engine, create_atom):
        """INVARIANT atoms should return 'never_decays'"""
        atom = create_atom(atom_type=AtomType.INVARIANT)
        
        schedule = decay_engine.get_decay_schedule(atom)
        
        assert "never_decays" in schedule
        assert schedule["never_decays"] is None
    
    def test_schedule_has_all_thresholds(self, decay_engine, create_atom):
        """Schedule should include all threshold points"""
        atom = create_atom(atom_type=AtomType.PREFERENCE)
        
        schedule = decay_engine.get_decay_schedule(atom)
        
        expected_thresholds = ["90%", "75%", "50%", "25%", "10%"]
        for threshold in expected_thresholds:
            assert threshold in schedule
            assert isinstance(schedule[threshold], datetime)
    
    def test_schedule_chronological_order(self, decay_engine, create_atom):
        """Schedule times should be in chronological order"""
        atom = create_atom(atom_type=AtomType.PREFERENCE)
        
        schedule = decay_engine.get_decay_schedule(atom)
        
        # 90% should come before 50% which comes before 10%
        assert schedule["90%"] < schedule["50%"] < schedule["10%"]
    
    def test_time_to_dissolution(self, decay_engine, create_atom):
        """Should calculate time until dissolution"""
        atom = create_atom(
            atom_type=AtomType.STATE,
            graph=GraphType.UNSUBSTANTIATED
        )
        
        time_remaining = decay_engine.get_time_to_dissolution(atom)
        
        assert time_remaining is not None
        assert isinstance(time_remaining, timedelta)
        assert time_remaining.total_seconds() > 0
    
    def test_substantiated_no_dissolution_time(self, decay_engine, create_atom):
        """Substantiated atoms should return None for dissolution time"""
        atom = create_atom(graph=GraphType.SUBSTANTIATED)
        
        time_remaining = decay_engine.get_time_to_dissolution(atom)
        
        assert time_remaining is None


class TestBatchOperations:
    """Test batch processing"""
    
    def test_batch_calculate_stability(self, decay_engine, create_atom):
        """Should calculate stability for multiple atoms"""
        atoms = [
            create_atom(atom_type=AtomType.ENTITY, hours_ago=24),
            create_atom(atom_type=AtomType.STATE, hours_ago=24),
            create_atom(atom_type=AtomType.PREFERENCE, hours_ago=24),
        ]
        
        results = decay_engine.batch_calculate_stability(atoms)
        
        assert len(results) == 3
        for atom, stability in results:
            assert isinstance(atom, MemoryAtom)
            assert 0.0 <= stability <= 1.0
    
    def test_batch_sorted_by_stability(self, decay_engine, create_atom):
        """Batch results should be sorted by stability (lowest first)"""
        atoms = [
            create_atom(atom_type=AtomType.ENTITY, hours_ago=24),  # Slow decay
            create_atom(atom_type=AtomType.STATE, hours_ago=24),   # Fast decay
        ]
        
        results = decay_engine.batch_calculate_stability(atoms)
        
        # STATE should come first (lowest stability)
        assert results[0][0].atom_type == AtomType.STATE
        assert results[1][0].atom_type == AtomType.ENTITY
    
    def test_get_atoms_needing_reconsolidation(self, decay_engine, create_atom):
        """Should identify atoms needing reconsolidation"""
        atoms = [
            create_atom(
                atom_type=AtomType.STATE,
                graph=GraphType.SUBSTANTIATED,
                hours_ago=48  # Decayed
            ),
            create_atom(
                atom_type=AtomType.ENTITY,
                graph=GraphType.SUBSTANTIATED,
                hours_ago=1  # Fresh
            ),
        ]
        
        needing_reconsolidation = decay_engine.get_atoms_needing_reconsolidation(
            atoms, threshold=0.5
        )
        
        # STATE atom should need reconsolidation
        assert len(needing_reconsolidation) >= 1
        assert any(a.atom_type == AtomType.STATE for a in needing_reconsolidation)


class TestStatistics:
    """Test statistics tracking"""
    
    def test_stats_tracking(self, decay_engine, create_atom):
        """Should track operation statistics"""
        atom = create_atom(graph=GraphType.UNSUBSTANTIATED)
        
        # Perform operations
        decay_engine.calculate_stability(atom)
        decay_engine.calculate_stability(atom)
        decay_engine.reconsolidate(atom)
        decay_engine.should_dissolve(atom)
        
        stats = decay_engine.get_stats()
        
        assert stats["stability_checks"] == 3  # 2 explicit + 1 from should_dissolve
        assert stats["reconsolidations"] == 1
        assert stats["dissolutions"] == 0
    
    def test_reset_stats(self, decay_engine, create_atom):
        """Should reset statistics"""
        atom = create_atom()
        
        decay_engine.calculate_stability(atom)
        decay_engine.reconsolidate(atom)
        
        decay_engine.reset_stats()
        
        stats = decay_engine.get_stats()
        assert stats["stability_checks"] == 0
        assert stats["reconsolidations"] == 0
        assert stats["dissolutions"] == 0


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_zero_confidence_atom(self, decay_engine, create_atom):
        """Should handle atoms with zero confidence"""
        atom = create_atom(confidence=0.0)
        
        stability = decay_engine.calculate_stability(atom)
        
        # Should still return valid stability
        assert 0.0 <= stability <= 1.0
    
    def test_very_old_atom(self, decay_engine, create_atom):
        """Should handle very old atoms gracefully"""
        atom = create_atom(hours_ago=1000000)  # ~114 years
        
        stability = decay_engine.calculate_stability(atom)
        
        # Should be clamped to 0
        assert stability == 0.0
    
    def test_future_last_accessed(self, decay_engine, create_atom):
        """Should handle atoms with future last_accessed (clock skew)"""
        atom = create_atom()
        atom.last_accessed = datetime.now() + timedelta(hours=1)
        
        stability = decay_engine.calculate_stability(atom)
        
        # Should handle gracefully (treat as fresh)
        assert stability >= 0.99
