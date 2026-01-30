"""Unit tests for core data models"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from src.core.models import (
    AtomType,
    GraphType,
    JudgeVerdict,
    JuryDecision,
    MemoryAtom,
    Provenance,
    ReconciliationAction,
    ReconciliationDecision,
)


class TestMemoryAtom:
    """Test MemoryAtom model and promotion logic"""

    def test_atom_creation(self) -> None:
        """Test basic atom creation"""
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )

        assert atom.atom_type == AtomType.RELATION
        assert atom.subject == "user_001"
        assert atom.predicate == "likes"
        assert atom.object == "jazz"
        assert atom.graph == GraphType.UNSUBSTANTIATED  # Default
        assert atom.assertion_count == 1
        assert len(atom.explicit_confirms) == 0

    def test_instant_promotion_user_stated(self) -> None:
        """High-confidence user statement promotes instantly"""
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            confidence=0.95,  # Above INSTANT_CONFIDENCE threshold
            strength=0.8,
        )

        assert atom.promotion_eligible is True
        assert atom.promotion_tier == "INSTANT"

    def test_instant_promotion_explicit_confirm(self) -> None:
        """User confirmation triggers instant promotion"""
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.INFERRED,
            confidence=0.5,
            strength=0.3,
            explicit_confirms=[datetime.now()],
        )

        assert atom.promotion_eligible is True
        assert atom.promotion_tier == "INSTANT_CONFIRM"

    def test_no_promotion_flip_flopping(self) -> None:
        """Recent contradiction prevents instant promotion"""
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            confidence=0.95,
            strength=0.8,
            last_contradicted=datetime.now() - timedelta(minutes=30),  # Within last hour
        )

        assert atom.promotion_eligible is False

    def test_fast_track_promotion(self) -> None:
        """High confidence (>=0.8) with 4 hours observation"""
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.INFERRED,
            confidence=0.85,
            strength=0.3,
            first_observed=datetime.now() - timedelta(hours=5),  # 5 hours ago
            assertion_count=3,
        )

        assert atom.promotion_eligible is True
        assert atom.promotion_tier == "FAST"

    def test_standard_track_promotion(self) -> None:
        """Medium confidence (>=0.7) with 12 hours observation"""
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.INFERRED,
            confidence=0.75,
            strength=0.3,
            first_observed=datetime.now() - timedelta(hours=13),  # 13 hours ago
            assertion_count=3,
        )

        assert atom.promotion_eligible is True
        assert atom.promotion_tier == "STANDARD"

    def test_slow_track_requires_24_hours(self) -> None:
        """Low confidence (<0.7) requires 24 hours"""
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.INFERRED,
            confidence=0.6,
            strength=0.3,
            first_observed=datetime.now() - timedelta(hours=20),  # Only 20 hours
            assertion_count=3,
        )

        assert atom.promotion_eligible is False  # Needs 24 hours
        assert atom.promotion_tier == "SLOW"

    def test_no_promotion_without_evidence(self) -> None:
        """Atom without sufficient evidence doesn't promote"""
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.INFERRED,
            confidence=0.5,
            strength=0.3,
            assertion_count=1,  # Not enough assertions
        )

        assert atom.promotion_eligible is False

    def test_substantiated_atoms_not_eligible(self) -> None:
        """Already substantiated atoms don't re-promote"""
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            confidence=0.95,
            strength=1.0,
            graph=GraphType.SUBSTANTIATED,
        )

        assert atom.promotion_eligible is False

    def test_json_serialization(self) -> None:
        """Test JSON-safe serialization"""
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
            explicit_confirms=[datetime.now()],
        )

        json_data = atom.model_dump_json_safe()

        assert isinstance(json_data["id"], str)
        assert json_data["atom_type"] == "relation"
        assert json_data["provenance"] == "user_stated"
        assert isinstance(json_data["explicit_confirms"], list)
        assert isinstance(json_data["explicit_confirms"][0], str)  # ISO format


class TestJuryDecision:
    """Test JuryDecision model"""

    def test_jury_decision_creation(self) -> None:
        """Test basic jury decision creation"""
        decision = JuryDecision(
            stage=1,
            memory_judge=JudgeVerdict.APPROVE,
            safety_judge=JudgeVerdict.APPROVE,
            final_verdict=JudgeVerdict.APPROVE,
            confidence_adjustment=0.1,
            reasoning="Atom passes all checks",
        )

        assert decision.stage == 1
        assert decision.final_verdict == JudgeVerdict.APPROVE
        assert decision.confidence_adjustment == 0.1

    def test_safety_judge_veto(self) -> None:
        """Safety judge can override other judges"""
        decision = JuryDecision(
            stage=1,
            memory_judge=JudgeVerdict.APPROVE,
            safety_judge=JudgeVerdict.REJECT,  # Safety veto
            final_verdict=JudgeVerdict.REJECT,
            reasoning="Safety concern detected",
        )

        assert decision.final_verdict == JudgeVerdict.REJECT


class TestReconciliationDecision:
    """Test ReconciliationDecision model"""

    def test_supersede_decision(self) -> None:
        """Test supersede reconciliation"""
        decision = ReconciliationDecision(
            action=ReconciliationAction.SUPERSEDE,
            keep_existing=False,
            keep_candidate=True,
            archive_existing=True,
            explanation="User changed their mind",
        )

        assert decision.action == ReconciliationAction.SUPERSEDE
        assert decision.keep_candidate is True
        assert decision.archive_existing is True

    def test_contextualize_decision(self) -> None:
        """Test contextualize reconciliation"""
        decision = ReconciliationDecision(
            action=ReconciliationAction.CONTEXTUALIZE,
            keep_existing=True,
            keep_candidate=True,
            archive_existing=False,
            explanation="Both true in different contexts",
            contexts_added=["working", "leisure"],
        )

        assert decision.action == ReconciliationAction.CONTEXTUALIZE
        assert len(decision.contexts_added) == 2
