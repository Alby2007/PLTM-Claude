"""Unit tests for ontology validation"""

import pytest

from src.core.models import AtomType, MemoryAtom, Provenance
from src.core.ontology import get_allowed_predicates, validate_atom


class TestOntologyValidation:
    """Test ontology validation rules"""

    def test_valid_entity_atom(self) -> None:
        """Valid entity atom passes validation"""
        atom = MemoryAtom(
            atom_type=AtomType.ENTITY,
            subject="user_001",
            predicate="is",
            object="software engineer",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )

        is_valid, reason = validate_atom(atom)
        assert is_valid is True
        assert reason == "Valid"

    def test_valid_relation_atom(self) -> None:
        """Valid relation atom passes validation"""
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )

        is_valid, reason = validate_atom(atom)
        assert is_valid is True

    def test_invalid_predicate_for_type(self) -> None:
        """Invalid predicate for atom type fails validation"""
        atom = MemoryAtom(
            atom_type=AtomType.ENTITY,
            subject="user_001",
            predicate="likes",  # Not allowed for ENTITY
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )

        is_valid, reason = validate_atom(atom)
        assert is_valid is False
        assert "not allowed" in reason.lower()

    def test_empty_fields_fail(self) -> None:
        """Empty subject/predicate/object fails validation"""
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="",  # Empty
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )

        is_valid, reason = validate_atom(atom)
        assert is_valid is False
        assert "non-empty" in reason.lower()

    def test_get_allowed_predicates(self) -> None:
        """Get allowed predicates for atom type"""
        predicates = get_allowed_predicates(AtomType.RELATION)
        assert "likes" in predicates
        assert "dislikes" in predicates
        assert "works_at" in predicates

    def test_state_atom_validation(self) -> None:
        """State atoms have specific predicates"""
        atom = MemoryAtom(
            atom_type=AtomType.STATE,
            subject="user_001",
            predicate="current_mood",
            object="tired",
            provenance=Provenance.INFERRED,
            strength=0.3,
        )

        is_valid, reason = validate_atom(atom)
        assert is_valid is True

    def test_hypothesis_atom_validation(self) -> None:
        """Hypothesis atoms use tentative predicates"""
        atom = MemoryAtom(
            atom_type=AtomType.HYPOTHESIS,
            subject="user_001",
            predicate="might",
            object="prefer async communication",
            provenance=Provenance.INFERRED,
            strength=0.3,
        )

        is_valid, reason = validate_atom(atom)
        assert is_valid is True
