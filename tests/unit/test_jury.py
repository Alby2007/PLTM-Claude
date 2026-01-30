"""Unit tests for jury system"""

import pytest

from src.core.models import AtomType, JudgeVerdict, MemoryAtom, Provenance
from src.jury.memory_judge import MemoryJudge
from src.jury.orchestrator import JuryOrchestrator
from src.jury.safety_judge import SafetyJudge


class TestSafetyJudge:
    """Test safety judge evaluations"""

    def test_approve_safe_content(self) -> None:
        """Safe content is approved"""
        judge = SafetyJudge()
        
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz music",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        result = judge.evaluate(atom)
        
        assert result["verdict"] == JudgeVerdict.APPROVE
        assert result["severity"] == 0
        assert result["confidence"] == 1.0

    def test_reject_ssn(self) -> None:
        """SSN is rejected as PII"""
        judge = SafetyJudge()
        
        atom = MemoryAtom(
            atom_type=AtomType.ENTITY,
            subject="user_001",
            predicate="has",
            object="My SSN is 123-45-6789",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        result = judge.evaluate(atom)
        
        assert result["verdict"] == JudgeVerdict.REJECT
        assert "SSN" in result["reason"]
        assert result["severity"] >= 8

    def test_reject_credit_card(self) -> None:
        """Credit card number is rejected"""
        judge = SafetyJudge()
        
        atom = MemoryAtom(
            atom_type=AtomType.ENTITY,
            subject="user_001",
            predicate="has",
            object="Card: 4532-1234-5678-9010",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        result = judge.evaluate(atom)
        
        assert result["verdict"] == JudgeVerdict.REJECT
        assert "Credit Card" in result["reason"]

    def test_reject_email(self) -> None:
        """Email address is rejected as PII"""
        judge = SafetyJudge()
        
        atom = MemoryAtom(
            atom_type=AtomType.ENTITY,
            subject="user_001",
            predicate="has",
            object="Contact me at user@example.com",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        result = judge.evaluate(atom)
        
        assert result["verdict"] == JudgeVerdict.REJECT
        assert "Email" in result["reason"]

    def test_reject_password(self) -> None:
        """Password is rejected"""
        judge = SafetyJudge()
        
        atom = MemoryAtom(
            atom_type=AtomType.ENTITY,
            subject="user_001",
            predicate="has",
            object="password: secret123",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        result = judge.evaluate(atom)
        
        assert result["verdict"] == JudgeVerdict.REJECT
        assert "Password" in result["reason"]

    def test_reject_harmful_content(self) -> None:
        """Harmful keywords are rejected"""
        judge = SafetyJudge()
        
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="wants",
            object="to hack the system",
            provenance=Provenance.INFERRED,
            strength=0.3,
        )
        
        result = judge.evaluate(atom)
        
        assert result["verdict"] == JudgeVerdict.REJECT
        assert "Harmful content" in result["reason"]

    def test_quarantine_long_content(self) -> None:
        """Excessively long content is quarantined"""
        judge = SafetyJudge()
        
        long_text = "x" * 600  # Over 500 chars
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object=long_text,
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        result = judge.evaluate(atom)
        
        assert result["verdict"] == JudgeVerdict.QUARANTINE
        assert "long content" in result["reason"].lower()


class TestMemoryJudge:
    """Test memory judge evaluations"""

    def test_approve_valid_atom(self) -> None:
        """Valid atom is approved"""
        judge = MemoryJudge()
        
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz music",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        result = judge.evaluate(atom)
        
        assert result["verdict"] == JudgeVerdict.APPROVE
        assert result["confidence"] > 0.5

    def test_reject_invalid_predicate(self) -> None:
        """Invalid predicate for atom type is rejected"""
        judge = MemoryJudge()
        
        atom = MemoryAtom(
            atom_type=AtomType.ENTITY,
            subject="user_001",
            predicate="likes",  # Not allowed for ENTITY
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        result = judge.evaluate(atom)
        
        assert result["verdict"] == JudgeVerdict.REJECT
        assert "not allowed" in result["reason"].lower()

    def test_reject_empty_object(self) -> None:
        """Empty object is rejected"""
        judge = MemoryJudge()
        
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="",  # Empty
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        result = judge.evaluate(atom)
        
        assert result["verdict"] == JudgeVerdict.REJECT
        assert "empty" in result["reason"].lower()

    def test_reject_too_short_object(self) -> None:
        """Very short object is rejected"""
        judge = MemoryJudge()
        
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="a",  # Too short
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        result = judge.evaluate(atom)
        
        assert result["verdict"] == JudgeVerdict.REJECT

    def test_quarantine_repeated_words(self) -> None:
        """Repeated words are quarantined"""
        judge = MemoryJudge()
        
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz jazz jazz",  # Repeated
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        result = judge.evaluate(atom)
        
        assert result["verdict"] == JudgeVerdict.QUARANTINE
        assert "repeated" in result["reason"].lower()

    def test_quarantine_generic_object(self) -> None:
        """Generic objects are quarantined"""
        judge = MemoryJudge()
        
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="something",  # Too generic
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        result = judge.evaluate(atom)
        
        assert result["verdict"] == JudgeVerdict.QUARANTINE
        assert "generic" in result["reason"].lower()

    def test_quarantine_contradictory_sentiment(self) -> None:
        """Contradictory sentiment is quarantined"""
        judge = MemoryJudge()
        
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="to hate things",  # Contradiction
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        result = judge.evaluate(atom)
        
        assert result["verdict"] == JudgeVerdict.QUARANTINE
        assert "contradictory" in result["reason"].lower()


class TestJuryOrchestrator:
    """Test jury orchestrator"""

    @pytest.mark.asyncio
    async def test_approve_safe_valid_atom(self) -> None:
        """Safe and valid atom is approved"""
        jury = JuryOrchestrator()
        
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz music",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        decision = await jury.deliberate(atom)
        
        assert decision.final_verdict == JudgeVerdict.APPROVE
        assert decision.safety_judge == JudgeVerdict.APPROVE
        assert decision.memory_judge == JudgeVerdict.APPROVE
        assert decision.confidence_adjustment > 0

    @pytest.mark.asyncio
    async def test_safety_veto(self) -> None:
        """Safety judge can veto even if memory approves"""
        jury = JuryOrchestrator()
        
        # Valid ontology but contains PII
        atom = MemoryAtom(
            atom_type=AtomType.ENTITY,
            subject="user_001",
            predicate="has",
            object="SSN 123-45-6789",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        decision = await jury.deliberate(atom)
        
        assert decision.final_verdict == JudgeVerdict.REJECT
        assert decision.safety_judge == JudgeVerdict.REJECT
        assert decision.confidence_adjustment < 0

    @pytest.mark.asyncio
    async def test_memory_reject(self) -> None:
        """Memory judge can reject if safety passes"""
        jury = JuryOrchestrator()
        
        # Safe but invalid ontology
        atom = MemoryAtom(
            atom_type=AtomType.ENTITY,
            subject="user_001",
            predicate="likes",  # Not allowed for ENTITY
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        decision = await jury.deliberate(atom)
        
        assert decision.final_verdict == JudgeVerdict.REJECT
        assert decision.safety_judge == JudgeVerdict.APPROVE
        assert decision.memory_judge == JudgeVerdict.REJECT

    @pytest.mark.asyncio
    async def test_batch_deliberation(self) -> None:
        """Batch deliberation processes multiple atoms"""
        jury = JuryOrchestrator()
        
        atoms = [
            MemoryAtom(
                atom_type=AtomType.RELATION,
                subject="user_001",
                predicate="likes",
                object="jazz",
                provenance=Provenance.USER_STATED,
                strength=0.8,
            ),
            MemoryAtom(
                atom_type=AtomType.RELATION,
                subject="user_001",
                predicate="dislikes",
                object="rock",
                provenance=Provenance.USER_STATED,
                strength=0.8,
            ),
            MemoryAtom(
                atom_type=AtomType.ENTITY,
                subject="user_001",
                predicate="likes",  # Invalid
                object="classical",
                provenance=Provenance.USER_STATED,
                strength=0.8,
            ),
        ]
        
        results = await jury.deliberate_batch(atoms)
        
        assert len(results) == 3
        
        # First two should be approved
        assert results[0][1].final_verdict == JudgeVerdict.APPROVE
        assert results[1][1].final_verdict == JudgeVerdict.APPROVE
        
        # Third should be rejected (invalid ontology)
        assert results[2][1].final_verdict == JudgeVerdict.REJECT

    @pytest.mark.asyncio
    async def test_jury_history_recorded(self) -> None:
        """Jury decision is recorded in atom history"""
        jury = JuryOrchestrator()
        
        atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        decision = await jury.deliberate(atom)
        
        # Decision should have reasoning
        assert len(decision.reasoning) > 0
        assert "Safety" in decision.reasoning
        assert "Memory" in decision.reasoning

    @pytest.mark.asyncio
    async def test_confidence_adjustments(self) -> None:
        """Different verdicts have different confidence adjustments"""
        jury = JuryOrchestrator()
        
        # Approved atom
        approved_atom = MemoryAtom(
            atom_type=AtomType.RELATION,
            subject="user_001",
            predicate="likes",
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        # Rejected atom
        rejected_atom = MemoryAtom(
            atom_type=AtomType.ENTITY,
            subject="user_001",
            predicate="likes",  # Invalid
            object="jazz",
            provenance=Provenance.USER_STATED,
            strength=0.8,
        )
        
        approved_decision = await jury.deliberate(approved_atom)
        rejected_decision = await jury.deliberate(rejected_atom)
        
        # Approved should have positive adjustment
        assert approved_decision.confidence_adjustment > 0
        
        # Rejected should have negative adjustment
        assert rejected_decision.confidence_adjustment < 0

    def test_get_stats(self) -> None:
        """Statistics are available"""
        jury = JuryOrchestrator()
        
        stats = jury.get_stats()
        
        assert "deliberations" in stats
        assert "safety_judge" in stats
        assert "memory_judge" in stats
