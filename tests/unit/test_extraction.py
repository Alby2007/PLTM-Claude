"""Unit tests for extraction pipeline"""

import pytest

from src.core.models import AtomType, Provenance
from src.extraction.rule_based import RuleBasedExtractor
from src.extraction.validator import ExtractionValidator
from src.extraction.hybrid import HybridExtractor


class TestRuleBasedExtractor:
    """Test rule-based extraction patterns"""

    def test_extract_identity(self) -> None:
        """Extract 'I am X' pattern"""
        extractor = RuleBasedExtractor()
        
        atoms = extractor.extract("I am a software engineer", "user_001")
        
        assert len(atoms) == 1
        assert atoms[0].predicate == "is"
        assert atoms[0].object == "software engineer"
        assert atoms[0].atom_type == AtomType.ENTITY
        assert atoms[0].provenance == Provenance.USER_STATED

    def test_extract_name(self) -> None:
        """Extract 'my name is X' pattern"""
        extractor = RuleBasedExtractor()
        
        atoms = extractor.extract("My name is Alice Johnson", "user_001")
        
        assert len(atoms) == 1
        assert atoms[0].predicate == "named"
        assert atoms[0].object == "Alice Johnson"

    def test_extract_work_location(self) -> None:
        """Extract 'I work at X' pattern"""
        extractor = RuleBasedExtractor()
        
        atoms = extractor.extract("I work at Anthropic", "user_001")
        
        assert len(atoms) == 1
        assert atoms[0].predicate == "works_at"
        assert atoms[0].object == "Anthropic"
        assert atoms[0].atom_type == AtomType.RELATION

    def test_extract_likes(self) -> None:
        """Extract preference patterns"""
        extractor = RuleBasedExtractor()
        
        atoms = extractor.extract("I love jazz music", "user_001")
        
        assert len(atoms) == 1
        assert atoms[0].predicate == "likes"
        assert atoms[0].object == "jazz music"

    def test_extract_dislikes(self) -> None:
        """Extract negative preference patterns"""
        extractor = RuleBasedExtractor()
        
        atoms = extractor.extract("I hate waiting in lines", "user_001")
        
        assert len(atoms) == 1
        assert atoms[0].predicate == "dislikes"
        assert atoms[0].object == "waiting in lines"

    def test_extract_skills(self) -> None:
        """Extract skill/tool usage patterns"""
        extractor = RuleBasedExtractor()
        
        atoms = extractor.extract("I use Python and JavaScript", "user_001")
        
        # Should extract "Python and JavaScript" as one object
        assert len(atoms) >= 1
        assert atoms[0].predicate == "uses"

    def test_extract_location(self) -> None:
        """Extract location patterns"""
        extractor = RuleBasedExtractor()
        
        atoms = extractor.extract("I live in San Francisco", "user_001")
        
        assert len(atoms) == 1
        assert atoms[0].predicate == "located_at"
        assert atoms[0].object == "San Francisco"

    def test_extract_multiple_atoms(self) -> None:
        """Extract multiple atoms from one message"""
        extractor = RuleBasedExtractor()
        
        message = "I am a software engineer. I work at Anthropic and I love Python."
        atoms = extractor.extract(message, "user_001")
        
        # Should extract: identity, work location, and preference
        assert len(atoms) >= 3
        
        predicates = {atom.predicate for atom in atoms}
        assert "is" in predicates
        assert "works_at" in predicates
        assert "likes" in predicates

    def test_high_confidence_for_patterns(self) -> None:
        """Rule-based extraction has high confidence"""
        extractor = RuleBasedExtractor()
        
        atoms = extractor.extract("I am a developer", "user_001")
        
        assert len(atoms) == 1
        assert atoms[0].confidence >= 0.9  # High confidence

    def test_user_stated_provenance(self) -> None:
        """First-person statements get USER_STATED provenance"""
        extractor = RuleBasedExtractor()
        
        atoms = extractor.extract("I like jazz", "user_001")
        
        assert len(atoms) == 1
        assert atoms[0].provenance == Provenance.USER_STATED

    def test_skip_short_objects(self) -> None:
        """Skip extractions with very short objects"""
        extractor = RuleBasedExtractor()
        
        # "I am a" would extract "a" which is too short
        atoms = extractor.extract("I am a", "user_001")
        
        # Should skip due to length check
        assert len(atoms) == 0

    def test_coverage_estimate(self) -> None:
        """Estimate coverage of rule-based extraction"""
        extractor = RuleBasedExtractor()
        
        message = "I am a software engineer and I love Python"
        coverage = extractor.get_coverage_estimate(message)
        
        # Should have decent coverage for this simple message
        assert coverage > 0.3

    def test_session_id_preserved(self) -> None:
        """Session ID is preserved in extracted atoms"""
        extractor = RuleBasedExtractor()
        
        atoms = extractor.extract("I like jazz", "user_001", session_id="session_123")
        
        assert len(atoms) == 1
        assert atoms[0].session_id == "session_123"


class TestExtractionValidator:
    """Test extraction quality validation"""

    def test_high_quality_extraction(self) -> None:
        """High quality extraction passes validation"""
        from src.core.models import MemoryAtom
        
        validator = ExtractionValidator()
        
        message = "I am a software engineer"
        atoms = [
            MemoryAtom(
                atom_type=AtomType.ENTITY,
                subject="user_001",
                predicate="is",
                object="software engineer",
                provenance=Provenance.USER_STATED,
                strength=0.8,
            )
        ]
        
        result = validator.validate(message, atoms)
        
        assert result["token_coverage"] > 0.3
        assert result["needs_reextraction"] is False
        assert result["quality_score"] > 0.5

    def test_low_quality_extraction(self) -> None:
        """Low quality extraction fails validation"""
        from src.core.models import MemoryAtom
        
        validator = ExtractionValidator()
        
        # Complex message but simple extraction
        message = "I'm passionate about machine learning, especially the intersection of RL and multi-agent systems"
        atoms = [
            MemoryAtom(
                atom_type=AtomType.RELATION,
                subject="user_001",
                predicate="likes",
                object="ML",  # Lost all nuance
                provenance=Provenance.INFERRED,
                strength=0.3,
            )
        ]
        
        result = validator.validate(message, atoms)
        
        assert result["needs_reextraction"] is True
        assert len(result["issues"]) > 0

    def test_empty_extraction(self) -> None:
        """Empty extraction is flagged"""
        validator = ExtractionValidator()
        
        result = validator.validate("Some message", [])
        
        assert result["needs_reextraction"] is True
        assert "No atoms extracted" in result["issues"]

    def test_entity_coverage(self) -> None:
        """Entity coverage is calculated correctly"""
        from src.core.models import MemoryAtom
        
        validator = ExtractionValidator()
        
        message = "I work at Anthropic in San Francisco"
        atoms = [
            MemoryAtom(
                atom_type=AtomType.RELATION,
                subject="user_001",
                predicate="works_at",
                object="Anthropic",
                provenance=Provenance.USER_STATED,
                strength=0.8,
            ),
            MemoryAtom(
                atom_type=AtomType.ENTITY,
                subject="user_001",
                predicate="located_at",
                object="San Francisco",
                provenance=Provenance.USER_STATED,
                strength=0.8,
            ),
        ]
        
        result = validator.validate(message, atoms)
        
        # Should have good entity coverage (Anthropic, San Francisco)
        assert result["entity_coverage"] > 0.5

    def test_extraction_method_confidence(self) -> None:
        """Different extraction methods have different confidence scores"""
        validator = ExtractionValidator()
        
        assert validator.get_extraction_method_confidence("rule_based") == 0.95
        assert validator.get_extraction_method_confidence("small_model") == 0.70
        assert validator.get_extraction_method_confidence("api_fallback") == 0.85


class TestHybridExtractor:
    """Test hybrid extraction orchestrator"""

    @pytest.mark.asyncio
    async def test_basic_extraction(self) -> None:
        """Basic extraction works"""
        extractor = HybridExtractor()
        
        atoms = await extractor.extract_atoms("I am a developer", "user_001")
        
        assert len(atoms) >= 1
        assert atoms[0].predicate == "is"

    @pytest.mark.asyncio
    async def test_confidence_adjustment(self) -> None:
        """Confidence is adjusted based on extraction method"""
        extractor = HybridExtractor()
        
        atoms = await extractor.extract_atoms("I like jazz", "user_001")
        
        assert len(atoms) >= 1
        # Confidence should be blended with method confidence
        assert 0.5 <= atoms[0].confidence <= 1.0

    @pytest.mark.asyncio
    async def test_session_id_propagation(self) -> None:
        """Session ID is propagated to atoms"""
        extractor = HybridExtractor()
        
        atoms = await extractor.extract_atoms(
            "I like jazz",
            "user_001",
            session_id="session_123"
        )
        
        assert len(atoms) >= 1
        assert atoms[0].session_id == "session_123"

    def test_complexity_detection(self) -> None:
        """Complex messages are detected"""
        extractor = HybridExtractor()
        
        simple = "I like jazz"
        complex_msg = (
            "I'm passionate about machine learning, especially the intersection "
            "of reinforcement learning and multi-agent systems, and I find "
            "the sample efficiency challenges, the exploration strategies, "
            "and the credit assignment problems particularly frustrating."
        )
        
        assert extractor._is_complex(simple) is False
        assert extractor._is_complex(complex_msg) is True

    @pytest.mark.asyncio
    async def test_multiple_atoms_extracted(self) -> None:
        """Multiple atoms can be extracted from one message"""
        extractor = HybridExtractor()
        
        message = "I am Alice, I work at Anthropic, and I love Python programming."
        atoms = await extractor.extract_atoms(message, "user_001")
        
        # Should extract multiple facts
        assert len(atoms) >= 2

    def test_get_stats(self) -> None:
        """Statistics are available"""
        extractor = HybridExtractor()
        
        stats = extractor.get_stats()
        
        assert "rule_based" in stats
        assert "small_model_enabled" in stats
        assert stats["small_model_enabled"] is False  # MVP only has rules
