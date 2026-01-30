"""Hybrid extraction orchestrator with progressive fallback"""

from typing import List, Optional

from loguru import logger

from src.core.models import MemoryAtom
from src.extraction.rule_based import RuleBasedExtractor
from src.extraction.small_model import SmallModelExtractor
from src.extraction.validator import ExtractionValidator


class HybridExtractor:
    """
    Multi-stage extraction: Rules → Small Model → Large Model (fallback).
    
    For MVP: Rule-based + small model (placeholder).
    API fallback is optional enhancement.
    """

    def __init__(self, enable_small_model: bool = False) -> None:
        self.rule_extractor = RuleBasedExtractor()
        self.validator = ExtractionValidator()
        
        # Small model extractor (optional)
        self.small_model_extractor = SmallModelExtractor() if enable_small_model else None
        self.api_fallback_enabled = False
        
        if enable_small_model:
            logger.info("HybridExtractor initialized (rule-based + small model)")
        else:
            logger.info("HybridExtractor initialized (rule-based only)")

    async def extract_atoms(
        self,
        message: str,
        user_id: str,
        session_id: str = "",
        context: Optional[str] = None,
    ) -> List[MemoryAtom]:
        """
        Extract atoms with progressive fallback.
        
        Progressive extraction: Rules → Small Model → API (optional)
        
        Args:
            message: User's message text
            user_id: Canonical user identifier
            session_id: Optional session identifier
            context: Optional conversation context
            
        Returns:
            List of extracted MemoryAtoms
        """
        atoms: List[MemoryAtom] = []
        extraction_method = "rule_based"
        
        # STAGE 1: Rule-based extraction (instant)
        logger.debug("Stage 1: Rule-based extraction")
        rule_atoms = self.rule_extractor.extract(message, user_id, session_id)
        atoms.extend(rule_atoms)
        
        # Validate quality
        validation = self.validator.validate(message, atoms)
        
        # STAGE 2: Small model fallback if needed
        if validation["needs_reextraction"] and self.small_model_extractor:
            logger.debug("Stage 2: Small model extraction (quality too low)")
            model_atoms = self.small_model_extractor.extract(message, user_id, session_id)
            atoms.extend(model_atoms)
            extraction_method = "hybrid"
            
            # Re-validate with combined results
            validation = self.validator.validate(message, atoms)
        elif self._needs_small_model(message) and self.small_model_extractor:
            logger.debug("Stage 2: Small model extraction (complex message)")
            model_atoms = self.small_model_extractor.extract(message, user_id, session_id)
            atoms.extend(model_atoms)
            extraction_method = "hybrid"
        
        # Deduplicate atoms (keep highest confidence)
        atoms = self._deduplicate(atoms)
        
        logger.info(
            "Extraction complete: {count} atoms, quality: {quality:.1%}, method: {method}",
            count=len(atoms),
            quality=validation["quality_score"],
            method=extraction_method,
        )
        
        # Adjust confidence based on extraction method
        method_confidence = self.validator.get_extraction_method_confidence(extraction_method)
        for atom in atoms:
            # Blend atom's base confidence with method confidence
            atom.confidence = (atom.confidence + method_confidence) / 2
        
        return atoms

    def _needs_small_model(self, message: str) -> bool:
        """
        Detect if message needs model extraction.
        
        Signals:
        - Uncertain/inferential language (might, possibly, seems)
        - Long messages (>20 words)
        - Multiple clauses (2+ commas)
        """
        complexity_signals = [
            any(
                word in message.lower()
                for word in [
                    "might",
                    "possibly",
                    "seems",
                    "appears",
                    "could",
                    "would",
                    "prefer",
                    "believe",
                    "think",
                ]
            ),  # Uncertain/inferential language
            len(message.split()) > 20,  # Long message
            message.count(",") >= 2,  # Multiple clauses
        ]
        return any(complexity_signals)

    def _deduplicate(self, atoms: List[MemoryAtom]) -> List[MemoryAtom]:
        """
        Remove duplicate triples, keep highest confidence.
        
        Args:
            atoms: List of potentially duplicate atoms
            
        Returns:
            Deduplicated list
        """
        seen = {}
        for atom in atoms:
            key = (atom.subject, atom.predicate, atom.object.lower())
            if key not in seen or atom.confidence > seen[key].confidence:
                seen[key] = atom
        return list(seen.values())

    def _is_complex(self, message: str) -> bool:
        """
        Detect if message needs large model.
        
        Complexity signals:
        - Long message (>30 words)
        - Multiple clauses (3+ commas)
        - Complex connectors (especially, however, etc.)
        - Parenthetical asides
        - Semicolons
        """
        complexity_signals = [
            len(message.split()) > 30,
            message.count(',') >= 3,
            any(
                word in message.lower()
                for word in [
                    'especially', 'however', 'although', 'whereas',
                    'intersection', 'contrast', 'nuance', 'specifically',
                    'particularly', 'moreover', 'furthermore', 'nevertheless'
                ]
            ),
            message.count('(') >= 1,
            message.count(';') >= 1,
        ]
        
        complexity_score = sum(complexity_signals)
        is_complex = complexity_score >= 2  # Require 2 signals for complexity
        
        if is_complex:
            logger.debug(
                "Complex message detected (score: {score}/5)",
                score=complexity_score,
            )
        
        return is_complex

    def get_stats(self) -> dict[str, any]:
        """Get extraction statistics"""
        stats = {
            "rule_based": self.rule_extractor.get_stats(),
            "small_model_enabled": self.small_model_extractor is not None,
            "api_fallback_enabled": self.api_fallback_enabled,
        }
        return stats
