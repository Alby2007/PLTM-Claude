"""Extraction quality validation and metrics"""

import re
from typing import Any

from loguru import logger

from src.core.models import MemoryAtom


class ExtractionValidator:
    """
    Validate extraction quality using coverage metrics.
    
    Helps determine if extraction needs to be re-run with better model.
    """

    def __init__(
        self,
        min_token_coverage: float = 0.5,
        min_entity_coverage: float = 0.7,
    ) -> None:
        self.min_token_coverage = min_token_coverage
        self.min_entity_coverage = min_entity_coverage

    def validate(
        self,
        message: str,
        atoms: list[MemoryAtom],
    ) -> dict[str, Any]:
        """
        Check if extraction captured key information.
        
        Returns:
            {
                "token_coverage": float,
                "entity_coverage": float,
                "needs_reextraction": bool,
                "quality_score": float,
                "issues": list[str]
            }
        """
        # Token coverage: What % of message is captured?
        token_coverage = self._calculate_token_coverage(message, atoms)
        
        # Entity coverage: Did we extract key entities?
        entity_coverage = self._calculate_entity_coverage(message, atoms)
        
        # Overall quality score (weighted average)
        quality_score = (0.6 * token_coverage) + (0.4 * entity_coverage)
        
        # Determine if reextraction is needed
        needs_reextraction = (
            token_coverage < self.min_token_coverage or
            entity_coverage < self.min_entity_coverage
        )
        
        # Identify specific issues
        issues = []
        if token_coverage < self.min_token_coverage:
            issues.append(f"Low token coverage: {token_coverage:.1%}")
        if entity_coverage < self.min_entity_coverage:
            issues.append(f"Low entity coverage: {entity_coverage:.1%}")
        if len(atoms) == 0:
            issues.append("No atoms extracted")
        
        result = {
            "token_coverage": token_coverage,
            "entity_coverage": entity_coverage,
            "needs_reextraction": needs_reextraction,
            "quality_score": quality_score,
            "issues": issues,
        }
        
        if needs_reextraction:
            logger.warning(
                "Low extraction quality: {score:.1%} (issues: {issues})",
                score=quality_score,
                issues=", ".join(issues),
            )
        else:
            logger.debug(
                "Extraction quality: {score:.1%}",
                score=quality_score,
            )
        
        return result

    def _calculate_token_coverage(
        self,
        message: str,
        atoms: list[MemoryAtom],
    ) -> float:
        """Calculate what percentage of message tokens are captured"""
        message_tokens = set(self._tokenize(message))
        
        if not message_tokens:
            return 0.0
        
        captured_tokens = set()
        for atom in atoms:
            captured_tokens.update(self._tokenize(atom.object))
            # Also consider predicate as it represents semantic meaning
            captured_tokens.add(atom.predicate.lower())
        
        overlap = message_tokens & captured_tokens
        coverage = len(overlap) / len(message_tokens)
        
        return coverage

    def _calculate_entity_coverage(
        self,
        message: str,
        atoms: list[MemoryAtom],
    ) -> float:
        """Calculate what percentage of entities are captured"""
        entities_in_message = self._extract_entities(message)
        
        if not entities_in_message:
            return 1.0  # No entities to capture
        
        entities_captured = 0
        for entity in entities_in_message:
            # Check if entity appears in any atom's object
            if any(
                entity.lower() in atom.object.lower()
                for atom in atoms
            ):
                entities_captured += 1
        
        coverage = entities_captured / len(entities_in_message)
        return coverage

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization (lowercase, alphanumeric only)"""
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
            'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
            'against', 'between', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
        }
        
        return [t for t in tokens if t not in stop_words]

    def _extract_entities(self, text: str) -> list[str]:
        """
        Simple entity extraction (proper nouns, capitalized words).
        
        For MVP: Basic heuristic. Full version would use NER model.
        """
        # Find capitalized words (likely proper nouns)
        # Exclude sentence-initial words
        sentences = re.split(r'[.!?]+', text)
        entities = []
        
        for sentence in sentences:
            words = sentence.strip().split()
            # Skip first word (might be capitalized due to sentence start)
            for word in words[1:]:
                # Check if word is capitalized and alphanumeric
                if word and word[0].isupper() and word.isalpha():
                    entities.append(word)
        
        # Also catch entities at sentence start if they appear multiple times
        all_words = text.split()
        for word in all_words:
            if word and word[0].isupper() and word.isalpha():
                if text.count(word) > 1:  # Appears multiple times
                    entities.append(word)
        
        # Deduplicate
        return list(set(entities))

    def get_extraction_method_confidence(self, method: str) -> float:
        """
        Score confidence based on extraction method.
        
        Used to adjust atom confidence based on how it was extracted.
        """
        confidence_scores = {
            "rule_based": 0.95,      # High confidence - direct patterns
            "small_model": 0.70,     # Medium confidence - 3B model
            "large_model": 0.85,     # High confidence - better model
            "api_fallback": 0.85,    # High confidence - Claude API
        }
        
        return confidence_scores.get(method, 0.5)
