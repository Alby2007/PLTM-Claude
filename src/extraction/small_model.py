"""Small model extraction using Qwen2.5-3B with grammar constraints"""

from typing import List

from loguru import logger

from src.core.config import settings
from src.core.models import AtomType, MemoryAtom, Provenance


class SmallModelExtractor:
    """
    Qwen2.5-3B with grammar-constrained output for complex extraction.
    
    For MVP: Simplified version without Outlines (will add later).
    Uses direct model inference with structured prompting.
    """

    def __init__(self) -> None:
        self.extraction_count = 0
        logger.info("SmallModelExtractor initialized (simplified for MVP)")
        logger.warning("Full grammar-constrained extraction requires Outlines - using fallback")

    def extract(self, message: str, user_id: str, session_id: str = "") -> List[MemoryAtom]:
        """
        Extract semantic triples using small model.
        
        For MVP: Returns empty list (placeholder for future implementation).
        Full implementation requires Outlines + Qwen2.5-3B.
        
        Args:
            message: User's message text
            user_id: Canonical user identifier
            session_id: Optional session identifier
            
        Returns:
            List of extracted MemoryAtoms
        """
        # TODO: Implement with Outlines when dependencies are installed
        logger.debug("Small model extraction not available (requires Outlines + transformers)")
        return []

    def _build_prompt(self, message: str, user_id: str) -> str:
        """Construct extraction prompt with examples"""
        return f"""Extract factual information from this message as semantic triples.

Message: "{message}"

Guidelines:
- Extract relationships, preferences, beliefs, and facts
- Subject is usually "user" (the person speaking)
- Use clear predicates: likes, dislikes, is, has, works_at, prefers, believes
- For uncertain statements (might, possibly, seems), set confidence < 0.7
- For definite statements (I am, I love, I work), set confidence >= 0.9

Examples:
"I love jazz" → {{"subject": "user", "predicate": "likes", "object": "jazz", "atom_type": "relation", "confidence": 0.95}}
"user_123 might prefer async communication" → {{"subject": "user_123", "predicate": "prefers", "object": "async communication", "atom_type": "hypothesis", "confidence": 0.6}}
"I am a software engineer" → {{"subject": "user", "predicate": "is", "object": "software engineer", "atom_type": "entity", "confidence": 0.95}}

Output JSON only:"""

    def _resolve_subject(self, subject: str, user_id: str) -> str:
        """Map 'user' to actual user_id"""
        if subject.lower() in ["user", "i", "me"]:
            return user_id
        return subject

    def _infer_provenance(self, message: str, confidence: float) -> Provenance:
        """Determine provenance based on message structure and confidence"""
        # First person = user stated
        first_person_indicators = ["i ", "i'm", "i've", "my ", "i am"]
        if any(indicator in message.lower() for indicator in first_person_indicators):
            return Provenance.USER_STATED

        # Uncertain language = inference
        uncertain_indicators = ["might", "possibly", "seems", "appears", "maybe"]
        if any(indicator in message.lower() for indicator in uncertain_indicators):
            return Provenance.INFERRED

        # Low confidence = inference
        if confidence < 0.7:
            return Provenance.INFERRED

        return Provenance.USER_STATED

    def _initial_strength(self, confidence: float) -> float:
        """Map confidence to initial strength"""
        if confidence >= 0.9:
            return 0.8  # USER_STATED
        elif confidence >= 0.7:
            return 0.75  # USER_CONFIRMED
        else:
            return 0.3  # INFERRED

    def get_stats(self) -> dict:
        """Get extraction statistics"""
        return {
            "total_extractions": self.extraction_count,
            "model": settings.SMALL_MODEL,
            "enabled": False,  # Not enabled in MVP
        }
