"""
Synthesize coherent personality profile from accumulated traits.

This module aggregates personality atoms into a coherent profile that can be
used to adapt AI behavior.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict

from loguru import logger

from src.core.models import MemoryAtom, AtomType, GraphType
from src.storage.sqlite_store import SQLiteGraphStore


class PersonalitySynthesizer:
    """
    Synthesize coherent personality from accumulated trait atoms.
    
    Aggregates:
    - Personality traits (stable characteristics)
    - Communication styles (preferences)
    - Interaction patterns (behavioral tendencies)
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        logger.info("PersonalitySynthesizer initialized")
    
    async def synthesize_personality(self, user_id: str) -> Dict[str, Any]:
        """
        Generate personality profile from accumulated memory atoms.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with:
            - core_traits: List of stable personality traits
            - communication_style: List of style preferences
            - interaction_patterns: List of behavioral patterns
            - formality_level: "casual", "professional", or "formal"
            - humor_preference: Whether user appreciates humor
            - detail_level: "high-level" or "detailed"
        """
        # Get all personality-related atoms
        traits = await self._get_atoms_by_type(user_id, AtomType.PERSONALITY_TRAIT)
        styles = await self._get_atoms_by_type(user_id, AtomType.COMMUNICATION_STYLE)
        patterns = await self._get_atoms_by_type(user_id, AtomType.INTERACTION_PATTERN)
        
        logger.debug(
            f"Synthesizing personality from {len(traits)} traits, "
            f"{len(styles)} styles, {len(patterns)} patterns"
        )
        
        # Aggregate into coherent profile
        personality = {
            "core_traits": self._aggregate_traits(traits),
            "communication_style": self._aggregate_styles(styles),
            "interaction_patterns": self._aggregate_patterns(patterns),
            "formality_level": self._infer_formality(styles, patterns),
            "humor_preference": self._infer_humor(traits, patterns),
            "detail_level": self._infer_detail_preference(styles),
        }
        
        logger.info(f"Synthesized personality profile for {user_id}")
        return personality
    
    async def _get_atoms_by_type(
        self,
        user_id: str,
        atom_type: AtomType
    ) -> List[MemoryAtom]:
        """Get all atoms of a specific type for user"""
        # Get all atoms for this subject
        all_atoms = await self.store.get_atoms_by_subject(
            user_id,
            graph=GraphType.SUBSTANTIATED
        )
        
        # Filter by atom type
        return [atom for atom in all_atoms if atom.atom_type == atom_type]
    
    def _aggregate_traits(self, traits: List[MemoryAtom]) -> List[str]:
        """
        Aggregate personality traits by confidence and recency.
        
        Returns list of trait names that are well-supported.
        """
        if not traits:
            return []
        
        # Group by trait (object)
        trait_scores: Dict[str, List[float]] = defaultdict(list)
        
        for atom in traits:
            trait = atom.object
            # Weight by confidence and strength
            score = atom.confidence * atom.strength
            trait_scores[trait].append(score)
        
        # Average scores for each trait
        aggregated = {
            trait: sum(scores) / len(scores)
            for trait, scores in trait_scores.items()
        }
        
        # Return traits above threshold (0.5)
        return [
            trait for trait, score in aggregated.items()
            if score > 0.5
        ]
    
    def _aggregate_styles(self, styles: List[MemoryAtom]) -> List[str]:
        """Aggregate communication style preferences"""
        if not styles:
            return []
        
        style_scores: Dict[str, List[float]] = defaultdict(list)
        
        for atom in styles:
            style = atom.object
            
            # Positive preference (prefers, responds_well_to)
            if atom.predicate in ["prefers_style", "responds_well_to"]:
                score = atom.confidence * atom.strength
                style_scores[style].append(score)
            
            # Negative preference (dislikes)
            elif atom.predicate == "dislikes_style":
                # Don't include dislikes in positive style list
                continue
        
        # Average scores
        aggregated = {
            style: sum(scores) / len(scores)
            for style, scores in style_scores.items()
        }
        
        return [
            style for style, score in aggregated.items()
            if score > 0.5
        ]
    
    def _aggregate_patterns(self, patterns: List[MemoryAtom]) -> List[str]:
        """Aggregate interaction patterns"""
        if not patterns:
            return []
        
        pattern_scores: Dict[str, List[float]] = defaultdict(list)
        
        for atom in patterns:
            pattern = atom.object
            score = atom.confidence * atom.strength
            pattern_scores[pattern].append(score)
        
        aggregated = {
            pattern: sum(scores) / len(scores)
            for pattern, scores in pattern_scores.items()
        }
        
        return [
            pattern for pattern, score in aggregated.items()
            if score > 0.5
        ]
    
    def _infer_formality(
        self,
        styles: List[MemoryAtom],
        patterns: List[MemoryAtom]
    ) -> str:
        """
        Infer formality level from styles and patterns.
        
        Returns: "casual", "professional", or "formal"
        """
        formal_score = 0.0
        
        for atom in styles + patterns:
            obj_lower = atom.object.lower()
            
            # Formal indicators
            if any(word in obj_lower for word in ["formal", "professional"]):
                formal_score += atom.confidence * atom.strength
            
            # Casual indicators
            elif any(word in obj_lower for word in ["casual", "friendly", "informal"]):
                formal_score -= atom.confidence * atom.strength
        
        if formal_score > 0.3:
            return "formal"
        elif formal_score < -0.3:
            return "casual"
        else:
            return "professional"
    
    def _infer_humor(
        self,
        traits: List[MemoryAtom],
        patterns: List[MemoryAtom]
    ) -> bool:
        """Infer if user appreciates humor"""
        humor_score = 0.0
        
        for atom in traits + patterns:
            obj_lower = atom.object.lower()
            
            if "humor" in obj_lower or "humorous" in obj_lower:
                humor_score += atom.confidence * atom.strength
        
        return humor_score > 0.4
    
    def _infer_detail_preference(self, styles: List[MemoryAtom]) -> str:
        """
        Infer preference for detail level.
        
        Returns: "high-level" or "detailed"
        """
        detail_score = 0.0
        
        for atom in styles:
            obj_lower = atom.object.lower()
            
            # Detailed indicators
            if any(word in obj_lower for word in ["technical", "detailed", "depth"]):
                detail_score += atom.confidence * atom.strength
            
            # High-level indicators
            elif any(word in obj_lower for word in ["concise", "brief", "summary"]):
                detail_score -= atom.confidence * atom.strength
        
        return "detailed" if detail_score > 0.2 else "high-level"
