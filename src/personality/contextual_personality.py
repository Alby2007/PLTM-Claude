"""
Context-aware personality tracking.

People behave differently in different contexts. This module tracks personality
traits specific to different interaction contexts (technical, casual, formal, etc.).
"""

from typing import Dict, Any, List, Optional

from loguru import logger

from src.core.models import MemoryAtom, AtomType, GraphType
from src.storage.sqlite_store import SQLiteGraphStore
from src.personality.personality_synthesizer import PersonalitySynthesizer


class ContextualPersonality:
    """
    Track personality by interaction context.
    
    Same user might be:
    - Technical and direct in code reviews
    - Casual and humorous in general chat
    - Formal in professional correspondence
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self.base_synthesizer = PersonalitySynthesizer(store)
        logger.info("ContextualPersonality initialized")
    
    async def get_personality_for_context(
        self,
        user_id: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Get personality profile specific to a context.
        
        Args:
            user_id: User identifier
            context: Context name (e.g., "technical", "casual", "formal")
            
        Returns:
            Personality profile adapted to the context
        """
        # Get all personality atoms
        all_atoms = await self.store.get_atoms_by_subject(
            user_id,
            graph=GraphType.SUBSTANTIATED
        )
        
        # Filter to personality atoms in this context
        context_atoms = [
            atom for atom in all_atoms
            if atom.atom_type in [
                AtomType.PERSONALITY_TRAIT,
                AtomType.COMMUNICATION_STYLE,
                AtomType.INTERACTION_PATTERN
            ] and context in atom.contexts
        ]
        
        # If no context-specific atoms, fall back to general personality
        if not context_atoms:
            logger.debug(f"No context-specific traits for '{context}', using general profile")
            return await self.base_synthesizer.synthesize_personality(user_id)
        
        # Synthesize context-specific profile
        traits = [a for a in context_atoms if a.atom_type == AtomType.PERSONALITY_TRAIT]
        styles = [a for a in context_atoms if a.atom_type == AtomType.COMMUNICATION_STYLE]
        patterns = [a for a in context_atoms if a.atom_type == AtomType.INTERACTION_PATTERN]
        
        profile = {
            "context": context,
            "core_traits": self.base_synthesizer._aggregate_traits(traits),
            "communication_style": self.base_synthesizer._aggregate_styles(styles),
            "interaction_patterns": self.base_synthesizer._aggregate_patterns(patterns),
            "formality_level": self.base_synthesizer._infer_formality(styles, patterns),
            "humor_preference": self.base_synthesizer._infer_humor(traits, patterns),
            "detail_level": self.base_synthesizer._infer_detail_preference(styles),
        }
        
        logger.info(f"Generated context-specific profile for '{context}'")
        return profile
    
    async def get_all_contexts(self, user_id: str) -> List[str]:
        """
        Get all contexts where user has personality data.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of context names
        """
        all_atoms = await self.store.get_atoms_by_subject(
            user_id,
            graph=GraphType.SUBSTANTIATED
        )
        
        # Extract unique contexts from personality atoms
        contexts = set()
        for atom in all_atoms:
            if atom.atom_type in [
                AtomType.PERSONALITY_TRAIT,
                AtomType.COMMUNICATION_STYLE,
                AtomType.INTERACTION_PATTERN
            ]:
                contexts.update(atom.contexts)
        
        # Filter out non-context tags
        context_list = [
            c for c in contexts
            if not c.startswith("detected_at:") and
               not c.startswith("indicators:") and
               c not in ["extracted_from_message_style", "positive_feedback", "negative_feedback"]
        ]
        
        return sorted(context_list)
    
    async def compare_contexts(
        self,
        user_id: str,
        context_a: str,
        context_b: str
    ) -> Dict[str, Any]:
        """
        Compare personality across two contexts.
        
        Args:
            user_id: User identifier
            context_a: First context
            context_b: Second context
            
        Returns:
            Comparison showing differences between contexts
        """
        profile_a = await self.get_personality_for_context(user_id, context_a)
        profile_b = await self.get_personality_for_context(user_id, context_b)
        
        comparison = {
            "context_a": context_a,
            "context_b": context_b,
            "differences": {
                "formality": {
                    context_a: profile_a["formality_level"],
                    context_b: profile_b["formality_level"],
                    "different": profile_a["formality_level"] != profile_b["formality_level"]
                },
                "humor": {
                    context_a: profile_a["humor_preference"],
                    context_b: profile_b["humor_preference"],
                    "different": profile_a["humor_preference"] != profile_b["humor_preference"]
                },
                "detail_level": {
                    context_a: profile_a["detail_level"],
                    context_b: profile_b["detail_level"],
                    "different": profile_a["detail_level"] != profile_b["detail_level"]
                }
            },
            "unique_to_a": [
                trait for trait in profile_a["core_traits"]
                if trait not in profile_b["core_traits"]
            ],
            "unique_to_b": [
                trait for trait in profile_b["core_traits"]
                if trait not in profile_a["core_traits"]
            ],
            "shared": [
                trait for trait in profile_a["core_traits"]
                if trait in profile_b["core_traits"]
            ]
        }
        
        logger.info(f"Compared contexts '{context_a}' vs '{context_b}'")
        return comparison
    
    def infer_context_from_message(self, message: str) -> str:
        """
        Infer interaction context from message content.
        
        Args:
            message: User's message
            
        Returns:
            Inferred context name
        """
        msg_lower = message.lower()
        
        # Technical context
        technical_indicators = [
            "algorithm", "implementation", "code", "function", "class",
            "api", "database", "query", "performance", "optimization",
            "bug", "error", "debug", "test", "deploy"
        ]
        if any(ind in msg_lower for ind in technical_indicators):
            return "technical"
        
        # Formal/professional context
        formal_indicators = [
            "please", "kindly", "would you", "could you",
            "appreciate", "regards", "sincerely", "respectfully"
        ]
        if any(ind in msg_lower for ind in formal_indicators):
            return "formal"
        
        # Casual context
        casual_indicators = [
            "hey", "yo", "lol", "haha", "btw", "gonna", "wanna",
            "yeah", "nah", "cool", "awesome"
        ]
        if any(ind in msg_lower for ind in casual_indicators):
            return "casual"
        
        # Support/help context
        help_indicators = [
            "help", "stuck", "confused", "don't understand",
            "how do i", "what is", "explain"
        ]
        if any(ind in msg_lower for ind in help_indicators):
            return "support"
        
        # Default to general
        return "general"
