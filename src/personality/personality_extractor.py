"""
Extract personality traits from user interactions.

This module learns personality traits, communication styles, and interaction patterns
from how users communicate with the AI.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import re

from loguru import logger

from src.core.models import MemoryAtom, AtomType, Provenance, GraphType


class PersonalityExtractor:
    """
    Extract personality traits from user interactions.
    
    Learns:
    - Communication style preferences (concise, technical, etc.)
    - Interaction patterns (formal, casual, direct)
    - Personality traits (humorous, analytical, etc.)
    """
    
    def __init__(self):
        logger.info("PersonalityExtractor initialized")
    
    async def extract_from_interaction(
        self,
        user_id: str,
        user_message: str,
        ai_response: Optional[str] = None,
        user_reaction: Optional[str] = None
    ) -> List[MemoryAtom]:
        """
        Extract personality traits from a single interaction.
        
        Args:
            user_id: User identifier
            user_message: What the user said
            ai_response: How the AI responded
            user_reaction: User's reaction to AI response (optional)
            
        Returns:
            List of personality-related memory atoms
        """
        traits: List[MemoryAtom] = []
        
        # Extract from user's message style
        message_traits = await self._extract_from_message_style(user_id, user_message)
        traits.extend(message_traits)
        
        # Extract from user's reaction to AI response (if provided)
        if ai_response and user_reaction:
            reaction_traits = await self._extract_from_reaction(
                user_id, ai_response, user_reaction
            )
            traits.extend(reaction_traits)
        
        logger.debug(f"Extracted {len(traits)} personality traits from interaction")
        return traits
    
    async def _extract_from_message_style(
        self,
        user_id: str,
        message: str
    ) -> List[MemoryAtom]:
        """Extract traits from how user writes messages"""
        traits: List[MemoryAtom] = []
        
        # Analyze message characteristics
        analysis = self._analyze_message_style(message)
        
        # Direct/blunt communication
        if analysis["direct"]:
            traits.append(MemoryAtom(
                atom_type=AtomType.INTERACTION_PATTERN,
                subject=user_id,
                predicate="communication_style_is",
                object="direct and to-the-point",
                confidence=0.7,
                strength=0.7,
                provenance=Provenance.INFERRED,
                source_user=user_id,
                contexts=["extracted_from_message_style"]
            ))
        
        # Formal vs casual
        if analysis["formal"]:
            traits.append(MemoryAtom(
                atom_type=AtomType.INTERACTION_PATTERN,
                subject=user_id,
                predicate="prefers_formality",
                object="formal and professional",
                confidence=0.6,
                strength=0.6,
                provenance=Provenance.INFERRED,
                source_user=user_id,
                contexts=["extracted_from_message_style"]
            ))
        elif analysis["casual"]:
            traits.append(MemoryAtom(
                atom_type=AtomType.INTERACTION_PATTERN,
                subject=user_id,
                predicate="prefers_formality",
                object="casual and friendly",
                confidence=0.6,
                strength=0.6,
                provenance=Provenance.INFERRED,
                source_user=user_id,
                contexts=["extracted_from_message_style"]
            ))
        
        # Technical language
        if analysis["technical"]:
            traits.append(MemoryAtom(
                atom_type=AtomType.COMMUNICATION_STYLE,
                subject=user_id,
                predicate="prefers_style",
                object="technical depth",
                confidence=0.7,
                strength=0.7,
                provenance=Provenance.INFERRED,
                source_user=user_id,
                contexts=["extracted_from_message_style"]
            ))
        
        # Concise vs verbose
        if analysis["concise"]:
            traits.append(MemoryAtom(
                atom_type=AtomType.COMMUNICATION_STYLE,
                subject=user_id,
                predicate="prefers_style",
                object="concise communication",
                confidence=0.6,
                strength=0.6,
                provenance=Provenance.INFERRED,
                source_user=user_id,
                contexts=["extracted_from_message_style"]
            ))
        
        # Humor
        if analysis["humorous"]:
            traits.append(MemoryAtom(
                atom_type=AtomType.PERSONALITY_TRAIT,
                subject=user_id,
                predicate="has_trait",
                object="uses humor",
                confidence=0.7,
                strength=0.7,
                provenance=Provenance.INFERRED,
                source_user=user_id,
                contexts=["extracted_from_message_style"]
            ))
        
        return traits
    
    async def _extract_from_reaction(
        self,
        user_id: str,
        ai_response: str,
        user_reaction: str
    ) -> List[MemoryAtom]:
        """Extract preferences from user's reaction to AI response"""
        traits: List[MemoryAtom] = []
        
        # Analyze if reaction is positive or negative
        is_positive = self._is_positive_reaction(user_reaction)
        is_negative = self._is_negative_reaction(user_reaction)
        
        if not (is_positive or is_negative):
            return traits  # Neutral reaction, nothing to learn
        
        # Analyze what made the response good/bad
        response_analysis = self._analyze_response_style(ai_response)
        
        if is_positive:
            # User liked this style - reinforce it
            if response_analysis["concise"]:
                traits.append(MemoryAtom(
                    atom_type=AtomType.COMMUNICATION_STYLE,
                    subject=user_id,
                    predicate="responds_well_to",
                    object="concise responses",
                    confidence=0.8,
                    strength=0.8,
                    provenance=Provenance.INFERRED,
                    source_user=user_id,
                    contexts=["positive_feedback"]
                ))
            
            if response_analysis["technical"]:
                traits.append(MemoryAtom(
                    atom_type=AtomType.COMMUNICATION_STYLE,
                    subject=user_id,
                    predicate="responds_well_to",
                    object="technical depth",
                    confidence=0.8,
                    strength=0.8,
                    provenance=Provenance.INFERRED,
                    source_user=user_id,
                    contexts=["positive_feedback"]
                ))
            
            if response_analysis["uses_examples"]:
                traits.append(MemoryAtom(
                    atom_type=AtomType.COMMUNICATION_STYLE,
                    subject=user_id,
                    predicate="responds_well_to",
                    object="examples and analogies",
                    confidence=0.7,
                    strength=0.7,
                    provenance=Provenance.INFERRED,
                    source_user=user_id,
                    contexts=["positive_feedback"]
                ))
        
        elif is_negative:
            # User disliked this style - avoid it
            if response_analysis["verbose"]:
                traits.append(MemoryAtom(
                    atom_type=AtomType.COMMUNICATION_STYLE,
                    subject=user_id,
                    predicate="dislikes_style",
                    object="verbose explanations",
                    confidence=0.8,
                    strength=0.8,
                    provenance=Provenance.INFERRED,
                    source_user=user_id,
                    contexts=["negative_feedback"]
                ))
            
            if response_analysis["too_simple"]:
                traits.append(MemoryAtom(
                    atom_type=AtomType.COMMUNICATION_STYLE,
                    subject=user_id,
                    predicate="dislikes_style",
                    object="oversimplified explanations",
                    confidence=0.7,
                    strength=0.7,
                    provenance=Provenance.INFERRED,
                    source_user=user_id,
                    contexts=["negative_feedback"]
                ))
        
        return traits
    
    def _analyze_message_style(self, message: str) -> Dict[str, bool]:
        """
        Analyze stylistic characteristics of a message.
        
        Returns dict with boolean flags for various style indicators.
        """
        msg_lower = message.lower()
        word_count = len(message.split())
        
        # Direct/blunt indicators
        direct_phrases = [
            "just give me", "just tell me", "no fluff", "get to the point",
            "bottom line", "tldr", "in short", "simply put"
        ]
        is_direct = any(phrase in msg_lower for phrase in direct_phrases)
        
        # Formal indicators
        formal_indicators = [
            "please", "kindly", "would you", "could you", "thank you",
            "appreciate", "regards"
        ]
        is_formal = any(ind in msg_lower for ind in formal_indicators)
        
        # Casual indicators
        casual_indicators = [
            "hey", "yo", "lol", "haha", "btw", "gonna", "wanna",
            "yeah", "nah", "cool", "awesome"
        ]
        is_casual = any(ind in msg_lower for ind in casual_indicators)
        
        # Technical language
        technical_indicators = [
            "algorithm", "implementation", "architecture", "optimization",
            "performance", "latency", "throughput", "scalability",
            "api", "database", "query", "function", "class", "method"
        ]
        is_technical = any(ind in msg_lower for ind in technical_indicators)
        
        # Concise (short messages)
        is_concise = word_count < 20
        
        # Humorous
        humor_indicators = ["lol", "haha", "😂", "🤣", "😄", "joke", "funny"]
        is_humorous = any(ind in msg_lower for ind in humor_indicators)
        
        return {
            "direct": is_direct,
            "formal": is_formal and not is_casual,
            "casual": is_casual and not is_formal,
            "technical": is_technical,
            "concise": is_concise,
            "humorous": is_humorous,
        }
    
    def _analyze_response_style(self, response: str) -> Dict[str, bool]:
        """Analyze characteristics of AI response"""
        word_count = len(response.split())
        
        return {
            "concise": word_count < 100,
            "verbose": word_count > 300,
            "technical": "algorithm" in response.lower() or "implementation" in response.lower(),
            "uses_examples": "example:" in response.lower() or "for instance" in response.lower(),
            "too_simple": word_count < 30,
        }
    
    def _is_positive_reaction(self, reaction: str) -> bool:
        """Check if reaction is positive"""
        positive_indicators = [
            "thanks", "thank you", "perfect", "great", "excellent",
            "exactly", "yes", "good", "helpful", "appreciate",
            "👍", "✅", "🎉", "😊"
        ]
        return any(ind in reaction.lower() for ind in positive_indicators)
    
    def _is_negative_reaction(self, reaction: str) -> bool:
        """Check if reaction is negative"""
        negative_indicators = [
            "too long", "too short", "too simple", "too complex",
            "not helpful", "wrong", "no", "bad", "confusing",
            "👎", "❌", "😞"
        ]
        return any(ind in reaction.lower() for ind in negative_indicators)
