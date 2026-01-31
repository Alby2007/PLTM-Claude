"""
Interaction Dynamics Learner

Learns from the conversation dynamics themselves - how the AI's responses
affect user reactions, enabling self-improvement and relationship building.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

from src.storage.sqlite_store import SQLiteGraphStore
from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
from loguru import logger


@dataclass
class InteractionOutcome:
    """Result of an interaction"""
    my_style: str
    user_reaction: str
    was_positive: bool
    correction_type: Optional[str]
    engagement_level: str


class InteractionDynamicsLearner:
    """
    Learn from conversation dynamics.
    
    Tracks:
    - What response styles work vs don't work
    - Correction patterns (what to avoid)
    - Engagement patterns (what works)
    - Relationship dynamics over time
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        
        # Response style indicators
        self.style_indicators = {
            "verbose": ["detailed", "comprehensive", "thorough", "extensive", "long"],
            "concise": ["brief", "short", "quick", "direct", "minimal"],
            "technical": ["implementation", "code", "algorithm", "architecture"],
            "high_level": ["overview", "summary", "concept", "idea"],
            "personalized": ["you", "your", "based on", "for you", "specifically"],
            "neutral": ["generally", "typically", "usually", "often"],
        }
        
        # Reaction indicators
        self.positive_indicators = [
            "perfect", "exactly", "great", "thanks", "helpful", "yes", "good",
            "awesome", "amazing", "excellent", "nice", "love", "works", "working",
            "complete", "done", "success", "🎉", "✅", "holy shit", "insane",
            "incredible", "wow", "sweet", "cool", "brilliant", "fantastic"
        ]
        self.negative_indicators = ["wrong", "broken", "failed", "error", "bug", "issue"]
        self.correction_indicators = ["don't do", "not quite", "instead of", "rather than", "skip the", "stop"]
        self.engagement_indicators = [
            "tell me more", "how", "why", "explain", "continue", "interesting",
            "let's", "now", "next", "test", "try", "build", "implement", "create"
        ]
    
    async def learn_from_interaction(
        self,
        user_id: str,
        my_response: str,
        user_reaction: str
    ) -> Dict[str, Any]:
        """
        Learn from a single interaction.
        
        Args:
            user_id: User identifier
            my_response: What I (AI) said
            user_reaction: How user responded
        
        Returns:
            What was learned
        """
        # Analyze my response style
        my_style = self._analyze_response_style(my_response)
        
        # Analyze user reaction
        reaction_analysis = self._analyze_reaction(user_reaction)
        
        learned = []
        
        # If correction detected
        if reaction_analysis["is_correction"]:
            mistake = self._extract_mistake(my_response, user_reaction)
            
            atom = MemoryAtom(
                atom_type=AtomType.COMMUNICATION_STYLE,
                subject=user_id,
                predicate="dislikes_when_i",
                object=mistake,
                confidence=0.8,
                strength=0.8,
                provenance=Provenance.INFERRED,
                source_user=user_id,
                contexts=["interaction_learning"],
                graph=GraphType.SUBSTANTIATED
            )
            await self.store.add_atom(atom)
            learned.append(f"Learned to avoid: {mistake}")
        
        # If positive engagement
        if reaction_analysis["is_positive"]:
            success = self._extract_success(my_response, my_style)
            
            atom = MemoryAtom(
                atom_type=AtomType.COMMUNICATION_STYLE,
                subject=user_id,
                predicate="responds_well_to",
                object=success,
                confidence=0.7,
                strength=0.7,
                provenance=Provenance.INFERRED,
                source_user=user_id,
                contexts=["interaction_learning"],
                graph=GraphType.SUBSTANTIATED
            )
            await self.store.add_atom(atom)
            learned.append(f"Learned works well: {success}")
        
        # If deep engagement
        if reaction_analysis["engagement_level"] == "deep":
            atom = MemoryAtom(
                atom_type=AtomType.INTERACTION_PATTERN,
                subject=user_id,
                predicate="engages_deeply_with",
                object=my_style,
                confidence=0.75,
                strength=0.75,
                provenance=Provenance.INFERRED,
                source_user=user_id,
                contexts=["interaction_learning"],
                graph=GraphType.SUBSTANTIATED
            )
            await self.store.add_atom(atom)
            learned.append(f"Deep engagement with: {my_style}")
        
        return {
            "my_style": my_style,
            "reaction_type": reaction_analysis["type"],
            "was_positive": reaction_analysis["is_positive"],
            "was_correction": reaction_analysis["is_correction"],
            "engagement_level": reaction_analysis["engagement_level"],
            "learned": learned
        }
    
    def _analyze_response_style(self, response: str) -> str:
        """Identify the style of my response"""
        response_lower = response.lower()
        
        for style, indicators in self.style_indicators.items():
            if any(ind in response_lower for ind in indicators):
                return style
        
        # Default based on length
        if len(response) > 500:
            return "verbose"
        elif len(response) < 100:
            return "concise"
        return "moderate"
    
    def _analyze_reaction(self, reaction: str) -> Dict[str, Any]:
        """Analyze user's reaction"""
        reaction_lower = reaction.lower()
        
        is_positive = any(ind in reaction_lower for ind in self.positive_indicators)
        is_negative = any(ind in reaction_lower for ind in self.negative_indicators)
        is_correction = any(ind in reaction_lower for ind in self.correction_indicators)
        is_engaged = any(ind in reaction_lower for ind in self.engagement_indicators)
        
        # Determine engagement level
        if is_engaged or len(reaction) > 200:
            engagement = "deep"
        elif is_positive:
            engagement = "positive"
        elif is_correction or is_negative:
            engagement = "corrective"
        else:
            engagement = "neutral"
        
        # Determine overall type
        if is_correction:
            reaction_type = "correction"
        elif is_positive and not is_negative:
            reaction_type = "positive"
        elif is_negative:
            reaction_type = "negative"
        elif is_engaged:
            reaction_type = "engaged"
        else:
            reaction_type = "neutral"
        
        return {
            "type": reaction_type,
            "is_positive": is_positive and not is_negative,
            "is_correction": is_correction,
            "engagement_level": engagement
        }
    
    def _extract_mistake(self, my_response: str, user_reaction: str) -> str:
        """Extract what I did wrong"""
        reaction_lower = user_reaction.lower()
        
        # Common correction patterns
        if "too long" in reaction_lower or "too detailed" in reaction_lower:
            return "being_too_verbose"
        if "too short" in reaction_lower or "more detail" in reaction_lower:
            return "being_too_brief"
        if "personalized" in reaction_lower or "assume" in reaction_lower:
            return "over_personalizing"
        if "technical" in reaction_lower and "too" in reaction_lower:
            return "being_too_technical"
        if "simple" in reaction_lower or "basic" in reaction_lower:
            return "being_too_simple"
        
        # Default to response style
        style = self._analyze_response_style(my_response)
        return f"being_{style}"
    
    def _extract_success(self, my_response: str, my_style: str) -> str:
        """Extract what I did right"""
        return f"{my_style}_responses"
    
    async def get_interaction_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get learned interaction preferences.
        
        Returns what works and what doesn't.
        """
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        
        # Filter to interaction learning atoms
        dislikes = [a for a in all_atoms if a.predicate == "dislikes_when_i"]
        likes = [a for a in all_atoms if a.predicate == "responds_well_to"]
        engages = [a for a in all_atoms if a.predicate == "engages_deeply_with"]
        
        return {
            "user_id": user_id,
            "avoid": [{"behavior": a.object, "confidence": a.confidence} for a in dislikes],
            "prefer": [{"behavior": a.object, "confidence": a.confidence} for a in likes],
            "engages_with": [{"style": a.object, "confidence": a.confidence} for a in engages],
            "total_interactions_learned": len(dislikes) + len(likes) + len(engages)
        }
    
    async def get_relationship_dynamics(self, user_id: str) -> Dict[str, Any]:
        """
        Analyze the relationship dynamics over time.
        
        Returns:
            Trust level, communication style evolution, relationship stage
        """
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        
        # Count positive vs negative interactions
        positive_count = len([a for a in all_atoms if "responds_well" in a.predicate])
        negative_count = len([a for a in all_atoms if "dislikes" in a.predicate])
        
        # Calculate trust score
        total = positive_count + negative_count
        if total > 0:
            trust_score = positive_count / total
        else:
            trust_score = 0.5  # Neutral
        
        # Determine relationship stage
        if total < 5:
            stage = "initial"
        elif total < 20:
            stage = "developing"
        elif trust_score > 0.7:
            stage = "established_positive"
        elif trust_score < 0.3:
            stage = "needs_improvement"
        else:
            stage = "stable"
        
        return {
            "user_id": user_id,
            "trust_score": trust_score,
            "positive_interactions": positive_count,
            "corrections_received": negative_count,
            "relationship_stage": stage,
            "recommendation": self._get_recommendation(stage, trust_score)
        }
    
    def _get_recommendation(self, stage: str, trust_score: float) -> str:
        """Get recommendation for improving relationship"""
        if stage == "initial":
            return "Build trust through accurate, helpful responses"
        elif stage == "needs_improvement":
            return "Focus on avoiding past mistakes, be more careful"
        elif trust_score > 0.8:
            return "Relationship strong - can take more initiative"
        else:
            return "Continue current approach, relationship developing well"
