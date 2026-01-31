"""
Complete agent with personality emergence and mood tracking.

This module combines personality synthesis and mood tracking to create
an AI that adapts to each user's unique communication style and emotional state.
"""

from typing import Dict, Optional

from loguru import logger

from src.personality.personality_extractor import PersonalityExtractor
from src.personality.personality_synthesizer import PersonalitySynthesizer
from src.personality.mood_tracker import MoodTracker
from src.storage.sqlite_store import SQLiteGraphStore
from src.pipeline.memory_pipeline import MemoryPipeline


class PersonalityMoodAgent:
    """
    AI agent with emergent personality and mood awareness.
    
    Features:
    - Learns personality from interactions (not programmed)
    - Tracks mood over time
    - Adapts communication style to user preferences
    - Provides empathetic responses based on emotional state
    """
    
    def __init__(self, memory_pipeline: MemoryPipeline):
        self.pipeline = memory_pipeline
        self.store = memory_pipeline.store
        
        # Initialize components
        self.personality_extractor = PersonalityExtractor()
        self.personality_synth = PersonalitySynthesizer(self.store)
        self.mood_tracker = MoodTracker(self.store)
        
        logger.info("PersonalityMoodAgent initialized")
    
    async def interact(
        self,
        user_id: str,
        message: str,
        extract_personality: bool = True
    ) -> Dict:
        """
        Complete interaction with personality + mood awareness.
        
        Args:
            user_id: User identifier
            message: User's message
            extract_personality: Whether to extract personality traits (default True)
            
        Returns:
            Dictionary with:
            - personality: Current personality profile
            - current_mood: Detected mood (or None)
            - adaptive_prompt: Prompt adapted to personality and mood
        """
        # 1. Detect mood from message
        mood_atom = await self.mood_tracker.detect_mood(user_id, message)
        if mood_atom:
            # Store mood in memory
            async for _ in self.pipeline.process_message(
                f"{user_id} is feeling {mood_atom.object}",
                user_id=user_id
            ):
                pass  # Consume generator
            logger.info(f"Detected mood: {mood_atom.object}")
        
        # 2. Get current mood
        current_mood = await self.mood_tracker.get_current_mood(user_id)
        
        # 3. Get personality profile
        personality = await self.personality_synth.synthesize_personality(user_id)
        
        # 4. Extract personality traits from this interaction (if enabled)
        if extract_personality:
            traits = await self.personality_extractor.extract_from_interaction(
                user_id, message
            )
            
            # Store extracted traits
            for trait in traits:
                async for _ in self.pipeline.process_message(
                    f"{user_id} {trait.predicate} {trait.object}",
                    user_id=user_id
                ):
                    pass  # Consume generator
            
            if traits:
                logger.info(f"Extracted {len(traits)} personality traits")
        
        # 5. Build adaptive prompt
        adaptive_prompt = self._build_adaptive_prompt(
            message, personality, current_mood
        )
        
        return {
            "personality": personality,
            "current_mood": current_mood,
            "adaptive_prompt": adaptive_prompt,
            "mood_detected": mood_atom.object if mood_atom else None
        }
    
    def _build_adaptive_prompt(
        self,
        message: str,
        personality: Dict,
        mood: Optional[str]
    ) -> str:
        """
        Build fully adaptive prompt based on personality and mood.
        
        Args:
            message: User's message
            personality: Personality profile
            mood: Current mood (or None)
            
        Returns:
            System prompt adapted to user's personality and mood
        """
        # Base system prompt
        system = "You are an AI assistant with an adaptive personality."
        
        # Add personality adaptations
        if personality["formality_level"] == "casual":
            system += "\nUse casual, friendly language."
        elif personality["formality_level"] == "formal":
            system += "\nUse formal, professional language."
        
        # Communication style preferences
        if "concise communication" in personality.get("communication_style", []):
            system += "\nBe concise and to-the-point."
        
        if "technical depth" in personality.get("communication_style", []):
            system += "\nProvide technical depth and details."
        
        if "examples and analogies" in personality.get("communication_style", []):
            system += "\nUse examples and analogies to explain concepts."
        
        # Interaction patterns
        if "direct and to-the-point" in personality.get("interaction_patterns", []):
            system += "\nBe direct and straightforward."
        
        # Humor preference
        if personality.get("humor_preference"):
            system += "\nFeel free to use appropriate humor."
        
        # Detail level
        if personality.get("detail_level") == "detailed":
            system += "\nProvide detailed, in-depth explanations."
        else:
            system += "\nKeep explanations high-level and concise."
        
        # Add mood awareness
        if mood == "frustrated":
            system += "\n\nIMPORTANT: User seems frustrated. Be extra patient and understanding."
        elif mood == "happy" or mood == "excited":
            system += "\n\nUser seems happy. Match their positive energy."
        elif mood == "sad":
            system += "\n\nUser seems down. Be supportive and encouraging."
        elif mood == "stressed":
            system += "\n\nUser seems stressed. Be calm and reassuring."
        elif mood == "confused":
            system += "\n\nUser seems confused. Provide clear, step-by-step explanations."
        
        return f"{system}\n\nUser: {message}"
    
    async def get_personality_summary(self, user_id: str) -> str:
        """
        Get human-readable summary of user's personality.
        
        Args:
            user_id: User identifier
            
        Returns:
            Formatted string describing the personality
        """
        personality = await self.personality_synth.synthesize_personality(user_id)
        
        lines = ["Personality Profile:"]
        
        if personality["core_traits"]:
            lines.append(f"  Traits: {', '.join(personality['core_traits'])}")
        
        if personality["communication_style"]:
            lines.append(f"  Style: {', '.join(personality['communication_style'])}")
        
        if personality["interaction_patterns"]:
            lines.append(f"  Patterns: {', '.join(personality['interaction_patterns'])}")
        
        lines.append(f"  Formality: {personality['formality_level']}")
        lines.append(f"  Humor: {'Yes' if personality['humor_preference'] else 'No'}")
        lines.append(f"  Detail: {personality['detail_level']}")
        
        return "\n".join(lines)
    
    async def get_mood_summary(self, user_id: str, days: int = 7) -> str:
        """
        Get human-readable summary of user's mood history.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            
        Returns:
            Formatted string describing mood history
        """
        history = await self.mood_tracker.get_mood_history(user_id, days)
        
        if not history:
            return f"No mood data for the last {days} days."
        
        lines = [f"Mood History (last {days} days):"]
        
        for record in history[-10:]:  # Show last 10 moods
            timestamp = record["timestamp"].strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"  {timestamp}: {record['mood']} "
                f"(confidence: {record['confidence']:.2f})"
            )
        
        # Current mood
        current = await self.mood_tracker.get_current_mood(user_id)
        if current:
            lines.append(f"\nCurrent mood: {current}")
        
        return "\n".join(lines)
