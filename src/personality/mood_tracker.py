"""
Track user mood over time.

Mood is volatile state that changes frequently, unlike stable personality traits.
This module detects mood from messages and tracks mood history.
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta

from loguru import logger

from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
from src.storage.sqlite_store import SQLiteGraphStore


class MoodTracker:
    """
    Track user mood over time.
    
    Detects mood from message content and maintains mood history.
    Mood atoms decay quickly (volatile state).
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        logger.info("MoodTracker initialized")
    
    async def detect_mood(
        self,
        user_id: str,
        message: str
    ) -> Optional[MemoryAtom]:
        """
        Detect mood from user message.
        
        Args:
            user_id: User identifier
            message: User's message
            
        Returns:
            MemoryAtom representing detected mood, or None if neutral/unclear
        """
        # Analyze message for mood indicators
        mood_analysis = self._analyze_mood(message)
        
        if mood_analysis["mood"] == "neutral" or mood_analysis["confidence"] < 0.6:
            return None
        
        # Create mood atom
        mood_atom = MemoryAtom(
            atom_type=AtomType.STATE,
            subject=user_id,
            predicate="is_feeling",
            object=mood_analysis["mood"],
            confidence=mood_analysis["confidence"],
            strength=0.5,  # Mood is volatile, decays quickly
            provenance=Provenance.INFERRED,
            source_user=user_id,
            contexts=[
                f"detected_at:{datetime.now().isoformat()}",
                f"indicators:{','.join(mood_analysis['indicators'])}"
            ]
        )
        
        logger.debug(
            f"Detected mood '{mood_analysis['mood']}' "
            f"(confidence: {mood_analysis['confidence']:.2f})"
        )
        
        return mood_atom
    
    async def get_current_mood(self, user_id: str) -> Optional[str]:
        """
        Get user's most recent mood that hasn't decayed.
        
        Args:
            user_id: User identifier
            
        Returns:
            Mood string (e.g., "happy", "frustrated") or None
        """
        # Get all mood atoms
        all_atoms = await self.store.get_atoms_by_subject(
            user_id,
            graph=GraphType.SUBSTANTIATED
        )
        
        # Filter to mood atoms
        mood_atoms = [
            atom for atom in all_atoms
            if atom.predicate == "is_feeling"
        ]
        
        if not mood_atoms:
            return None
        
        # Get most recent
        recent_mood = max(mood_atoms, key=lambda m: m.first_observed)
        
        # Check if it's still fresh (within last hour)
        age = datetime.now() - recent_mood.first_observed
        if age > timedelta(hours=1):
            return None  # Too old, mood has likely changed
        
        return recent_mood.object
    
    async def get_mood_history(
        self,
        user_id: str,
        days: int = 7
    ) -> List[Dict]:
        """
        Get mood history over time.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            
        Returns:
            List of mood records with timestamp and confidence
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        # Get all mood atoms
        all_atoms = await self.store.get_atoms_by_subject(
            user_id,
            graph=GraphType.SUBSTANTIATED
        )
        
        # Filter to mood atoms within time range
        mood_atoms = [
            atom for atom in all_atoms
            if atom.predicate == "is_feeling" and atom.first_observed > cutoff
        ]
        
        # Build history
        history = []
        for mood in sorted(mood_atoms, key=lambda m: m.first_observed):
            history.append({
                "mood": mood.object,
                "timestamp": mood.first_observed,
                "confidence": mood.confidence
            })
        
        logger.debug(f"Retrieved {len(history)} mood records for {user_id}")
        return history
    
    def _analyze_mood(self, message: str) -> Dict:
        """
        Analyze message for mood indicators.
        
        Returns dict with:
        - mood: Detected mood name
        - confidence: Confidence score (0.0-1.0)
        - indicators: List of words/phrases that indicated this mood
        """
        msg_lower = message.lower()
        
        # Define mood patterns
        mood_patterns = {
            "excited": {
                "indicators": ["excited", "can't wait", "amazing", "incredible",
                              "wow", "omg", "🤩", "🚀", "holy shit", "insane",
                              "fucking", "so cool", "breakthrough", "working",
                              "complete", "done", "finished", "built", "shipped"],
                "confidence_boost": 0.85
            },
            "triumphant": {
                "indicators": ["complete", "done", "finished", "success", "achieved",
                              "accomplished", "nailed", "crushed", "killed it",
                              "perfect", "exactly", "works", "working", "🎉", "✅"],
                "confidence_boost": 0.85
            },
            "happy": {
                "indicators": ["happy", "great", "awesome", "excellent", "wonderful", 
                              "love", "😊", "😄", "🎉", "yay", "nice", "sweet"],
                "confidence_boost": 0.8
            },
            "frustrated": {
                "indicators": ["frustrated", "annoying", "annoyed", "ugh", "argh",
                              "why won't", "doesn't work", "broken", "😤", "😠",
                              "not working", "bug", "issue", "problem", "stuck"],
                "confidence_boost": 0.8
            },
            "impatient": {
                "indicators": ["come on", "let's go", "hurry", "now", "quickly",
                              "duh", "obviously", "just", "already", "waiting"],
                "confidence_boost": 0.75
            },
            "curious": {
                "indicators": ["interesting", "hmm", "wonder", "what if", "could we",
                              "how about", "tell me", "explain", "why", "how"],
                "confidence_boost": 0.7
            },
            "focused": {
                "indicators": ["let's", "next", "continue", "proceed", "okay",
                              "alright", "moving on", "now", "step"],
                "confidence_boost": 0.65
            },
            "sad": {
                "indicators": ["sad", "depressed", "down", "unhappy", "terrible",
                              "awful", "worst", "😢", "😞", "☹️"],
                "confidence_boost": 0.8
            },
            "stressed": {
                "indicators": ["stressed", "overwhelmed", "anxious", "worried",
                              "pressure", "deadline", "too much", "can't handle"],
                "confidence_boost": 0.7
            },
            "confused": {
                "indicators": ["confused", "don't understand", "unclear", "what",
                              "huh", "🤔", "lost", "not sure"],
                "confidence_boost": 0.6
            },
            "calm": {
                "indicators": ["calm", "relaxed", "peaceful", "content", "fine",
                              "okay", "good"],
                "confidence_boost": 0.6
            }
        }
        
        # Check each mood pattern
        detected_moods = []
        for mood, pattern in mood_patterns.items():
            indicators_found = []
            for indicator in pattern["indicators"]:
                if indicator in msg_lower:
                    indicators_found.append(indicator)
            
            if indicators_found:
                confidence = pattern["confidence_boost"] * (len(indicators_found) / 3)
                confidence = min(confidence, 1.0)  # Cap at 1.0
                
                detected_moods.append({
                    "mood": mood,
                    "confidence": confidence,
                    "indicators": indicators_found
                })
        
        # Return strongest detected mood
        if detected_moods:
            strongest = max(detected_moods, key=lambda m: m["confidence"])
            return strongest
        
        # No mood detected
        return {
            "mood": "neutral",
            "confidence": 0.0,
            "indicators": []
        }
