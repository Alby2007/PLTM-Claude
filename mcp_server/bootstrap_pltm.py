"""
Bootstrap PLTM with Historical Conversation Data

Analyzes past conversations to extract personality traits, communication styles,
and interaction patterns, then populates PLTM with this historical data.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.sqlite_store import SQLiteGraphStore
from src.personality.personality_extractor import PersonalityExtractor
from src.personality.mood_tracker import MoodTracker
from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
from loguru import logger


class ConversationAnalyzer:
    """Analyzes conversation history to extract personality signals"""
    
    def __init__(self):
        self.personality_patterns = {
            # Communication style indicators
            "concise": ["brief", "short", "quick", "tldr", "summary", "just the"],
            "detailed": ["explain in detail", "elaborate", "comprehensive", "thorough", "deep dive"],
            "technical": ["implementation", "architecture", "algorithm", "code", "technical"],
            "casual": ["hey", "cool", "awesome", "nice", "thanks"],
            "formal": ["please", "could you", "would you", "appreciate", "kindly"],
            
            # Preference indicators
            "direct": ["don't", "just give me", "skip", "no need for", "directly"],
            "examples_preferred": ["example", "show me", "demonstrate", "sample"],
            "theory_preferred": ["why", "how does", "explain the concept", "theory"],
            
            # Feedback patterns
            "positive": ["perfect", "exactly", "great", "love it", "yes"],
            "negative": ["too", "not quite", "actually", "instead", "rather"],
            "corrective": ["don't", "not", "incorrect", "wrong", "fix"],
        }
        
        self.mood_indicators = {
            "frustrated": ["frustrated", "annoying", "not working", "broken", "issue"],
            "excited": ["awesome", "amazing", "perfect", "love", "great"],
            "focused": ["let's", "now", "next", "continue", "proceed"],
            "stressed": ["urgent", "quickly", "asap", "deadline", "pressure"],
            "calm": ["thanks", "appreciate", "good", "nice", "okay"],
        }
    
    def analyze_message(self, message: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze a single message for personality signals
        
        Returns:
            Dict with extracted traits, styles, and mood
        """
        message_lower = message.lower()
        
        results = {
            "traits": [],
            "styles": [],
            "mood": None,
            "preferences": []
        }
        
        # Extract communication style
        for style, indicators in self.personality_patterns.items():
            if any(ind in message_lower for ind in indicators):
                confidence = self._calculate_confidence(message_lower, indicators)
                results["styles"].append({
                    "style": style,
                    "confidence": confidence,
                    "context": context
                })
        
        # Extract mood
        for mood, indicators in self.mood_indicators.items():
            if any(ind in message_lower for ind in indicators):
                confidence = self._calculate_confidence(message_lower, indicators)
                if not results["mood"] or confidence > results["mood"]["confidence"]:
                    results["mood"] = {
                        "mood": mood,
                        "confidence": confidence
                    }
        
        # Extract preferences from feedback
        if any(word in message_lower for word in ["too", "don't", "not"]):
            # Negative feedback - extract what they don't want
            for pattern, indicators in self.personality_patterns.items():
                if any(ind in message_lower for ind in indicators):
                    results["preferences"].append({
                        "preference": f"dislikes_{pattern}",
                        "confidence": 0.8,
                        "context": context
                    })
        
        return results
    
    def _calculate_confidence(self, text: str, indicators: List[str]) -> float:
        """Calculate confidence based on number of matching indicators"""
        matches = sum(1 for ind in indicators if ind in text)
        base_confidence = 0.6
        boost = min(0.3, matches * 0.1)
        return min(0.95, base_confidence + boost)
    
    def analyze_conversation(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze entire conversation for patterns
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
        
        Returns:
            Aggregated personality insights
        """
        # Handle both formats: direct list or dict with "messages" key
        if isinstance(messages, dict) and "messages" in messages:
            messages = messages["messages"]
        
        user_messages = [m for m in messages if m.get("role") == "user"]
        
        all_styles = []
        all_moods = []
        all_preferences = []
        
        for msg in user_messages:
            analysis = self.analyze_message(msg["content"])
            all_styles.extend(analysis["styles"])
            if analysis["mood"]:
                all_moods.append(analysis["mood"])
            all_preferences.extend(analysis["preferences"])
        
        # Aggregate results
        return {
            "styles": self._aggregate_styles(all_styles),
            "moods": self._aggregate_moods(all_moods),
            "preferences": self._aggregate_preferences(all_preferences),
            "message_count": len(user_messages)
        }
    
    def _aggregate_styles(self, styles: List[Dict]) -> List[Dict]:
        """Aggregate and deduplicate communication styles"""
        style_map = {}
        for s in styles:
            style_name = s["style"]
            if style_name not in style_map:
                style_map[style_name] = {
                    "style": style_name,
                    "confidence": s["confidence"],
                    "count": 1,
                    "contexts": [s.get("context", "general")]
                }
            else:
                # Average confidence, track count
                existing = style_map[style_name]
                existing["count"] += 1
                existing["confidence"] = (existing["confidence"] + s["confidence"]) / 2
                if s.get("context") and s["context"] not in existing["contexts"]:
                    existing["contexts"].append(s["context"])
        
        return list(style_map.values())
    
    def _aggregate_moods(self, moods: List[Dict]) -> Dict[str, Any]:
        """Aggregate mood data"""
        if not moods:
            return {}
        
        mood_counts = {}
        for m in moods:
            mood_name = m["mood"]
            if mood_name not in mood_counts:
                mood_counts[mood_name] = {"count": 0, "total_confidence": 0.0}
            mood_counts[mood_name]["count"] += 1
            mood_counts[mood_name]["total_confidence"] += m["confidence"]
        
        # Find dominant mood
        dominant = max(mood_counts.items(), key=lambda x: x[1]["count"])
        
        return {
            "dominant_mood": dominant[0],
            "confidence": dominant[1]["total_confidence"] / dominant[1]["count"],
            "distribution": {k: v["count"] for k, v in mood_counts.items()}
        }
    
    def _aggregate_preferences(self, preferences: List[Dict]) -> List[Dict]:
        """Aggregate and deduplicate preferences"""
        pref_map = {}
        for p in preferences:
            pref_name = p["preference"]
            if pref_name not in pref_map:
                pref_map[pref_name] = {
                    "preference": pref_name,
                    "confidence": p["confidence"],
                    "count": 1
                }
            else:
                existing = pref_map[pref_name]
                existing["count"] += 1
                existing["confidence"] = (existing["confidence"] + p["confidence"]) / 2
        
        return list(pref_map.values())


class PLTMBootstrapper:
    """Bootstrap PLTM with historical conversation data"""
    
    def __init__(self, db_path: str = "pltm_mcp.db"):
        self.db_path = db_path
        self.store = None
        self.analyzer = ConversationAnalyzer()
        self.extractor = None
        self.mood_tracker = None
    
    async def initialize(self):
        """Initialize storage and components"""
        self.store = SQLiteGraphStore(self.db_path)
        await self.store.connect()
        
        # Initialize personality components (they need the store)
        self.extractor = PersonalityExtractor(self.store)
        self.mood_tracker = MoodTracker(self.store)
        
        logger.info(f"Initialized PLTM bootstrapper with database: {self.db_path}")
    
    async def bootstrap_from_conversations(
        self,
        conversations: List[Dict[str, Any]],
        user_id: str = "alby"
    ) -> Dict[str, Any]:
        """
        Bootstrap PLTM from conversation history
        
        Args:
            conversations: List of conversation dicts with messages
            user_id: User identifier
        
        Returns:
            Summary of bootstrapped data
        """
        logger.info(f"Bootstrapping PLTM from {len(conversations)} conversations")
        
        total_atoms = 0
        all_styles = []
        all_moods = []
        
        for i, conv in enumerate(conversations):
            logger.info(f"Processing conversation {i+1}/{len(conversations)}")
            
            # Analyze conversation
            analysis = self.analyzer.analyze_conversation(conv.get("messages", []))
            
            # Store communication styles
            for style_data in analysis["styles"]:
                atom = MemoryAtom(
                    atom_type=AtomType.COMMUNICATION_STYLE,
                    subject=user_id,
                    predicate="prefers_style",
                    object=style_data["style"],
                    confidence=style_data["confidence"],
                    strength=style_data["confidence"],
                    provenance=Provenance.INFERRED,
                    source_user=user_id,
                    contexts=style_data.get("contexts", ["general"]),
                    graph=GraphType.SUBSTANTIATED
                )
                await self.store.add_atom(atom)
                total_atoms += 1
                all_styles.append(style_data)
            
            # Store mood data
            if analysis["moods"]:
                mood_data = analysis["moods"]
                atom = MemoryAtom(
                    atom_type=AtomType.STATE,
                    subject=user_id,
                    predicate="typical_mood",
                    object=mood_data["dominant_mood"],
                    confidence=mood_data["confidence"],
                    strength=mood_data["confidence"],
                    provenance=Provenance.INFERRED,
                    source_user=user_id,
                    contexts=["historical_baseline"],
                    graph=GraphType.SUBSTANTIATED
                )
                await self.store.add_atom(atom)
                total_atoms += 1
                all_moods.append(mood_data)
            
            # Store preferences
            for pref_data in analysis["preferences"]:
                atom = MemoryAtom(
                    atom_type=AtomType.PREFERENCE,
                    subject=user_id,
                    predicate="has_preference",
                    object=pref_data["preference"],
                    confidence=pref_data["confidence"],
                    strength=pref_data["confidence"],
                    provenance=Provenance.INFERRED,
                    source_user=user_id,
                    contexts=["general"],
                    graph=GraphType.SUBSTANTIATED
                )
                await self.store.add_atom(atom)
                total_atoms += 1
        
        logger.info(f"Bootstrap complete: {total_atoms} atoms stored")
        
        return {
            "conversations_processed": len(conversations),
            "atoms_created": total_atoms,
            "styles_extracted": len(all_styles),
            "moods_analyzed": len(all_moods),
            "unique_styles": len(set(s["style"] for s in all_styles))
        }
    
    async def bootstrap_from_sample_data(self, user_id: str = "alby") -> Dict[str, Any]:
        """
        Bootstrap from sample conversation data (for testing)
        """
        sample_conversations = [
            {
                "title": "Technical Discussion",
                "messages": [
                    {"role": "user", "content": "Explain the PLTM conflict resolution algorithm"},
                    {"role": "assistant", "content": "The conflict resolution uses a multi-judge jury system..."},
                    {"role": "user", "content": "Too detailed, just give me the key steps"},
                    {"role": "assistant", "content": "1. Detect conflicts 2. Gather evidence 3. Jury deliberation"},
                    {"role": "user", "content": "Perfect, exactly what I needed"}
                ]
            },
            {
                "title": "Code Review",
                "messages": [
                    {"role": "user", "content": "Can you review this implementation?"},
                    {"role": "assistant", "content": "Here's a detailed analysis..."},
                    {"role": "user", "content": "Don't make this so personalized, just focus on the code"},
                    {"role": "assistant", "content": "Code review: 1. Logic is sound 2. Consider edge cases"},
                    {"role": "user", "content": "Great, that's helpful"}
                ]
            },
            {
                "title": "Debugging Session",
                "messages": [
                    {"role": "user", "content": "This isn't working, I'm so frustrated"},
                    {"role": "assistant", "content": "Let's debug step by step..."},
                    {"role": "user", "content": "Found it! Thanks for the systematic approach"},
                ]
            }
        ]
        
        return await self.bootstrap_from_conversations(sample_conversations, user_id)
    
    async def close(self):
        """Close database connection"""
        if self.store:
            await self.store.close()


async def main():
    """Main bootstrap function"""
    logger.info("Starting PLTM bootstrap process")
    
    bootstrapper = PLTMBootstrapper()
    await bootstrapper.initialize()
    
    # Bootstrap from sample data
    logger.info("Bootstrapping from sample conversations...")
    results = await bootstrapper.bootstrap_from_sample_data()
    
    print("\n" + "="*60)
    print("PLTM BOOTSTRAP COMPLETE")
    print("="*60)
    print(f"Conversations processed: {results['conversations_processed']}")
    print(f"Atoms created: {results['atoms_created']}")
    print(f"Styles extracted: {results['styles_extracted']}")
    print(f"Moods analyzed: {results['moods_analyzed']}")
    print(f"Unique styles: {results['unique_styles']}")
    print("="*60)
    
    # Query personality to verify
    from src.personality.personality_synthesizer import PersonalitySynthesizer
    synth = PersonalitySynthesizer(bootstrapper.store)
    personality = await synth.synthesize_personality("alby")
    
    print("\nBOOTSTRAPPED PERSONALITY PROFILE:")
    print(json.dumps(personality, indent=2))
    
    await bootstrapper.close()
    logger.info("Bootstrap process complete")


if __name__ == "__main__":
    asyncio.run(main())
