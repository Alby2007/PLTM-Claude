"""
Test suite for personality emergence and mood tracking (Experiment 8).
"""

import pytest
from datetime import datetime

from src.storage.sqlite_store import SQLiteGraphStore
from src.pipeline.memory_pipeline import MemoryPipeline
from src.personality.personality_extractor import PersonalityExtractor
from src.personality.personality_synthesizer import PersonalitySynthesizer
from src.personality.mood_tracker import MoodTracker
from src.personality.personality_mood_agent import PersonalityMoodAgent
from src.core.models import AtomType


@pytest.fixture
async def setup():
    """Setup test environment"""
    store = SQLiteGraphStore(":memory:")
    await store.connect()
    pipeline = MemoryPipeline(store)
    
    yield store, pipeline
    
    await store.close()


@pytest.mark.asyncio
async def test_personality_extractor_direct_style(setup):
    """Test extraction of direct communication style"""
    store, pipeline = setup
    extractor = PersonalityExtractor()
    
    # Direct message
    traits = await extractor.extract_from_interaction(
        "user_123",
        "Just give me the facts, no fluff"
    )
    
    assert len(traits) > 0
    # Should detect direct style
    direct_traits = [t for t in traits if "direct" in t.object.lower()]
    assert len(direct_traits) > 0


@pytest.mark.asyncio
async def test_personality_extractor_technical_style(setup):
    """Test extraction of technical communication preference"""
    store, pipeline = setup
    extractor = PersonalityExtractor()
    
    # Technical message
    traits = await extractor.extract_from_interaction(
        "user_123",
        "Explain the algorithm implementation and performance optimization"
    )
    
    assert len(traits) > 0
    # Should detect technical preference
    tech_traits = [t for t in traits if "technical" in t.object.lower()]
    assert len(tech_traits) > 0


@pytest.mark.asyncio
async def test_personality_extractor_humor(setup):
    """Test extraction of humor trait"""
    store, pipeline = setup
    extractor = PersonalityExtractor()
    
    # Humorous message
    traits = await extractor.extract_from_interaction(
        "user_123",
        "lol this is hilarious 😂"
    )
    
    assert len(traits) > 0
    # Should detect humor
    humor_traits = [t for t in traits if "humor" in t.object.lower()]
    assert len(humor_traits) > 0


@pytest.mark.asyncio
async def test_personality_synthesizer_aggregation(setup):
    """Test personality synthesis from multiple traits"""
    store, pipeline = setup
    agent = PersonalityMoodAgent(pipeline)
    
    user_id = "test_user"
    
    # Simulate multiple interactions
    await agent.interact(user_id, "Just give me the facts")
    await agent.interact(user_id, "I need technical details")
    await agent.interact(user_id, "hey, quick question")
    
    # Synthesize personality
    personality = await agent.personality_synth.synthesize_personality(user_id)
    
    # Should have detected some patterns
    assert personality is not None
    assert "formality_level" in personality
    assert "communication_style" in personality


@pytest.mark.asyncio
async def test_mood_tracker_happy(setup):
    """Test detection of happy mood"""
    store, pipeline = setup
    tracker = MoodTracker(store)
    
    # Happy message
    mood_atom = await tracker.detect_mood("user_123", "I'm so happy about this!")
    
    assert mood_atom is not None
    assert mood_atom.object == "happy"
    assert mood_atom.confidence > 0.6


@pytest.mark.asyncio
async def test_mood_tracker_frustrated(setup):
    """Test detection of frustrated mood"""
    store, pipeline = setup
    tracker = MoodTracker(store)
    
    # Frustrated message
    mood_atom = await tracker.detect_mood("user_123", "This is so frustrating!")
    
    assert mood_atom is not None
    assert mood_atom.object == "frustrated"
    assert mood_atom.confidence > 0.6


@pytest.mark.asyncio
async def test_mood_tracker_stressed(setup):
    """Test detection of stressed mood"""
    store, pipeline = setup
    tracker = MoodTracker(store)
    
    # Stressed message
    mood_atom = await tracker.detect_mood("user_123", "I'm so overwhelmed with all this")
    
    assert mood_atom is not None
    assert mood_atom.object == "stressed"
    assert mood_atom.confidence > 0.6


@pytest.mark.asyncio
async def test_mood_tracker_neutral(setup):
    """Test that neutral messages don't create mood atoms"""
    store, pipeline = setup
    tracker = MoodTracker(store)
    
    # Neutral message
    mood_atom = await tracker.detect_mood("user_123", "What is the weather today?")
    
    # Should not detect strong mood
    assert mood_atom is None or mood_atom.confidence < 0.6


@pytest.mark.asyncio
async def test_personality_mood_agent_integration(setup):
    """Test full PersonalityMoodAgent integration"""
    store, pipeline = setup
    agent = PersonalityMoodAgent(pipeline)
    
    user_id = "integration_user"
    
    # Interaction with mood
    result = await agent.interact(user_id, "I'm so excited about this project!")
    
    assert result is not None
    assert "personality" in result
    assert "current_mood" in result
    assert "adaptive_prompt" in result


@pytest.mark.asyncio
async def test_adaptive_prompt_formality(setup):
    """Test that adaptive prompt adjusts to formality level"""
    store, pipeline = setup
    agent = PersonalityMoodAgent(pipeline)
    
    user_id = "formal_user"
    
    # Formal interactions
    await agent.interact(user_id, "Please provide a comprehensive explanation")
    await agent.interact(user_id, "Thank you for your assistance")
    
    # Get adaptive prompt
    result = await agent.interact(user_id, "Could you explain this?")
    prompt = result["adaptive_prompt"]
    
    # Should adapt to formal style
    assert "formal" in prompt.lower() or "professional" in prompt.lower()


@pytest.mark.asyncio
async def test_adaptive_prompt_mood_awareness(setup):
    """Test that adaptive prompt responds to mood"""
    store, pipeline = setup
    agent = PersonalityMoodAgent(pipeline)
    
    user_id = "frustrated_user"
    
    # Frustrated interaction
    result = await agent.interact(user_id, "This is so frustrating, nothing works!")
    prompt = result["adaptive_prompt"]
    
    # Should include empathetic language
    assert "patient" in prompt.lower() or "understanding" in prompt.lower()


@pytest.mark.asyncio
async def test_personality_persistence(setup):
    """Test that personality traits persist across sessions"""
    store, pipeline = setup
    agent = PersonalityMoodAgent(pipeline)
    
    user_id = "persistent_user"
    
    # First interaction
    await agent.interact(user_id, "Just give me the facts, no fluff")
    
    # Get personality
    personality1 = await agent.personality_synth.synthesize_personality(user_id)
    
    # Second interaction (different session)
    await agent.interact(user_id, "I need technical details")
    
    # Get personality again
    personality2 = await agent.personality_synth.synthesize_personality(user_id)
    
    # Personality should build on previous traits
    assert len(personality2["communication_style"]) >= len(personality1["communication_style"])


@pytest.mark.asyncio
async def test_mood_history(setup):
    """Test mood history tracking"""
    store, pipeline = setup
    agent = PersonalityMoodAgent(pipeline)
    
    user_id = "mood_history_user"
    
    # Multiple moods over time
    await agent.interact(user_id, "I'm so happy!")
    await agent.interact(user_id, "This is frustrating")
    await agent.interact(user_id, "Feeling stressed")
    
    # Get mood history
    history = await agent.mood_tracker.get_mood_history(user_id, days=1)
    
    # Should have recorded moods
    assert len(history) > 0


@pytest.mark.asyncio
async def test_personality_summary(setup):
    """Test personality summary generation"""
    store, pipeline = setup
    agent = PersonalityMoodAgent(pipeline)
    
    user_id = "summary_user"
    
    # Build some personality
    await agent.interact(user_id, "Just the facts please")
    await agent.interact(user_id, "I need technical depth")
    
    # Get summary
    summary = await agent.get_personality_summary(user_id)
    
    assert summary is not None
    assert "Personality Profile" in summary


@pytest.mark.asyncio
async def test_different_users_different_personalities(setup):
    """Test that different users develop different personalities"""
    store, pipeline = setup
    agent = PersonalityMoodAgent(pipeline)
    
    # User A: Formal, detailed
    user_a = "user_a"
    await agent.interact(user_a, "Please provide comprehensive details")
    await agent.interact(user_a, "Thank you for the thorough explanation")
    
    # User B: Casual, concise
    user_b = "user_b"
    await agent.interact(user_b, "hey, just the basics")
    await agent.interact(user_b, "cool, thanks")
    
    # Get personalities
    personality_a = await agent.personality_synth.synthesize_personality(user_a)
    personality_b = await agent.personality_synth.synthesize_personality(user_b)
    
    # Should have different formality levels
    assert personality_a["formality_level"] != personality_b["formality_level"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
