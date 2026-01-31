"""
Demo: Personality Emergence and Mood Tracking

This demo shows how personality emerges from interactions over time
and how mood tracking provides emotional awareness.
"""

import asyncio
from datetime import datetime

from src.storage.sqlite_store import SQLiteGraphStore
from src.pipeline.memory_pipeline import MemoryPipeline
from src.personality.personality_mood_agent import PersonalityMoodAgent


async def demo_personality_emergence():
    """
    Demo showing personality emerging over multiple interactions.
    """
    print("=" * 80)
    print("PERSONALITY EMERGENCE DEMO")
    print("=" * 80)
    print()
    
    # Initialize system
    store = SQLiteGraphStore(":memory:")
    await store.connect()
    pipeline = MemoryPipeline(store)
    agent = PersonalityMoodAgent(pipeline)
    
    user_id = "demo_user"
    
    # === Day 1: First Interactions ===
    print("=== DAY 1: First Interactions ===")
    print()
    
    # Interaction 1: User is direct
    print("User: Just give me the facts, no fluff")
    result1 = await agent.interact(user_id, "Just give me the facts, no fluff")
    print(f"Detected mood: {result1['mood_detected']}")
    print(f"Adaptive prompt: {result1['adaptive_prompt'][:100]}...")
    print()
    
    # Interaction 2: User wants technical depth
    print("User: That's too high-level, I need technical details")
    result2 = await agent.interact(user_id, "That's too high-level, I need technical details")
    print(f"Detected mood: {result2['mood_detected']}")
    print()
    
    # Interaction 3: User is frustrated
    print("User: This is frustrating, why won't it work?")
    result3 = await agent.interact(user_id, "This is frustrating, why won't it work?")
    print(f"Detected mood: {result3['mood_detected']}")
    print(f"Current mood: {result3['current_mood']}")
    print()
    
    # === Day 2: More Interactions ===
    print("=== DAY 2: Personality Starting to Form ===")
    print()
    
    # Interaction 4: Casual, technical
    print("User: hey, explain how the algorithm works")
    result4 = await agent.interact(user_id, "hey, explain how the algorithm works")
    print(f"Personality so far:")
    print(f"  Formality: {result4['personality']['formality_level']}")
    print(f"  Styles: {result4['personality']['communication_style']}")
    print()
    
    # Interaction 5: Direct question
    print("User: bottom line - will this scale?")
    result5 = await agent.interact(user_id, "bottom line - will this scale?")
    print(f"Interaction patterns: {result5['personality']['interaction_patterns']}")
    print()
    
    # === Day 7: Distinct Personality ===
    print("=== DAY 7: Distinct Personality Emerged ===")
    print()
    
    # Check full personality
    personality_summary = await agent.get_personality_summary(user_id)
    print(personality_summary)
    print()
    
    # Interaction with fully adapted prompt
    print("User: Explain quantum computing")
    result7 = await agent.interact(user_id, "Explain quantum computing")
    print("\nAdaptive prompt (should be: casual, technical, direct):")
    print(result7['adaptive_prompt'])
    print()
    
    # === Mood History ===
    print("=== MOOD HISTORY ===")
    print()
    mood_summary = await agent.get_mood_summary(user_id, days=7)
    print(mood_summary)
    print()
    
    await store.close()
    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)


async def demo_mood_tracking():
    """
    Demo showing mood detection and tracking.
    """
    print("\n\n")
    print("=" * 80)
    print("MOOD TRACKING DEMO")
    print("=" * 80)
    print()
    
    # Initialize system
    store = SQLiteGraphStore(":memory:")
    await store.connect()
    pipeline = MemoryPipeline(store)
    agent = PersonalityMoodAgent(pipeline)
    
    user_id = "mood_demo_user"
    
    # Test different moods
    test_messages = [
        ("I'm so happy about this!", "happy"),
        ("This is frustrating, nothing works", "frustrated"),
        ("I'm feeling overwhelmed with all this", "stressed"),
        ("Wow, this is amazing!", "excited"),
        ("I don't understand what's happening", "confused"),
        ("Everything is going great today!", "happy"),
    ]
    
    print("Testing mood detection:")
    print()
    
    for message, expected_mood in test_messages:
        result = await agent.interact(user_id, message, extract_personality=False)
        detected = result['mood_detected']
        current = result['current_mood']
        
        match = "✅" if detected == expected_mood else "❌"
        print(f"{match} Message: '{message}'")
        print(f"   Expected: {expected_mood}, Detected: {detected}, Current: {current}")
        print()
    
    # Show mood history
    print("\nMood History:")
    mood_summary = await agent.get_mood_summary(user_id, days=1)
    print(mood_summary)
    print()
    
    await store.close()
    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)


async def demo_adaptive_responses():
    """
    Demo showing how responses adapt to personality and mood.
    """
    print("\n\n")
    print("=" * 80)
    print("ADAPTIVE RESPONSE DEMO")
    print("=" * 80)
    print()
    
    # Initialize system
    store = SQLiteGraphStore(":memory:")
    await store.connect()
    pipeline = MemoryPipeline(store)
    agent = PersonalityMoodAgent(pipeline)
    
    # User A: Formal, detailed
    print("=== USER A: Formal, Detailed ===")
    user_a = "user_a"
    
    await agent.interact(user_a, "Please provide a comprehensive explanation")
    await agent.interact(user_a, "I would appreciate technical details")
    await agent.interact(user_a, "Thank you for the thorough response")
    
    result_a = await agent.interact(user_a, "Could you explain machine learning?")
    print("\nPersonality:")
    print(await agent.get_personality_summary(user_a))
    print("\nAdaptive Prompt:")
    print(result_a['adaptive_prompt'])
    print()
    
    # User B: Casual, concise
    print("\n=== USER B: Casual, Concise ===")
    user_b = "user_b"
    
    await agent.interact(user_b, "hey, just give me the basics")
    await agent.interact(user_b, "too long, tldr please")
    await agent.interact(user_b, "lol perfect, thanks")
    
    result_b = await agent.interact(user_b, "explain machine learning")
    print("\nPersonality:")
    print(await agent.get_personality_summary(user_b))
    print("\nAdaptive Prompt:")
    print(result_b['adaptive_prompt'])
    print()
    
    # User C: Frustrated, needs patience
    print("\n=== USER C: Frustrated ===")
    user_c = "user_c"
    
    result_c = await agent.interact(user_c, "This is so frustrating, nothing makes sense")
    print("\nCurrent Mood:", result_c['current_mood'])
    print("\nAdaptive Prompt (should be empathetic):")
    print(result_c['adaptive_prompt'])
    print()
    
    await store.close()
    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)


async def main():
    """Run all demos"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "EXPERIMENT 8: PERSONALITY & MOOD" + " " * 26 + "║")
    print("║" + " " * 78 + "║")
    print("║" + " " * 15 + "Emergent Personality + Mood Tracking" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # Run demos
    await demo_personality_emergence()
    await demo_mood_tracking()
    await demo_adaptive_responses()
    
    print("\n")
    print("=" * 80)
    print("ALL DEMOS COMPLETE")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("1. Personality emerges from interactions (not programmed)")
    print("2. Mood is detected and tracked over time")
    print("3. Responses adapt to both personality and mood")
    print("4. Different users get different AI 'characters'")
    print()
    print("This is the 8th experiment - ready for testing!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
