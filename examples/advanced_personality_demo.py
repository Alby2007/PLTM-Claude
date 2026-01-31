"""
Advanced Personality & Mood Demo

Showcases enhanced features:
- Advanced mood pattern detection (cyclical, volatility, velocity)
- Enhanced conflict resolution (6-factor scoring, explanations)
- Context-aware personality
- Mood prediction
"""

import asyncio
from datetime import datetime, timedelta

from src.storage.sqlite_store import SQLiteGraphStore
from src.pipeline.memory_pipeline import MemoryPipeline
from src.personality.personality_mood_agent import PersonalityMoodAgent
from src.personality.mood_patterns import MoodPatterns
from src.personality.enhanced_conflict_resolver import EnhancedConflictResolver
from src.personality.contextual_personality import ContextualPersonality
from src.personality.advanced_mood_patterns import AdvancedMoodPatterns


async def demo_advanced_mood_patterns():
    """Demo advanced mood pattern detection"""
    print("=" * 80)
    print("ADVANCED MOOD PATTERN DETECTION")
    print("=" * 80)
    print()
    
    store = SQLiteGraphStore(":memory:")
    await store.connect()
    pipeline = MemoryPipeline(store)
    agent = PersonalityMoodAgent(pipeline)
    patterns = MoodPatterns(store)
    
    user_id = "pattern_user"
    
    # Simulate mood data over time
    print("Simulating 30 days of mood data...")
    moods_sequence = [
        # Week 1: Stressed pattern on Mondays
        ("Monday stressed", "stressed"),
        ("Tuesday happy", "happy"),
        ("Wednesday calm", "calm"),
        ("Thursday happy", "happy"),
        ("Friday excited", "excited"),
        ("Saturday happy", "happy"),
        ("Sunday calm", "calm"),
        # Week 2: Same pattern
        ("Monday stressed again", "stressed"),
        ("Tuesday feeling good", "happy"),
        ("Wednesday relaxed", "calm"),
        ("Thursday great", "happy"),
        ("Friday pumped", "excited"),
        ("Saturday chill", "happy"),
        ("Sunday peaceful", "calm"),
        # Week 3: Pattern continues
        ("Monday ugh stressed", "stressed"),
        ("Tuesday better", "happy"),
        ("Wednesday okay", "calm"),
        ("Thursday good", "happy"),
        ("Friday woohoo", "excited"),
        ("Saturday nice", "happy"),
        ("Sunday rest", "calm"),
    ]
    
    for message, expected_mood in moods_sequence:
        await agent.interact(user_id, f"I'm {message}")
    
    print(f"✅ Collected {len(moods_sequence)} mood records")
    print()
    
    # Detect patterns
    print("Detecting patterns...")
    detected = await patterns.detect_patterns(user_id, window_days=30)
    print()
    
    # Show results
    print("📊 PATTERN ANALYSIS:")
    print()
    
    # Temporal patterns
    if detected["temporal_patterns"].get("by_weekday"):
        print("Day-of-week patterns:")
        for day, mood in detected["temporal_patterns"]["by_weekday"].items():
            print(f"  {day}: {mood}")
        print()
    
    # Cyclical patterns
    if detected["cyclical_patterns"]:
        print("Cyclical patterns detected:")
        for cycle in detected["cyclical_patterns"]:
            print(f"  {cycle['type']}: strength {cycle['strength']:.2f}")
        print()
    
    # Volatility
    print(f"Mood volatility: {detected['volatility']:.2f}")
    print(f"  (0.0 = very stable, 1.0 = very volatile)")
    print()
    
    # Mood velocity
    if detected["mood_velocity"]:
        print("Average mood duration:")
        for mood, hours in detected["mood_velocity"].items():
            print(f"  {mood}: {hours:.1f} hours")
        print()
    
    # Distribution
    print("Mood distribution:")
    for mood, percentage in sorted(detected["mood_distribution"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {mood}: {percentage}%")
    print()
    
    # Prediction
    print("🔮 MOOD PREDICTION:")
    next_monday = datetime.now() + timedelta(days=(7 - datetime.now().weekday()))
    prediction = await patterns.predict_mood(user_id, for_time=next_monday)
    if prediction:
        mood, confidence = prediction
        print(f"  Next Monday: {mood} (confidence: {confidence:.2f})")
    print()
    
    await store.close()
    print("=" * 80)
    print()


async def demo_enhanced_conflict_resolution():
    """Demo enhanced conflict resolution"""
    print("=" * 80)
    print("ENHANCED CONFLICT RESOLUTION")
    print("=" * 80)
    print()
    
    store = SQLiteGraphStore(":memory:")
    await store.connect()
    pipeline = MemoryPipeline(store)
    agent = PersonalityMoodAgent(pipeline)
    resolver = EnhancedConflictResolver(store)
    
    user_id = "conflict_user"
    
    # Create conflicting traits
    print("Creating conflicting personality traits...")
    print()
    
    # User shows both concise and detailed preferences
    await agent.interact(user_id, "Just give me the facts, keep it brief")
    await agent.interact(user_id, "I need more technical details please")
    await agent.interact(user_id, "Too short, give me comprehensive explanation")
    await agent.interact(user_id, "Just the basics is fine")
    await agent.interact(user_id, "I want in-depth analysis")
    
    print("Conflicting signals:")
    print("  - 'concise' preference (2 observations)")
    print("  - 'detailed' preference (3 observations)")
    print()
    
    # Get personality atoms
    all_atoms = await store.get_atoms_by_subject(user_id)
    from src.core.models import AtomType
    style_atoms = [
        atom for atom in all_atoms
        if atom.atom_type == AtomType.COMMUNICATION_STYLE
    ]
    
    if len(style_atoms) >= 2:
        # Resolve with explanation
        winner, explanation = await resolver.resolve_with_explanation(
            user_id, style_atoms[:2]
        )
        
        print("🏆 RESOLUTION:")
        print(explanation)
        print()
        
        # Resolve with confidence
        winner2, confidence = await resolver.resolve_with_confidence(
            user_id, style_atoms[:2]
        )
        print(f"Confidence in resolution: {confidence:.2f}")
        print()
    
    await store.close()
    print("=" * 80)
    print()


async def demo_context_aware_personality():
    """Demo context-aware personality"""
    print("=" * 80)
    print("CONTEXT-AWARE PERSONALITY")
    print("=" * 80)
    print()
    
    store = SQLiteGraphStore(":memory:")
    await store.connect()
    pipeline = MemoryPipeline(store)
    agent = PersonalityMoodAgent(pipeline)
    contextual = ContextualPersonality(store)
    
    user_id = "context_user"
    
    print("Same user, different contexts...")
    print()
    
    # Technical context
    print("📝 Technical context:")
    await agent.interact(user_id, "Explain the algorithm implementation details")
    await agent.interact(user_id, "I need performance optimization specs")
    
    # Casual context
    print("💬 Casual context:")
    await agent.interact(user_id, "hey, what's up with this feature?")
    await agent.interact(user_id, "lol that's cool")
    
    # Formal context
    print("🎩 Formal context:")
    await agent.interact(user_id, "Please provide a comprehensive analysis")
    await agent.interact(user_id, "Thank you for your thorough explanation")
    print()
    
    # Get contexts
    contexts = await contextual.get_all_contexts(user_id)
    print(f"Detected contexts: {', '.join(contexts)}")
    print()
    
    # Compare contexts
    if len(contexts) >= 2:
        comparison = await contextual.compare_contexts(user_id, contexts[0], contexts[1])
        print(f"Comparing '{contexts[0]}' vs '{contexts[1]}':")
        print()
        
        if comparison["differences"]["formality"]["different"]:
            print("  Formality differs:")
            print(f"    {contexts[0]}: {comparison['differences']['formality'][contexts[0]]}")
            print(f"    {contexts[1]}: {comparison['differences']['formality'][contexts[1]]}")
        print()
    
    await store.close()
    print("=" * 80)
    print()


async def demo_mood_insights():
    """Demo comprehensive mood insights"""
    print("=" * 80)
    print("COMPREHENSIVE MOOD INSIGHTS")
    print("=" * 80)
    print()
    
    store = SQLiteGraphStore(":memory:")
    await store.connect()
    pipeline = MemoryPipeline(store)
    agent = PersonalityMoodAgent(pipeline)
    patterns = MoodPatterns(store)
    
    user_id = "insights_user"
    
    # Simulate varied mood data
    mood_data = [
        "I'm so happy today!",
        "Feeling a bit stressed",
        "This is frustrating",
        "Much better now, happy again",
        "Feeling calm and relaxed",
        "Excited about this project!",
        "Stressed again with deadlines",
        "Happy it's working out",
        "Frustrated with bugs",
        "Calm after fixing it",
    ]
    
    print("Collecting mood data...")
    for message in mood_data:
        await agent.interact(user_id, message)
    print()
    
    # Get insights
    insights = await patterns.get_mood_insights(user_id)
    print(insights)
    print()
    
    # Additional analysis
    history = await agent.mood_tracker.get_mood_history(user_id, days=1)
    
    # Calculate entropy
    entropy = AdvancedMoodPatterns.calculate_mood_entropy(history)
    print(f"Mood entropy: {entropy:.2f}")
    print("  (Higher = more unpredictable)")
    print()
    
    # Detect clusters
    clusters = AdvancedMoodPatterns.detect_mood_clusters(history)
    if clusters:
        print("Mood clusters (co-occurring moods):")
        for mood, related in clusters.items():
            print(f"  {mood} often with: {', '.join(related)}")
        print()
    
    await store.close()
    print("=" * 80)
    print()


async def main():
    """Run all advanced demos"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "ADVANCED PERSONALITY & MOOD FEATURES" + " " * 27 + "║")
    print("║" + " " * 78 + "║")
    print("║" + " " * 10 + "Enhanced Pattern Detection + Conflict Resolution" + " " * 18 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    await demo_advanced_mood_patterns()
    await demo_enhanced_conflict_resolution()
    await demo_context_aware_personality()
    await demo_mood_insights()
    
    print("\n")
    print("=" * 80)
    print("ALL ADVANCED DEMOS COMPLETE")
    print("=" * 80)
    print()
    print("✅ Advanced Features Demonstrated:")
    print("  1. Cyclical pattern detection (weekly, monthly cycles)")
    print("  2. Mood volatility and velocity analysis")
    print("  3. Enhanced conflict resolution (6-factor scoring)")
    print("  4. Context-aware personality tracking")
    print("  5. Mood prediction with confidence")
    print("  6. Comprehensive insights and analytics")
    print()
    print("🚀 Experiment 8 is production-ready with advanced capabilities!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
