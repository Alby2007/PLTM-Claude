"""
Simple Integration Test for All 7 Experiments

Tests that all experiments can be instantiated and basic methods work.
"""

import asyncio
import sys
from datetime import datetime

from src.pipeline.memory_pipeline import MemoryPipeline
from src.storage.sqlite_store import SQLiteGraphStore

# Import all experiment modules
from src.agents.lifelong_learning_agent import LifelongLearningAgent
from src.agents.multi_agent_workspace import SharedMemoryWorkspace
from src.agents.adaptive_prompts import AdaptivePromptSystem
from src.agents.temporal_reasoning import TemporalReasoningEngine
from src.agents.personalized_tutor import PersonalizedTutor
from src.agents.contextual_copilot import ContextualCopilot
from src.agents.memory_aware_rag import MemoryAwareRAG


async def process_message_fully(pipeline, user_id, message):
    """Helper to consume async generator"""
    async for update in pipeline.process_message(message, user_id):
        pass  # Consume all updates


async def test_lifelong_learning(pipeline: MemoryPipeline) -> bool:
    """Test Lifelong Learning Agent"""
    print("\n" + "="*70)
    print("TEST 1: Lifelong Learning Agent")
    print("="*70)
    
    try:
        user_id = "test_user_1"
        agent = LifelongLearningAgent(pipeline, user_id)
        
        # Process some messages
        print("📝 Processing interactions...")
        await process_message_fully(pipeline, user_id, "I love Python programming")
        await process_message_fully(pipeline, user_id, "I'm learning machine learning")
        
        # Get context
        context = await agent.get_context("Tell me about programming")
        
        assert agent.interaction_count > 0
        print(f"✅ Agent tracked {agent.interaction_count} interactions")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def test_multi_agent(pipeline: MemoryPipeline) -> bool:
    """Test Multi-Agent Collaboration"""
    print("\n" + "="*70)
    print("TEST 2: Multi-Agent Collaboration")
    print("="*70)
    
    try:
        workspace = SharedMemoryWorkspace(pipeline, workspace_id="test_workspace")
        
        # Add agents
        print("👥 Adding agents...")
        await workspace.add_agent("researcher", "Research specialist")
        await workspace.add_agent("writer", "Content writer")
        
        assert len(workspace.agents) == 2
        print(f"✅ {len(workspace.agents)} agents added to workspace")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def test_adaptive_prompts(pipeline: MemoryPipeline) -> bool:
    """Test Memory-Guided Prompt Engineering"""
    print("\n" + "="*70)
    print("TEST 3: Memory-Guided Prompt Engineering")
    print("="*70)
    
    try:
        user_id = "test_user_3"
        prompt_system = AdaptivePromptSystem(pipeline, user_id)
        
        # Add preferences
        print("🎯 Learning preferences...")
        await process_message_fully(pipeline, user_id, "I prefer concise explanations")
        
        # Generate prompt
        prompt = await prompt_system.generate_adaptive_prompt(
            task="explain recursion",
            domain="programming"
        )
        
        assert prompt.base_prompt is not None
        print(f"✅ Generated adaptive prompt")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def test_temporal_reasoning(pipeline: MemoryPipeline) -> bool:
    """Test Temporal Reasoning & Prediction"""
    print("\n" + "="*70)
    print("TEST 4: Temporal Reasoning & Prediction")
    print("="*70)
    
    try:
        user_id = "test_user_4"
        engine = TemporalReasoningEngine(pipeline)
        
        # Add data
        print("⏰ Adding temporal patterns...")
        await process_message_fully(pipeline, user_id, "I love Python")
        
        # Predict decay
        predictions = await engine.predict_decay(user_id, days_ahead=30)
        
        assert isinstance(predictions, list)
        print(f"✅ Generated {len(predictions)} decay predictions")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def test_personalized_tutor(pipeline: MemoryPipeline) -> bool:
    """Test Personalized Tutor"""
    print("\n" + "="*70)
    print("TEST 5: Personalized Tutor")
    print("="*70)
    
    try:
        user_id = "test_user_5"
        tutor = PersonalizedTutor(pipeline, user_id)
        
        # Add learning history
        print("📚 Adding learning history...")
        await process_message_fully(pipeline, user_id, "I'm learning Python")
        
        # Assess skill
        level, confidence, reasoning = await tutor.assess_skill_level("Python")
        
        assert confidence >= 0.0 and confidence <= 1.0
        print(f"✅ Assessed skill level: {level.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def test_contextual_copilot(pipeline: MemoryPipeline) -> bool:
    """Test Contextual Copilot"""
    print("\n" + "="*70)
    print("TEST 6: Contextual Copilot")
    print("="*70)
    
    try:
        user_id = "test_user_6"
        copilot = ContextualCopilot(pipeline, user_id)
        
        # Add preferences
        print("💻 Learning coding preferences...")
        await process_message_fully(pipeline, user_id, "I prefer spaces for indentation")
        
        # Learn preferences
        await copilot.learn_preferences()
        
        assert isinstance(copilot.style_preferences, dict)
        print(f"✅ Learned {len(copilot.style_preferences)} style preferences")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def test_memory_aware_rag(pipeline: MemoryPipeline) -> bool:
    """Test Memory-Aware RAG"""
    print("\n" + "="*70)
    print("TEST 7: Memory-Aware RAG")
    print("="*70)
    
    try:
        user_id = "test_user_7"
        rag = MemoryAwareRAG(pipeline, user_id)
        
        # Add interests
        print("🔍 Building user profile...")
        await process_message_fully(pipeline, user_id, "I love machine learning")
        
        # Build profile
        await rag.build_user_profile()
        
        assert isinstance(rag.user_interests, list)
        print(f"✅ Built profile with {len(rag.user_interests)} interests")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def main():
    """Run all experiment tests"""
    print("\n" + "="*70)
    print("COMPREHENSIVE EXPERIMENT TEST SUITE")
    print("Testing all 7 experiment capabilities")
    print("="*70)
    
    start_time = datetime.now()
    
    # Initialize pipeline
    print("\n🔧 Initializing test environment...")
    store = SQLiteGraphStore(":memory:")
    await store.connect()
    pipeline = MemoryPipeline(store)
    print("✅ Test environment ready")
    
    # Run all tests
    tests = [
        ("Lifelong Learning Agent", test_lifelong_learning),
        ("Multi-Agent Collaboration", test_multi_agent),
        ("Memory-Guided Prompts", test_adaptive_prompts),
        ("Temporal Reasoning", test_temporal_reasoning),
        ("Personalized Tutor", test_personalized_tutor),
        ("Contextual Copilot", test_contextual_copilot),
        ("Memory-Aware RAG", test_memory_aware_rag),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = await test_func(pipeline)
            results.append((name, "✅ PASS" if passed else "❌ FAIL"))
        except Exception as e:
            results.append((name, f"❌ FAIL: {str(e)[:50]}"))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = result.split()[0]
        print(f"{status} {name}")
    
    passed = sum(1 for _, r in results if "PASS" in r)
    total = len(tests)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*70)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"Duration: {elapsed:.2f} seconds")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL EXPERIMENTS WORKING! System is ready for production/research!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review output above.")
    
    await store.close()
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
