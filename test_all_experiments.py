"""
Comprehensive Test Suite for All 7 Experiments

Runs integration tests for:
1. Lifelong Learning Agent
2. Multi-Agent Collaboration
3. Memory-Guided Prompts
4. Temporal Reasoning & Prediction
5. Personalized Tutor
6. Contextual Copilot
7. Memory-Aware RAG

Usage:
    python test_all_experiments.py
"""

import asyncio
import sys
from datetime import datetime

from src.pipeline.memory_pipeline import MemoryPipeline
from src.storage.sqlite_store import SQLiteGraphStore

# Import all experiment modules
from src.agents.lifelong_learning_agent import LifelongLearningAgent, LifelongLearningExperiment
from src.agents.multi_agent_workspace import SharedMemoryWorkspace, MultiAgentExperiment
from src.agents.adaptive_prompts import AdaptivePromptSystem, PromptOptimizationExperiment
from src.agents.temporal_reasoning import TemporalReasoningEngine, TemporalReasoningExperiment
from src.agents.personalized_tutor import PersonalizedTutor, PersonalizedTutorExperiment
from src.agents.contextual_copilot import ContextualCopilot, ContextualCopilotExperiment
from src.agents.memory_aware_rag import MemoryAwareRAG, MemoryAwareRAGExperiment


async def test_lifelong_learning(pipeline: MemoryPipeline) -> bool:
    """Test Lifelong Learning Agent"""
    print("\n" + "="*70)
    print("TEST 1: Lifelong Learning Agent")
    print("="*70)
    
    try:
        user_id = "test_user_1"
        agent = LifelongLearningAgent(pipeline, user_id)
        
        # Simulate interactions
        print("📝 Processing interactions...")
        await pipeline.process_message(user_id, "I love Python programming")
        await pipeline.process_message(user_id, "I'm learning machine learning")
        
        # Get context
        context = await agent.get_context("Tell me about programming")
        
        assert context is not None
        assert agent.interaction_count > 0
        
        print(f"✅ Agent tracked {agent.interaction_count} interactions")
        print(f"✅ Context retrieved successfully")
        
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
        workspace = SharedMemoryWorkspace(pipeline)
        
        # Add agents
        print("👥 Adding agents to workspace...")
        await workspace.add_agent("researcher", "Research specialist")
        await workspace.add_agent("writer", "Content writer")
        
        # Simulate collaboration
        await pipeline.process_message("researcher", "Found key insights about AI")
        await pipeline.process_message("writer", "Writing summary based on research")
        
        # Get shared knowledge
        knowledge = await workspace.get_shared_knowledge()
        
        assert len(workspace.agents) == 2
        
        print(f"✅ {len(workspace.agents)} agents collaborating")
        print(f"✅ Shared knowledge base established")
        
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
        
        # Add user preferences
        print("🎯 Learning user preferences...")
        await pipeline.process_message(user_id, "I prefer concise explanations")
        await pipeline.process_message(user_id, "I'm an expert in Python")
        
        # Generate adaptive prompt
        prompt = await prompt_system.generate_adaptive_prompt(
            task="explain recursion",
            domain="programming"
        )
        
        assert prompt.base_prompt is not None
        assert prompt.personalized_prompt is not None
        
        print(f"✅ Generated adaptive prompt")
        print(f"✅ Applied {len(prompt.personalization_applied)} personalizations")
        
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
        experiment = TemporalReasoningExperiment(pipeline)
        
        # Add temporal data
        print("⏰ Adding temporal patterns...")
        await pipeline.process_message(user_id, "I love Python")
        await pipeline.process_message(user_id, "I'm learning React")
        
        # Run experiments
        decay_result = await experiment.run_decay_prediction_experiment(user_id, days_ahead=30)
        interest_result = await experiment.run_interest_tracking_experiment(
            user_id, 
            topics=["Python", "React"]
        )
        
        assert decay_result["experiment"] == "decay_prediction"
        assert interest_result["experiment"] == "interest_tracking"
        
        summary = experiment.get_summary()
        
        print(f"✅ Ran {summary['total_experiments']} temporal experiments")
        print(f"✅ Predicted decay for {decay_result['num_predictions']} memories")
        
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
        experiment = PersonalizedTutorExperiment(pipeline, user_id)
        
        # Add learning history
        print("📚 Adding learning history...")
        await pipeline.process_message(user_id, "I'm learning Python")
        await pipeline.process_message(user_id, "I'm proficient in JavaScript")
        
        # Run experiments
        assessment_result = await experiment.run_skill_assessment_experiment(
            topics=["Python", "JavaScript"]
        )
        
        assert assessment_result["experiment"] == "skill_assessment"
        
        summary = experiment.get_summary()
        
        print(f"✅ Assessed skills in {len(assessment_result['topics'])} topics")
        print(f"✅ Ran {summary['total_experiments']} tutor experiments")
        
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
        experiment = ContextualCopilotExperiment(pipeline, user_id)
        
        # Add coding preferences
        print("💻 Learning coding preferences...")
        await pipeline.process_message(user_id, "I prefer spaces for indentation")
        await pipeline.process_message(user_id, "I use React")
        await pipeline.process_message(user_id, "I use React often")
        
        # Run experiments
        pref_result = await experiment.run_preference_learning_experiment()
        pattern_result = await experiment.run_pattern_detection_experiment()
        
        assert pref_result["experiment"] == "preference_learning"
        assert pattern_result["experiment"] == "pattern_detection"
        
        summary = experiment.get_summary()
        
        print(f"✅ Learned {len(pref_result['style_preferences'])} style preferences")
        print(f"✅ Detected {pattern_result['patterns_found']} coding patterns")
        
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
        experiment = MemoryAwareRAGExperiment(pipeline, user_id)
        
        # Add user interests
        print("🔍 Building user profile...")
        await pipeline.process_message(user_id, "I love machine learning")
        await pipeline.process_message(user_id, "I'm an expert in Python")
        
        # Run experiments
        aug_result = await experiment.run_query_augmentation_experiment(
            queries=["python tutorial", "machine learning"]
        )
        
        mock_docs = [
            {"id": "doc1", "content": "Python for experts", "score": 0.7},
            {"id": "doc2", "content": "Beginner Python", "score": 0.8},
        ]
        pers_result = await experiment.run_personalization_experiment(
            "python tutorial",
            mock_docs
        )
        
        assert aug_result["experiment"] == "query_augmentation"
        assert pers_result["experiment"] == "personalization"
        
        summary = experiment.get_summary()
        
        print(f"✅ Augmented {aug_result['queries_tested']} queries")
        print(f"✅ Personalized {pers_result['documents_processed']} documents")
        
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
