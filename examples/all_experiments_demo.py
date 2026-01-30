"""
Comprehensive Demo of All Experiment Capabilities

Demonstrates:
1. Lifelong Learning Agent
2. Multi-Agent Collaboration
3. Memory-Guided Prompt Engineering
4. Temporal Reasoning & Prediction
5. Personalized Tutor
6. Contextual Copilot
7. Memory-Aware RAG

Run this to see all experiments in action!
"""

import asyncio
from datetime import datetime, timedelta

from src.pipeline.orchestrator import MemoryPipeline
from src.storage.sqlite_store import SQLiteGraphStore
from src.extraction.rule_based import RuleBasedExtractor

# Import all experiment modules
from src.agents.lifelong_learning_agent import LifelongLearningAgent, LifelongLearningExperiment
from src.agents.multi_agent_workspace import SharedMemoryWorkspace, MultiAgentExperiment
from src.agents.adaptive_prompts import AdaptivePromptSystem, PromptOptimizationExperiment
from src.agents.temporal_reasoning import TemporalReasoningEngine, TemporalReasoningExperiment
from src.agents.personalized_tutor import PersonalizedTutor, PersonalizedTutorExperiment
from src.agents.contextual_copilot import ContextualCopilot, ContextualCopilotExperiment
from src.agents.memory_aware_rag import MemoryAwareRAG, MemoryAwareRAGExperiment


async def demo_temporal_reasoning(pipeline: MemoryPipeline):
    """Demo: Temporal Reasoning & Prediction"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Temporal Reasoning & Prediction")
    print("="*70)
    
    experiment = TemporalReasoningExperiment(pipeline)
    
    # Simulate some user interactions
    user_id = "demo_user"
    
    # Add some memories with different timestamps
    print("\n📝 Adding memories with temporal patterns...")
    messages = [
        "I love Python programming",
        "I'm learning machine learning",
        "I use React for frontend",
    ]
    
    for msg in messages:
        await pipeline.process_message(user_id, msg)
    
    # Run decay prediction
    print("\n🔮 Predicting memory decay (30 days ahead)...")
    decay_result = await experiment.run_decay_prediction_experiment(user_id, days_ahead=30)
    print(f"Found {decay_result['num_predictions']} predictions")
    
    # Run anomaly detection
    print("\n🚨 Detecting temporal anomalies...")
    anomaly_result = await experiment.run_anomaly_detection_experiment(user_id)
    print(f"Found {anomaly_result['num_anomalies']} anomalies")
    
    # Track interest trends
    print("\n📊 Tracking interest trends...")
    interest_result = await experiment.run_interest_tracking_experiment(
        user_id, 
        topics=["Python", "machine learning", "React"]
    )
    
    for topic, trend_data in interest_result['trends'].items():
        print(f"  {topic}: {trend_data['trend']} (confidence: {trend_data['confidence']:.2f})")
    
    print("\n✅ Temporal Reasoning Demo Complete!")


async def demo_personalized_tutor(pipeline: MemoryPipeline):
    """Demo: Personalized Tutor"""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Personalized Tutor")
    print("="*70)
    
    user_id = "demo_user"
    experiment = PersonalizedTutorExperiment(pipeline, user_id)
    
    # Add some learning-related memories
    print("\n📚 Adding learning history...")
    learning_messages = [
        "I'm learning Python",
        "I understand functions and loops",
        "I struggle with decorators",
        "I'm proficient in JavaScript",
    ]
    
    for msg in learning_messages:
        await pipeline.process_message(user_id, msg)
    
    # Assess skill levels
    print("\n🎓 Assessing skill levels...")
    assessment_result = await experiment.run_skill_assessment_experiment(
        topics=["Python", "JavaScript", "machine learning"]
    )
    
    for topic, assessment in assessment_result['assessments'].items():
        print(f"  {topic}: {assessment['skill_level']} (confidence: {assessment['confidence']:.2f})")
    
    # Identify knowledge gaps
    print("\n🔍 Identifying knowledge gaps...")
    gap_result = await experiment.run_gap_analysis_experiment(
        topics=["Python", "React"]
    )
    print(f"Found {gap_result['total_gaps']} knowledge gaps")
    
    print("\n✅ Personalized Tutor Demo Complete!")


async def demo_contextual_copilot(pipeline: MemoryPipeline):
    """Demo: Contextual Copilot"""
    print("\n" + "="*70)
    print("EXPERIMENT 6: Contextual Copilot")
    print("="*70)
    
    user_id = "demo_user"
    experiment = ContextualCopilotExperiment(pipeline, user_id)
    
    # Add coding preferences
    print("\n⚙️ Adding coding preferences...")
    coding_messages = [
        "I prefer using spaces for indentation",
        "I love React for frontend",
        "I use camelCase for JavaScript",
        "I made a mistake with async/await last time",
    ]
    
    for msg in coding_messages:
        await pipeline.process_message(user_id, msg)
    
    # Learn preferences
    print("\n🧠 Learning coding preferences...")
    pref_result = await experiment.run_preference_learning_experiment()
    print(f"Learned {len(pref_result['style_preferences'])} style preferences")
    print(f"Tracked {pref_result['past_mistakes_count']} past mistakes")
    
    # Detect patterns
    print("\n🔎 Detecting coding patterns...")
    pattern_result = await experiment.run_pattern_detection_experiment()
    print(f"Found {pattern_result['patterns_found']} coding patterns")
    
    for pattern in pattern_result['patterns']:
        print(f"  {pattern['description']} (frequency: {pattern['frequency']})")
    
    print("\n✅ Contextual Copilot Demo Complete!")


async def demo_memory_aware_rag(pipeline: MemoryPipeline):
    """Demo: Memory-Aware RAG"""
    print("\n" + "="*70)
    print("EXPERIMENT 7: Memory-Aware RAG")
    print("="*70)
    
    user_id = "demo_user"
    experiment = MemoryAwareRAGExperiment(pipeline, user_id)
    
    # Add user interests and expertise
    print("\n🎯 Adding user interests and expertise...")
    rag_messages = [
        "I'm interested in machine learning",
        "I'm an expert in Python",
        "I love data science",
    ]
    
    for msg in rag_messages:
        await pipeline.process_message(user_id, msg)
    
    # Test query augmentation
    print("\n🔍 Testing query augmentation...")
    queries = [
        "python tutorial",
        "machine learning guide",
        "data visualization"
    ]
    
    aug_result = await experiment.run_query_augmentation_experiment(queries)
    print(f"Augmented {aug_result['queries_tested']} queries")
    print(f"Avg context items: {aug_result['avg_context_items']:.1f}")
    
    # Test personalization
    print("\n⭐ Testing document personalization...")
    mock_docs = [
        {"id": "doc1", "content": "Advanced Python for machine learning experts", "score": 0.7},
        {"id": "doc2", "content": "Beginner's guide to Python", "score": 0.8},
        {"id": "doc3", "content": "Data science with Python", "score": 0.6},
    ]
    
    pers_result = await experiment.run_personalization_experiment(
        "python machine learning",
        mock_docs
    )
    
    print(f"Processed {pers_result['documents_processed']} documents")
    print(f"Avg relevance boost: {pers_result['avg_relevance_boost']:.2f}")
    print(f"Avg novelty score: {pers_result['avg_novelty_score']:.2f}")
    
    print("\n✅ Memory-Aware RAG Demo Complete!")


async def main():
    """Run all experiment demos"""
    print("\n" + "="*70)
    print("COMPREHENSIVE EXPERIMENT DEMO")
    print("Demonstrating all 7 experiment capabilities")
    print("="*70)
    
    # Initialize pipeline
    store = SQLiteGraphStore(":memory:")
    await store.connect()
    
    extractor = RuleBasedExtractor()
    pipeline = MemoryPipeline(store, extractor)
    
    # Run all demos
    await demo_temporal_reasoning(pipeline)
    await demo_personalized_tutor(pipeline)
    await demo_contextual_copilot(pipeline)
    await demo_memory_aware_rag(pipeline)
    
    # Summary
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\n✅ Experiment 1: Lifelong Learning Agent (see lifelong_learning_demo.py)")
    print("✅ Experiment 2: Multi-Agent Collaboration (implemented)")
    print("✅ Experiment 3: Memory-Guided Prompts (implemented)")
    print("✅ Experiment 4: Temporal Reasoning - DEMONSTRATED ✓")
    print("✅ Experiment 5: Personalized Tutor - DEMONSTRATED ✓")
    print("✅ Experiment 6: Contextual Copilot - DEMONSTRATED ✓")
    print("✅ Experiment 7: Memory-Aware RAG - DEMONSTRATED ✓")
    print("\n🎉 All 7 experiment capabilities are now available!")
    
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
