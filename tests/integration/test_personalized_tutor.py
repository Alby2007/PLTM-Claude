"""
Integration tests for Personalized Tutor
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from src.pipeline.memory_pipeline import MemoryPipeline
from src.storage.sqlite_store import SQLiteGraphStore
from src.extraction.rule_based import RuleBasedExtractor
from src.agents.personalized_tutor import PersonalizedTutor, PersonalizedTutorExperiment, SkillLevel


@pytest.fixture
async def setup_pipeline():
    """Setup test pipeline"""
    store = SQLiteGraphStore(":memory:")
    await store.connect()
    extractor = RuleBasedExtractor()
    pipeline = MemoryPipeline(store, extractor)
    
    yield pipeline
    
    await store.close()


@pytest.mark.asyncio
async def test_skill_assessment(setup_pipeline):
    """Test skill level assessment"""
    pipeline = await setup_pipeline.__anext__()
    user_id = "test_user"
    tutor = PersonalizedTutor(pipeline, user_id)
    
    # Add learning history
    await pipeline.process_message(user_id, "I'm learning Python")
    await pipeline.process_message(user_id, "I understand functions")
    
    # Assess skill
    level, confidence, reasoning = await tutor.assess_skill_level("Python")
    
    assert isinstance(level, SkillLevel)
    assert 0.0 <= confidence <= 1.0
    assert isinstance(reasoning, str)
    
    print(f"✅ Skill assessment: {level.value} (confidence: {confidence:.2f})")


@pytest.mark.asyncio
async def test_knowledge_gaps(setup_pipeline):
    """Test knowledge gap identification"""
    pipeline = await setup_pipeline.__anext__()
    user_id = "test_user"
    tutor = PersonalizedTutor(pipeline, user_id)
    
    # Add knowledge
    await pipeline.process_message(user_id, "I know React")
    
    # Identify gaps
    gaps = await tutor.identify_knowledge_gaps("React")
    
    assert isinstance(gaps, list)
    print(f"✅ Identified {len(gaps)} knowledge gaps")


@pytest.mark.asyncio
async def test_personalized_explanation(setup_pipeline):
    """Test personalized explanation generation"""
    pipeline = await setup_pipeline.__anext__()
    user_id = "test_user"
    tutor = PersonalizedTutor(pipeline, user_id)
    
    # Add skill level
    await pipeline.process_message(user_id, "I'm a beginner in Python")
    
    # Generate explanation
    explanation = await tutor.generate_personalized_explanation("recursion")
    
    assert isinstance(explanation, str)
    assert len(explanation) > 0
    
    print(f"✅ Generated personalized explanation: {explanation[:100]}...")


@pytest.mark.asyncio
async def test_learning_recommendations(setup_pipeline):
    """Test next step recommendations"""
    pipeline = await setup_pipeline.__anext__()
    user_id = "test_user"
    tutor = PersonalizedTutor(pipeline, user_id)
    
    # Add current knowledge
    await pipeline.process_message(user_id, "I know Python basics")
    
    # Get recommendations
    recommendations = await tutor.recommend_next_steps("Python")
    
    assert isinstance(recommendations, list)
    print(f"✅ Generated {len(recommendations)} learning recommendations")


@pytest.mark.asyncio
async def test_tutor_experiment(setup_pipeline):
    """Test full personalized tutor experiment"""
    pipeline = await setup_pipeline.__anext__()
    user_id = "test_user"
    experiment = PersonalizedTutorExperiment(pipeline, user_id)
    
    # Add learning data
    await pipeline.process_message(user_id, "I'm learning Python")
    await pipeline.process_message(user_id, "I'm proficient in JavaScript")
    
    # Run experiments
    assessment_result = await experiment.run_skill_assessment_experiment(
        topics=["Python", "JavaScript"]
    )
    gap_result = await experiment.run_gap_analysis_experiment(
        topics=["Python", "React"]
    )
    
    # Verify results
    assert assessment_result["experiment"] == "skill_assessment"
    assert gap_result["experiment"] == "gap_analysis"
    
    # Get summary
    summary = experiment.get_summary()
    assert summary["total_experiments"] == 2
    
    print(f"✅ Personalized Tutor Experiment complete: {summary['total_experiments']} tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
