"""
Integration tests for Memory-Aware RAG
"""

import pytest
import asyncio

from src.pipeline.memory_pipeline import MemoryPipeline
from src.storage.sqlite_store import SQLiteGraphStore
from src.extraction.rule_based import RuleBasedExtractor
from src.agents.memory_aware_rag import MemoryAwareRAG, MemoryAwareRAGExperiment


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
async def test_user_profile_building(setup_pipeline):
    """Test building user profile from memory"""
    pipeline = await setup_pipeline.__anext__()
    user_id = "test_user"
    rag = MemoryAwareRAG(pipeline, user_id)
    
    # Add user data
    await pipeline.process_message(user_id, "I love machine learning")
    await pipeline.process_message(user_id, "I'm an expert in Python")
    
    # Build profile
    await rag.build_user_profile()
    
    assert isinstance(rag.user_interests, list)
    assert isinstance(rag.user_expertise, dict)
    
    print(f"✅ Built profile: {len(rag.user_interests)} interests, {len(rag.user_expertise)} expertise areas")


@pytest.mark.asyncio
async def test_query_augmentation(setup_pipeline):
    """Test query augmentation with user context"""
    pipeline = await setup_pipeline.__anext__()
    user_id = "test_user"
    rag = MemoryAwareRAG(pipeline, user_id)
    
    # Add expertise
    await pipeline.process_message(user_id, "I'm an expert in Python")
    
    # Augment query
    augmented = await rag.augment_query("python tutorial")
    
    assert augmented.original_query == "python tutorial"
    assert isinstance(augmented.augmented_query, str)
    assert isinstance(augmented.user_context, list)
    
    print(f"✅ Augmented query: '{augmented.original_query}' → '{augmented.augmented_query}'")


@pytest.mark.asyncio
async def test_result_personalization(setup_pipeline):
    """Test document personalization"""
    pipeline = await setup_pipeline.__anext__()
    user_id = "test_user"
    rag = MemoryAwareRAG(pipeline, user_id)
    
    # Add interests
    await pipeline.process_message(user_id, "I love data science")
    
    # Mock documents
    docs = [
        {"id": "doc1", "content": "Data science tutorial", "score": 0.7},
        {"id": "doc2", "content": "Random topic", "score": 0.8},
    ]
    
    # Personalize
    personalized = await rag.personalize_results("data science", docs)
    
    assert isinstance(personalized, list)
    assert len(personalized) == 2
    assert all(hasattr(d, 'personalized_relevance') for d in personalized)
    
    print(f"✅ Personalized {len(personalized)} documents")


@pytest.mark.asyncio
async def test_personalized_answer_generation(setup_pipeline):
    """Test personalized answer generation"""
    pipeline = await setup_pipeline.__anext__()
    user_id = "test_user"
    rag = MemoryAwareRAG(pipeline, user_id)
    
    # Add expertise
    await pipeline.process_message(user_id, "I'm an expert in Python")
    
    # Generate answer
    answer, notes = await rag.generate_personalized_answer(
        "What is Python?",
        ["Python is a programming language"]
    )
    
    assert isinstance(answer, str)
    assert isinstance(notes, list)
    assert len(answer) > 0
    
    print(f"✅ Generated personalized answer with {len(notes)} personalization notes")


@pytest.mark.asyncio
async def test_rag_experiment(setup_pipeline):
    """Test full memory-aware RAG experiment"""
    pipeline = await setup_pipeline.__anext__()
    user_id = "test_user"
    experiment = MemoryAwareRAGExperiment(pipeline, user_id)
    
    # Add user data
    await pipeline.process_message(user_id, "I love machine learning")
    await pipeline.process_message(user_id, "I'm an expert in Python")
    
    # Run experiments
    aug_result = await experiment.run_query_augmentation_experiment(
        queries=["python tutorial", "machine learning guide"]
    )
    
    mock_docs = [
        {"id": "doc1", "content": "Python for experts", "score": 0.7},
        {"id": "doc2", "content": "Beginner Python", "score": 0.8},
    ]
    pers_result = await experiment.run_personalization_experiment(
        "python tutorial",
        mock_docs
    )
    
    # Verify results
    assert aug_result["experiment"] == "query_augmentation"
    assert pers_result["experiment"] == "personalization"
    
    # Get summary
    summary = experiment.get_summary()
    assert summary["total_experiments"] == 2
    
    print(f"✅ Memory-Aware RAG Experiment complete: {summary['total_experiments']} tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
