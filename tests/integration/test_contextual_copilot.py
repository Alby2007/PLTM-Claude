"""
Integration tests for Contextual Copilot
"""

import pytest
import asyncio

from src.pipeline.memory_pipeline import MemoryPipeline
from src.storage.sqlite_store import SQLiteGraphStore
from src.extraction.rule_based import RuleBasedExtractor
from src.agents.contextual_copilot import ContextualCopilot, ContextualCopilotExperiment


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
async def test_preference_learning(setup_pipeline):
    """Test learning coding preferences"""
    pipeline = await setup_pipeline.__anext__()
    user_id = "test_user"
    copilot = ContextualCopilot(pipeline, user_id)
    
    # Add preferences
    await pipeline.process_message(user_id, "I prefer spaces for indentation")
    await pipeline.process_message(user_id, "I love React")
    
    # Learn preferences
    await copilot.learn_preferences()
    
    assert isinstance(copilot.style_preferences, dict)
    assert isinstance(copilot.library_preferences, dict)
    
    print(f"✅ Learned {len(copilot.style_preferences)} style preferences")


@pytest.mark.asyncio
async def test_code_suggestion(setup_pipeline):
    """Test personalized code suggestions"""
    pipeline = await setup_pipeline.__anext__()
    user_id = "test_user"
    copilot = ContextualCopilot(pipeline, user_id)
    
    # Add preferences
    await pipeline.process_message(user_id, "I prefer camelCase")
    
    # Get suggestion
    suggestion = await copilot.suggest_code(
        task="create user function",
        language="javascript"
    )
    
    assert suggestion.code is not None
    assert isinstance(suggestion.reasoning, str)
    assert 0.0 <= suggestion.confidence <= 1.0
    
    print(f"✅ Generated code suggestion (confidence: {suggestion.confidence:.2f})")


@pytest.mark.asyncio
async def test_antipattern_detection(setup_pipeline):
    """Test antipattern detection"""
    pipeline = await setup_pipeline.__anext__()
    user_id = "test_user"
    copilot = ContextualCopilot(pipeline, user_id)
    
    # Add past mistake
    await pipeline.process_message(user_id, "I made a mistake with async await")
    
    # Detect antipatterns
    code = "async function test() { await something(); }"
    warnings = await copilot.detect_antipatterns(code, "javascript")
    
    assert isinstance(warnings, list)
    print(f"✅ Detected {len(warnings)} potential issues")


@pytest.mark.asyncio
async def test_pattern_tracking(setup_pipeline):
    """Test coding pattern tracking"""
    pipeline = await setup_pipeline.__anext__()
    user_id = "test_user"
    copilot = ContextualCopilot(pipeline, user_id)
    
    # Add usage patterns
    await pipeline.process_message(user_id, "I use React")
    await pipeline.process_message(user_id, "I use React for projects")
    await pipeline.process_message(user_id, "I prefer React")
    
    # Track patterns
    patterns = await copilot.track_coding_patterns()
    
    assert isinstance(patterns, list)
    print(f"✅ Tracked {len(patterns)} coding patterns")


@pytest.mark.asyncio
async def test_copilot_experiment(setup_pipeline):
    """Test full contextual copilot experiment"""
    pipeline = await setup_pipeline.__anext__()
    user_id = "test_user"
    experiment = ContextualCopilotExperiment(pipeline, user_id)
    
    # Add coding data
    await pipeline.process_message(user_id, "I prefer spaces")
    await pipeline.process_message(user_id, "I use React")
    await pipeline.process_message(user_id, "I use React often")
    
    # Run experiments
    pref_result = await experiment.run_preference_learning_experiment()
    pattern_result = await experiment.run_pattern_detection_experiment()
    
    # Verify results
    assert pref_result["experiment"] == "preference_learning"
    assert pattern_result["experiment"] == "pattern_detection"
    
    # Get summary
    summary = experiment.get_summary()
    assert summary["total_experiments"] == 2
    
    print(f"✅ Contextual Copilot Experiment complete: {summary['total_experiments']} tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
