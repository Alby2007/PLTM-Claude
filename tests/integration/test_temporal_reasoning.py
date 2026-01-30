"""
Integration tests for Temporal Reasoning & Prediction
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from src.pipeline.memory_pipeline import MemoryPipeline
from src.storage.sqlite_store import SQLiteGraphStore
from src.extraction.rule_based import RuleBasedExtractor
from src.agents.temporal_reasoning import TemporalReasoningEngine, TemporalReasoningExperiment


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
async def test_decay_prediction(setup_pipeline):
    """Test that decay prediction works"""
    pipeline = await setup_pipeline.__anext__()
    engine = TemporalReasoningEngine(pipeline)
    
    user_id = "test_user"
    
    # Add some memories
    await pipeline.process_message(user_id, "I love Python programming")
    await pipeline.process_message(user_id, "I'm learning machine learning")
    
    # Predict decay
    predictions = await engine.predict_decay(user_id, days_ahead=30)
    
    assert isinstance(predictions, list)
    print(f"✅ Generated {len(predictions)} decay predictions")


@pytest.mark.asyncio
async def test_anomaly_detection(setup_pipeline):
    """Test temporal anomaly detection"""
    pipeline = await setup_pipeline.__anext__()
    engine = TemporalReasoningEngine(pipeline)
    
    user_id = "test_user"
    
    # Add memories
    await pipeline.process_message(user_id, "I use React for frontend")
    
    # Detect anomalies
    anomalies = await engine.detect_anomalies(user_id, lookback_days=7)
    
    assert isinstance(anomalies, list)
    print(f"✅ Detected {len(anomalies)} temporal anomalies")


@pytest.mark.asyncio
async def test_interest_tracking(setup_pipeline):
    """Test interest shift prediction"""
    pipeline = await setup_pipeline.__anext__()
    engine = TemporalReasoningEngine(pipeline)
    
    user_id = "test_user"
    
    # Add interest
    await pipeline.process_message(user_id, "I love Python")
    
    # Track interest
    trend, confidence, reasoning = await engine.predict_interest_shift(user_id, "Python")
    
    assert trend in ["increasing", "decreasing", "stable", "unknown"]
    assert 0.0 <= confidence <= 1.0
    assert isinstance(reasoning, str)
    
    print(f"✅ Interest trend: {trend} (confidence: {confidence:.2f})")


@pytest.mark.asyncio
async def test_temporal_experiment(setup_pipeline):
    """Test full temporal reasoning experiment"""
    pipeline = await setup_pipeline.__anext__()
    experiment = TemporalReasoningExperiment(pipeline)
    
    user_id = "test_user"
    
    # Add test data
    await pipeline.process_message(user_id, "I love Python")
    await pipeline.process_message(user_id, "I'm learning React")
    
    # Run experiments
    decay_result = await experiment.run_decay_prediction_experiment(user_id, days_ahead=30)
    anomaly_result = await experiment.run_anomaly_detection_experiment(user_id)
    interest_result = await experiment.run_interest_tracking_experiment(
        user_id, 
        topics=["Python", "React"]
    )
    
    # Verify results
    assert decay_result["experiment"] == "decay_prediction"
    assert anomaly_result["experiment"] == "anomaly_detection"
    assert interest_result["experiment"] == "interest_tracking"
    
    # Get summary
    summary = experiment.get_summary()
    assert summary["total_experiments"] == 3
    
    print(f"✅ Temporal Reasoning Experiment complete: {summary['total_experiments']} tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
