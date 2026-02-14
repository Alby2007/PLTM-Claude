"""
Tests for Sensory Input System

Tests the multimodal perception infrastructure:
- Sensory buffer (temporal integration)
- Camera capture (visual observations)
- Microphone capture (audio observations)
- Memory integration (sensory → atoms)
"""

import pytest
import time
from unittest.mock import Mock, patch
import numpy as np

from src.sensory.sensory_buffer import SensoryBuffer, SensoryObservation
from src.sensory.memory_integration import SensoryMemoryIntegrator
from src.core.models import AtomType
from src.storage.sqlite_store import SQLiteGraphStore


class TestSensoryBuffer:
    """Test sensory buffer temporal integration"""
    
    def test_buffer_initialization(self):
        """Test buffer initializes correctly"""
        buffer = SensoryBuffer(max_size=50, max_age_seconds=30.0)
        
        assert buffer.max_size == 50
        assert buffer.max_age_seconds == 30.0
        assert len(buffer.visual_buffer) == 0
        assert len(buffer.audio_buffer) == 0
    
    def test_add_visual_observation(self):
        """Test adding visual observations"""
        buffer = SensoryBuffer()
        
        data = {
            "faces_detected": 1,
            "scene": {"brightness": 120, "description": "bright, static"}
        }
        
        buffer.add_visual(data, confidence=0.8)
        
        assert len(buffer.visual_buffer) == 1
        obs = buffer.get_current_visual()
        assert obs is not None
        assert obs.modality == "visual"
        assert obs.data["faces_detected"] == 1
        assert obs.confidence == 0.8
    
    def test_add_audio_observation(self):
        """Test adding audio observations"""
        buffer = SensoryBuffer()
        
        data = {
            "speech_detected": True,
            "transcript": "Hello world",
            "tone": "neutral"
        }
        
        buffer.add_audio(data, confidence=0.9)
        
        assert len(buffer.audio_buffer) == 1
        obs = buffer.get_current_audio()
        assert obs is not None
        assert obs.modality == "audio"
        assert obs.data["transcript"] == "Hello world"
        assert obs.confidence == 0.9
    
    def test_get_current_state(self):
        """Test getting unified multimodal state"""
        buffer = SensoryBuffer()
        
        # Add visual
        buffer.add_visual({"faces_detected": 1})
        time.sleep(0.1)
        
        # Add audio
        buffer.add_audio({"speech_detected": True, "transcript": "test"})
        
        state = buffer.get_current_state()
        
        assert "visual" in state
        assert "audio" in state
        assert "cross_modal" in state
        assert state["visual"] is not None
        assert state["audio"] is not None
    
    def test_get_history(self):
        """Test getting recent history"""
        buffer = SensoryBuffer()
        
        # Add multiple observations
        buffer.add_visual({"test": 1})
        time.sleep(0.05)
        buffer.add_audio({"test": 2})
        time.sleep(0.05)
        buffer.add_visual({"test": 3})
        
        history = buffer.get_history(seconds=1.0)
        
        assert len(history) == 3
        assert all(isinstance(obs, SensoryObservation) for obs in history)
    
    def test_get_timeline(self):
        """Test getting temporal timeline"""
        buffer = SensoryBuffer()
        
        buffer.add_visual({"test": 1})
        time.sleep(0.05)
        buffer.add_audio({"test": 2})
        
        timeline = buffer.get_timeline(seconds=1.0)
        
        assert "events" in timeline
        assert "visual_count" in timeline
        assert "audio_count" in timeline
        assert timeline["visual_count"] == 1
        assert timeline["audio_count"] == 1
        assert len(timeline["events"]) == 2
    
    def test_buffer_pruning(self):
        """Test old observations are pruned"""
        buffer = SensoryBuffer(max_size=100, max_age_seconds=0.5)
        
        # Add observation
        buffer.add_visual({"test": 1})
        
        # Wait for it to age beyond max_age
        time.sleep(0.7)
        
        # Add new observation (triggers prune)
        buffer.add_visual({"test": 2})
        time.sleep(0.2)
        
        # Force another prune
        buffer._maybe_prune()
        
        # Old observation should be pruned (only get recent ones)
        history = buffer.get_history(seconds=0.5)
        assert len(history) == 1  # Only recent observation within 0.5s window
    
    def test_clear_buffer(self):
        """Test clearing buffer"""
        buffer = SensoryBuffer()
        
        buffer.add_visual({"test": 1})
        buffer.add_audio({"test": 2})
        
        assert len(buffer.visual_buffer) > 0
        assert len(buffer.audio_buffer) > 0
        
        buffer.clear()
        
        assert len(buffer.visual_buffer) == 0
        assert len(buffer.audio_buffer) == 0


@pytest.mark.asyncio
class TestSensoryMemoryIntegration:
    """Test sensory memory integration"""
    
    async def test_store_visual_observation(self):
        """Test storing visual observation as atoms"""
        import tempfile
        import os
        
        # Use temporary database for testing
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()
        
        try:
            store = SQLiteGraphStore(db_path=temp_db.name)
            await store.connect()
            integrator = SensoryMemoryIntegrator(store)
            
            observation = {
                "timestamp": time.time(),
                "faces_detected": 1,
                "faces": [
                    {
                        "emotion": "happy",
                        "confidence": 0.8,
                        "position": {"x": 100, "y": 100, "w": 50, "h": 50}
                    }
                ],
                "scene": {
                    "description": "bright, static",
                    "brightness": 120,
                    "motion_detected": False
                }
            }
            
            atom_ids = await integrator.store_visual_observation(observation, user_id="test_user")
            
            assert len(atom_ids) > 0
            
            # Verify atoms were created
            atoms = await store.get_atoms_by_subject("test_user")
            sensory_atoms = [a for a in atoms if a.atom_type == AtomType.SENSORY_OBSERVATION]
            assert len(sensory_atoms) > 0
        finally:
            os.unlink(temp_db.name)
    
    async def test_store_audio_observation(self):
        """Test storing audio observation as atoms"""
        import tempfile
        import os
        
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()
        
        try:
            store = SQLiteGraphStore(db_path=temp_db.name)
            await store.connect()
            integrator = SensoryMemoryIntegrator(store)
            
            observation = {
                "timestamp": time.time(),
                "speech_detected": True,
                "transcript": "Hello world",
                "confidence": 0.9,
                "tone": "positive",
                "ambient": {
                    "volume": "moderate",
                    "sound_type": "mid_frequency",
                    "energy": 0.05
                }
            }
            
            atom_ids = await integrator.store_audio_observation(observation, user_id="test_user")
            
            assert len(atom_ids) > 0
        finally:
            os.unlink(temp_db.name)
    
    async def test_store_cross_modal_observation(self):
        """Test storing cross-modal correlations"""
        import tempfile
        import os
        
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()
        
        try:
            store = SQLiteGraphStore(db_path=temp_db.name)
            await store.connect()
            integrator = SensoryMemoryIntegrator(store)
            
            visual = {
                "faces": [{"emotion": "happy"}]
            }
            
            audio = {
                "tone": "happy",
                "transcript": "I'm excited!"
            }
            
            correlation = {
                "aligned": True,
                "time_delta": 0.5,
                "emotion_tone_match": True,
                "face_and_speech": True
            }
            
            atom_ids = await integrator.store_cross_modal_observation(
                visual, audio, correlation, user_id="test_user"
            )
            
            assert len(atom_ids) > 0
        finally:
            os.unlink(temp_db.name)
    
    async def test_get_recent_sensory_atoms(self):
        """Test retrieving recent sensory atoms"""
        import tempfile
        import os
        
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_db.close()
        
        try:
            store = SQLiteGraphStore(db_path=temp_db.name)
            await store.connect()
            integrator = SensoryMemoryIntegrator(store)
            
            # Store some observations
            visual_obs = {
                "timestamp": time.time(),
                "faces_detected": 1,
                "faces": [{"emotion": "neutral", "confidence": 0.7, "position": {"x": 0, "y": 0, "w": 50, "h": 50}}],
                "scene": {"description": "test", "brightness": 100, "motion_detected": False}
            }
            
            await integrator.store_visual_observation(visual_obs, user_id="test_user")
            
            # Retrieve recent atoms
            recent = await integrator.get_recent_sensory_atoms(
                user_id="test_user",
                seconds=60.0,
                modality="visual"
            )
            
            assert len(recent) > 0
            assert all(a.atom_type == AtomType.SENSORY_OBSERVATION for a in recent)
        finally:
            os.unlink(temp_db.name)


class TestSensoryObservation:
    """Test sensory observation data structure"""
    
    def test_observation_creation(self):
        """Test creating sensory observation"""
        obs = SensoryObservation(
            timestamp=time.time(),
            modality="visual",
            data={"test": "data"},
            confidence=0.8
        )
        
        assert obs.modality == "visual"
        assert obs.data["test"] == "data"
        assert obs.confidence == 0.8
    
    def test_observation_age(self):
        """Test observation age calculation"""
        obs = SensoryObservation(
            timestamp=time.time() - 5.0,
            modality="visual",
            data={}
        )
        
        age = obs.age_seconds()
        assert age >= 5.0
        assert age < 6.0
    
    def test_observation_to_dict(self):
        """Test converting observation to dictionary"""
        obs = SensoryObservation(
            timestamp=time.time(),
            modality="audio",
            data={"transcript": "test"},
            confidence=0.9
        )
        
        d = obs.to_dict()
        
        assert "timestamp" in d
        assert "modality" in d
        assert "data" in d
        assert "confidence" in d
        assert "age_seconds" in d
        assert d["modality"] == "audio"
        assert d["data"]["transcript"] == "test"
