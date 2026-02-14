"""
Sensory Buffer - Temporal Integration Layer

Ring buffer of recent sensory observations with cross-modal correlation.
Provides unified multimodal state for Claude's grounded perception.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
import time

from loguru import logger


@dataclass
class SensoryObservation:
    """A single sensory observation (visual or audio)"""
    
    timestamp: float
    modality: str  # "visual" or "audio"
    data: Dict[str, Any]
    confidence: float = 0.8
    
    def age_seconds(self) -> float:
        """How old is this observation?"""
        return time.time() - self.timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "mod": self.modality,
            "data": self.data
        }


class SensoryBuffer:
    """
    Temporal integration buffer for multimodal sensory data.
    
    Maintains a rolling window of recent observations with:
    - Temporal coherence (timestamped observations)
    - Cross-modal correlation (visual + audio alignment)
    - Efficient access patterns (current state, recent history)
    """
    
    def __init__(self, max_size: int = 100, max_age_seconds: float = 60.0):
        """
        Initialize sensory buffer.
        
        Args:
            max_size: Maximum number of observations to keep
            max_age_seconds: Maximum age of observations before auto-pruning
        """
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        
        self.visual_buffer: deque = deque(maxlen=max_size)
        self.audio_buffer: deque = deque(maxlen=max_size)
        
        self._last_prune = time.time()
        
        logger.info(f"SensoryBuffer initialized: max_size={max_size}, max_age={max_age_seconds}s")
    
    def add_visual(self, data: Dict[str, Any], confidence: float = 0.8) -> None:
        """Add a visual observation"""
        obs = SensoryObservation(
            timestamp=time.time(),
            modality="visual",
            data=data,
            confidence=confidence
        )
        self.visual_buffer.append(obs)
        self._maybe_prune()
    
    def add_audio(self, data: Dict[str, Any], confidence: float = 0.8) -> None:
        """Add an audio observation"""
        obs = SensoryObservation(
            timestamp=time.time(),
            modality="audio",
            data=data,
            confidence=confidence
        )
        self.audio_buffer.append(obs)
        self._maybe_prune()
    
    def get_current_visual(self) -> Optional[SensoryObservation]:
        """Get most recent visual observation"""
        if not self.visual_buffer:
            return None
        return self.visual_buffer[-1]
    
    def get_current_audio(self) -> Optional[SensoryObservation]:
        """Get most recent audio observation"""
        if not self.audio_buffer:
            return None
        return self.audio_buffer[-1]
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get unified multimodal state (visual + audio + temporal context).
        
        Returns:
            Dictionary with current visual, audio, and cross-modal correlations
        """
        visual = self.get_current_visual()
        audio = self.get_current_audio()
        
        state = {
            "vis": visual.to_dict() if visual else None,
            "aud": audio.to_dict() if audio else None,
            "aligned": self._compute_cross_modal_correlation(visual, audio).get("aligned", False)
        }
        
        return state
    
    def get_history(self, seconds: float = 10.0, modality: Optional[str] = None) -> List[SensoryObservation]:
        """
        Get recent sensory history.
        
        Args:
            seconds: How far back to look
            modality: Filter by "visual", "audio", or None for both
        
        Returns:
            List of observations within the time window
        """
        cutoff = time.time() - seconds
        history = []
        
        if modality in (None, "visual"):
            history.extend([obs for obs in self.visual_buffer if obs.timestamp >= cutoff])
        
        if modality in (None, "audio"):
            history.extend([obs for obs in self.audio_buffer if obs.timestamp >= cutoff])
        
        # Sort by timestamp
        history.sort(key=lambda x: x.timestamp)
        
        return history
    
    def get_timeline(self, seconds: float = 10.0) -> Dict[str, Any]:
        """
        Get temporal timeline of recent observations.
        
        Returns:
            Dictionary with timeline events and temporal patterns
        """
        history = self.get_history(seconds)
        
        if not history:
            return {
                "duration_seconds": seconds,
                "events": [],
                "visual_count": 0,
                "audio_count": 0,
                "temporal_density": 0.0
            }
        
        events = [
            {
                "timestamp": obs.timestamp,
                "modality": obs.modality,
                "data": obs.data,
                "confidence": obs.confidence
            }
            for obs in history
        ]
        
        visual_count = sum(1 for obs in history if obs.modality == "visual")
        audio_count = sum(1 for obs in history if obs.modality == "audio")
        
        return {
            "events": events,
            "n_vis": visual_count,
            "n_aud": audio_count
        }
    
    def clear(self) -> None:
        """Clear all observations"""
        self.visual_buffer.clear()
        self.audio_buffer.clear()
        logger.info("SensoryBuffer cleared")
    
    def _compute_cross_modal_correlation(
        self, 
        visual: Optional[SensoryObservation], 
        audio: Optional[SensoryObservation]
    ) -> Dict[str, Any]:
        """
        Compute cross-modal correlations between visual and audio.
        
        Returns:
            Dictionary with correlation metrics and aligned observations
        """
        if not visual or not audio:
            return {"aligned": False, "time_delta": None}
        
        time_delta = abs(visual.timestamp - audio.timestamp)
        aligned = time_delta < 2.0  # Within 2 seconds = temporally aligned
        
        correlation = {
            "aligned": aligned,
            "time_delta": time_delta,
        }
        
        # Extract cross-modal patterns if aligned
        if aligned:
            visual_data = visual.data
            audio_data = audio.data
            
            # Example: face emotion + voice tone correlation
            if "emotion" in visual_data and "tone" in audio_data:
                correlation["emotion_tone_match"] = visual_data["emotion"] == audio_data["tone"]
            
            # Example: face detected + speech detected
            if "faces_detected" in visual_data and "speech_detected" in audio_data:
                correlation["face_and_speech"] = visual_data["faces_detected"] > 0 and audio_data["speech_detected"]
        
        return correlation
    
    def _maybe_prune(self) -> None:
        """Prune old observations if needed (rate-limited to once per second)"""
        now = time.time()
        if now - self._last_prune < 1.0:
            return
        
        self._last_prune = now
        cutoff = now - self.max_age_seconds
        
        # Prune visual buffer
        while self.visual_buffer and self.visual_buffer[0].timestamp < cutoff:
            self.visual_buffer.popleft()
        
        # Prune audio buffer
        while self.audio_buffer and self.audio_buffer[0].timestamp < cutoff:
            self.audio_buffer.popleft()
