"""
Sensory Input System for PLTM

Real-time multimodal perception:
- Camera: Visual observation (faces, objects, scenes)
- Microphone: Audio observation (speech, tone, ambient)
- Buffer: Temporal integration and cross-modal correlation

Enables grounded sensory memory atoms for increased Φ.
"""

from src.sensory.sensory_buffer import SensoryBuffer, SensoryObservation
from src.sensory.camera import CameraCapture
from src.sensory.microphone import MicrophoneCapture

__all__ = [
    "SensoryBuffer",
    "SensoryObservation",
    "CameraCapture",
    "MicrophoneCapture",
]
