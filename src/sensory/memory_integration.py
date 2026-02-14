"""
Memory Integration for Sensory Data

Converts sensory observations into memory atoms for PLTM storage.
Enables grounded perceptual memories with cross-modal integration.
"""

from typing import Dict, Any, List
from datetime import datetime
import time

from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
from src.storage.sqlite_store import SQLiteGraphStore
from src.learning.universal_learning import SourceType
from loguru import logger


class SensoryMemoryIntegrator:
    """
    Integrates sensory observations into PLTM memory system.
    
    Converts visual and audio observations into memory atoms with:
    - Grounded perceptual data (faces, speech, emotions)
    - Cross-modal correlations (face + voice alignment)
    - Temporal context (when observations occurred)
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        logger.info("SensoryMemoryIntegrator initialized")
    
    async def store_visual_observation(
        self, 
        observation: Dict[str, Any],
        user_id: str = "alby"
    ) -> List[str]:
        """
        Store visual observation as memory atoms.
        
        Returns:
            List of atom IDs created
        """
        atom_ids = []
        timestamp = observation.get("timestamp", time.time())
        
        # Store face detections
        for i, face in enumerate(observation.get("faces", [])):
            emotion = face.get("emotion", "neutral")
            confidence = face.get("confidence", 0.7)
            
            # Create sensory observation atom
            atom = MemoryAtom(
                atom_type=AtomType.SENSORY_OBSERVATION,
                subject=user_id,
                predicate="visual_face_detected",
                object=f"emotion:{emotion}",
                confidence=confidence,
                strength=confidence,
                provenance=Provenance.INFERRED,
                source_user="sensory_camera",
                contexts=[
                    "visual",
                    "face_detection",
                    f"timestamp:{timestamp}",
                    f"position:x{face['position']['x']}y{face['position']['y']}"
                ],
                graph=GraphType.UNSUBSTANTIATED
            )
            
            atom_id = await self.store.add_atom(atom)
            atom_ids.append(atom_id)
        
        # Store scene description
        scene = observation.get("scene", {})
        if scene:
            description = scene.get("description", "")
            
            atom = MemoryAtom(
                atom_type=AtomType.SENSORY_OBSERVATION,
                subject=user_id,
                predicate="visual_scene",
                object=description,
                confidence=0.8,
                strength=0.8,
                provenance=Provenance.INFERRED,
                source_user="sensory_camera",
                contexts=[
                    "visual",
                    "scene_description",
                    f"timestamp:{timestamp}",
                    f"brightness:{scene.get('brightness', 0)}",
                    f"motion:{scene.get('motion_detected', False)}"
                ],
                graph=GraphType.UNSUBSTANTIATED
            )
            
            atom_id = await self.store.add_atom(atom)
            atom_ids.append(atom_id)
        
        return atom_ids
    
    async def store_audio_observation(
        self,
        observation: Dict[str, Any],
        user_id: str = "alby"
    ) -> List[str]:
        """
        Store audio observation as memory atoms.
        
        Returns:
            List of atom IDs created
        """
        atom_ids = []
        timestamp = observation.get("timestamp", time.time())
        
        # Store speech transcript if detected
        if observation.get("speech_detected"):
            transcript = observation.get("transcript", "")
            tone = observation.get("tone", "neutral")
            confidence = observation.get("confidence", 0.7)
            
            if transcript:
                # Create speech atom
                atom = MemoryAtom(
                    atom_type=AtomType.SENSORY_OBSERVATION,
                    subject=user_id,
                    predicate="audio_speech",
                    object=transcript,
                    confidence=confidence,
                    strength=confidence,
                    provenance=Provenance.INFERRED,
                    source_user="sensory_microphone",
                    contexts=[
                        "audio",
                        "speech",
                        f"timestamp:{timestamp}",
                        f"tone:{tone}"
                    ],
                    graph=GraphType.UNSUBSTANTIATED
                )
                
                atom_id = await self.store.add_atom(atom)
                atom_ids.append(atom_id)
        
        # Store ambient sound characteristics
        ambient = observation.get("ambient", {})
        if ambient:
            sound_type = ambient.get("sound_type", "unknown")
            volume = ambient.get("volume", "quiet")
            
            atom = MemoryAtom(
                atom_type=AtomType.SENSORY_OBSERVATION,
                subject=user_id,
                predicate="audio_ambient",
                object=f"{volume}_{sound_type}",
                confidence=0.6,
                strength=0.6,
                provenance=Provenance.INFERRED,
                source_user="sensory_microphone",
                contexts=[
                    "audio",
                    "ambient",
                    f"timestamp:{timestamp}",
                    f"energy:{ambient.get('energy', 0)}"
                ],
                graph=GraphType.UNSUBSTANTIATED
            )
            
            atom_id = await self.store.add_atom(atom)
            atom_ids.append(atom_id)
        
        return atom_ids
    
    async def store_cross_modal_observation(
        self,
        visual: Dict[str, Any],
        audio: Dict[str, Any],
        correlation: Dict[str, Any],
        user_id: str = "alby"
    ) -> List[str]:
        """
        Store cross-modal correlations as memory atoms.
        
        Creates high-integration atoms that link visual and audio modalities.
        
        Returns:
            List of atom IDs created
        """
        atom_ids = []
        
        if not correlation.get("aligned"):
            return atom_ids  # Only store if temporally aligned
        
        timestamp = time.time()
        
        # Check for emotion-tone correlation
        if correlation.get("emotion_tone_match"):
            # Extract emotion and tone
            faces = visual.get("faces", [])
            if faces:
                emotion = faces[0].get("emotion", "neutral")
                tone = audio.get("tone", "neutral")
                
                # Create cross-modal atom
                atom = MemoryAtom(
                    atom_type=AtomType.SENSORY_OBSERVATION,
                    subject=user_id,
                    predicate="cross_modal_emotion_tone",
                    object=f"face:{emotion}_voice:{tone}",
                    confidence=0.85,  # Higher confidence for aligned observations
                    strength=0.85,
                    provenance=Provenance.INFERRED,
                    source_user="sensory_integration",
                    contexts=[
                        "cross_modal",
                        "emotion_tone_correlation",
                        f"timestamp:{timestamp}",
                        f"time_delta:{correlation.get('time_delta', 0)}"
                    ],
                    graph=GraphType.UNSUBSTANTIATED
                )
                
                atom_id = await self.store.add_atom(atom)
                atom_ids.append(atom_id)
        
        # Check for face + speech correlation
        if correlation.get("face_and_speech"):
            atom = MemoryAtom(
                atom_type=AtomType.SENSORY_OBSERVATION,
                subject=user_id,
                predicate="cross_modal_face_speech",
                object="user_present_and_speaking",
                confidence=0.9,
                strength=0.9,
                provenance=Provenance.INFERRED,
                source_user="sensory_integration",
                contexts=[
                    "cross_modal",
                    "face_speech_correlation",
                    f"timestamp:{timestamp}",
                    f"transcript:{audio.get('transcript', '')[:100]}"
                ],
                graph=GraphType.UNSUBSTANTIATED
            )
            
            atom_id = await self.store.add_atom(atom)
            atom_ids.append(atom_id)
        
        return atom_ids
    
    async def get_recent_sensory_atoms(
        self,
        user_id: str = "alby",
        seconds: float = 60.0,
        modality: str = None
    ) -> List[MemoryAtom]:
        """
        Retrieve recent sensory observation atoms.
        
        Args:
            user_id: User to query
            seconds: How far back to look
            modality: Filter by "visual", "audio", "cross_modal", or None for all
        
        Returns:
            List of sensory observation atoms
        """
        # Get all atoms for user
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        
        # Filter to sensory observations
        sensory_atoms = [
            atom for atom in all_atoms
            if atom.atom_type == AtomType.SENSORY_OBSERVATION
        ]
        
        # Filter by modality if specified
        if modality:
            sensory_atoms = [
                atom for atom in sensory_atoms
                if modality in atom.contexts
            ]
        
        # Filter by time (using contexts timestamp)
        cutoff = time.time() - seconds
        recent_atoms = []
        
        for atom in sensory_atoms:
            # Extract timestamp from contexts
            for context in atom.contexts:
                if context.startswith("timestamp:"):
                    try:
                        ts = float(context.split(":")[1])
                        if ts >= cutoff:
                            recent_atoms.append(atom)
                            break
                    except (ValueError, IndexError):
                        continue
        
        return recent_atoms
