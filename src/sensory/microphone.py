"""
Microphone Capture - Audio Sensory Input

Microphone capture with audio analysis:
- Speech-to-text (Whisper)
- Tone/sentiment analysis
- Ambient sound classification
- Continuous recording in chunks
"""

import sounddevice as sd
import numpy as np
from typing import Dict, List, Optional, Any
import threading
import time
import queue
from io import BytesIO

from loguru import logger


class MicrophoneCapture:
    """
    Microphone capture with audio analysis.
    
    Runs in background thread, continuously records audio chunks and analyzes:
    - Speech transcription (Whisper)
    - Tone/sentiment (from transcribed text)
    - Ambient sound level
    - Voice activity detection
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000,
        chunk_duration: float = 5.0,
        device: Optional[int] = None
    ):
        """
        Initialize microphone capture.
        
        Args:
            sample_rate: Audio sample rate in Hz (16kHz for Whisper)
            chunk_duration: Duration of each audio chunk in seconds
            device: Audio device ID (None = default)
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.device = device
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.audio_queue: queue.Queue = queue.Queue(maxsize=10)
        
        # Whisper model (lazy load)
        self._whisper_model = None
        self._whisper_processor = None
        
        # Last observation
        self.last_observation: Optional[Dict[str, Any]] = None
        
        logger.info(f"MicrophoneCapture initialized: sample_rate={sample_rate}, chunk={chunk_duration}s")
    
    def start(self) -> bool:
        """
        Start microphone capture.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Microphone already running")
            return True
        
        # Test audio device
        try:
            sd.check_input_settings(device=self.device, samplerate=self.sample_rate)
        except Exception as e:
            logger.error(f"Failed to access microphone: {e}")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        logger.info("Microphone capture started")
        return True
    
    def stop(self) -> None:
        """Stop microphone capture"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        logger.info("Microphone capture stopped")
    
    def get_current_observation(self) -> Optional[Dict[str, Any]]:
        """Get most recent audio observation"""
        return self.last_observation
    
    def _capture_loop(self) -> None:
        """Background capture loop"""
        # Audio callback
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            # Add to queue (non-blocking)
            try:
                self.audio_queue.put_nowait(indata.copy())
            except queue.Full:
                pass  # Drop frame if queue full
        
        # Start audio stream
        with sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_samples,
            callback=audio_callback
        ):
            while self.running:
                try:
                    # Get audio chunk from queue
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                    
                    # Analyze chunk
                    observation = self._analyze_audio(audio_chunk)
                    self.last_observation = observation
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in capture loop: {e}")
    
    def _analyze_audio(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Analyze audio chunk.
        
        Returns:
            Dictionary with audio analysis results
        """
        # Flatten to 1D if needed
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()
        
        # Voice activity detection (simple energy-based)
        energy = np.mean(np.abs(audio_chunk))
        speech_detected = energy > 0.01  # Threshold for speech vs silence
        
        # Transcribe if speech detected
        transcript = ""
        confidence = 0.0
        
        if speech_detected:
            try:
                transcript, confidence = self._transcribe_audio(audio_chunk)
            except Exception as e:
                logger.debug(f"Transcription failed: {e}")
        
        # Analyze tone/sentiment from transcript
        tone = self._analyze_tone(transcript) if transcript else "neutral"
        
        # Ambient sound analysis
        ambient = self._analyze_ambient(audio_chunk, energy)
        
        return {
            "ts": time.time(),
            "speech": speech_detected,
            "text": transcript,
            "tone": tone
        }
    
    def _transcribe_audio(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Transcribe audio using Whisper.
        
        Returns:
            (transcript, confidence)
        """
        # Lazy load Whisper model
        if self._whisper_model is None:
            try:
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                
                # Use tiny model for speed (can upgrade to base if needed)
                model_name = "openai/whisper-tiny"
                logger.info(f"Loading Whisper model: {model_name}")
                
                self._whisper_processor = WhisperProcessor.from_pretrained(model_name)
                self._whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
                
                logger.info("Whisper model loaded")
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
                return "", 0.0
        
        try:
            # Prepare audio for Whisper (expects 16kHz)
            input_features = self._whisper_processor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_features
            
            # Generate transcription
            predicted_ids = self._whisper_model.generate(input_features)
            transcription = self._whisper_processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            # Simple confidence heuristic (length-based)
            confidence = min(0.9, len(transcription) / 100.0)
            
            return transcription.strip(), confidence
            
        except Exception as e:
            logger.debug(f"Whisper transcription error: {e}")
            return "", 0.0
    
    def _analyze_tone(self, text: str) -> str:
        """
        Analyze tone/sentiment from text.
        
        Uses simple keyword-based heuristics (could upgrade to sentiment model).
        """
        if not text:
            return "neutral"
        
        text_lower = text.lower()
        
        # Positive indicators
        positive_words = ["happy", "great", "good", "excellent", "wonderful", "love", "yes", "thanks"]
        positive_count = sum(1 for word in positive_words if word in text_lower)
        
        # Negative indicators
        negative_words = ["sad", "bad", "terrible", "hate", "no", "angry", "frustrated", "upset"]
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Excited indicators
        excited_words = ["wow", "amazing", "incredible", "awesome", "!", "?!"]
        excited_count = sum(1 for word in excited_words if word in text_lower)
        
        # Question indicators
        is_question = "?" in text
        
        # Classify tone
        if excited_count > 0:
            return "excited"
        elif positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        elif is_question:
            return "curious"
        else:
            return "neutral"
    
    def _analyze_ambient(self, audio: np.ndarray, energy: float) -> Dict[str, Any]:
        """
        Analyze ambient sound characteristics.
        
        Returns:
            Dictionary with ambient sound analysis
        """
        # Volume level
        if energy > 0.1:
            volume = "loud"
        elif energy > 0.01:
            volume = "moderate"
        else:
            volume = "quiet"
        
        # Frequency analysis (simple)
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # Find dominant frequency
        dominant_idx = np.argmax(np.abs(fft))
        dominant_freq = freqs[dominant_idx]
        
        # Classify sound type by frequency
        if dominant_freq < 200:
            sound_type = "low_frequency"  # Rumble, bass
        elif dominant_freq < 2000:
            sound_type = "mid_frequency"  # Voice, music
        else:
            sound_type = "high_frequency"  # Whistle, beep
        
        return {
            "vol": volume,
            "type": sound_type
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop()
