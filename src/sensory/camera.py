"""
Camera Capture - Visual Sensory Input

Webcam capture with computer vision analysis:
- Face detection (OpenCV Haar cascades)
- Emotion estimation (from facial landmarks)
- Scene description (object detection)
- Periodic snapshots for temporal continuity
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any
import threading
import time
from pathlib import Path

from loguru import logger


class CameraCapture:
    """
    Webcam capture with CV analysis.
    
    Runs in background thread, periodically captures frames and analyzes:
    - Faces (count, positions, estimated emotions)
    - Scene (brightness, motion, objects)
    - Temporal changes
    """
    
    def __init__(self, camera_id: int = 0, fps: float = 1.0):
        """
        Initialize camera capture.
        
        Args:
            camera_id: Camera device ID (0 = default webcam)
            fps: Capture rate in frames per second
        """
        self.camera_id = camera_id
        self.fps = fps
        self.interval = 1.0 / fps
        
        self.capture: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Load Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load eye cascade for emotion hints
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        # Last observation
        self.last_observation: Optional[Dict[str, Any]] = None
        self.last_frame: Optional[np.ndarray] = None
        
        logger.info(f"CameraCapture initialized: camera_id={camera_id}, fps={fps}")
    
    def start(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Camera already running")
            return True
        
        # Open camera
        self.capture = cv2.VideoCapture(self.camera_id)
        if not self.capture.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return False
        
        # Set resolution (lower = faster processing)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        logger.info("Camera capture started")
        return True
    
    def stop(self) -> None:
        """Stop camera capture"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        logger.info("Camera capture stopped")
    
    def get_current_observation(self) -> Optional[Dict[str, Any]]:
        """Get most recent visual observation"""
        return self.last_observation
    
    def _capture_loop(self) -> None:
        """Background capture loop"""
        while self.running:
            start_time = time.time()
            
            try:
                # Capture frame
                ret, frame = self.capture.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(self.interval)
                    continue
                
                # Analyze frame
                observation = self._analyze_frame(frame)
                self.last_observation = observation
                self.last_frame = frame
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval - elapsed)
            time.sleep(sleep_time)
    
    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a single frame.
        
        Returns:
            Dictionary with visual analysis results
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Analyze each face
        face_data = []
        for (x, y, w, h) in faces:
            face_roi_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes in face region
            eyes = self.eye_cascade.detectMultiScale(face_roi_gray)
            
            # Estimate emotion from face region (simple heuristic)
            emotion = self._estimate_emotion(face_roi_gray, len(eyes))
            
            face_data.append({
                "emotion": emotion,
                "conf": 0.7
            })
        
        # Scene analysis
        scene = self._analyze_scene(frame, gray)
        
        return {
            "ts": time.time(),
            "n_faces": len(faces),
            "faces": face_data,
            "scene": scene
        }
    
    def _estimate_emotion(self, face_roi: np.ndarray, eyes_count: int) -> str:
        """
        Estimate emotion from face region (simple heuristic).
        
        This is a placeholder - in production would use a proper emotion model.
        For now, uses basic heuristics from face region characteristics.
        """
        # Calculate brightness (proxy for facial expression)
        brightness = np.mean(face_roi)
        
        # Calculate variance (proxy for facial detail/tension)
        variance = np.var(face_roi)
        
        # Simple heuristic rules
        if eyes_count >= 2:
            if brightness > 120:
                return "happy"  # Bright face, eyes visible
            elif variance > 1000:
                return "surprised"  # High variance, eyes visible
            else:
                return "neutral"
        elif eyes_count == 1:
            return "curious"  # One eye visible (head turned)
        else:
            if brightness < 80:
                return "serious"  # Dark face, no eyes
            else:
                return "neutral"
    
    def _analyze_scene(self, frame: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """
        Analyze overall scene characteristics.
        
        Returns:
            Dictionary with scene analysis
        """
        # Brightness
        brightness = np.mean(gray)
        
        # Contrast
        contrast = np.std(gray)
        
        # Motion detection (if we have previous frame)
        motion_detected = False
        motion_intensity = 0.0
        
        if self.last_frame is not None:
            try:
                last_gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
                frame_diff = cv2.absdiff(gray, last_gray)
                motion_intensity = np.mean(frame_diff)
                motion_detected = motion_intensity > 10.0
            except Exception:
                pass
        
        # Dominant colors (simple histogram)
        colors = self._get_dominant_colors(frame)
        
        # Scene description
        description = self._describe_scene(brightness, contrast, motion_detected)
        
        return {
            "desc": description,
            "motion": motion_detected
        }
    
    def _get_dominant_colors(self, frame: np.ndarray, n_colors: int = 3) -> List[str]:
        """
        Get dominant colors in frame.
        
        Returns:
            List of color names
        """
        # Calculate average color in each channel
        avg_b = np.mean(frame[:, :, 0])
        avg_g = np.mean(frame[:, :, 1])
        avg_r = np.mean(frame[:, :, 2])
        
        # Simple color classification
        colors = []
        
        if avg_r > avg_g and avg_r > avg_b:
            colors.append("red")
        if avg_g > avg_r and avg_g > avg_b:
            colors.append("green")
        if avg_b > avg_r and avg_b > avg_g:
            colors.append("blue")
        
        # Check for grayscale
        if abs(avg_r - avg_g) < 20 and abs(avg_g - avg_b) < 20:
            if avg_r > 150:
                colors.append("bright")
            elif avg_r < 80:
                colors.append("dark")
            else:
                colors.append("gray")
        
        return colors if colors else ["neutral"]
    
    def _describe_scene(self, brightness: float, contrast: float, motion: bool) -> str:
        """Generate text description of scene"""
        parts = []
        
        # Lighting
        if brightness > 150:
            parts.append("bright")
        elif brightness < 80:
            parts.append("dim")
        else:
            parts.append("moderate lighting")
        
        # Detail
        if contrast > 60:
            parts.append("high detail")
        elif contrast < 30:
            parts.append("low detail")
        
        # Motion
        if motion:
            parts.append("motion detected")
        else:
            parts.append("static")
        
        return ", ".join(parts)
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop()
