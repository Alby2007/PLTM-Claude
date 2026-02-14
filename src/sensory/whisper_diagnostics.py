"""
Whisper Diagnostics - Capture detailed transcription attempts
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

DIAGNOSTICS_FILE = Path(__file__).parent.parent.parent / "whisper_diagnostics.json"


class WhisperDiagnostics:
    """Track Whisper transcription attempts for debugging"""
    
    @staticmethod
    def log_attempt(
        audio_shape: tuple,
        audio_dtype: str,
        audio_min: float,
        audio_max: float,
        energy: float,
        success: bool,
        transcription: str = "",
        error: str = "",
        raw_output: str = ""
    ):
        """Log a transcription attempt"""
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "audio": {
                "shape": str(audio_shape),
                "dtype": audio_dtype,
                "min": float(audio_min),
                "max": float(audio_max),
                "energy": float(energy)
            },
            "success": success,
            "transcription": transcription,
            "raw_output": raw_output,
            "error": error
        }
        
        # Append to diagnostics file
        try:
            if DIAGNOSTICS_FILE.exists():
                with open(DIAGNOSTICS_FILE, 'r') as f:
                    data = json.load(f)
            else:
                data = {"attempts": []}
            
            data["attempts"].append(entry)
            
            # Keep only last 50 attempts
            data["attempts"] = data["attempts"][-50:]
            
            with open(DIAGNOSTICS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            # Don't let diagnostics break the main flow
            print(f"Failed to write diagnostics: {e}")
    
    @staticmethod
    def get_recent_attempts(n: int = 10) -> list:
        """Get recent transcription attempts"""
        try:
            if DIAGNOSTICS_FILE.exists():
                with open(DIAGNOSTICS_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get("attempts", [])[-n:]
            return []
        except Exception:
            return []
    
    @staticmethod
    def clear():
        """Clear diagnostics file"""
        if DIAGNOSTICS_FILE.exists():
            DIAGNOSTICS_FILE.unlink()
