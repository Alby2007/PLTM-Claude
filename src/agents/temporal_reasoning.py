"""
Temporal Reasoning & Prediction Engine

Uses memory decay and temporal patterns to predict future states and detect
temporal inconsistencies.

Key features:
- Predict future states based on decay patterns
- Detect temporal anomalies (sudden changes in behavior)
- Forecast interest decay ("likely lost interest in X")
- Temporal conflict detection (past vs present vs future)

Research potential: "Temporal Dynamics in AI Memory Systems"
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import math
from dataclasses import dataclass

from loguru import logger

from src.core.models import MemoryAtom
from src.pipeline.memory_pipeline import MemoryPipeline


@dataclass
class TemporalPrediction:
    """Prediction about future state"""
    atom_id: str
    prediction_type: str  # "decay", "revival", "anomaly"
    confidence: float
    predicted_date: datetime
    reasoning: str
    current_strength: float
    predicted_strength: float


@dataclass
class TemporalAnomaly:
    """Detected temporal anomaly"""
    atom_id: str
    anomaly_type: str  # "sudden_revival", "rapid_decay", "inconsistent_pattern"
    severity: float  # 0.0-1.0
    detected_at: datetime
    description: str


class TemporalReasoningEngine:
    """
    Analyzes temporal patterns in memory to make predictions and detect anomalies.
    
    Uses memory decay formula: R(t) = e^(-t/S)
    where R = retention, t = time since last access, S = strength
    """
    
    def __init__(self, memory_pipeline: MemoryPipeline):
        self.pipeline = memory_pipeline
        
        # Thresholds
        self.decay_threshold = 0.1  # Below this, consider "forgotten"
        self.revival_threshold = 0.5  # Above this after decay, consider "revived"
        self.anomaly_threshold = 0.3  # Change rate above this is anomalous
        
        logger.info("TemporalReasoningEngine initialized")
    
    async def predict_decay(
        self,
        user_id: str,
        days_ahead: int = 30
    ) -> List[TemporalPrediction]:
        """
        Predict which memories will decay below threshold in next N days.
        
        Use case: "User likely losing interest in Python if not mentioned in 30 days"
        """
        predictions = []
        
        # Get all user atoms
        atoms = await self.pipeline.store.get_atoms_by_subject(user_id)
        
        future_date = datetime.now() + timedelta(days=days_ahead)
        
        for atom in atoms:
            # Calculate current strength
            current_strength = self._calculate_retention(atom)
            
            # Calculate predicted strength after N days
            time_delta = (future_date - atom.last_accessed).total_seconds()
            predicted_strength = math.exp(-time_delta / (atom.strength * 86400))  # S in days
            
            # If will decay below threshold
            if current_strength > self.decay_threshold and predicted_strength < self.decay_threshold:
                predictions.append(TemporalPrediction(
                    atom_id=str(atom.id),
                    prediction_type="decay",
                    confidence=0.8,
                    predicted_date=future_date,
                    reasoning=f"Memory strength will decay from {current_strength:.2f} to {predicted_strength:.2f}",
                    current_strength=current_strength,
                    predicted_strength=predicted_strength
                ))
        
        return sorted(predictions, key=lambda p: p.predicted_strength)
    
    async def detect_anomalies(
        self,
        user_id: str,
        lookback_days: int = 7
    ) -> List[TemporalAnomaly]:
        """
        Detect temporal anomalies (sudden changes in access patterns).
        
        Use case: "User suddenly started mentioning Python again after 6 months"
        """
        anomalies = []
        
        atoms = await self.pipeline.store.get_atoms_by_subject(user_id)
        
        for atom in atoms:
            # Check for sudden revival
            time_since_access = (datetime.now() - atom.last_accessed).total_seconds() / 86400
            
            if time_since_access > 30 and atom.access_count > 0:  # Not accessed in 30 days
                # Check if recently accessed
                if time_since_access < lookback_days:
                    anomalies.append(TemporalAnomaly(
                        atom_id=str(atom.id),
                        anomaly_type="sudden_revival",
                        severity=0.8,
                        detected_at=datetime.now(),
                        description=f"Memory revived after {time_since_access:.0f} days of inactivity"
                    ))
            
            # Check for rapid decay
            if atom.access_count > 5:  # Was frequently accessed
                current_strength = self._calculate_retention(atom)
                if current_strength < 0.2:  # Now very weak
                    anomalies.append(TemporalAnomaly(
                        atom_id=str(atom.id),
                        anomaly_type="rapid_decay",
                        severity=0.6,
                        detected_at=datetime.now(),
                        description=f"Frequently accessed memory ({atom.access_count} times) now decaying rapidly"
                    ))
        
        return sorted(anomalies, key=lambda a: a.severity, reverse=True)
    
    async def predict_interest_shift(
        self,
        user_id: str,
        topic: str
    ) -> Tuple[str, float, str]:
        """
        Predict if user is gaining or losing interest in a topic.
        
        Returns:
            (trend, confidence, reasoning)
            trend: "increasing", "decreasing", "stable"
        """
        # Get atoms related to topic
        atoms = await self.pipeline.store.get_atoms_by_subject(user_id)
        topic_atoms = [a for a in atoms if topic.lower() in a.object.lower()]
        
        if not topic_atoms:
            return "unknown", 0.0, f"No memories found for topic: {topic}"
        
        # Calculate average access recency
        avg_recency = sum(
            (datetime.now() - a.last_accessed).total_seconds() / 86400
            for a in topic_atoms
        ) / len(topic_atoms)
        
        # Calculate average strength
        avg_strength = sum(self._calculate_retention(a) for a in topic_atoms) / len(topic_atoms)
        
        # Determine trend
        if avg_recency < 7 and avg_strength > 0.7:
            return "increasing", 0.9, f"Recently accessed ({avg_recency:.1f} days ago) with high retention ({avg_strength:.2f})"
        elif avg_recency > 30 and avg_strength < 0.3:
            return "decreasing", 0.8, f"Not accessed in {avg_recency:.0f} days, low retention ({avg_strength:.2f})"
        else:
            return "stable", 0.6, f"Moderate activity (last: {avg_recency:.1f} days, strength: {avg_strength:.2f})"
    
    async def forecast_memory_consolidation(
        self,
        user_id: str
    ) -> Dict[str, List[str]]:
        """
        Forecast which memories will consolidate (move to long-term) vs decay.
        
        Returns:
            {
                "will_consolidate": [atom_ids],
                "will_decay": [atom_ids],
                "uncertain": [atom_ids]
            }
        """
        atoms = await self.pipeline.store.get_atoms_by_subject(user_id)
        
        forecast = {
            "will_consolidate": [],
            "will_decay": [],
            "uncertain": []
        }
        
        for atom in atoms:
            strength = self._calculate_retention(atom)
            access_frequency = atom.access_count / max(1, (datetime.now() - atom.first_observed).days)
            
            # High strength + frequent access = consolidate
            if strength > 0.7 and access_frequency > 0.1:
                forecast["will_consolidate"].append(str(atom.id))
            
            # Low strength + rare access = decay
            elif strength < 0.3 and access_frequency < 0.01:
                forecast["will_decay"].append(str(atom.id))
            
            # Everything else = uncertain
            else:
                forecast["uncertain"].append(str(atom.id))
        
        return forecast
    
    def _calculate_retention(self, atom: MemoryAtom) -> float:
        """
        Calculate current retention using decay formula.
        
        R(t) = e^(-t/S)
        where t = time since last access (in days)
              S = strength parameter (in days)
        """
        time_delta = (datetime.now() - atom.last_accessed).total_seconds() / 86400  # Convert to days
        
        # Avoid division by zero
        if atom.strength == 0:
            return 0.0
        
        retention = math.exp(-time_delta / atom.strength)
        
        return max(0.0, min(1.0, retention))  # Clamp to [0, 1]


class TemporalReasoningExperiment:
    """
    Experiment framework for temporal reasoning research.
    
    Research questions:
    1. Can we predict user interest decay?
    2. Can we detect behavioral anomalies?
    3. Can we forecast memory consolidation?
    """
    
    def __init__(self, memory_pipeline: MemoryPipeline):
        self.engine = TemporalReasoningEngine(memory_pipeline)
        self.results = []
    
    async def run_decay_prediction_experiment(
        self,
        user_id: str,
        days_ahead: int = 30
    ) -> Dict:
        """
        Experiment: Predict which memories will decay.
        
        Metrics:
        - Number of predictions
        - Confidence distribution
        - Predicted decay timeline
        """
        predictions = await self.engine.predict_decay(user_id, days_ahead)
        
        result = {
            "experiment": "decay_prediction",
            "user_id": user_id,
            "days_ahead": days_ahead,
            "num_predictions": len(predictions),
            "predictions": [
                {
                    "atom_id": p.atom_id,
                    "type": p.prediction_type,
                    "confidence": p.confidence,
                    "current_strength": p.current_strength,
                    "predicted_strength": p.predicted_strength,
                    "reasoning": p.reasoning
                }
                for p in predictions[:10]  # Top 10
            ]
        }
        
        self.results.append(result)
        return result
    
    async def run_anomaly_detection_experiment(
        self,
        user_id: str,
        lookback_days: int = 7
    ) -> Dict:
        """
        Experiment: Detect temporal anomalies.
        
        Metrics:
        - Number of anomalies
        - Severity distribution
        - Anomaly types
        """
        anomalies = await self.engine.detect_anomalies(user_id, lookback_days)
        
        result = {
            "experiment": "anomaly_detection",
            "user_id": user_id,
            "lookback_days": lookback_days,
            "num_anomalies": len(anomalies),
            "anomalies": [
                {
                    "atom_id": a.atom_id,
                    "type": a.anomaly_type,
                    "severity": a.severity,
                    "description": a.description
                }
                for a in anomalies
            ]
        }
        
        self.results.append(result)
        return result
    
    async def run_interest_tracking_experiment(
        self,
        user_id: str,
        topics: List[str]
    ) -> Dict:
        """
        Experiment: Track interest trends across multiple topics.
        
        Metrics:
        - Trend distribution (increasing/decreasing/stable)
        - Confidence levels
        - Topic comparisons
        """
        trends = {}
        
        for topic in topics:
            trend, confidence, reasoning = await self.engine.predict_interest_shift(user_id, topic)
            trends[topic] = {
                "trend": trend,
                "confidence": confidence,
                "reasoning": reasoning
            }
        
        result = {
            "experiment": "interest_tracking",
            "user_id": user_id,
            "topics": topics,
            "trends": trends
        }
        
        self.results.append(result)
        return result
    
    def get_summary(self) -> Dict:
        """Get summary of all experiments"""
        return {
            "total_experiments": len(self.results),
            "experiments": self.results
        }
