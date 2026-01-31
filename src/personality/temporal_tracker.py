"""
Temporal Personality Tracker

Tracks how personality traits evolve over time, detecting trends,
inflection points, and predicting future trajectories.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import statistics

from src.storage.sqlite_store import SQLiteGraphStore
from src.core.models import MemoryAtom, AtomType
from loguru import logger


class TemporalPersonalityTracker:
    """
    Track how personality evolves over time.
    
    Enables:
    - Trait evolution tracking
    - Trend detection (increasing, decreasing, stable)
    - Inflection point detection
    - Trajectory prediction
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
    
    async def track_trait_evolution(
        self,
        user_id: str,
        trait: str,
        window_days: int = 90
    ) -> Dict[str, Any]:
        """
        Show how a trait has changed over time.
        
        Returns:
            Timeline, trend, inflection points, trajectory prediction
        """
        # Get all atoms for this user
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        
        # Filter to atoms mentioning this trait
        trait_atoms = [
            a for a in all_atoms
            if trait.lower() in a.object.lower() or trait.lower() in a.predicate.lower()
        ]
        
        if not trait_atoms:
            return {
                "trait": trait,
                "timeline": [],
                "trend": "unknown",
                "message": f"No data found for trait: {trait}"
            }
        
        # Build timeline
        timeline = []
        for atom in sorted(trait_atoms, key=lambda a: a.first_observed):
            timeline.append({
                "date": atom.first_observed.isoformat() if atom.first_observed else None,
                "confidence": atom.confidence,
                "strength": atom.strength,
                "context": atom.contexts,
                "value": atom.object
            })
        
        # Detect trend
        trend = self._detect_trend(timeline)
        
        # Find inflection points
        inflection_points = self._find_inflections(timeline)
        
        # Predict trajectory
        trajectory = self._predict_trajectory(timeline)
        
        return {
            "trait": trait,
            "timeline": timeline,
            "data_points": len(timeline),
            "trend": trend,
            "inflection_points": inflection_points,
            "trajectory": trajectory,
            "first_observed": timeline[0]["date"] if timeline else None,
            "last_observed": timeline[-1]["date"] if timeline else None
        }
    
    def _detect_trend(self, timeline: List[Dict]) -> Dict[str, Any]:
        """Detect if trait is increasing, decreasing, or stable"""
        if len(timeline) < 2:
            return {"direction": "insufficient_data", "strength": 0.0}
        
        confidences = [t["confidence"] for t in timeline]
        
        # Calculate trend using linear regression approximation
        n = len(confidences)
        if n < 2:
            return {"direction": "stable", "strength": 0.0}
        
        # Simple slope calculation
        first_half = statistics.mean(confidences[:n//2]) if n >= 2 else confidences[0]
        second_half = statistics.mean(confidences[n//2:]) if n >= 2 else confidences[-1]
        
        change = second_half - first_half
        
        if abs(change) < 0.05:
            direction = "stable"
        elif change > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        return {
            "direction": direction,
            "strength": abs(change),
            "change": change,
            "first_half_avg": first_half,
            "second_half_avg": second_half
        }
    
    def _find_inflections(self, timeline: List[Dict]) -> List[Dict]:
        """Find points where trait significantly changed"""
        if len(timeline) < 3:
            return []
        
        inflections = []
        confidences = [t["confidence"] for t in timeline]
        
        for i in range(1, len(confidences) - 1):
            prev_change = confidences[i] - confidences[i-1]
            next_change = confidences[i+1] - confidences[i]
            
            # Inflection if direction changes significantly
            if (prev_change > 0.1 and next_change < -0.1) or \
               (prev_change < -0.1 and next_change > 0.1):
                inflections.append({
                    "index": i,
                    "date": timeline[i]["date"],
                    "confidence_at_inflection": confidences[i],
                    "type": "peak" if prev_change > 0 else "trough",
                    "context": timeline[i].get("context", [])
                })
        
        return inflections
    
    def _predict_trajectory(self, timeline: List[Dict]) -> Dict[str, Any]:
        """Predict future trajectory based on current trend"""
        if len(timeline) < 3:
            return {"prediction": "insufficient_data", "confidence": 0.0}
        
        trend = self._detect_trend(timeline)
        current = timeline[-1]["confidence"]
        
        # Simple linear extrapolation
        if trend["direction"] == "increasing":
            predicted_30d = min(1.0, current + trend["strength"])
            predicted_90d = min(1.0, current + trend["strength"] * 3)
        elif trend["direction"] == "decreasing":
            predicted_30d = max(0.0, current - trend["strength"])
            predicted_90d = max(0.0, current - trend["strength"] * 3)
        else:
            predicted_30d = current
            predicted_90d = current
        
        return {
            "current": current,
            "predicted_30d": predicted_30d,
            "predicted_90d": predicted_90d,
            "trend_direction": trend["direction"],
            "confidence_in_prediction": 0.6 if len(timeline) > 5 else 0.4
        }
    
    async def get_all_trait_trends(
        self,
        user_id: str,
        min_data_points: int = 2
    ) -> Dict[str, Any]:
        """Get trends for all tracked traits"""
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        
        # Group by trait/object
        trait_groups = defaultdict(list)
        for atom in all_atoms:
            if atom.atom_type in [AtomType.PERSONALITY_TRAIT, AtomType.COMMUNICATION_STYLE]:
                trait_groups[atom.object].append(atom)
        
        trends = {}
        for trait, atoms in trait_groups.items():
            if len(atoms) >= min_data_points:
                timeline = [
                    {"confidence": a.confidence, "date": a.first_observed}
                    for a in sorted(atoms, key=lambda x: x.first_observed or datetime.min)
                ]
                trends[trait] = self._detect_trend(timeline)
        
        return {
            "user_id": user_id,
            "traits_tracked": len(trends),
            "trends": trends,
            "increasing": [t for t, d in trends.items() if d["direction"] == "increasing"],
            "decreasing": [t for t, d in trends.items() if d["direction"] == "decreasing"],
            "stable": [t for t, d in trends.items() if d["direction"] == "stable"]
        }
    
    async def detect_learning_curve(
        self,
        user_id: str,
        domain: str
    ) -> Dict[str, Any]:
        """
        Detect learning curve in a specific domain.
        
        Useful for predicting: "Based on past learning curves, 
        Alby will master X in ~2 weeks"
        """
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        
        # Filter to domain-related atoms
        domain_atoms = [
            a for a in all_atoms
            if domain.lower() in str(a.contexts).lower() or 
               domain.lower() in a.object.lower()
        ]
        
        if len(domain_atoms) < 3:
            return {
                "domain": domain,
                "learning_curve": "insufficient_data",
                "data_points": len(domain_atoms)
            }
        
        # Sort by time
        sorted_atoms = sorted(domain_atoms, key=lambda x: x.first_observed or datetime.min)
        
        # Calculate expertise growth
        confidences = [a.confidence for a in sorted_atoms]
        
        # Detect curve type
        if len(confidences) >= 4:
            early_growth = confidences[len(confidences)//4] - confidences[0]
            late_growth = confidences[-1] - confidences[3*len(confidences)//4]
            
            if early_growth > late_growth * 1.5:
                curve_type = "exponential_early"  # Fast start, slowing
            elif late_growth > early_growth * 1.5:
                curve_type = "exponential_late"  # Slow start, accelerating
            else:
                curve_type = "linear"
        else:
            curve_type = "unknown"
        
        return {
            "domain": domain,
            "learning_curve": curve_type,
            "data_points": len(domain_atoms),
            "current_level": confidences[-1] if confidences else 0,
            "growth_rate": (confidences[-1] - confidences[0]) / len(confidences) if len(confidences) > 1 else 0,
            "time_to_mastery_estimate": self._estimate_mastery_time(confidences)
        }
    
    def _estimate_mastery_time(self, confidences: List[float], mastery_threshold: float = 0.9) -> Optional[str]:
        """Estimate time to reach mastery level"""
        if not confidences or len(confidences) < 2:
            return None
        
        current = confidences[-1]
        if current >= mastery_threshold:
            return "already_mastered"
        
        # Calculate average growth rate
        growth_rate = (confidences[-1] - confidences[0]) / len(confidences)
        
        if growth_rate <= 0:
            return "not_progressing"
        
        # Estimate data points needed
        remaining = mastery_threshold - current
        points_needed = remaining / growth_rate
        
        # Assume ~1 data point per interaction session
        if points_needed < 5:
            return "~1 week"
        elif points_needed < 15:
            return "~2-3 weeks"
        elif points_needed < 30:
            return "~1 month"
        else:
            return "~2+ months"
