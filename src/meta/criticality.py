"""
Self-Organized Criticality Monitoring

Universal Principle #5: Critical Point Operation

Keep the system at the "edge of chaos" - the optimal zone where:
- Too much order = rigid, unable to learn
- Too much chaos = unstable, unable to retain
- Critical point = maximum adaptability + stability

This is the missing piece for AGI architecture.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import math

from src.storage.sqlite_store import SQLiteGraphStore
from loguru import logger


@dataclass
class CriticalityState:
    """Current criticality metrics"""
    entropy: float  # Disorder measure (0-1)
    integration: float  # Order measure (0-1)
    criticality_ratio: float  # entropy/integration
    zone: str  # "subcritical", "critical", "supercritical"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CriticalityConfig:
    """Configuration for criticality maintenance"""
    critical_ratio_min: float = 0.8  # Below = too ordered
    critical_ratio_max: float = 1.2  # Above = too chaotic
    entropy_weight: float = 0.5
    integration_weight: float = 0.5
    adjustment_rate: float = 0.1  # How fast to adjust toward criticality


class SelfOrganizedCriticality:
    """
    Monitor and maintain system at edge of chaos.
    
    Key insight: Complex adaptive systems perform best at critical points
    where phase transitions occur. This is where:
    - Information propagates optimally
    - Learning is maximized
    - Emergence happens
    
    Like a sandpile at critical angle - one grain can cause avalanche.
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self.config = CriticalityConfig()
        self.history: List[CriticalityState] = []
        self.adjustments: List[Dict[str, Any]] = []
        
        logger.info("SelfOrganizedCriticality initialized - edge of chaos monitoring enabled")
    
    async def measure_entropy(self) -> float:
        """
        Measure system entropy (disorder).
        
        High entropy indicators:
        - Many unconnected concepts
        - High variance in confidence scores
        - Many unresolved superpositions
        - Diverse, scattered knowledge
        """
        atoms = await self.store.get_all_atoms()
        if not atoms:
            return 0.5  # Neutral
        
        # Factor 1: Confidence variance (high variance = high entropy)
        confidences = [a.confidence for a in atoms if hasattr(a, 'confidence')]
        if confidences:
            mean_conf = sum(confidences) / len(confidences)
            variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
            conf_entropy = min(1.0, variance * 4)  # Normalize
        else:
            conf_entropy = 0.5
        
        # Factor 2: Type diversity (many types = higher entropy)
        types = set(a.atom_type.value if hasattr(a, 'atom_type') else 'unknown' for a in atoms)
        type_entropy = min(1.0, len(types) / 10)  # Normalize to 10 types
        
        # Factor 3: Recency spread (old + new = higher entropy)
        timestamps = [a.created_at for a in atoms if hasattr(a, 'created_at')]
        if len(timestamps) > 1:
            timestamps_sorted = sorted(timestamps)
            time_range = (timestamps_sorted[-1] - timestamps_sorted[0]).total_seconds()
            time_entropy = min(1.0, time_range / (86400 * 30))  # Normalize to 30 days
        else:
            time_entropy = 0.5
        
        # Weighted combination
        entropy = (conf_entropy * 0.4 + type_entropy * 0.3 + time_entropy * 0.3)
        return round(entropy, 3)
    
    async def measure_integration(self) -> float:
        """
        Measure information integration (order).
        
        High integration indicators:
        - Dense knowledge graph connections
        - Consistent confidence scores
        - Resolved conflicts
        - Coherent knowledge clusters
        """
        atoms = await self.store.get_all_atoms()
        if not atoms:
            return 0.5  # Neutral
        
        # Factor 1: Average confidence (high = more integrated)
        confidences = [a.confidence for a in atoms if hasattr(a, 'confidence')]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Factor 2: Subject clustering (repeated subjects = more integrated)
        subjects = [a.subject for a in atoms if hasattr(a, 'subject')]
        if subjects:
            unique_ratio = len(set(subjects)) / len(subjects)
            clustering = 1 - unique_ratio  # More repeats = more integration
        else:
            clustering = 0.5
        
        # Factor 3: Predicate consistency (common predicates = more integrated)
        predicates = [a.predicate for a in atoms if hasattr(a, 'predicate')]
        if predicates:
            pred_counts = {}
            for p in predicates:
                pred_counts[p] = pred_counts.get(p, 0) + 1
            max_pred = max(pred_counts.values())
            pred_consistency = min(1.0, max_pred / len(predicates) * 5)
        else:
            pred_consistency = 0.5
        
        # Weighted combination
        integration = (avg_confidence * 0.4 + clustering * 0.3 + pred_consistency * 0.3)
        return round(integration, 3)
    
    async def get_criticality_state(self) -> Dict[str, Any]:
        """
        Get current criticality state.
        
        Returns zone classification:
        - subcritical: Too ordered, needs more exploration
        - critical: Optimal learning zone
        - supercritical: Too chaotic, needs consolidation
        """
        entropy = await self.measure_entropy()
        integration = await self.measure_integration()
        
        # Avoid division by zero
        ratio = entropy / max(integration, 0.01)
        
        # Classify zone
        if ratio < self.config.critical_ratio_min:
            zone = "subcritical"
        elif ratio > self.config.critical_ratio_max:
            zone = "supercritical"
        else:
            zone = "critical"
        
        state = CriticalityState(
            entropy=entropy,
            integration=integration,
            criticality_ratio=round(ratio, 3),
            zone=zone
        )
        
        # Track history (keep last 100)
        self.history.append(state)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        return {"e": entropy, "i": integration, "r": round(ratio, 3), "z": zone}
    
    async def get_adjustment_recommendation(self) -> Dict[str, Any]:
        """
        Get recommendation for maintaining criticality.
        
        Returns specific actions to move toward critical point.
        """
        state = await self.get_criticality_state()
        zone = state["z"]
        
        if zone == "critical":
            return {"ok": True, "action": "maintain", "msg": "At critical point"}
        
        elif zone == "subcritical":
            # Too ordered - need more exploration/diversity
            return {
                "ok": False,
                "action": "explore",
                "recommendations": [
                    "Add diverse knowledge sources",
                    "Explore new domains",
                    "Reduce confidence thresholds",
                    "Allow more superposition states"
                ],
                "params": {"exploration_boost": self.config.adjustment_rate}
            }
        
        else:  # supercritical
            # Too chaotic - need consolidation
            return {
                "ok": False,
                "action": "consolidate",
                "recommendations": [
                    "Resolve superposition states",
                    "Strengthen high-confidence memories",
                    "Prune low-confidence atoms",
                    "Focus on core domains"
                ],
                "params": {"consolidation_boost": self.config.adjustment_rate}
            }
    
    async def auto_adjust(self) -> Dict[str, Any]:
        """
        Automatically adjust system toward criticality.
        
        This is the key AGI capability: self-regulation to optimal learning zone.
        """
        recommendation = await self.get_adjustment_recommendation()
        
        if recommendation.get("ok"):
            return {"adjusted": False, "reason": "Already at critical point"}
        
        action = recommendation["action"]
        
        if action == "explore":
            # Increase entropy by lowering confidence thresholds
            adjustment = {
                "type": "exploration",
                "timestamp": datetime.now().isoformat(),
                "effect": "Lowered consolidation pressure"
            }
        else:
            # Decrease entropy by increasing consolidation
            adjustment = {
                "type": "consolidation", 
                "timestamp": datetime.now().isoformat(),
                "effect": "Increased integration pressure"
            }
        
        self.adjustments.append(adjustment)
        if len(self.adjustments) > 50:
            self.adjustments = self.adjustments[-50:]
        
        return {"adjusted": True, "action": action, "effect": adjustment["effect"]}
    
    async def get_criticality_history(self) -> Dict[str, Any]:
        """Get history of criticality states"""
        if not self.history:
            return {"n": 0, "history": []}
        
        # Summarize recent history
        recent = self.history[-10:]
        zones = [s.zone for s in recent]
        zone_counts = {"subcritical": 0, "critical": 0, "supercritical": 0}
        for z in zones:
            zone_counts[z] = zone_counts.get(z, 0) + 1
        
        avg_ratio = sum(s.criticality_ratio for s in recent) / len(recent)
        
        return {
            "n": len(self.history),
            "recent": zone_counts,
            "avg_r": round(avg_ratio, 3),
            "trend": self._calculate_trend()
        }
    
    def _calculate_trend(self) -> str:
        """Calculate criticality trend"""
        if len(self.history) < 3:
            return "insufficient_data"
        
        recent = self.history[-5:]
        ratios = [s.criticality_ratio for s in recent]
        
        # Simple trend detection
        if ratios[-1] > ratios[0] * 1.1:
            return "increasing_chaos"
        elif ratios[-1] < ratios[0] * 0.9:
            return "increasing_order"
        else:
            return "stable"
