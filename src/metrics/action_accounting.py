"""
True Action Accounting for PLTM

Implements Georgiev's Average Action Efficiency (AAE) framework
with REAL computational cost tracking, not proxy metrics.

AAE = events / total_action
unit_action = total_action / events

Where:
- events = successful operations (hypothesis validated, memory stored, etc.)
- total_action = sum of (tokens * latency_weight * complexity_weight)

This replaces the proxy metric (validated/hypotheses) with true measurements.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import math
import json

from loguru import logger


@dataclass
class ActionRecord:
    """Single action/operation record"""
    record_id: str
    operation: str          # hypothesis_gen, memory_store, retrieval, etc.
    tokens_used: int        # Actual token count
    latency_ms: float       # Wall-clock time
    success: bool           # Did it achieve its goal?
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Optional metadata
    cycle_id: Optional[str] = None
    context: Optional[str] = None
    
    @property
    def action_cost(self) -> float:
        """
        Compute action cost for this operation.
        
        Action = tokens * latency_weight
        
        Latency weight: penalizes slow operations
        - <100ms: weight = 1.0
        - 100-500ms: weight = 1.0 + (latency-100)/400
        - >500ms: weight = 2.0 + (latency-500)/1000
        """
        if self.latency_ms < 100:
            latency_weight = 1.0
        elif self.latency_ms < 500:
            latency_weight = 1.0 + (self.latency_ms - 100) / 400
        else:
            latency_weight = 2.0 + (self.latency_ms - 500) / 1000
        
        return self.tokens_used * latency_weight
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.record_id,
            "op": self.operation,
            "tok": self.tokens_used,
            "ms": self.latency_ms,
            "ok": self.success,
            "cost": round(self.action_cost, 1),
            "t": self.timestamp.isoformat()
        }


@dataclass
class AAEMetrics:
    """Aggregated AAE metrics for a time window"""
    total_events: int           # Successful operations
    total_action: float         # Sum of action costs
    aae: float                  # events / total_action
    unit_action: float          # total_action / events
    window_start: datetime
    window_end: datetime
    by_operation: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "events": self.total_events,
            "action": round(self.total_action, 1),
            "aae": round(self.aae, 4),
            "unit": round(self.unit_action, 2),
            "by_op": {k: {"n": v["events"], "aae": round(v["aae"], 4)} 
                      for k, v in self.by_operation.items()}
        }


class ActionAccounting:
    """
    True action accounting system.
    
    Tracks every operation's computational cost and success rate
    to compute real AAE metrics per Georgiev's framework.
    """
    
    MAX_HISTORY = 10000  # Keep last 10k records
    
    def __init__(self):
        self.records: deque = deque(maxlen=self.MAX_HISTORY)
        self.cycle_records: Dict[str, List[ActionRecord]] = {}
        self.current_cycle: Optional[str] = None
        self._record_counter = 0
        
        logger.info("ActionAccounting initialized - true AAE tracking active")
    
    def start_cycle(self, cycle_id: str) -> None:
        """Start a new measurement cycle"""
        self.current_cycle = cycle_id
        self.cycle_records[cycle_id] = []
        logger.debug(f"Started cycle {cycle_id}")
    
    def end_cycle(self, cycle_id: Optional[str] = None) -> AAEMetrics:
        """End cycle and compute metrics"""
        cid = cycle_id or self.current_cycle
        if cid and cid in self.cycle_records:
            metrics = self._compute_metrics(self.cycle_records[cid])
            self.current_cycle = None
            return metrics
        return self._compute_metrics([])
    
    def record(
        self,
        operation: str,
        tokens_used: int,
        latency_ms: float,
        success: bool,
        context: Optional[str] = None
    ) -> ActionRecord:
        """
        Record an action/operation.
        
        Args:
            operation: Type of operation (hypothesis_gen, memory_store, etc.)
            tokens_used: Actual token count consumed
            latency_ms: Wall-clock time in milliseconds
            success: Whether operation achieved its goal
            context: Optional context string
        
        Returns:
            The recorded ActionRecord
        """
        self._record_counter += 1
        record = ActionRecord(
            record_id=f"act_{self._record_counter}_{datetime.now().strftime('%H%M%S')}",
            operation=operation,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            success=success,
            cycle_id=self.current_cycle,
            context=context
        )
        
        self.records.append(record)
        
        if self.current_cycle and self.current_cycle in self.cycle_records:
            self.cycle_records[self.current_cycle].append(record)
        
        logger.debug(f"Recorded action: {operation} tok={tokens_used} ms={latency_ms} ok={success}")
        return record
    
    def get_aae(self, last_n: Optional[int] = None) -> AAEMetrics:
        """
        Get current AAE metrics.
        
        Args:
            last_n: Only consider last N records (None = all)
        
        Returns:
            AAEMetrics with current state
        """
        if last_n:
            records = list(self.records)[-last_n:]
        else:
            records = list(self.records)
        
        return self._compute_metrics(records)
    
    def get_aae_by_cycle(self, cycle_id: str) -> AAEMetrics:
        """Get AAE metrics for a specific cycle"""
        if cycle_id in self.cycle_records:
            return self._compute_metrics(self.cycle_records[cycle_id])
        return self._compute_metrics([])
    
    def _compute_metrics(self, records: List[ActionRecord]) -> AAEMetrics:
        """Compute AAE metrics from records"""
        if not records:
            return AAEMetrics(
                total_events=0,
                total_action=0.0,
                aae=0.0,
                unit_action=float('inf'),
                window_start=datetime.now(),
                window_end=datetime.now()
            )
        
        # Count events (successful operations)
        events = sum(1 for r in records if r.success)
        
        # Sum action costs
        total_action = sum(r.action_cost for r in records)
        
        # Compute AAE and unit_action
        aae = events / total_action if total_action > 0 else 0.0
        unit_action = total_action / events if events > 0 else float('inf')
        
        # Breakdown by operation type
        by_op: Dict[str, Dict[str, Any]] = {}
        for r in records:
            if r.operation not in by_op:
                by_op[r.operation] = {"events": 0, "action": 0.0}
            if r.success:
                by_op[r.operation]["events"] += 1
            by_op[r.operation]["action"] += r.action_cost
        
        for op in by_op:
            e = by_op[op]["events"]
            a = by_op[op]["action"]
            by_op[op]["aae"] = e / a if a > 0 else 0.0
        
        return AAEMetrics(
            total_events=events,
            total_action=total_action,
            aae=aae,
            unit_action=unit_action,
            window_start=records[0].timestamp,
            window_end=records[-1].timestamp,
            by_operation=by_op
        )
    
    def get_history(self, last_n: int = 20) -> List[Dict[str, Any]]:
        """Get recent action history"""
        records = list(self.records)[-last_n:]
        return [r.to_dict() for r in records]
    
    def get_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """
        Get AAE trend over recent windows.
        
        Returns trend direction and magnitude.
        """
        records = list(self.records)
        if len(records) < window_size * 2:
            return {"trend": "insufficient_data", "windows": 0}
        
        # Split into windows
        n_windows = min(5, len(records) // window_size)
        windows = []
        
        for i in range(n_windows):
            start = len(records) - (n_windows - i) * window_size
            end = start + window_size
            window_records = records[start:end]
            metrics = self._compute_metrics(window_records)
            windows.append(metrics.aae)
        
        # Compute trend
        if len(windows) >= 2:
            first_half = sum(windows[:len(windows)//2]) / (len(windows)//2)
            second_half = sum(windows[len(windows)//2:]) / (len(windows) - len(windows)//2)
            
            if first_half > 0:
                change = (second_half - first_half) / first_half
            else:
                change = 0.0
            
            if change > 0.05:
                trend = "improving"
            elif change < -0.05:
                trend = "declining"
            else:
                trend = "stable"
            
            return {
                "trend": trend,
                "change": round(change, 3),
                "windows": n_windows,
                "latest_aae": round(windows[-1], 4)
            }
        
        return {"trend": "insufficient_data", "windows": len(windows)}


# Global instance
_action_accounting: Optional[ActionAccounting] = None

def get_action_accounting() -> ActionAccounting:
    """Get or create global ActionAccounting instance"""
    global _action_accounting
    if _action_accounting is None:
        _action_accounting = ActionAccounting()
    return _action_accounting
