"""
Quantum-Inspired Memory Superposition

Universal Principle #2: Parallel Processing (Superposition)

Instead of immediately resolving conflicts, hold multiple contradictory
memories in superposition until a query forces "measurement" (collapse).

Benefits:
- Preserves maximum information
- Defers computation until needed
- Allows context-dependent truth
- Mirrors quantum mechanics principle
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import hashlib

from src.storage.sqlite_store import SQLiteGraphStore
from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
from loguru import logger


class SuperpositionState(str, Enum):
    """State of a memory superposition"""
    SUPERPOSED = "superposed"  # Multiple states held simultaneously
    COLLAPSED = "collapsed"    # Resolved to single state
    ENTANGLED = "entangled"    # Linked to other superpositions


@dataclass
class QuantumMemory:
    """A memory in superposition - multiple possible states"""
    memory_id: str
    subject: str
    predicate: str
    possible_states: List[Dict[str, Any]]  # Multiple possible values
    state: SuperpositionState
    created_at: datetime = field(default_factory=datetime.now)
    collapsed_value: Optional[str] = None
    collapse_context: Optional[str] = None
    entangled_with: List[str] = field(default_factory=list)


class QuantumMemorySystem:
    """
    Quantum-inspired memory that holds contradictions in superposition.
    
    Key insight: Don't resolve conflicts until you NEED to.
    This preserves information and allows context-dependent truth.
    
    Like Schrödinger's cat - the memory exists in multiple states
    until observation (query) forces collapse.
    """
    
    # Configuration
    MAX_SUPERPOSITIONS = 1000  # LRU cache limit
    MAX_COLLAPSE_HISTORY = 500
    PRUNE_AGE_HOURS = 24  # Auto-prune collapsed states older than this
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self.superpositions: OrderedDict[str, QuantumMemory] = OrderedDict()  # LRU cache
        self.collapse_history: List[Dict[str, Any]] = []
        
        logger.info("QuantumMemorySystem initialized - superposition enabled")
    
    def _touch_superposition(self, sp_id: str) -> None:
        """Move superposition to end of LRU cache (most recently used)"""
        if sp_id in self.superpositions:
            self.superpositions.move_to_end(sp_id)
    
    def _enforce_lru_limit(self) -> int:
        """Remove oldest superpositions if over limit. Returns count removed."""
        removed = 0
        while len(self.superpositions) > self.MAX_SUPERPOSITIONS:
            oldest_id, oldest = self.superpositions.popitem(last=False)
            # Only remove if collapsed (preserve active superpositions)
            if oldest.state == SuperpositionState.COLLAPSED:
                removed += 1
            else:
                # Put it back, remove next oldest
                self.superpositions[oldest_id] = oldest
                self.superpositions.move_to_end(oldest_id, last=False)
                break
        return removed
    
    async def cleanup_old_states(self) -> Dict[str, Any]:
        """
        Garbage collect old collapsed states.
        
        Removes:
        - Collapsed states older than PRUNE_AGE_HOURS
        - Excess history entries
        """
        cutoff = datetime.now() - timedelta(hours=self.PRUNE_AGE_HOURS)
        
        # Find old collapsed states
        to_remove = []
        for sp_id, sp in self.superpositions.items():
            if sp.state == SuperpositionState.COLLAPSED and sp.created_at < cutoff:
                to_remove.append(sp_id)
        
        # Remove them
        for sp_id in to_remove:
            del self.superpositions[sp_id]
        
        # Trim history
        history_trimmed = 0
        if len(self.collapse_history) > self.MAX_COLLAPSE_HISTORY:
            history_trimmed = len(self.collapse_history) - self.MAX_COLLAPSE_HISTORY
            self.collapse_history = self.collapse_history[-self.MAX_COLLAPSE_HISTORY:]
        
        return {"p": len(to_remove), "h": history_trimmed}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get quantum memory statistics"""
        s, c, e = 0, 0, 0
        for sp in self.superpositions.values():
            if sp.state == SuperpositionState.SUPERPOSED: s += 1
            elif sp.state == SuperpositionState.COLLAPSED: c += 1
            else: e += 1
        
        return {"n": len(self.superpositions), "s": s, "c": c, "e": e, "h": len(self.collapse_history)}
    
    def _generate_superposition_id(self, subject: str, predicate: str) -> str:
        """Generate unique ID for a superposition"""
        content = f"{subject}:{predicate}"
        return f"qm_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
    
    async def add_to_superposition(
        self,
        subject: str,
        predicate: str,
        value: str,
        confidence: float,
        source: str
    ) -> Dict[str, Any]:
        """
        Add a memory state to superposition.
        
        If conflicting values exist for same subject+predicate,
        they're held in superposition rather than resolved.
        """
        sp_id = self._generate_superposition_id(subject, predicate)
        
        new_state = {
            "value": value,
            "confidence": confidence,
            "source": source,
            "added_at": datetime.now().isoformat()
        }
        
        if sp_id in self.superpositions:
            # Add to existing superposition
            sp = self.superpositions[sp_id]
            self._touch_superposition(sp_id)  # LRU update
            
            # Check if this value already exists
            existing_values = [s["value"] for s in sp.possible_states]
            if value not in existing_values:
                sp.possible_states.append(new_state)
                sp.state = SuperpositionState.SUPERPOSED
                logger.info(f"Added state to superposition {sp_id}: now {len(sp.possible_states)} states")
        else:
            # Create new superposition
            sp = QuantumMemory(
                memory_id=sp_id,
                subject=subject,
                predicate=predicate,
                possible_states=[new_state],
                state=SuperpositionState.SUPERPOSED
            )
            self.superpositions[sp_id] = sp
            self._enforce_lru_limit()  # Cleanup if over limit
        
        return {"id": sp_id, "n": len(self.superpositions[sp_id].possible_states)}
    
    async def query_with_collapse(
        self,
        subject: str,
        predicate: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query a superposition - forces collapse to single value.
        
        The collapse is context-dependent:
        - With context: Choose state most relevant to context
        - Without context: Choose highest confidence state
        
        This is the "measurement" that collapses the wave function.
        """
        sp_id = self._generate_superposition_id(subject, predicate)
        
        if sp_id not in self.superpositions:
            return {"found": False, "sp_id": sp_id}
        
        sp = self.superpositions[sp_id]
        self._touch_superposition(sp_id)  # LRU update
        
        if sp.state == SuperpositionState.COLLAPSED:
            # Already collapsed - return cached value
            return {
                "found": True,
                "value": sp.collapsed_value,
                "was_superposed": False,
                "collapse_context": sp.collapse_context
            }
        
        # Perform collapse
        collapsed_value, collapse_reason = await self._collapse_superposition(sp, context)
        
        # Record collapse
        sp.collapsed_value = collapsed_value
        sp.collapse_context = context
        sp.state = SuperpositionState.COLLAPSED
        
        self.collapse_history.append({
            "sp_id": sp_id,
            "collapsed_to": collapsed_value,
            "context": context,
            "reason": collapse_reason,
            "timestamp": datetime.now().isoformat(),
            "states_before": len(sp.possible_states)
        })
        
        return {"v": collapsed_value, "from": len(sp.possible_states), "why": collapse_reason[:30]}
    
    async def _collapse_superposition(
        self,
        sp: QuantumMemory,
        context: Optional[str]
    ) -> Tuple[str, str]:
        """
        Collapse superposition to single value.
        
        Collapse rules:
        1. If context provided: Score states by context relevance
        2. If no context: Use confidence-weighted selection
        3. Tie-breaker: Most recent state
        """
        states = sp.possible_states
        
        if len(states) == 1:
            return states[0]["value"], "single_state"
        
        if context:
            # Context-dependent collapse
            scored_states = []
            context_lower = context.lower()
            
            for state in states:
                score = state["confidence"]
                value_lower = state["value"].lower()
                
                # Boost if value relates to context
                if any(word in value_lower for word in context_lower.split()):
                    score *= 1.5
                
                scored_states.append((state, score))
            
            best = max(scored_states, key=lambda x: x[1])
            return best[0]["value"], f"context_match:{context[:30]}"
        else:
            # Confidence-based collapse
            best = max(states, key=lambda s: s["confidence"])
            return best["value"], "highest_confidence"
    
    async def peek_superposition(
        self,
        subject: str,
        predicate: str
    ) -> Dict[str, Any]:
        """
        Peek at superposition WITHOUT collapsing.
        
        Returns all possible states - useful for understanding
        the uncertainty before committing to a value.
        """
        sp_id = self._generate_superposition_id(subject, predicate)
        
        if sp_id not in self.superpositions:
            return {"found": False}
        
        sp = self.superpositions[sp_id]
        self._touch_superposition(sp_id)  # LRU update
        
        return {
            "n": len(sp.possible_states),
            "vals": [s["value"][:50] for s in sp.possible_states[:5]],
            "collapsed": sp.collapsed_value
        }
    
    async def entangle_memories(
        self,
        sp_id_1: str,
        sp_id_2: str,
        relationship: str
    ) -> Dict[str, Any]:
        """
        Entangle two superpositions - when one collapses, it affects the other.
        
        Like quantum entanglement: correlated states that influence each other.
        """
        if sp_id_1 not in self.superpositions or sp_id_2 not in self.superpositions:
            return {"ok": False, "error": "Superposition not found"}
        
        sp1 = self.superpositions[sp_id_1]
        sp2 = self.superpositions[sp_id_2]
        
        if sp_id_2 not in sp1.entangled_with:
            sp1.entangled_with.append(sp_id_2)
        if sp_id_1 not in sp2.entangled_with:
            sp2.entangled_with.append(sp_id_1)
        
        sp1.state = SuperpositionState.ENTANGLED
        sp2.state = SuperpositionState.ENTANGLED
        
        return {
            "ok": True,
            "entangled": [sp_id_1, sp_id_2],
            "relationship": relationship
        }
    
    async def get_uncertainty(self, user_id: str) -> Dict[str, Any]:
        """
        Get overall uncertainty in the memory system.
        
        High uncertainty = many unresolved superpositions
        Low uncertainty = most memories collapsed
        """
        user_superpositions = [
            sp for sp in self.superpositions.values()
            if sp.subject.startswith(user_id) or user_id in sp.subject
        ]
        
        total = len(user_superpositions)
        superposed = sum(1 for sp in user_superpositions if sp.state == SuperpositionState.SUPERPOSED)
        collapsed = sum(1 for sp in user_superpositions if sp.state == SuperpositionState.COLLAPSED)
        entangled = sum(1 for sp in user_superpositions if sp.state == SuperpositionState.ENTANGLED)
        
        total_states = sum(len(sp.possible_states) for sp in user_superpositions)
        
        uncertainty = superposed / total if total > 0 else 0
        
        return {"total": total, "open": superposed, "closed": collapsed, "unc": round(uncertainty, 2)}
    
    def _interpret_uncertainty(self, ratio: float) -> str:
        """Interpret uncertainty ratio"""
        if ratio < 0.2:
            return "Low uncertainty - most knowledge is definite"
        elif ratio < 0.5:
            return "Moderate uncertainty - some unresolved contradictions"
        elif ratio < 0.8:
            return "High uncertainty - many competing beliefs"
        else:
            return "Very high uncertainty - knowledge is highly fluid"
    
    async def force_decoherence(self, user_id: str) -> Dict[str, Any]:
        """
        Force all superpositions to collapse (decoherence).
        
        Use when you need definite answers for everything.
        Warning: This loses information!
        """
        collapsed_count = 0
        
        for sp_id, sp in self.superpositions.items():
            if sp.state == SuperpositionState.SUPERPOSED:
                if sp.subject.startswith(user_id) or user_id in sp.subject:
                    # Force collapse with no context (highest confidence wins)
                    value, reason = await self._collapse_superposition(sp, None)
                    sp.collapsed_value = value
                    sp.state = SuperpositionState.COLLAPSED
                    collapsed_count += 1
        
        return {
            "decoherence_complete": True,
            "collapsed_count": collapsed_count,
            "warning": "Information may have been lost"
        }
