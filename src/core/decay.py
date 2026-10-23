"""
Ebbinghaus decay engine for realistic memory forgetting.

Implements the Ebbinghaus forgetting curve with type-specific decay rates
and memory reconsolidation mechanics.

Key Formula: R(t) = e^(-t/S)
Where:
  R = Retrieval probability (0-1)
  t = Time since last access (hours)
  S = Strength (decay_rate * confidence)
"""

import math
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from loguru import logger

from src.core.models import MemoryAtom, GraphType
from src.core.ontology import get_decay_rate


class DecayEngine:
    """
    Implements Ebbinghaus forgetting curve with type-specific decay rates.
    
    Biological Inspiration:
    - Different memory types decay at different rates (semantic vs episodic)
    - Retrieval strengthens memories (reconsolidation)
    - Unused memories fade over time
    - Repeated access increases stability
    
    Type-Specific Decay Rates (from ontology):
    - ENTITY: 0.01 (identity rarely changes)
    - AFFILIATION: 0.03 (jobs change slowly)
    - PREFERENCE: 0.08 (preferences change)
    - STATE: 0.50 (states are volatile)
    - INVARIANT: 0.00 (never decay)
    """
    
    def __init__(self):
        """Initialize decay engine"""
        self.stability_checks = 0
        self.reconsolidations = 0
        self.dissolutions = 0
        logger.info("DecayEngine initialized with type-specific decay rates")
    
    def calculate_stability(self, atom: MemoryAtom) -> float:
        """
        Calculate current retrieval probability using Ebbinghaus curve.
        
        Formula: R(t) = e^(-t/S)
        
        Args:
            atom: Memory atom to calculate stability for
            
        Returns:
            Stability score (0.0 = forgotten, 1.0 = perfect recall)
            
        Example:
            ENTITY atom (decay=0.01) with confidence=0.9:
              After 1 day: 0.99 (barely decayed)
              After 1 month: 0.97 (still strong)
              
            STATE atom (decay=0.50) with confidence=0.8:
              After 1 hour: 0.95 (already fading)
              After 1 day: 0.61 (significantly decayed)
        """
        self.stability_checks += 1
        
        # Get type-specific decay rate from ontology
        decay_rate = get_decay_rate(atom.atom_type)
        
        # INVARIANT atoms never decay
        if decay_rate == 0.0:
            return 1.0
        
        # Calculate time since last access
        hours_elapsed = (
            datetime.now() - atom.last_accessed
        ).total_seconds() / 3600
        
        # Strength parameter (adjusted by confidence)
        # Higher confidence = stronger memory = slower decay
        # S = conf * 8 / rate² keeps ordering across widely different decay
        # rates: ENTITY (0.01) persists for months while STATE (0.50) fades
        # within a day. Linear rate scaling cannot produce that spread.
        confidence = max(atom.confidence, 0.01)
        strength = (confidence * 8.0) / max(decay_rate * decay_rate, 1e-9)
        
        # Ebbinghaus forgetting curve
        stability = math.exp(-hours_elapsed / strength)
        
        # Clamp to [0, 1]
        stability = max(0.0, min(1.0, stability))
        
        logger.debug(
            f"Stability for {atom.atom_type.value} atom {atom.id}: {stability:.3f} "
            f"(elapsed: {hours_elapsed:.1f}h, decay_rate: {decay_rate}, "
            f"confidence: {atom.confidence:.2f})"
        )
        
        return stability
    
    def should_dissolve(self, atom: MemoryAtom, threshold: float = 0.1) -> bool:
        """
        Determine if atom should be deleted from unsubstantiated graph.
        
        Rule: Dissolve if stability < threshold AND in unsubstantiated graph
        
        Substantiated atoms are never dissolved - they're moved to historical
        graph when superseded, but not deleted due to decay.
        
        Args:
            atom: Memory atom to check
            threshold: Dissolution threshold (default: 0.1 = 10% stability)
            
        Returns:
            True if atom should be dissolved
        """
        # Only unsubstantiated atoms can be dissolved
        if atom.graph != GraphType.UNSUBSTANTIATED:
            return False
        
        # Check stability
        stability = self.calculate_stability(atom)
        
        if stability < threshold:
            logger.info(
                f"Atom {atom.id} ({atom.atom_type.value}) marked for dissolution "
                f"(stability: {stability:.3f} < {threshold})"
            )
            self.dissolutions += 1
            return True
        
        return False
    
    def reconsolidate(
        self,
        atom: MemoryAtom,
        boost_factor: float = 1.5
    ) -> None:
        """
        Strengthen memory on retrieval (memory reconsolidation).
        
        Biological Basis:
        Human memories get stronger when recalled. This is called
        reconsolidation - the memory is re-encoded with increased strength.
        
        Args:
            atom: Memory atom to reconsolidate
            boost_factor: Strength multiplier (default: 1.5 = 50% boost)
            
        Side Effects:
            - Increases atom confidence (capped at 1.0)
            - Resets last_accessed timestamp
            - Increments access_count
        """
        old_confidence = atom.confidence
        
        # Boost confidence (capped at 1.0)
        atom.confidence = min(1.0, atom.confidence * boost_factor)
        
        # Reset decay timer
        atom.last_accessed = datetime.now()
        
        # Track access count
        if not hasattr(atom, 'access_count'):
            atom.access_count = 0
        atom.access_count += 1
        
        self.reconsolidations += 1
        
        logger.debug(
            f"Reconsolidated atom {atom.id} ({atom.atom_type.value}): "
            f"confidence {old_confidence:.2f} → {atom.confidence:.2f}, "
            f"access_count: {atom.access_count}"
        )
    
    def get_decay_schedule(self, atom: MemoryAtom) -> Dict[str, Optional[datetime]]:
        """
        Calculate when atom will reach key stability thresholds.
        
        Useful for:
        - Predicting when memories will fade
        - Scheduling reconsolidation reminders
        - Planning background decay processing
        
        Args:
            atom: Memory atom to analyze
            
        Returns:
            Dict mapping threshold names to datetime when reached
            
        Example:
            {
                "50%": datetime(2026, 2, 15, 10, 30),
                "25%": datetime(2026, 3, 1, 14, 20),
                "10%": datetime(2026, 3, 15, 8, 45)  # Dissolution point
            }
        """
        thresholds = {
            "90%": 0.9,
            "75%": 0.75,
            "50%": 0.5,
            "25%": 0.25,
            "10%": 0.1,  # Dissolution point
        }
        
        decay_rate = get_decay_rate(atom.atom_type)
        
        # INVARIANT atoms never decay
        if decay_rate == 0.0:
            return {"never_decays": None}
        
        schedule = {}
        strength = decay_rate * atom.confidence * 100  # Scale to hours
        
        for name, threshold in thresholds.items():
            # Solve for t: threshold = e^(-t/S)
            # t = -S * ln(threshold)
            hours_to_threshold = -strength * math.log(threshold)
            target_time = atom.last_accessed + timedelta(hours=hours_to_threshold)
            schedule[name] = target_time
        
        return schedule
    
    def get_time_to_dissolution(self, atom: MemoryAtom) -> Optional[timedelta]:
        """
        Calculate time until atom will be dissolved.
        
        Args:
            atom: Memory atom to analyze
            
        Returns:
            Timedelta until dissolution, or None if never dissolves
        """
        if atom.graph != GraphType.UNSUBSTANTIATED:
            return None  # Substantiated atoms don't dissolve
        
        decay_rate = get_decay_rate(atom.atom_type)
        
        if decay_rate == 0.0:
            return None  # INVARIANT atoms never decay
        
        # Calculate time to 10% stability (dissolution threshold)
        strength = decay_rate * atom.confidence * 100
        hours_to_dissolution = -strength * math.log(0.1)
        
        # Subtract time already elapsed
        hours_elapsed = (datetime.now() - atom.last_accessed).total_seconds() / 3600
        hours_remaining = hours_to_dissolution - hours_elapsed
        
        if hours_remaining <= 0:
            return timedelta(0)  # Already past dissolution point
        
        return timedelta(hours=hours_remaining)
    
    def batch_calculate_stability(
        self,
        atoms: List[MemoryAtom]
    ) -> List[tuple[MemoryAtom, float]]:
        """
        Calculate stability for multiple atoms efficiently.
        
        Args:
            atoms: List of memory atoms
            
        Returns:
            List of (atom, stability) tuples, sorted by stability (lowest first)
        """
        results = []
        
        for atom in atoms:
            stability = self.calculate_stability(atom)
            results.append((atom, stability))
        
        # Sort by stability (lowest first = most at risk)
        results.sort(key=lambda x: x[1])
        
        return results
    
    def get_atoms_needing_reconsolidation(
        self,
        atoms: List[MemoryAtom],
        threshold: float = 0.5
    ) -> List[MemoryAtom]:
        """
        Find atoms that have decayed below threshold and need reconsolidation.
        
        Args:
            atoms: List of memory atoms
            threshold: Stability threshold (default: 0.5 = 50%)
            
        Returns:
            List of atoms needing reconsolidation
        """
        needing_reconsolidation = []
        
        for atom in atoms:
            stability = self.calculate_stability(atom)
            if stability < threshold and atom.graph == GraphType.SUBSTANTIATED:
                needing_reconsolidation.append(atom)
        
        logger.info(
            f"Found {len(needing_reconsolidation)} atoms needing reconsolidation "
            f"(stability < {threshold})"
        )
        
        return needing_reconsolidation
    
    def get_stats(self) -> dict:
        """Get decay engine statistics"""
        return {
            "stability_checks": self.stability_checks,
            "reconsolidations": self.reconsolidations,
            "dissolutions": self.dissolutions,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters"""
        self.stability_checks = 0
        self.reconsolidations = 0
        self.dissolutions = 0
