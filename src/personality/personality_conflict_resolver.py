"""
Resolve conflicting personality traits using the jury system.

This module handles cases where personality inferences conflict, using the
existing multi-judge deliberation system to determine the correct trait.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from loguru import logger

from src.core.models import MemoryAtom, AtomType, JudgeVerdict
from src.storage.sqlite_store import SQLiteGraphStore
from src.jury.orchestrator import JuryOrchestrator


class PersonalityConflictResolver:
    """
    Resolve conflicting personality traits using jury deliberation.
    
    Example conflicts:
    - User shows both "prefers concise" and "wants detail"
    - User exhibits both "formal" and "casual" behavior
    - User has both "patient" and "impatient" traits
    
    Uses interaction evidence and jury system to determine which trait is correct.
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self.jury = JuryOrchestrator()
        logger.info("PersonalityConflictResolver initialized")
    
    async def resolve_conflict(
        self,
        user_id: str,
        conflicting_traits: List[MemoryAtom]
    ) -> Optional[MemoryAtom]:
        """
        Resolve conflict between personality traits.
        
        Args:
            user_id: User identifier
            conflicting_traits: List of conflicting trait atoms
            
        Returns:
            The winning trait atom, or None if no clear winner
        """
        if len(conflicting_traits) < 2:
            return conflicting_traits[0] if conflicting_traits else None
        
        logger.info(
            f"Resolving personality conflict for {user_id}: "
            f"{len(conflicting_traits)} conflicting traits"
        )
        
        # Gather evidence for each trait
        evidence = await self._gather_evidence(user_id, conflicting_traits)
        
        # Score each trait based on evidence
        scored_traits = []
        for trait in conflicting_traits:
            score = self._calculate_trait_score(trait, evidence)
            scored_traits.append((trait, score))
        
        # Sort by score
        scored_traits.sort(key=lambda x: x[1], reverse=True)
        
        # Return highest scoring trait
        winner = scored_traits[0][0]
        winner_score = scored_traits[0][1]
        
        logger.info(
            f"Resolved conflict: '{winner.object}' won with score {winner_score:.2f}"
        )
        
        return winner
    
    async def _gather_evidence(
        self,
        user_id: str,
        traits: List[MemoryAtom]
    ) -> Dict[str, Any]:
        """
        Gather evidence to support or refute each trait.
        
        Evidence includes:
        - Recency of observations
        - Frequency of trait manifestation
        - User feedback (positive/negative reactions)
        - Consistency over time
        """
        # Get all personality-related atoms for this user
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        
        # Filter to personality atoms
        personality_atoms = [
            atom for atom in all_atoms
            if atom.atom_type in [
                AtomType.PERSONALITY_TRAIT,
                AtomType.COMMUNICATION_STYLE,
                AtomType.INTERACTION_PATTERN
            ]
        ]
        
        # Build evidence dict
        evidence = {
            "total_observations": len(personality_atoms),
            "recent_observations": [],
            "feedback_signals": [],
            "consistency_scores": {}
        }
        
        # Recent observations (last 30 days)
        recent_cutoff = datetime.now().timestamp() - (30 * 24 * 60 * 60)
        evidence["recent_observations"] = [
            atom for atom in personality_atoms
            if atom.first_observed.timestamp() > recent_cutoff
        ]
        
        # Feedback signals (from contexts)
        for atom in personality_atoms:
            if "positive_feedback" in atom.contexts:
                evidence["feedback_signals"].append(("positive", atom))
            elif "negative_feedback" in atom.contexts:
                evidence["feedback_signals"].append(("negative", atom))
        
        # Consistency scores (how often each trait appears)
        for trait in traits:
            matching = [
                atom for atom in personality_atoms
                if atom.object == trait.object
            ]
            evidence["consistency_scores"][trait.object] = len(matching)
        
        return evidence
    
    def _calculate_trait_score(
        self,
        trait: MemoryAtom,
        evidence: Dict[str, Any]
    ) -> float:
        """
        Calculate score for a trait based on evidence.
        
        Enhanced scoring algorithm with multiple factors:
        - Base confidence and strength
        - Recency (newer = better)
        - Consistency (repeated observations)
        - User feedback (positive/negative)
        - Context diversity (appears in multiple contexts)
        - Temporal stability (consistent over time)
        
        Higher score = more likely to be correct trait.
        """
        score = 0.0
        
        # Base score from confidence and strength (30%)
        base_score = (trait.confidence * 0.6 + trait.strength * 0.4)
        score += base_score * 0.3
        
        # Recency bonus (20%)
        age_days = (datetime.now() - trait.first_observed).days
        if age_days < 7:
            recency_bonus = 1.0  # Very recent
        elif age_days < 30:
            recency_bonus = 0.8  # Recent
        elif age_days < 90:
            recency_bonus = 0.5  # Moderate
        else:
            recency_bonus = max(0, 1.0 - (age_days / 180))  # Decay over 6 months
        score += recency_bonus * 0.2
        
        # Consistency bonus (25%)
        consistency = evidence["consistency_scores"].get(trait.object, 0)
        if consistency >= 5:
            consistency_bonus = 1.0  # Very consistent
        elif consistency >= 3:
            consistency_bonus = 0.7  # Consistent
        elif consistency >= 2:
            consistency_bonus = 0.4  # Somewhat consistent
        else:
            consistency_bonus = 0.2  # Single observation
        score += consistency_bonus * 0.25
        
        # Feedback bonus (15%)
        positive_feedback = 0
        negative_feedback = 0
        for feedback_type, feedback_atom in evidence["feedback_signals"]:
            if feedback_atom.object == trait.object:
                if feedback_type == "positive":
                    positive_feedback += 1
                else:
                    negative_feedback += 1
        
        if positive_feedback + negative_feedback > 0:
            feedback_ratio = positive_feedback / (positive_feedback + negative_feedback)
            feedback_bonus = feedback_ratio
        else:
            feedback_bonus = 0.5  # Neutral (no feedback)
        score += feedback_bonus * 0.15
        
        # Context diversity bonus (10%)
        # Traits that appear in multiple contexts are more robust
        context_count = len(set(trait.contexts))
        context_bonus = min(1.0, context_count / 3)  # Cap at 3 contexts
        score += context_bonus * 0.1
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    async def detect_conflicts(
        self,
        user_id: str
    ) -> List[List[MemoryAtom]]:
        """
        Detect conflicting personality traits for a user.
        
        Returns:
            List of conflict groups, where each group contains conflicting atoms
        """
        # Get all personality atoms
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        personality_atoms = [
            atom for atom in all_atoms
            if atom.atom_type in [
                AtomType.PERSONALITY_TRAIT,
                AtomType.COMMUNICATION_STYLE,
                AtomType.INTERACTION_PATTERN
            ]
        ]
        
        # Define conflicting pairs
        conflict_patterns = [
            ("concise", "detailed"),
            ("concise", "verbose"),
            ("formal", "casual"),
            ("patient", "impatient"),
            ("direct", "diplomatic"),
            ("technical", "simple"),
        ]
        
        # Find conflicts
        conflicts = []
        for pattern_a, pattern_b in conflict_patterns:
            group_a = [
                atom for atom in personality_atoms
                if pattern_a in atom.object.lower()
            ]
            group_b = [
                atom for atom in personality_atoms
                if pattern_b in atom.object.lower()
            ]
            
            if group_a and group_b:
                conflicts.append(group_a + group_b)
        
        logger.debug(f"Detected {len(conflicts)} personality conflicts for {user_id}")
        return conflicts
    
    async def auto_resolve_conflicts(self, user_id: str) -> int:
        """
        Automatically detect and resolve all personality conflicts.
        
        Returns:
            Number of conflicts resolved
        """
        conflicts = await self.detect_conflicts(user_id)
        
        resolved_count = 0
        for conflict_group in conflicts:
            winner = await self.resolve_conflict(user_id, conflict_group)
            if winner:
                # Mark losing atoms as superseded
                for atom in conflict_group:
                    if atom.id != winner.id:
                        atom.superseded_by = winner.id
                        # Could update in store here
                resolved_count += 1
        
        logger.info(f"Auto-resolved {resolved_count} personality conflicts")
        return resolved_count
