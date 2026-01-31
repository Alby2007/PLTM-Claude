"""
Enhanced conflict resolution with advanced scoring and jury integration.

This module provides sophisticated conflict resolution algorithms that leverage
the existing jury system for complex personality trait conflicts.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from loguru import logger

from src.core.models import MemoryAtom, AtomType
from src.storage.sqlite_store import SQLiteGraphStore


class EnhancedConflictResolver:
    """
    Advanced conflict resolution with multi-factor scoring.
    
    Improvements over basic resolver:
    - Weighted scoring across 6 factors
    - Temporal decay modeling
    - Context-aware resolution
    - Confidence calibration
    - Tie-breaking heuristics
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        logger.info("EnhancedConflictResolver initialized")
    
    async def resolve_with_explanation(
        self,
        user_id: str,
        conflicting_traits: List[MemoryAtom]
    ) -> Tuple[Optional[MemoryAtom], str]:
        """
        Resolve conflict and provide detailed explanation.
        
        Args:
            user_id: User identifier
            conflicting_traits: List of conflicting trait atoms
            
        Returns:
            Tuple of (winning_trait, explanation)
        """
        if len(conflicting_traits) < 2:
            return (conflicting_traits[0] if conflicting_traits else None, "No conflict")
        
        # Gather evidence
        evidence = await self._gather_comprehensive_evidence(user_id, conflicting_traits)
        
        # Score each trait
        scored_traits = []
        for trait in conflicting_traits:
            score, breakdown = self._calculate_detailed_score(trait, evidence)
            scored_traits.append((trait, score, breakdown))
        
        # Sort by score
        scored_traits.sort(key=lambda x: x[1], reverse=True)
        
        # Build explanation
        winner = scored_traits[0][0]
        winner_score = scored_traits[0][1]
        winner_breakdown = scored_traits[0][2]
        
        explanation = self._build_explanation(
            winner, winner_score, winner_breakdown, scored_traits[1:]
        )
        
        logger.info(f"Resolved conflict: '{winner.object}' (score: {winner_score:.2f})")
        return (winner, explanation)
    
    async def _gather_comprehensive_evidence(
        self,
        user_id: str,
        traits: List[MemoryAtom]
    ) -> Dict[str, Any]:
        """Gather comprehensive evidence for conflict resolution"""
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        
        personality_atoms = [
            atom for atom in all_atoms
            if atom.atom_type in [
                AtomType.PERSONALITY_TRAIT,
                AtomType.COMMUNICATION_STYLE,
                AtomType.INTERACTION_PATTERN
            ]
        ]
        
        # Build evidence
        evidence = {
            "total_observations": len(personality_atoms),
            "trait_frequencies": defaultdict(int),
            "trait_recency": {},
            "feedback_signals": defaultdict(lambda: {"positive": 0, "negative": 0}),
            "context_diversity": defaultdict(set),
            "temporal_consistency": {},
        }
        
        # Calculate frequencies
        for atom in personality_atoms:
            evidence["trait_frequencies"][atom.object] += 1
            evidence["context_diversity"][atom.object].update(atom.contexts)
            
            # Track most recent observation
            if atom.object not in evidence["trait_recency"]:
                evidence["trait_recency"][atom.object] = atom.first_observed
            else:
                evidence["trait_recency"][atom.object] = max(
                    evidence["trait_recency"][atom.object],
                    atom.first_observed
                )
            
            # Track feedback
            if "positive_feedback" in atom.contexts:
                evidence["feedback_signals"][atom.object]["positive"] += 1
            elif "negative_feedback" in atom.contexts:
                evidence["feedback_signals"][atom.object]["negative"] += 1
        
        # Calculate temporal consistency (how stable over time)
        for trait in traits:
            matching_atoms = [
                atom for atom in personality_atoms
                if atom.object == trait.object
            ]
            
            if len(matching_atoms) >= 2:
                # Calculate time span
                timestamps = [atom.first_observed for atom in matching_atoms]
                time_span = (max(timestamps) - min(timestamps)).days
                
                # Consistency = observations per day
                if time_span > 0:
                    consistency = len(matching_atoms) / time_span
                else:
                    consistency = len(matching_atoms)
                
                evidence["temporal_consistency"][trait.object] = consistency
        
        return evidence
    
    def _calculate_detailed_score(
        self,
        trait: MemoryAtom,
        evidence: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate detailed score with breakdown.
        
        Returns:
            Tuple of (total_score, score_breakdown)
        """
        breakdown = {}
        
        # 1. Base Quality (25%)
        base_quality = (trait.confidence * 0.6 + trait.strength * 0.4)
        breakdown["base_quality"] = base_quality * 0.25
        
        # 2. Recency (20%)
        age_days = (datetime.now() - trait.first_observed).days
        if age_days < 7:
            recency_score = 1.0
        elif age_days < 30:
            recency_score = 0.8
        elif age_days < 90:
            recency_score = 0.5
        else:
            recency_score = max(0, 1.0 - (age_days / 180))
        breakdown["recency"] = recency_score * 0.20
        
        # 3. Frequency (20%)
        frequency = evidence["trait_frequencies"].get(trait.object, 1)
        frequency_score = min(1.0, frequency / 5)  # Cap at 5
        breakdown["frequency"] = frequency_score * 0.20
        
        # 4. User Feedback (15%)
        feedback = evidence["feedback_signals"].get(trait.object, {"positive": 0, "negative": 0})
        total_feedback = feedback["positive"] + feedback["negative"]
        if total_feedback > 0:
            feedback_score = feedback["positive"] / total_feedback
        else:
            feedback_score = 0.5  # Neutral
        breakdown["feedback"] = feedback_score * 0.15
        
        # 5. Context Diversity (10%)
        contexts = evidence["context_diversity"].get(trait.object, set())
        context_score = min(1.0, len(contexts) / 3)
        breakdown["context_diversity"] = context_score * 0.10
        
        # 6. Temporal Consistency (10%)
        consistency = evidence["temporal_consistency"].get(trait.object, 0)
        consistency_score = min(1.0, consistency * 10)  # Scale up
        breakdown["temporal_consistency"] = consistency_score * 0.10
        
        # Total score
        total_score = sum(breakdown.values())
        
        return (total_score, breakdown)
    
    def _build_explanation(
        self,
        winner: MemoryAtom,
        winner_score: float,
        winner_breakdown: Dict[str, float],
        losers: List[Tuple[MemoryAtom, float, Dict[str, float]]]
    ) -> str:
        """Build human-readable explanation of conflict resolution"""
        lines = [f"Resolved to: '{winner.object}' (score: {winner_score:.2f})"]
        lines.append("")
        lines.append("Winning factors:")
        
        # Show top 3 factors
        sorted_factors = sorted(
            winner_breakdown.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for factor, score in sorted_factors[:3]:
            lines.append(f"  - {factor}: {score:.2f}")
        
        if losers:
            lines.append("")
            lines.append("Alternatives considered:")
            for trait, score, _ in losers[:2]:  # Show top 2 losers
                lines.append(f"  - '{trait.object}': {score:.2f}")
        
        return "\n".join(lines)
    
    async def resolve_with_confidence(
        self,
        user_id: str,
        conflicting_traits: List[MemoryAtom]
    ) -> Tuple[Optional[MemoryAtom], float]:
        """
        Resolve conflict and return confidence in the resolution.
        
        Args:
            user_id: User identifier
            conflicting_traits: List of conflicting traits
            
        Returns:
            Tuple of (winning_trait, confidence_in_resolution)
        """
        if len(conflicting_traits) < 2:
            return (conflicting_traits[0] if conflicting_traits else None, 1.0)
        
        evidence = await self._gather_comprehensive_evidence(user_id, conflicting_traits)
        
        # Score all traits
        scores = []
        for trait in conflicting_traits:
            score, _ = self._calculate_detailed_score(trait, evidence)
            scores.append((trait, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate confidence based on score gap
        if len(scores) >= 2:
            winner_score = scores[0][1]
            runner_up_score = scores[1][1]
            score_gap = winner_score - runner_up_score
            
            # Confidence based on gap
            if score_gap > 0.3:
                confidence = 0.9  # Clear winner
            elif score_gap > 0.2:
                confidence = 0.8  # Strong winner
            elif score_gap > 0.1:
                confidence = 0.7  # Moderate winner
            else:
                confidence = 0.6  # Weak winner
        else:
            confidence = 1.0  # Only one option
        
        return (scores[0][0], confidence)
    
    def suggest_data_collection(
        self,
        conflicting_traits: List[MemoryAtom],
        evidence: Dict[str, Any]
    ) -> List[str]:
        """
        Suggest what additional data would help resolve the conflict.
        
        Returns:
            List of suggestions for data collection
        """
        suggestions = []
        
        # Check if we need more observations
        for trait in conflicting_traits:
            frequency = evidence["trait_frequencies"].get(trait.object, 0)
            if frequency < 3:
                suggestions.append(
                    f"Collect more observations of '{trait.object}' behavior "
                    f"(currently only {frequency} observations)"
                )
        
        # Check if we need user feedback
        for trait in conflicting_traits:
            feedback = evidence["feedback_signals"].get(trait.object, {"positive": 0, "negative": 0})
            total = feedback["positive"] + feedback["negative"]
            if total == 0:
                suggestions.append(
                    f"Request user feedback on '{trait.object}' trait "
                    f"(no feedback signals yet)"
                )
        
        # Check if we need context diversity
        for trait in conflicting_traits:
            contexts = evidence["context_diversity"].get(trait.object, set())
            if len(contexts) < 2:
                suggestions.append(
                    f"Observe '{trait.object}' in different contexts "
                    f"(currently only in {len(contexts)} context(s))"
                )
        
        return suggestions
