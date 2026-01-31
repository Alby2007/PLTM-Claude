"""
Meta Pattern Detector

Finds patterns that transcend specific contexts - behaviors that appear
across multiple domains, revealing core personality traits.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

from src.storage.sqlite_store import SQLiteGraphStore
from src.core.models import MemoryAtom, AtomType
from loguru import logger


@dataclass
class MetaPattern:
    """A pattern that appears across multiple contexts"""
    behavior: str
    contexts: List[str]
    strength: float
    occurrences: int
    examples: List[str]
    is_core_trait: bool


class MetaPatternDetector:
    """
    Find patterns that transcend specific contexts.
    
    Example:
        "Alby applies rigorous validation to trading, coding, AND research"
        → This is a META-PATTERN: rigorous validation is domain-independent
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        
        # Behavior categories for clustering
        self.behavior_categories = {
            "validation": ["test", "verify", "validate", "confirm", "check", "rigorous"],
            "systematic": ["systematic", "methodical", "structured", "organized", "step-by-step"],
            "direct": ["direct", "concise", "no-fluff", "straight", "brief"],
            "ambitious": ["ambitious", "high-scope", "big", "comprehensive", "extensive"],
            "rapid_execution": ["rapid", "fast", "quick", "immediately", "now"],
            "deep_analysis": ["deep", "thorough", "comprehensive", "detailed", "analysis"],
            "novel_seeking": ["novel", "new", "innovative", "breakthrough", "creative"],
            "risk_tolerance": ["risk", "bold", "aggressive", "high-stakes"],
        }
    
    async def detect_meta_patterns(
        self,
        user_id: str,
        min_contexts: int = 2
    ) -> List[MetaPattern]:
        """
        Find behaviors that appear across multiple domains.
        
        Args:
            user_id: User to analyze
            min_contexts: Minimum contexts for a pattern to be "meta"
        
        Returns:
            List of MetaPattern objects
        """
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        
        # Cluster atoms by behavior category
        behavior_clusters = defaultdict(lambda: {
            "contexts": set(),
            "occurrences": 0,
            "examples": [],
            "total_confidence": 0.0
        })
        
        for atom in all_atoms:
            # Identify behavior category
            category = self._categorize_behavior(atom)
            if category:
                cluster = behavior_clusters[category]
                cluster["contexts"].update(atom.contexts if atom.contexts else ["general"])
                cluster["occurrences"] += 1
                cluster["examples"].append(f"{atom.predicate}: {atom.object}")
                cluster["total_confidence"] += atom.confidence
        
        # Filter to meta-patterns (appear in multiple contexts)
        meta_patterns = []
        total_contexts = len(set(
            ctx for atom in all_atoms for ctx in (atom.contexts or ["general"])
        ))
        
        for behavior, data in behavior_clusters.items():
            if len(data["contexts"]) >= min_contexts:
                strength = len(data["contexts"]) / max(total_contexts, 1)
                avg_confidence = data["total_confidence"] / max(data["occurrences"], 1)
                
                meta_patterns.append(MetaPattern(
                    behavior=behavior,
                    contexts=list(data["contexts"]),
                    strength=min(1.0, strength * avg_confidence),
                    occurrences=data["occurrences"],
                    examples=data["examples"][:5],
                    is_core_trait=len(data["contexts"]) >= 3 and data["occurrences"] >= 5
                ))
        
        # Sort by strength
        meta_patterns.sort(key=lambda x: x.strength, reverse=True)
        
        return meta_patterns
    
    def _categorize_behavior(self, atom: MemoryAtom) -> Optional[str]:
        """Categorize an atom's behavior"""
        text = f"{atom.predicate} {atom.object}".lower()
        
        for category, keywords in self.behavior_categories.items():
            if any(kw in text for kw in keywords):
                return category
        
        return None
    
    async def get_core_traits(self, user_id: str) -> Dict[str, Any]:
        """
        Get the user's core traits - behaviors that are domain-independent.
        
        These are the most fundamental aspects of personality.
        """
        meta_patterns = await self.detect_meta_patterns(user_id, min_contexts=2)
        
        core_traits = [p for p in meta_patterns if p.is_core_trait]
        secondary_traits = [p for p in meta_patterns if not p.is_core_trait]
        
        return {
            "user_id": user_id,
            "core_traits": [
                {
                    "trait": p.behavior,
                    "strength": p.strength,
                    "contexts": p.contexts,
                    "occurrences": p.occurrences
                }
                for p in core_traits
            ],
            "secondary_traits": [
                {
                    "trait": p.behavior,
                    "strength": p.strength,
                    "contexts": p.contexts
                }
                for p in secondary_traits
            ],
            "total_meta_patterns": len(meta_patterns),
            "core_trait_count": len(core_traits)
        }
    
    async def predict_behavior_in_new_domain(
        self,
        user_id: str,
        new_domain: str
    ) -> Dict[str, Any]:
        """
        Predict how user will behave in a new domain based on meta-patterns.
        
        Example:
            predict_behavior_in_new_domain("alby", "machine_learning")
            → "Will apply rigorous validation, systematic approach, rapid execution"
        """
        meta_patterns = await self.detect_meta_patterns(user_id)
        
        # Strong meta-patterns will likely apply to new domain
        predictions = []
        for pattern in meta_patterns:
            if pattern.strength > 0.5:
                predictions.append({
                    "behavior": pattern.behavior,
                    "confidence": pattern.strength,
                    "based_on_contexts": pattern.contexts,
                    "prediction": f"Will likely apply {pattern.behavior} approach"
                })
        
        return {
            "new_domain": new_domain,
            "predicted_behaviors": predictions,
            "confidence": sum(p["confidence"] for p in predictions) / max(len(predictions), 1),
            "reasoning": f"Based on {len(predictions)} meta-patterns observed across {len(set(c for p in meta_patterns for c in p.contexts))} contexts"
        }
    
    async def find_common_thread(
        self,
        user_id: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Find what's common across specified contexts.
        
        Example:
            find_common_thread("alby", ["trading", "coding", "research"])
            → "Common thread: rigorous validation before deployment"
        """
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        
        # Filter to specified contexts
        context_atoms = defaultdict(list)
        for atom in all_atoms:
            for ctx in (atom.contexts or ["general"]):
                if ctx in contexts:
                    context_atoms[ctx].append(atom)
        
        # Find behaviors present in ALL specified contexts
        behavior_in_contexts = defaultdict(set)
        for ctx, atoms in context_atoms.items():
            for atom in atoms:
                category = self._categorize_behavior(atom)
                if category:
                    behavior_in_contexts[category].add(ctx)
        
        # Common behaviors = present in all contexts
        common_behaviors = [
            behavior for behavior, ctxs in behavior_in_contexts.items()
            if len(ctxs) == len(contexts)
        ]
        
        return {
            "contexts_analyzed": contexts,
            "common_behaviors": common_behaviors,
            "common_thread": common_behaviors[0] if common_behaviors else None,
            "explanation": f"'{common_behaviors[0]}' appears across all {len(contexts)} contexts" if common_behaviors else "No common thread found"
        }
