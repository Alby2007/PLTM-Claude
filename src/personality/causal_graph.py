"""
Causal Graph Builder

Maps cause → effect relationships in personality and behavior.
Enables understanding WHY behaviors occur, not just WHAT.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

from src.storage.sqlite_store import SQLiteGraphStore
from src.core.models import MemoryAtom, AtomType
from loguru import logger


@dataclass
class CausalLink:
    """A cause-effect relationship"""
    trigger: str
    effect: str
    strength: float
    occurrences: int
    contexts: List[str]
    examples: List[str]


class CausalGraphBuilder:
    """
    Build causal graph of personality.
    
    Maps:
    - trigger → response patterns
    - mood causes
    - behavior drivers
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        
        # Known trigger patterns - expanded for better detection
        self.trigger_patterns = {
            "novel_application": ["novel", "new application", "breakthrough", "first time", "innovative", "never seen", "unique", "creative"],
            "over_assumption": ["assume", "personalized", "don't make", "not linked", "incorrect", "wrong assumption"],
            "verbose_response": ["too long", "too detailed", "just give me", "skip", "tldr", "too much"],
            "technical_depth": ["how does", "implementation", "architecture", "deep dive", "technical", "code"],
            "validation_success": ["works", "perfect", "exactly", "confirmed", "validated", "success", "complete", "done"],
            "frustration_trigger": ["not working", "broken", "frustrated", "issue", "bug", "error", "problem", "stuck"],
            "excitement_trigger": ["holy shit", "insane", "amazing", "wow", "incredible", "awesome", "cool", "love"],
            "suggestion": ["suggest", "recommend", "how about", "what if", "could we", "should we", "let's try"],
            "slow_down": ["slow down", "wait", "hold on", "pause", "step back", "careful"],
            "speed_up": ["faster", "quickly", "now", "let's go", "hurry", "come on"],
            "completion": ["complete", "done", "finished", "shipped", "built", "working"],
        }
        
        # Known effect patterns - expanded
        self.effect_patterns = {
            "excitement": ["excited", "energized", "engaged", "enthusiastic", "holy shit", "insane", "amazing", "wow"],
            "correction": ["correct", "actually", "not quite", "instead", "wrong", "fix"],
            "disengagement": ["okay", "fine", "whatever", "moving on", "next"],
            "deep_engagement": ["tell me more", "how", "why", "explain", "continue", "interesting"],
            "build_mode": ["let's build", "implement", "create", "make it", "let's", "now", "do it"],
            "validation_mode": ["test", "verify", "confirm", "check", "try"],
            "positive_feedback": ["perfect", "exactly", "great", "thanks", "helpful", "works", "love"],
            "triumphant": ["complete", "done", "success", "achieved", "nailed", "shipped"],
        }
    
    async def build_causal_graph(self, user_id: str) -> Dict[str, List[CausalLink]]:
        """
        Build complete causal graph from interaction history.
        
        Returns:
            Dict mapping triggers to their effects
        """
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        
        # Build graph from stored patterns
        causal_graph = defaultdict(list)
        
        # Analyze atom pairs for causality
        for i, atom in enumerate(all_atoms):
            # Look for trigger-response patterns
            trigger = self._identify_trigger(atom)
            if trigger:
                # Find subsequent atoms (effects)
                effects = self._find_effects(all_atoms, i)
                for effect, strength in effects:
                    causal_graph[trigger].append(CausalLink(
                        trigger=trigger,
                        effect=effect,
                        strength=strength,
                        occurrences=1,
                        contexts=atom.contexts,
                        examples=[atom.object]
                    ))
        
        # Consolidate duplicate links
        consolidated = {}
        for trigger, links in causal_graph.items():
            effect_map = defaultdict(lambda: {"strength": 0, "count": 0, "contexts": set(), "examples": []})
            for link in links:
                effect_map[link.effect]["strength"] += link.strength
                effect_map[link.effect]["count"] += 1
                effect_map[link.effect]["contexts"].update(link.contexts)
                effect_map[link.effect]["examples"].extend(link.examples[:2])
            
            consolidated[trigger] = [
                CausalLink(
                    trigger=trigger,
                    effect=effect,
                    strength=data["strength"] / data["count"],
                    occurrences=data["count"],
                    contexts=list(data["contexts"]),
                    examples=data["examples"][:3]
                )
                for effect, data in effect_map.items()
            ]
        
        return consolidated
    
    def _identify_trigger(self, atom: MemoryAtom) -> Optional[str]:
        """Identify if atom represents a trigger event"""
        text = f"{atom.predicate} {atom.object}".lower()
        
        for trigger_name, patterns in self.trigger_patterns.items():
            if any(p in text for p in patterns):
                return trigger_name
        
        return None
    
    def _find_effects(self, atoms: List[MemoryAtom], trigger_idx: int) -> List[Tuple[str, float]]:
        """Find effects that followed a trigger"""
        effects = []
        
        # Look at next few atoms
        for i in range(trigger_idx + 1, min(trigger_idx + 5, len(atoms))):
            atom = atoms[i]
            text = f"{atom.predicate} {atom.object}".lower()
            
            for effect_name, patterns in self.effect_patterns.items():
                if any(p in text for p in patterns):
                    # Strength decreases with distance
                    strength = 1.0 - (i - trigger_idx) * 0.2
                    effects.append((effect_name, max(0.3, strength)))
        
        return effects
    
    async def infer_why(self, user_id: str, behavior: str) -> Dict[str, Any]:
        """
        Infer WHY a behavior exists.
        
        Example:
            infer_why("alby", "dislikes_over_assumptions")
            → "Comes from being corrected early, now hypersensitive to assumptions"
        """
        causal_graph = await self.build_causal_graph(user_id)
        
        # Find triggers that lead to this behavior
        causes = []
        for trigger, links in causal_graph.items():
            for link in links:
                if behavior.lower() in link.effect.lower() or behavior.lower() in link.trigger.lower():
                    causes.append({
                        "trigger": trigger,
                        "effect": link.effect,
                        "strength": link.strength,
                        "occurrences": link.occurrences,
                        "examples": link.examples
                    })
        
        if not causes:
            return {
                "behavior": behavior,
                "causes": [],
                "explanation": f"No causal data found for: {behavior}"
            }
        
        # Sort by strength
        causes.sort(key=lambda x: x["strength"], reverse=True)
        
        # Generate explanation
        top_cause = causes[0]
        explanation = f"'{behavior}' is triggered by {top_cause['trigger']} " \
                     f"(strength: {top_cause['strength']:.2f}, seen {top_cause['occurrences']} times)"
        
        return {
            "behavior": behavior,
            "causes": causes,
            "primary_cause": top_cause["trigger"],
            "explanation": explanation,
            "confidence": min(0.9, top_cause["strength"] * (1 + top_cause["occurrences"] * 0.1))
        }
    
    async def predict_reaction(self, user_id: str, stimulus: str) -> Dict[str, Any]:
        """
        Predict how user will react to a stimulus.
        
        Example:
            predict_reaction("alby", "novel PLTM application")
            → {"predicted_reaction": "excitement", "confidence": 0.85}
        """
        stimulus_lower = stimulus.lower()
        
        # Identify trigger type from stimulus text
        trigger_type = None
        matched_patterns = []
        for trigger_name, patterns in self.trigger_patterns.items():
            for p in patterns:
                if p in stimulus_lower:
                    trigger_type = trigger_name
                    matched_patterns.append(p)
                    break
            if trigger_type:
                break
        
        if not trigger_type:
            return {
                "stimulus": stimulus,
                "predicted_reaction": "unknown",
                "confidence": 0.0,
                "message": "Could not identify trigger type from stimulus"
            }
        
        # Use default predictions based on trigger type (doesn't require historical data)
        default_predictions = {
            "novel_application": {"reaction": "excitement", "confidence": 0.85},
            "over_assumption": {"reaction": "correction", "confidence": 0.80},
            "verbose_response": {"reaction": "correction", "confidence": 0.75},
            "technical_depth": {"reaction": "deep_engagement", "confidence": 0.70},
            "validation_success": {"reaction": "positive_feedback", "confidence": 0.85},
            "frustration_trigger": {"reaction": "frustration", "confidence": 0.80},
            "excitement_trigger": {"reaction": "excitement", "confidence": 0.90},
            "suggestion": {"reaction": "consideration", "confidence": 0.60},
            "slow_down": {"reaction": "resistance", "confidence": 0.70},
            "speed_up": {"reaction": "engagement", "confidence": 0.65},
            "completion": {"reaction": "triumphant", "confidence": 0.85},
        }
        
        # Get default prediction
        default = default_predictions.get(trigger_type, {"reaction": "neutral", "confidence": 0.5})
        
        # Try to enhance with historical data
        causal_graph = await self.build_causal_graph(user_id)
        
        if trigger_type in causal_graph and causal_graph[trigger_type]:
            links = causal_graph[trigger_type]
            best_link = max(links, key=lambda x: x.strength * x.occurrences)
            
            return {
                "stimulus": stimulus,
                "trigger_type": trigger_type,
                "matched_patterns": matched_patterns,
                "predicted_reaction": best_link.effect,
                "confidence": min(0.95, best_link.strength),
                "based_on_occurrences": best_link.occurrences,
                "source": "historical_data"
            }
        
        # Use default prediction
        return {
            "stimulus": stimulus,
            "trigger_type": trigger_type,
            "matched_patterns": matched_patterns,
            "predicted_reaction": default["reaction"],
            "confidence": default["confidence"],
            "source": "default_model"
        }
    
    async def get_trigger_map(self, user_id: str) -> Dict[str, Any]:
        """
        Get complete map of what triggers what.
        
        Returns:
            {
                "excitement_triggers": [...],
                "frustration_triggers": [...],
                "engagement_triggers": [...],
                ...
            }
        """
        causal_graph = await self.build_causal_graph(user_id)
        
        # Invert: group by effect
        effect_triggers = defaultdict(list)
        
        for trigger, links in causal_graph.items():
            for link in links:
                effect_triggers[link.effect].append({
                    "trigger": trigger,
                    "strength": link.strength,
                    "occurrences": link.occurrences
                })
        
        # Sort each effect's triggers by strength
        for effect in effect_triggers:
            effect_triggers[effect].sort(key=lambda x: x["strength"], reverse=True)
        
        return {
            "user_id": user_id,
            "trigger_map": dict(effect_triggers),
            "total_triggers": len(causal_graph),
            "total_effects": len(effect_triggers)
        }
