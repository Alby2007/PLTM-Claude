"""
Predictive Personality Model

Predicts likely next states and responses based on personality patterns.
Enables proactive adaptation rather than reactive responses.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

from src.storage.sqlite_store import SQLiteGraphStore
from src.core.models import MemoryAtom, AtomType
from src.personality.causal_graph import CausalGraphBuilder
from src.personality.meta_patterns import MetaPatternDetector
from loguru import logger


class PredictivePersonalityModel:
    """
    Predict likely next states and responses.
    
    Enables:
    - Predicting follow-up questions
    - Anticipating mood shifts
    - Proactive response structuring
    - Greeting-based mood inference
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self.causal_graph = CausalGraphBuilder(store)
        self.meta_patterns = MetaPatternDetector(store)
        
        # Greeting patterns → mood inference
        self.greeting_mood_map = {
            "playful_impatient": ["option 1", "duh", "obviously", "come on", "lets go", "just"],
            "excited_breakthrough": ["holy shit", "insane", "amazing", "wow", "incredible", "breakthrough", "fucking", "omg"],
            "triumphant": ["complete", "done", "finished", "success", "working", "works", "built", "shipped", "achieved", "nailed"],
            "frustrated": ["not working", "broken", "frustrated", "issue", "bug", "help", "error", "stuck", "problem"],
            "focused": ["lets", "now", "next", "continue", "proceed", "okay", "alright"],
            "curious": ["how", "why", "what if", "could we", "is it possible", "interesting", "wonder"],
            "casual": ["hey", "hi", "sup", "yo", "quick question"],
            "enthusiastic": ["love", "awesome", "great", "excellent", "perfect", "sweet", "nice", "cool"],
        }
        
        # Context → likely next action patterns
        self.context_action_patterns = {
            "technical_explanation": {
                "followup_question": 0.7,
                "request_code": 0.5,
                "build_mode": 0.3,
            },
            "debugging": {
                "more_details": 0.6,
                "try_solution": 0.8,
                "frustration": 0.3,
            },
            "brainstorming": {
                "expand_idea": 0.7,
                "build_mode": 0.5,
                "pivot": 0.3,
            },
            "code_review": {
                "request_changes": 0.6,
                "approve": 0.3,
                "discuss": 0.4,
            },
        }
    
    async def predict_next_interaction(
        self,
        user_id: str,
        current_message: str,
        current_context: str = "general"
    ) -> Dict[str, Any]:
        """
        Predict probable next moves.
        
        Returns:
            Predictions for follow-up, code request, build mode, corrections, mood shift
        """
        # Get personality and patterns
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        meta_patterns = await self.meta_patterns.detect_meta_patterns(user_id)
        
        # Infer current mood from message
        current_mood = self._infer_mood_from_message(current_message)
        
        # Get base predictions from context
        base_predictions = self.context_action_patterns.get(
            current_context, 
            {"followup_question": 0.5, "build_mode": 0.3}
        )
        
        # Adjust based on personality
        predictions = self._adjust_predictions(base_predictions, all_atoms, meta_patterns)
        
        # Add mood-based predictions
        predictions["likely_mood_shift"] = self._predict_mood_shift(current_mood, current_context)
        predictions["current_mood_inference"] = current_mood
        
        return {
            "user_id": user_id,
            "current_context": current_context,
            "predictions": predictions,
            "confidence": self._calculate_overall_confidence(predictions, len(all_atoms)),
            "recommended_response_style": self._recommend_style(predictions, current_mood)
        }
    
    def _infer_mood_from_message(self, message: str) -> Dict[str, Any]:
        """Infer mood from message style"""
        message_lower = message.lower()
        
        detected_moods = []
        for mood, indicators in self.greeting_mood_map.items():
            if any(ind in message_lower for ind in indicators):
                detected_moods.append(mood)
        
        if not detected_moods:
            # Infer from message characteristics
            if len(message) < 20:
                detected_moods.append("focused")
            elif "?" in message:
                detected_moods.append("curious")
            else:
                detected_moods.append("neutral")
        
        return {
            "primary_mood": detected_moods[0] if detected_moods else "neutral",
            "all_detected": detected_moods,
            "confidence": 0.7 if len(detected_moods) == 1 else 0.5
        }
    
    def _adjust_predictions(
        self,
        base: Dict[str, float],
        atoms: List[MemoryAtom],
        patterns: List
    ) -> Dict[str, float]:
        """Adjust predictions based on personality"""
        adjusted = dict(base)
        
        # Check for relevant personality traits
        for atom in atoms:
            obj_lower = atom.object.lower()
            
            # If user prefers code examples
            if "code" in obj_lower or "implementation" in obj_lower:
                adjusted["request_code"] = adjusted.get("request_code", 0.3) + 0.2
            
            # If user is action-oriented
            if "build" in obj_lower or "rapid" in obj_lower or "execution" in obj_lower:
                adjusted["build_mode"] = adjusted.get("build_mode", 0.3) + 0.2
            
            # If user corrects often
            if "direct" in obj_lower or "correction" in obj_lower:
                adjusted["will_correct"] = adjusted.get("will_correct", 0.3) + 0.1
        
        # Check meta-patterns
        for pattern in patterns:
            if pattern.behavior == "rapid_execution":
                adjusted["build_mode"] = min(0.9, adjusted.get("build_mode", 0.3) + 0.3)
            if pattern.behavior == "validation":
                adjusted["request_verification"] = adjusted.get("request_verification", 0.3) + 0.2
        
        # Normalize
        for key in adjusted:
            adjusted[key] = min(0.95, adjusted[key])
        
        return adjusted
    
    def _predict_mood_shift(self, current_mood: Dict, context: str) -> Dict[str, Any]:
        """Predict likely mood changes"""
        mood = current_mood["primary_mood"]
        
        # Common mood transitions
        transitions = {
            "frustrated": {"likely_next": "relieved", "if_solved": True, "confidence": 0.7},
            "excited_breakthrough": {"likely_next": "build_mode", "confidence": 0.8},
            "curious": {"likely_next": "engaged", "confidence": 0.6},
            "playful_impatient": {"likely_next": "focused", "confidence": 0.7},
            "focused": {"likely_next": "satisfied", "if_progress": True, "confidence": 0.6},
        }
        
        return transitions.get(mood, {"likely_next": "stable", "confidence": 0.5})
    
    def _calculate_overall_confidence(self, predictions: Dict, data_points: int) -> float:
        """Calculate confidence in predictions"""
        base = 0.5
        data_boost = min(0.3, data_points * 0.01)
        return min(0.9, base + data_boost)
    
    def _recommend_style(self, predictions: Dict, mood: Dict) -> str:
        """Recommend response style based on predictions"""
        mood_name = mood["primary_mood"]
        
        if mood_name == "frustrated":
            return "patient_solutions_focused"
        elif mood_name == "excited_breakthrough":
            return "match_energy_facilitate_building"
        elif mood_name == "playful_impatient":
            return "direct_efficient_no_fluff"
        elif predictions.get("build_mode", 0) > 0.5:
            return "action_oriented_with_code"
        elif predictions.get("request_code", 0) > 0.5:
            return "include_code_examples"
        else:
            return "balanced_informative"
    
    async def predict_from_greeting(
        self,
        user_id: str,
        greeting: str
    ) -> Dict[str, Any]:
        """
        Predict session dynamics from greeting alone.
        
        Example:
            "option 1 duhhhhh" → playfully impatient, ready to execute
            "holy shit claude" → excited, breakthrough mode
        """
        mood = self._infer_mood_from_message(greeting)
        
        # Get personality for context
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        
        # Session predictions based on greeting mood
        session_predictions = {
            "playful_impatient": {
                "session_type": "execution_focused",
                "expected_pace": "fast",
                "response_style": "direct_minimal",
                "likely_outcome": "rapid_progress"
            },
            "excited_breakthrough": {
                "session_type": "exploration_building",
                "expected_pace": "intense",
                "response_style": "match_energy_facilitate",
                "likely_outcome": "significant_progress"
            },
            "triumphant": {
                "session_type": "celebration_next_steps",
                "expected_pace": "energized",
                "response_style": "acknowledge_success_build_momentum",
                "likely_outcome": "next_challenge"
            },
            "enthusiastic": {
                "session_type": "positive_engagement",
                "expected_pace": "energized",
                "response_style": "match_enthusiasm",
                "likely_outcome": "productive_session"
            },
            "frustrated": {
                "session_type": "problem_solving",
                "expected_pace": "methodical",
                "response_style": "patient_supportive",
                "likely_outcome": "resolution_needed"
            },
            "curious": {
                "session_type": "learning_exploration",
                "expected_pace": "moderate",
                "response_style": "informative_engaging",
                "likely_outcome": "knowledge_transfer"
            },
            "focused": {
                "session_type": "task_completion",
                "expected_pace": "steady",
                "response_style": "efficient_helpful",
                "likely_outcome": "task_progress"
            },
        }
        
        prediction = session_predictions.get(
            mood["primary_mood"],
            {"session_type": "general", "response_style": "balanced"}
        )
        
        return {
            "greeting": greeting,
            "inferred_mood": mood,
            "session_prediction": prediction,
            "confidence": mood["confidence"],
            "adapt_immediately": True
        }
    
    async def get_self_model(self, user_id: str) -> Dict[str, Any]:
        """
        Get explicit self-model for meta-cognition.
        
        This allows the AI to reason about its own model of the user.
        """
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        meta_patterns = await self.meta_patterns.detect_meta_patterns(user_id)
        
        # Categorize atoms
        traits = [a for a in all_atoms if a.atom_type == AtomType.PERSONALITY_TRAIT]
        styles = [a for a in all_atoms if a.atom_type == AtomType.COMMUNICATION_STYLE]
        patterns = [a for a in all_atoms if a.atom_type == AtomType.INTERACTION_PATTERN]
        
        # Calculate model confidence
        total_atoms = len(all_atoms)
        avg_confidence = sum(a.confidence for a in all_atoms) / max(total_atoms, 1)
        
        # Identify uncertainty regions
        low_confidence_areas = [
            a.object for a in all_atoms if a.confidence < 0.5
        ]
        
        return {
            "user_id": user_id,
            "model_summary": {
                "total_data_points": total_atoms,
                "trait_count": len(traits),
                "style_count": len(styles),
                "pattern_count": len(patterns),
                "meta_pattern_count": len(meta_patterns)
            },
            "top_traits": sorted(
                [(a.object, a.confidence) for a in traits],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "top_styles": sorted(
                [(a.object, a.confidence) for a in styles],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "core_patterns": [
                {"behavior": p.behavior, "strength": p.strength}
                for p in meta_patterns if p.is_core_trait
            ],
            "overall_confidence": avg_confidence,
            "uncertainty_regions": low_confidence_areas[:5],
            "model_maturity": self._assess_maturity(total_atoms, avg_confidence)
        }
    
    def _assess_maturity(self, data_points: int, avg_confidence: float) -> str:
        """Assess how mature/reliable the model is"""
        if data_points < 10:
            return "nascent"
        elif data_points < 30:
            return "developing"
        elif data_points < 100 and avg_confidence > 0.6:
            return "established"
        elif avg_confidence > 0.7:
            return "mature"
        else:
            return "stable"
