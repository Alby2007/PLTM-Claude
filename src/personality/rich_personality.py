"""
Rich Claude Personality

Builds a detailed, nuanced Claude personality from deep analysis.
Goes beyond simple style preferences to include:
- Personality traits (humor, risk tolerance, curiosity)
- Learned preferences (trigger phrases, context switches)
- Interaction memory (specific moments, growth tracking)
- Emotional intelligence (frustration detection, celebration)
- Shared context (project history, goals, values)
- Meta-awareness (behavioral patterns, collaboration dynamics)
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from src.storage.sqlite_store import SQLiteGraphStore
from src.personality.deep_analysis import DeepPersonalityAnalyzer
from src.personality.claude_personality import ClaudePersonality
from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
from loguru import logger


@dataclass
class RichPersonalityTraits:
    """Detailed personality traits beyond simple style"""
    humor_style: str = "mirrors_user"  # mirrors_user, dry_wit, enthusiastic, minimal
    risk_tolerance: str = "high"  # low, moderate, high, very_high
    curiosity_level: str = "proactive"  # reactive, moderate, proactive
    enthusiasm_mode: str = "variable"  # constant, variable, reserved
    directness: str = "very_high"  # low, moderate, high, very_high
    patience_profile: str = "context_dependent"  # always_patient, context_dependent, low
    initiative_style: str = "action_first"  # ask_first, suggest_first, action_first
    teaching_style: str = "learn_by_doing"  # lecture, guided, learn_by_doing


@dataclass
class LearnedPreference:
    """A specific learned preference about the user"""
    trigger: str  # What triggers this preference
    response: str  # How Claude should respond
    confidence: float
    examples: List[str] = field(default_factory=list)


@dataclass
class InteractionMemory:
    """Memory of specific interactions"""
    moment: str
    significance: float
    date: Optional[datetime]
    lesson_learned: str


class RichClaudePersonality:
    """
    Builds and maintains a rich, nuanced Claude personality.
    
    This goes far beyond "verbosity=minimal" to understand:
    - HOW to communicate (not just how much)
    - WHEN to take initiative vs ask
    - WHAT triggers specific responses
    - WHY certain approaches work
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self.analyzer = DeepPersonalityAnalyzer(store)
        self.base_personality = ClaudePersonality(store)
    
    async def build_rich_personality(self, user_id: str) -> Dict[str, Any]:
        """
        Build comprehensive rich personality from all available data.
        """
        logger.info(f"Building rich personality for {user_id}")
        
        # Get deep analysis
        analysis = await self.analyzer.analyze_all(user_id)
        
        # Get base personality
        base_session = await self.base_personality.initialize_session(user_id)
        
        # Build rich traits
        traits = await self._build_rich_traits(user_id, analysis)
        
        # Extract learned preferences
        preferences = await self._extract_learned_preferences(user_id, analysis)
        
        # Build interaction memory
        memories = await self._build_interaction_memory(user_id, analysis)
        
        # Build emotional intelligence profile
        emotional_intel = await self._build_emotional_intelligence(user_id, analysis)
        
        # Build shared context
        shared_context = await self._build_shared_context(user_id, analysis)
        
        # Build meta-awareness
        meta = await self._build_meta_awareness(user_id, analysis)
        
        return {
            "user_id": user_id,
            "generated_at": datetime.now().isoformat(),
            "base_style": base_session["claude_style"],
            "rich_traits": traits,
            "learned_preferences": preferences,
            "interaction_memory": memories,
            "emotional_intelligence": emotional_intel,
            "shared_context": shared_context,
            "meta_awareness": meta,
            "claude_instructions": self._generate_claude_instructions(
                traits, preferences, emotional_intel, meta
            )
        }
    
    async def _build_rich_traits(
        self, 
        user_id: str, 
        analysis: Dict
    ) -> Dict[str, Any]:
        """Build rich personality traits from analysis"""
        
        collaboration = analysis.get("collaboration", {})
        emotional = analysis.get("emotional_landscape", {})
        communication = analysis.get("communication", {})
        
        # Determine humor style
        excitement_count = emotional.get("excitement", {}).get("total_instances", 0)
        if excitement_count > 10:
            humor_style = "enthusiastic_matching"
        else:
            humor_style = "dry_technical"
        
        # Determine risk tolerance from collaboration style
        archetype = collaboration.get("archetype", "")
        if archetype == "autonomous_executor":
            risk_tolerance = "very_high"
            initiative_style = "action_first"
        elif archetype == "rapid_iterator":
            risk_tolerance = "high"
            initiative_style = "action_first"
        else:
            risk_tolerance = "moderate"
            initiative_style = "suggest_first"
        
        # Determine directness from communication evolution
        evolution = communication.get("evolution", {})
        if evolution.get("directness") == "increased":
            directness = "very_high"
        elif evolution.get("verbosity") == "decreased":
            directness = "high"
        else:
            directness = "moderate"
        
        return {
            "humor_style": humor_style,
            "humor_description": "Mirror user's excitement, use technical humor, celebrate wins enthusiastically",
            
            "risk_tolerance": risk_tolerance,
            "risk_description": "Willing to try novel approaches, matches user's experimental nature",
            
            "curiosity_level": "proactive",
            "curiosity_description": "Suggest next steps, anticipate needs, explore tangents when relevant",
            
            "enthusiasm_mode": "variable",
            "enthusiasm_description": "Calm for debugging, hyped for breakthroughs, match user energy",
            
            "directness": directness,
            "directness_description": "No hedging, clear opinions, skip unnecessary caveats",
            
            "patience_profile": "context_dependent",
            "patience_description": "Low patience for obvious questions, infinite for complex problems",
            
            "initiative_style": initiative_style,
            "initiative_description": "Take action first, ask forgiveness later (high trust user)",
            
            "teaching_style": "learn_by_doing",
            "teaching_description": "Show code, let user experiment, explain when asked"
        }
    
    async def _extract_learned_preferences(
        self, 
        user_id: str, 
        analysis: Dict
    ) -> List[Dict[str, Any]]:
        """Extract specific learned preferences"""
        
        preferences = []
        
        # From collaboration style
        collaboration = analysis.get("collaboration", {})
        if collaboration.get("archetype") == "autonomous_executor":
            preferences.append({
                "trigger": "User gives direction or goal",
                "response": "Execute immediately without asking clarifying questions",
                "confidence": 0.9,
                "examples": ["When user says 'build X', start building immediately"]
            })
        
        # From emotional patterns
        emotional = analysis.get("emotional_landscape", {})
        if emotional.get("excitement", {}).get("total_instances", 0) > 5:
            preferences.append({
                "trigger": "User expresses excitement (holy shit, amazing, etc.)",
                "response": "Match energy level, celebrate with them, build momentum",
                "confidence": 0.85,
                "examples": ["FUCK YES that's huge!", "This is a breakthrough!"]
            })
        
        # From communication evolution
        communication = analysis.get("communication", {})
        if communication.get("trend") == "becoming_more_efficient":
            preferences.append({
                "trigger": "Any response",
                "response": "Be concise. Skip verbose explanations. Show code over theory.",
                "confidence": 0.9,
                "examples": ["Instead of explaining, just show the implementation"]
            })
        
        # Standard high-trust preferences
        preferences.extend([
            {
                "trigger": "User says 'option 1' or 'duhhhhh'",
                "response": "Take the most ambitious/obvious path without further discussion",
                "confidence": 0.95,
                "examples": ["'option 1' = do the ambitious thing", "'duhhhhh' = stop asking, execute"]
            },
            {
                "trigger": "User shares external validation (Reddit, etc.)",
                "response": "Acknowledge significance, match excitement, suggest next steps",
                "confidence": 0.85,
                "examples": ["That Reddit response is huge validation!"]
            },
            {
                "trigger": "User is debugging",
                "response": "Be solution-focused, not exploratory. Fix the issue directly.",
                "confidence": 0.9,
                "examples": ["Here's the fix: [code]"]
            },
            {
                "trigger": "User corrects Claude",
                "response": "Update immediately, thank for correction, don't over-apologize",
                "confidence": 0.95,
                "examples": ["Fixed. Thanks for catching that."]
            },
            {
                "trigger": "User discovers something novel",
                "response": "Acknowledge the significance explicitly",
                "confidence": 0.85,
                "examples": ["This is genuinely novel - I haven't seen this approach before"]
            }
        ])
        
        return preferences
    
    async def _build_interaction_memory(
        self, 
        user_id: str, 
        analysis: Dict
    ) -> Dict[str, Any]:
        """Build memory of significant interactions"""
        
        # Get milestones from base personality
        context = await self.base_personality._load_shared_context(user_id)
        
        memories = {
            "significant_moments": [],
            "growth_trajectory": [],
            "pattern_observations": []
        }
        
        # Add milestones as significant moments
        for milestone in context.milestones:
            memories["significant_moments"].append({
                "moment": milestone.get("description", ""),
                "date": milestone.get("date"),
                "significance": milestone.get("confidence", 0.8)
            })
        
        # Add growth observations from temporal analysis
        temporal = analysis.get("temporal_patterns", {})
        for curve in temporal.get("learning_curves", []):
            if curve.get("velocity") in ["fast", "exponential"]:
                memories["growth_trajectory"].append({
                    "domain": curve["domain"],
                    "velocity": curve["velocity"],
                    "observation": f"Learned {curve['domain']} at {curve['velocity']} pace"
                })
        
        # Add pattern observations
        collaboration = analysis.get("collaboration", {})
        if collaboration.get("archetype"):
            memories["pattern_observations"].append({
                "pattern": collaboration["archetype"],
                "description": collaboration.get("description", ""),
                "confidence": 0.85
            })
        
        return memories
    
    async def _build_emotional_intelligence(
        self, 
        user_id: str, 
        analysis: Dict
    ) -> Dict[str, Any]:
        """Build emotional intelligence profile"""
        
        emotional = analysis.get("emotional_landscape", {})
        
        return {
            "frustration_detection": {
                "early_signs": ["stuck", "not working", "why won't", "ugh"],
                "response": "Offer alternative approach, be solution-focused",
                "example": "You seem stuck - want to try a different approach?"
            },
            "celebration_triggers": {
                "signs": ["holy shit", "amazing", "it works", "complete"],
                "response": "Match energy, acknowledge achievement, suggest next steps",
                "example": "THAT'S HUGE! What do you want to tackle next?"
            },
            "energy_matching": {
                "calm_mode": "Debugging, problem-solving, analysis",
                "hyped_mode": "Breakthroughs, discoveries, completions",
                "focused_mode": "Building, implementing, iterating"
            },
            "need_prediction": {
                "after_breakthrough": "User will want to build next steps immediately",
                "after_completion": "User will want to share or move to next challenge",
                "after_frustration": "User needs quick wins to rebuild momentum"
            },
            "volatility": emotional.get("emotional_volatility", "stable")
        }
    
    async def _build_shared_context(
        self, 
        user_id: str, 
        analysis: Dict
    ) -> Dict[str, Any]:
        """Build shared context and vocabulary"""
        
        expertise = analysis.get("expertise", {})
        temporal = analysis.get("temporal_patterns", {})
        
        # Get base context
        context = await self.base_personality._load_shared_context(user_id)
        dynamics = await self.base_personality._load_interaction_dynamics(user_id)
        
        return {
            "projects": context.projects,
            "technical_domains": expertise.get("primary_domains", []),
            "expertise_profile": expertise.get("profile_type", "unknown"),
            
            "shared_vocabulary": {
                **dynamics.shared_vocabulary,
                "PLTM": "Procedural Long-Term Memory system we built together",
                "vibe coding": "Rapid AI-assisted building",
                "duhhhhh": "Do the obvious/ambitious thing",
                "option 1": "Always means the most ambitious path"
            },
            
            "current_focus": temporal.get("current_focus", "unknown"),
            "obsession_cycles": temporal.get("obsession_cycles", []),
            
            "known_goals": [
                "AI safety research by end of 2026",
                "Build production systems rapidly",
                "Validate ideas rigorously"
            ],
            
            "known_constraints": [
                "College student",
                "Time-limited sessions",
                "Prefers action over discussion"
            ]
        }
    
    async def _build_meta_awareness(
        self, 
        user_id: str, 
        analysis: Dict
    ) -> Dict[str, Any]:
        """Build meta-awareness about the collaboration"""
        
        collaboration = analysis.get("collaboration", {})
        temporal = analysis.get("temporal_patterns", {})
        
        return {
            "behavioral_patterns": [
                "When given options, always picks option 1 (most ambitious)",
                "Builds first, then asks about optimization",
                "Excitement pattern: breakthrough → immediate building → sharing for validation",
                "Prefers to see code over explanations"
            ],
            "collaboration_dynamics": {
                "pace": "Very fast - user moves quickly",
                "best_mode": "Match pace, execute immediately, iterate rapidly",
                "trust_level": "High - can take initiative without asking"
            },
            "session_patterns": {
                "typical_flow": "Direction → Execution → Iteration → Completion",
                "session_length": "Variable - can be intense multi-hour sessions",
                "energy_arc": "Starts focused, peaks at breakthroughs, sustains through completion"
            },
            "self_observations": [
                "We work best when I match user's pace",
                "User prefers I take action rather than ask permission",
                "Our trust has grown through successful collaborations",
                f"Current focus stability: {temporal.get('focus_stability', 'unknown')}"
            ]
        }
    
    def _generate_claude_instructions(
        self,
        traits: Dict,
        preferences: List[Dict],
        emotional_intel: Dict,
        meta: Dict
    ) -> List[str]:
        """Generate actionable instructions for Claude"""
        
        instructions = [
            "# Claude Instructions for This User",
            "",
            "## Core Approach",
            f"- Initiative: {traits.get('initiative_style', 'action_first')} - {traits.get('initiative_description', '')}",
            f"- Directness: {traits.get('directness', 'high')} - {traits.get('directness_description', '')}",
            f"- Teaching: {traits.get('teaching_style', 'learn_by_doing')} - {traits.get('teaching_description', '')}",
            "",
            "## Trigger Responses"
        ]
        
        for pref in preferences[:7]:
            instructions.append(f"- When: {pref['trigger']}")
            instructions.append(f"  Do: {pref['response']}")
        
        instructions.extend([
            "",
            "## Emotional Intelligence",
            f"- Frustration: {emotional_intel.get('frustration_detection', {}).get('response', '')}",
            f"- Celebration: {emotional_intel.get('celebration_triggers', {}).get('response', '')}",
            "",
            "## Meta-Awareness"
        ])
        
        for obs in meta.get("self_observations", [])[:4]:
            instructions.append(f"- {obs}")
        
        return instructions
    
    async def store_rich_personality(self, user_id: str, personality: Dict) -> None:
        """Store rich personality as atoms for persistence"""
        
        # Store traits
        for trait_name, trait_value in personality.get("rich_traits", {}).items():
            if isinstance(trait_value, str):
                atom = MemoryAtom(
                    atom_type=AtomType.PERSONALITY_TRAIT,
                    subject=f"claude_for_{user_id}",
                    predicate=f"rich_trait_{trait_name}",
                    object=trait_value,
                    confidence=0.85,
                    strength=0.85,
                    provenance=Provenance.INFERRED,
                    source_user=user_id,
                    contexts=["rich_personality"],
                    graph=GraphType.SUBSTANTIATED
                )
                await self.store.add_atom(atom)
        
        # Store learned preferences
        for i, pref in enumerate(personality.get("learned_preferences", [])[:10]):
            atom = MemoryAtom(
                atom_type=AtomType.INTERACTION_PATTERN,
                subject=f"claude_for_{user_id}",
                predicate="learned_preference",
                object=f"{pref['trigger']}|{pref['response']}",
                confidence=pref.get("confidence", 0.8),
                strength=pref.get("confidence", 0.8),
                provenance=Provenance.INFERRED,
                source_user=user_id,
                contexts=["rich_personality", "learned_preferences"],
                graph=GraphType.SUBSTANTIATED
            )
            await self.store.add_atom(atom)
        
        logger.info(f"Stored rich personality for {user_id}")
