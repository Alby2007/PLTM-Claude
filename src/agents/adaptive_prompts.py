"""
Memory-Guided Prompt Engineering

Automatically optimizes prompts based on accumulated user knowledge.
This is OPTIONAL and does not modify the core memory system.

Key features:
- Prompts adapt to user's expertise level
- Prompts match user's communication style
- Prompts incorporate user preferences
- Self-optimization through feedback

Research potential: "Self-Optimizing Prompts via User Memory"
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

from loguru import logger

from src.core.models import AtomType
from src.pipeline.memory_pipeline import MemoryPipeline


class ExpertiseLevel(str, Enum):
    """User expertise levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class CommunicationStyle(str, Enum):
    """User communication preferences"""
    CONCISE = "concise"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    CASUAL = "casual"
    FORMAL = "formal"


class AdaptivePromptSystem:
    """
    Prompt system that improves through accumulated user knowledge.
    
    Over time:
    - Day 1: Generic prompts
    - Day 30: Personalized to user's style and expertise
    - Day 90: Highly optimized for individual user
    
    Result: Better responses without manual prompt engineering.
    """
    
    def __init__(
        self,
        memory_pipeline: MemoryPipeline,
        user_id: str
    ):
        """
        Initialize adaptive prompt system.
        
        Args:
            memory_pipeline: Existing memory pipeline
            user_id: User identifier
        """
        self.pipeline = memory_pipeline
        self.user_id = user_id
        
        # Cache for performance
        self._expertise_cache: Optional[ExpertiseLevel] = None
        self._style_cache: Optional[CommunicationStyle] = None
        self._preferences_cache: List[str] = []
        self._cache_timestamp: Optional[datetime] = None
        
        logger.info(f"AdaptivePromptSystem initialized for user {user_id}")
    
    async def generate_prompt(
        self,
        task: str,
        base_prompt: Optional[str] = None,
        include_examples: bool = True
    ) -> str:
        """
        Generate personalized prompt for user.
        
        Args:
            task: Task description
            base_prompt: Optional base prompt to personalize
            include_examples: Whether to include user-specific examples
            
        Returns:
            Personalized prompt
        """
        # Get user profile
        expertise = await self._infer_expertise()
        style = await self._infer_communication_style()
        preferences = await self._get_key_preferences()
        
        # Build personalized prompt
        sections = []
        
        # Task
        sections.append(f"Task: {task}")
        
        # Expertise adaptation
        if expertise == ExpertiseLevel.BEGINNER:
            sections.append("\nProvide step-by-step explanations with examples.")
        elif expertise == ExpertiseLevel.INTERMEDIATE:
            sections.append("\nProvide clear explanations with key concepts highlighted.")
        elif expertise == ExpertiseLevel.ADVANCED:
            sections.append("\nFocus on advanced concepts and best practices.")
        elif expertise == ExpertiseLevel.EXPERT:
            sections.append("\nProvide expert-level insights and edge cases.")
        
        # Style adaptation
        if style == CommunicationStyle.CONCISE:
            sections.append("Keep response brief and to the point.")
        elif style == CommunicationStyle.DETAILED:
            sections.append("Provide comprehensive, detailed explanations.")
        elif style == CommunicationStyle.TECHNICAL:
            sections.append("Use technical terminology and precise language.")
        elif style == CommunicationStyle.CASUAL:
            sections.append("Use conversational, friendly tone.")
        elif style == CommunicationStyle.FORMAL:
            sections.append("Maintain professional, formal tone.")
        
        # Preferences
        if preferences:
            sections.append(f"\nUser preferences:")
            for pref in preferences[:5]:  # Top 5 preferences
                sections.append(f"- {pref}")
        
        # Examples from user's history
        if include_examples:
            examples = await self._get_relevant_examples(task)
            if examples:
                sections.append("\nRelevant from user's history:")
                for ex in examples[:3]:  # Top 3 examples
                    sections.append(f"- {ex}")
        
        prompt = "\n".join(sections)
        
        logger.debug(f"Generated personalized prompt (expertise={expertise}, style={style})")
        
        return prompt
    
    async def learn_from_feedback(
        self,
        prompt: str,
        response: str,
        feedback: str,
        feedback_type: str = "quality"
    ):
        """
        Learn from user feedback to improve future prompts.
        
        Args:
            prompt: The prompt that was used
            response: The response that was generated
            feedback: User's feedback
            feedback_type: Type of feedback (quality, style, technical_level, etc.)
        """
        # Store feedback as learning
        if feedback_type == "too_technical":
            await self.pipeline.process(
                user_id=self.user_id,
                message="User prefers less technical explanations"
            )
            self._expertise_cache = ExpertiseLevel.INTERMEDIATE
            
        elif feedback_type == "too_simple":
            await self.pipeline.process(
                user_id=self.user_id,
                message="User prefers more technical depth"
            )
            self._expertise_cache = ExpertiseLevel.ADVANCED
            
        elif feedback_type == "too_verbose":
            await self.pipeline.process(
                user_id=self.user_id,
                message="User prefers concise responses"
            )
            self._style_cache = CommunicationStyle.CONCISE
            
        elif feedback_type == "too_brief":
            await self.pipeline.process(
                user_id=self.user_id,
                message="User prefers detailed explanations"
            )
            self._style_cache = CommunicationStyle.DETAILED
            
        elif feedback_type == "helpful":
            # Extract what made it helpful
            await self.pipeline.process(
                user_id=self.user_id,
                message=f"User found helpful: {feedback}"
            )
        
        # Invalidate cache
        self._cache_timestamp = None
        
        logger.info(f"Learned from feedback: {feedback_type}")
    
    async def _infer_expertise(self) -> ExpertiseLevel:
        """Infer user's expertise level from memory"""
        # Check cache
        if self._expertise_cache and self._is_cache_valid():
            return self._expertise_cache
        
        # Get expertise indicators from memory
        atoms = await self.pipeline.store.find_by_triple(
            subject=self.user_id,
            exclude_historical=True
        )
        
        # Count expertise signals
        expert_signals = 0
        beginner_signals = 0
        
        for atom in atoms:
            obj_lower = atom.object.lower()
            
            # Expert signals
            if any(word in obj_lower for word in [
                "expert", "advanced", "proficient", "experienced",
                "deep understanding", "mastered"
            ]):
                expert_signals += 1
            
            # Beginner signals
            if any(word in obj_lower for word in [
                "learning", "beginner", "new to", "just started",
                "confused", "struggling"
            ]):
                beginner_signals += 1
        
        # Determine level
        if expert_signals > 3:
            level = ExpertiseLevel.EXPERT
        elif expert_signals > 1:
            level = ExpertiseLevel.ADVANCED
        elif beginner_signals > 2:
            level = ExpertiseLevel.BEGINNER
        else:
            level = ExpertiseLevel.INTERMEDIATE
        
        self._expertise_cache = level
        self._cache_timestamp = datetime.now()
        
        return level
    
    async def _infer_communication_style(self) -> CommunicationStyle:
        """Infer user's preferred communication style"""
        # Check cache
        if self._style_cache and self._is_cache_valid():
            return self._style_cache
        
        # Get style indicators from memory
        atoms = await self.pipeline.store.find_by_triple(
            subject=self.user_id,
            exclude_historical=True
        )
        
        # Count style signals
        concise_signals = 0
        detailed_signals = 0
        technical_signals = 0
        
        for atom in atoms:
            obj_lower = atom.object.lower()
            
            if any(word in obj_lower for word in [
                "brief", "concise", "short", "quick", "tldr"
            ]):
                concise_signals += 1
            
            if any(word in obj_lower for word in [
                "detailed", "comprehensive", "thorough", "in-depth"
            ]):
                detailed_signals += 1
            
            if any(word in obj_lower for word in [
                "technical", "precise", "formal", "professional"
            ]):
                technical_signals += 1
        
        # Determine style
        if concise_signals > detailed_signals:
            style = CommunicationStyle.CONCISE
        elif detailed_signals > concise_signals:
            style = CommunicationStyle.DETAILED
        elif technical_signals > 2:
            style = CommunicationStyle.TECHNICAL
        else:
            style = CommunicationStyle.CASUAL
        
        self._style_cache = style
        self._cache_timestamp = datetime.now()
        
        return style
    
    async def _get_key_preferences(self) -> List[str]:
        """Get user's key preferences"""
        # Check cache
        if self._preferences_cache and self._is_cache_valid():
            return self._preferences_cache
        
        # Get preference atoms
        atoms = await self.pipeline.store.find_by_triple(
            subject=self.user_id,
            exclude_historical=True
        )
        
        # Filter for preferences
        preferences = []
        for atom in atoms:
            if atom.atom_type == AtomType.PREFERENCE:
                if atom.predicate in ["likes", "prefers", "loves"]:
                    preferences.append(f"{atom.predicate} {atom.object}")
        
        # Sort by confidence
        preferences.sort(
            key=lambda p: next(
                (a.confidence for a in atoms if f"{a.predicate} {a.object}" == p),
                0
            ),
            reverse=True
        )
        
        self._preferences_cache = preferences[:10]
        self._cache_timestamp = datetime.now()
        
        return preferences[:10]
    
    async def _get_relevant_examples(self, task: str) -> List[str]:
        """Get relevant examples from user's history"""
        # In a real implementation, you'd use semantic search
        # For now, return recent relevant atoms
        
        atoms = await self.pipeline.store.find_by_triple(
            subject=self.user_id,
            exclude_historical=True
        )
        
        # Simple keyword matching (would use embeddings in production)
        task_words = set(task.lower().split())
        relevant = []
        
        for atom in atoms:
            obj_words = set(atom.object.lower().split())
            overlap = len(task_words & obj_words)
            
            if overlap > 0:
                relevant.append((overlap, atom.object))
        
        # Sort by relevance
        relevant.sort(reverse=True)
        
        return [obj for _, obj in relevant[:5]]
    
    def _is_cache_valid(self, max_age_seconds: int = 3600) -> bool:
        """Check if cache is still valid"""
        if not self._cache_timestamp:
            return False
        
        age = (datetime.now() - self._cache_timestamp).total_seconds()
        return age < max_age_seconds


class PromptOptimizationExperiment:
    """
    Experiment to measure prompt optimization over time.
    
    Hypothesis: Prompts automatically improve as user memory accumulates.
    
    Metrics:
    - User satisfaction scores
    - Task completion rate
    - Need for clarification
    - Response relevance
    """
    
    def __init__(
        self,
        prompt_system: AdaptivePromptSystem,
        experiment_name: str = "Prompt Optimization"
    ):
        """
        Initialize experiment.
        
        Args:
            prompt_system: Adaptive prompt system
            experiment_name: Name for this experiment
        """
        self.prompt_system = prompt_system
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        # Metrics
        self.prompts_generated = 0
        self.feedback_received = 0
        self.satisfaction_scores: List[float] = []
        
        logger.info(f"PromptOptimizationExperiment '{experiment_name}' initialized")
    
    async def generate_and_track(
        self,
        task: str,
        satisfaction_score: Optional[float] = None
    ) -> str:
        """
        Generate prompt and track metrics.
        
        Args:
            task: Task description
            satisfaction_score: Optional user satisfaction (0-1)
            
        Returns:
            Generated prompt
        """
        prompt = await self.prompt_system.generate_prompt(task)
        self.prompts_generated += 1
        
        if satisfaction_score is not None:
            self.satisfaction_scores.append(satisfaction_score)
        
        return prompt
    
    async def record_feedback(
        self,
        prompt: str,
        response: str,
        feedback: str,
        feedback_type: str
    ):
        """Record user feedback"""
        await self.prompt_system.learn_from_feedback(
            prompt, response, feedback, feedback_type
        )
        self.feedback_received += 1
    
    def get_experiment_results(self) -> Dict[str, Any]:
        """Get experiment results"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        avg_satisfaction = (
            sum(self.satisfaction_scores) / len(self.satisfaction_scores)
            if self.satisfaction_scores else 0
        )
        
        return {
            "experiment_name": self.experiment_name,
            "duration_seconds": duration,
            "prompts_generated": self.prompts_generated,
            "feedback_received": self.feedback_received,
            "avg_satisfaction": avg_satisfaction,
            "satisfaction_trend": self.satisfaction_scores,
            "improvement": (
                ((self.satisfaction_scores[-1] - self.satisfaction_scores[0]) 
                 / self.satisfaction_scores[0] * 100)
                if len(self.satisfaction_scores) > 1 else 0
            )
        }
