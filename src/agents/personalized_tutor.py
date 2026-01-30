"""
Personalized Tutor System

Adapts teaching style and content based on accumulated knowledge about the user.

Key features:
- Tracks what user knows/doesn't know
- Adapts difficulty based on skill level
- Personalizes explanations to user's learning style
- Identifies knowledge gaps
- Suggests next learning steps

Research potential: "Adaptive Learning via Memory-Augmented AI"
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from loguru import logger

from src.core.models import MemoryAtom, AtomType
from src.pipeline.memory_pipeline import MemoryPipeline


class SkillLevel(str, Enum):
    """User's skill level in a topic"""
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class LearningStyle(str, Enum):
    """User's preferred learning style"""
    VISUAL = "visual"
    VERBAL = "verbal"
    HANDS_ON = "hands_on"
    THEORETICAL = "theoretical"
    EXAMPLE_BASED = "example_based"


@dataclass
class KnowledgeGap:
    """Identified gap in user's knowledge"""
    topic: str
    gap_type: str  # "prerequisite", "misconception", "incomplete"
    severity: float  # 0.0-1.0
    description: str
    suggested_resources: List[str]


@dataclass
class LearningRecommendation:
    """Personalized learning recommendation"""
    topic: str
    difficulty: SkillLevel
    reasoning: str
    confidence: float
    next_steps: List[str]


class PersonalizedTutor:
    """
    AI tutor that adapts to user's knowledge and learning style.
    """
    
    def __init__(self, memory_pipeline: MemoryPipeline, user_id: str):
        self.pipeline = memory_pipeline
        self.user_id = user_id
        
        # User profile (learned from memory)
        self.skill_levels: Dict[str, SkillLevel] = {}
        self.learning_style: Optional[LearningStyle] = None
        self.known_concepts: List[str] = []
        self.struggling_with: List[str] = []
        
        logger.info(f"PersonalizedTutor initialized for user {user_id}")
    
    async def assess_skill_level(self, topic: str) -> Tuple[SkillLevel, float, str]:
        """
        Assess user's skill level in a topic based on memory.
        
        Returns:
            (skill_level, confidence, reasoning)
        """
        # Get all atoms related to topic
        atoms = await self.pipeline.store.get_atoms_by_subject(self.user_id)
        topic_atoms = [a for a in atoms if topic.lower() in a.object.lower()]
        
        if not topic_atoms:
            return SkillLevel.NOVICE, 0.5, f"No knowledge found about {topic}"
        
        # Analyze skill indicators
        expert_indicators = ["expert", "proficient", "advanced", "mastered"]
        intermediate_indicators = ["familiar", "know", "understand", "can do"]
        beginner_indicators = ["learning", "studying", "trying"]
        struggle_indicators = ["struggle", "difficult", "don't understand", "confused"]
        
        expert_count = sum(1 for a in topic_atoms if any(ind in a.object.lower() for ind in expert_indicators))
        intermediate_count = sum(1 for a in topic_atoms if any(ind in a.object.lower() for ind in intermediate_indicators))
        beginner_count = sum(1 for a in topic_atoms if any(ind in a.object.lower() for ind in beginner_indicators))
        struggle_count = sum(1 for a in topic_atoms if any(ind in a.object.lower() for ind in struggle_indicators))
        
        # Determine skill level
        if expert_count > 0:
            level = SkillLevel.EXPERT
            confidence = 0.9
            reasoning = f"User has {expert_count} expert-level indicators for {topic}"
        elif intermediate_count > beginner_count and struggle_count == 0:
            level = SkillLevel.INTERMEDIATE
            confidence = 0.8
            reasoning = f"User shows {intermediate_count} intermediate-level indicators"
        elif struggle_count > 0:
            level = SkillLevel.BEGINNER
            confidence = 0.7
            reasoning = f"User struggling with {topic} ({struggle_count} struggle indicators)"
        elif beginner_count > 0:
            level = SkillLevel.BEGINNER
            confidence = 0.8
            reasoning = f"User is learning {topic} ({beginner_count} learning indicators)"
        else:
            level = SkillLevel.NOVICE
            confidence = 0.6
            reasoning = f"Limited information about {topic} knowledge"
        
        self.skill_levels[topic] = level
        return level, confidence, reasoning
    
    async def identify_knowledge_gaps(self, topic: str) -> List[KnowledgeGap]:
        """
        Identify gaps in user's knowledge about a topic.
        
        Use case: "User knows React but doesn't know JavaScript fundamentals"
        """
        gaps = []
        
        # Get user's knowledge atoms
        atoms = await self.pipeline.store.get_atoms_by_subject(self.user_id)
        
        # Define prerequisite relationships
        prerequisites = {
            "react": ["javascript", "html", "css"],
            "machine_learning": ["python", "linear_algebra", "statistics"],
            "backend_development": ["databases", "apis", "server"],
            "data_science": ["python", "statistics", "pandas"],
        }
        
        topic_lower = topic.lower()
        
        # Check if user knows topic but missing prerequisites
        if topic_lower in prerequisites:
            knows_topic = any(topic_lower in a.object.lower() for a in atoms)
            
            if knows_topic:
                for prereq in prerequisites[topic_lower]:
                    knows_prereq = any(prereq in a.object.lower() for a in atoms)
                    
                    if not knows_prereq:
                        gaps.append(KnowledgeGap(
                            topic=prereq,
                            gap_type="prerequisite",
                            severity=0.8,
                            description=f"Missing prerequisite: {prereq} is needed for {topic}",
                            suggested_resources=[
                                f"Learn {prereq} basics",
                                f"Review {prereq} fundamentals"
                            ]
                        ))
        
        # Check for misconceptions (user thinks they know but struggles)
        for atom in atoms:
            if topic_lower in atom.object.lower():
                if "expert" in atom.object.lower() or "proficient" in atom.object.lower():
                    # Check if also has struggle indicators
                    struggle_atoms = [a for a in atoms if topic_lower in a.object.lower() and 
                                    any(s in a.object.lower() for s in ["struggle", "difficult", "confused"])]
                    
                    if struggle_atoms:
                        gaps.append(KnowledgeGap(
                            topic=topic,
                            gap_type="misconception",
                            severity=0.6,
                            description=f"Claims expertise but shows struggles with {topic}",
                            suggested_resources=[
                                f"Review {topic} fundamentals",
                                f"Practice {topic} with guided examples"
                            ]
                        ))
        
        return sorted(gaps, key=lambda g: g.severity, reverse=True)
    
    async def generate_personalized_explanation(
        self,
        concept: str,
        context: Optional[str] = None
    ) -> str:
        """
        Generate explanation adapted to user's skill level and learning style.
        
        Use case: Explain "recursion" differently to novice vs expert
        """
        # Assess user's skill level
        skill_level, confidence, _ = await self.assess_skill_level(concept)
        
        # Infer learning style from memory (simplified)
        atoms = await self.pipeline.store.get_atoms_by_subject(self.user_id)
        
        visual_indicators = sum(1 for a in atoms if any(v in a.object.lower() for v in ["visual", "diagram", "chart"]))
        example_indicators = sum(1 for a in atoms if any(e in a.object.lower() for e in ["example", "demo", "hands-on"]))
        
        if visual_indicators > example_indicators:
            learning_style = LearningStyle.VISUAL
        elif example_indicators > 0:
            learning_style = LearningStyle.EXAMPLE_BASED
        else:
            learning_style = LearningStyle.VERBAL
        
        # Generate explanation template based on skill + style
        if skill_level == SkillLevel.NOVICE:
            explanation = f"[NOVICE] {concept}: Start with basics, use simple language"
        elif skill_level == SkillLevel.BEGINNER:
            explanation = f"[BEGINNER] {concept}: Build on fundamentals, provide examples"
        elif skill_level == SkillLevel.INTERMEDIATE:
            explanation = f"[INTERMEDIATE] {concept}: Dive deeper, compare with related concepts"
        else:
            explanation = f"[ADVANCED] {concept}: Technical details, edge cases, optimizations"
        
        if learning_style == LearningStyle.VISUAL:
            explanation += " | Use diagrams and visual aids"
        elif learning_style == LearningStyle.EXAMPLE_BASED:
            explanation += " | Provide concrete examples and code"
        
        return explanation
    
    async def recommend_next_steps(self, current_topic: str) -> List[LearningRecommendation]:
        """
        Recommend next learning steps based on current knowledge.
        
        Use case: "You know Python basics, try data structures next"
        """
        recommendations = []
        
        # Assess current skill
        skill_level, confidence, reasoning = await self.assess_skill_level(current_topic)
        
        # Define learning paths
        learning_paths = {
            "python": {
                SkillLevel.BEGINNER: ["data_structures", "functions", "loops"],
                SkillLevel.INTERMEDIATE: ["oop", "decorators", "generators"],
                SkillLevel.ADVANCED: ["async", "metaclasses", "performance"]
            },
            "javascript": {
                SkillLevel.BEGINNER: ["dom", "events", "async"],
                SkillLevel.INTERMEDIATE: ["promises", "closures", "prototypes"],
                SkillLevel.ADVANCED: ["webpack", "typescript", "performance"]
            }
        }
        
        topic_lower = current_topic.lower()
        
        if topic_lower in learning_paths and skill_level in learning_paths[topic_lower]:
            next_topics = learning_paths[topic_lower][skill_level]
            
            for next_topic in next_topics:
                recommendations.append(LearningRecommendation(
                    topic=next_topic,
                    difficulty=skill_level,
                    reasoning=f"Natural progression from {current_topic} at {skill_level.value} level",
                    confidence=0.8,
                    next_steps=[
                        f"Study {next_topic} basics",
                        f"Practice {next_topic} with examples",
                        f"Build project using {next_topic}"
                    ]
                ))
        
        return recommendations
    
    async def track_learning_progress(self, topic: str, days: int = 30) -> Dict:
        """
        Track user's learning progress over time.
        
        Returns:
            {
                "topic": str,
                "initial_level": SkillLevel,
                "current_level": SkillLevel,
                "progress": float,  # 0.0-1.0
                "milestones": List[str]
            }
        """
        # Get atoms from last N days
        atoms = await self.pipeline.store.get_atoms_by_subject(self.user_id)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_atoms = [a for a in atoms if a.first_observed >= cutoff_date and topic.lower() in a.object.lower()]
        
        # Simplified progress tracking
        if not recent_atoms:
            return {
                "topic": topic,
                "progress": 0.0,
                "milestones": []
            }
        
        # Count learning indicators over time
        milestones = []
        for atom in sorted(recent_atoms, key=lambda a: a.first_observed):
            if "learned" in atom.object.lower() or "understand" in atom.object.lower():
                milestones.append(f"{atom.first_observed.strftime('%Y-%m-%d')}: {atom.object}")
        
        progress = min(1.0, len(milestones) / 5)  # 5 milestones = 100% progress
        
        return {
            "topic": topic,
            "progress": progress,
            "milestones": milestones,
            "days_tracked": days
        }


class PersonalizedTutorExperiment:
    """
    Experiment framework for personalized tutoring research.
    
    Research questions:
    1. Can we accurately assess skill levels from memory?
    2. Can we identify knowledge gaps?
    3. Can we personalize explanations effectively?
    """
    
    def __init__(self, memory_pipeline: MemoryPipeline, user_id: str):
        self.tutor = PersonalizedTutor(memory_pipeline, user_id)
        self.results = []
    
    async def run_skill_assessment_experiment(self, topics: List[str]) -> Dict:
        """
        Experiment: Assess skill levels across multiple topics.
        """
        assessments = {}
        
        for topic in topics:
            level, confidence, reasoning = await self.tutor.assess_skill_level(topic)
            assessments[topic] = {
                "skill_level": level.value,
                "confidence": confidence,
                "reasoning": reasoning
            }
        
        result = {
            "experiment": "skill_assessment",
            "topics": topics,
            "assessments": assessments
        }
        
        self.results.append(result)
        return result
    
    async def run_gap_analysis_experiment(self, topics: List[str]) -> Dict:
        """
        Experiment: Identify knowledge gaps.
        """
        all_gaps = {}
        
        for topic in topics:
            gaps = await self.tutor.identify_knowledge_gaps(topic)
            all_gaps[topic] = [
                {
                    "gap_topic": g.topic,
                    "type": g.gap_type,
                    "severity": g.severity,
                    "description": g.description
                }
                for g in gaps
            ]
        
        result = {
            "experiment": "gap_analysis",
            "topics": topics,
            "gaps": all_gaps,
            "total_gaps": sum(len(gaps) for gaps in all_gaps.values())
        }
        
        self.results.append(result)
        return result
    
    def get_summary(self) -> Dict:
        """Get summary of all experiments"""
        return {
            "total_experiments": len(self.results),
            "experiments": self.results
        }
