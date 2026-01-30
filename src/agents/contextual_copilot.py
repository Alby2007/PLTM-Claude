"""
Contextual Copilot

Code suggestions and assistance based on user's accumulated preferences and style.

Key features:
- Remembers user's coding style preferences
- Learns from past mistakes
- Suggests code based on user's skill level
- Adapts to user's preferred patterns
- Warns about previously encountered pitfalls

Research potential: "Memory-Guided Code Generation"
"""

from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass

from loguru import logger

from src.core.models import MemoryAtom
from src.pipeline.memory_pipeline import MemoryPipeline


@dataclass
class CodeSuggestion:
    """Code suggestion adapted to user"""
    code: str
    reasoning: str
    confidence: float
    based_on_preferences: List[str]
    warnings: List[str]


@dataclass
class CodingPattern:
    """Learned coding pattern"""
    pattern_type: str  # "style", "library", "architecture"
    description: str
    frequency: int
    examples: List[str]


class ContextualCopilot:
    """
    AI coding assistant that adapts to user's style and preferences.
    """
    
    def __init__(self, memory_pipeline: MemoryPipeline, user_id: str):
        self.pipeline = memory_pipeline
        self.user_id = user_id
        
        # Learned preferences
        self.style_preferences: Dict[str, str] = {}
        self.library_preferences: Dict[str, List[str]] = {}
        self.past_mistakes: List[str] = []
        self.coding_patterns: List[CodingPattern] = []
        
        logger.info(f"ContextualCopilot initialized for user {user_id}")
    
    async def learn_preferences(self):
        """
        Learn user's coding preferences from memory.
        """
        atoms = await self.pipeline.store.get_atoms_by_subject(self.user_id)
        
        # Extract style preferences
        for atom in atoms:
            obj_lower = atom.object.lower()
            
            # Indentation
            if "tabs" in obj_lower or "spaces" in obj_lower:
                if "prefer" in atom.predicate or "like" in atom.predicate:
                    self.style_preferences["indentation"] = "tabs" if "tabs" in obj_lower else "spaces"
            
            # Naming convention
            if "camelcase" in obj_lower or "snake_case" in obj_lower:
                if "prefer" in atom.predicate:
                    self.style_preferences["naming"] = "camelCase" if "camelcase" in obj_lower else "snake_case"
            
            # Framework preferences
            if any(fw in obj_lower for fw in ["react", "vue", "angular", "svelte"]):
                if "prefer" in atom.predicate or "love" in atom.predicate:
                    framework = next(fw for fw in ["react", "vue", "angular", "svelte"] if fw in obj_lower)
                    self.library_preferences.setdefault("frontend", []).append(framework)
            
            # Past mistakes
            if "mistake" in obj_lower or "bug" in obj_lower or "error" in obj_lower:
                self.past_mistakes.append(atom.object)
        
        logger.info(f"Learned {len(self.style_preferences)} style preferences, {len(self.past_mistakes)} past mistakes")
    
    async def suggest_code(
        self,
        task: str,
        language: str,
        context: Optional[str] = None
    ) -> CodeSuggestion:
        """
        Suggest code adapted to user's style and preferences.
        
        Use case: "Generate React component using user's preferred patterns"
        """
        await self.learn_preferences()
        
        # Build suggestion based on preferences
        preferences_used = []
        warnings = []
        
        # Apply style preferences
        code_template = f"# Code suggestion for: {task}\n"
        
        if "indentation" in self.style_preferences:
            indent = "\t" if self.style_preferences["indentation"] == "tabs" else "    "
            preferences_used.append(f"Using {self.style_preferences['indentation']} for indentation")
        else:
            indent = "    "
        
        # Check for relevant past mistakes
        for mistake in self.past_mistakes:
            if any(keyword in mistake.lower() for keyword in task.lower().split()):
                warnings.append(f"Warning: Previously encountered issue with {mistake}")
        
        # Generate code (simplified - in production would use LLM)
        if language.lower() == "python":
            code = f"{code_template}\ndef {task.replace(' ', '_')}():\n{indent}pass\n"
        elif language.lower() == "javascript":
            naming = self.style_preferences.get("naming", "camelCase")
            func_name = task.replace(" ", "_") if naming == "snake_case" else task.replace(" ", "")
            code = f"{code_template}\nfunction {func_name}() {{\n{indent}// TODO: Implement\n}}\n"
        else:
            code = f"{code_template}\n// TODO: Implement {task}\n"
        
        return CodeSuggestion(
            code=code,
            reasoning=f"Generated based on user's {language} preferences",
            confidence=0.8 if preferences_used else 0.5,
            based_on_preferences=preferences_used,
            warnings=warnings
        )
    
    async def detect_antipatterns(self, code: str, language: str) -> List[str]:
        """
        Detect antipatterns based on user's past mistakes.
        
        Use case: "Warn if user is about to repeat a past mistake"
        """
        await self.learn_preferences()
        
        warnings = []
        
        # Check against past mistakes
        for mistake in self.past_mistakes:
            # Simplified pattern matching
            if any(keyword in code.lower() for keyword in mistake.lower().split()[:3]):
                warnings.append(f"Potential issue: Similar to past mistake '{mistake}'")
        
        # Check against known antipatterns
        antipatterns = {
            "python": [
                ("except:", "Bare except clause - specify exception type"),
                ("== None", "Use 'is None' instead of '== None'"),
            ],
            "javascript": [
                ("== ", "Use === for strict equality"),
                ("var ", "Use let/const instead of var"),
            ]
        }
        
        if language.lower() in antipatterns:
            for pattern, warning in antipatterns[language.lower()]:
                if pattern in code:
                    warnings.append(warning)
        
        return warnings
    
    async def suggest_refactoring(self, code: str, language: str) -> List[str]:
        """
        Suggest refactorings based on user's preferred patterns.
        
        Use case: "Suggest converting to user's preferred style"
        """
        await self.learn_preferences()
        
        suggestions = []
        
        # Check indentation
        if "indentation" in self.style_preferences:
            preferred = self.style_preferences["indentation"]
            if preferred == "spaces" and "\t" in code:
                suggestions.append("Convert tabs to spaces (user preference)")
            elif preferred == "tabs" and "    " in code:
                suggestions.append("Convert spaces to tabs (user preference)")
        
        # Check naming convention
        if "naming" in self.style_preferences and language.lower() == "javascript":
            preferred = self.style_preferences["naming"]
            if preferred == "snake_case" and any(c.isupper() for c in code):
                suggestions.append("Convert to snake_case (user preference)")
            elif preferred == "camelCase" and "_" in code:
                suggestions.append("Convert to camelCase (user preference)")
        
        return suggestions
    
    async def track_coding_patterns(self) -> List[CodingPattern]:
        """
        Identify recurring coding patterns from user's memory.
        
        Use case: "User always uses async/await, suggest it by default"
        """
        atoms = await self.pipeline.store.get_atoms_by_subject(self.user_id)
        
        patterns = []
        
        # Track library usage
        libraries = {}
        for atom in atoms:
            for lib in ["react", "vue", "django", "flask", "express", "fastapi"]:
                if lib in atom.object.lower():
                    libraries[lib] = libraries.get(lib, 0) + 1
        
        for lib, count in libraries.items():
            if count >= 3:  # Used at least 3 times
                patterns.append(CodingPattern(
                    pattern_type="library",
                    description=f"Frequently uses {lib}",
                    frequency=count,
                    examples=[f"Uses {lib} for projects"]
                ))
        
        self.coding_patterns = patterns
        return patterns


class ContextualCopilotExperiment:
    """
    Experiment framework for contextual copilot research.
    """
    
    def __init__(self, memory_pipeline: MemoryPipeline, user_id: str):
        self.copilot = ContextualCopilot(memory_pipeline, user_id)
        self.results = []
    
    async def run_preference_learning_experiment(self) -> Dict:
        """
        Experiment: Learn user's coding preferences.
        """
        await self.copilot.learn_preferences()
        
        result = {
            "experiment": "preference_learning",
            "style_preferences": self.copilot.style_preferences,
            "library_preferences": self.copilot.library_preferences,
            "past_mistakes_count": len(self.copilot.past_mistakes)
        }
        
        self.results.append(result)
        return result
    
    async def run_pattern_detection_experiment(self) -> Dict:
        """
        Experiment: Detect coding patterns.
        """
        patterns = await self.copilot.track_coding_patterns()
        
        result = {
            "experiment": "pattern_detection",
            "patterns_found": len(patterns),
            "patterns": [
                {
                    "type": p.pattern_type,
                    "description": p.description,
                    "frequency": p.frequency
                }
                for p in patterns
            ]
        }
        
        self.results.append(result)
        return result
    
    def get_summary(self) -> Dict:
        """Get summary of all experiments"""
        return {
            "total_experiments": len(self.results),
            "experiments": self.results
        }
