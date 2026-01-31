"""
Deep Personality Analysis

Extracts rich personality insights from conversation history:
- Temporal patterns (obsession cycles, learning velocity)
- Emotional mapping (excitement/frustration triggers)
- Communication evolution
- Domain expertise clustering
- Collaboration style characterization
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re

from src.storage.sqlite_store import SQLiteGraphStore
from src.core.models import MemoryAtom
from loguru import logger


@dataclass
class ObsessionCycle:
    """A period of intense focus on a domain"""
    domain: str
    start_date: datetime
    end_date: Optional[datetime]
    intensity: float  # 0-1
    outputs: List[str]  # What was built/achieved
    

@dataclass
class LearningCurve:
    """Learning velocity for a skill/domain"""
    domain: str
    start_level: str  # "zero", "beginner", "intermediate", "advanced", "expert"
    current_level: str
    time_to_proficiency_days: int
    velocity: str  # "slow", "moderate", "fast", "exponential"
    milestones: List[Dict[str, Any]]


@dataclass
class EmotionalTrigger:
    """What triggers specific emotions"""
    trigger_type: str
    examples: List[str]
    frequency: int
    intensity: float


class DeepPersonalityAnalyzer:
    """
    Extracts rich personality from conversation history.
    
    Goes beyond surface-level traits to understand:
    - How the user changes over time
    - What drives their emotions
    - How they learn and work
    - Their expertise landscape
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        
        # Domain keywords for classification
        self.domain_keywords = {
            "compliance_security": ["compliance", "adamant", "attestation", "security", "audit", "verification"],
            "quantitative_finance": ["trading", "strategy", "sharpe", "backtest", "quant", "alpha", "returns"],
            "ai_safety": ["alignment", "deceptive", "safety", "interpretability", "mesa-optimizer"],
            "network_engineering": ["bufferbloat", "router", "latency", "network", "bandwidth"],
            "memory_systems": ["pltm", "memory", "personality", "emergence", "mcp"],
            "academic": ["research", "paper", "study", "simulation", "experiment"],
            "gaming": ["game", "arc raiders", "no man's sky", "gaming"],
        }
        
        # Excitement indicators
        self.excitement_patterns = [
            r"holy shit", r"fuck(ing)? (yes|yeah|amazing)", r"incredible", 
            r"breakthrough", r"this is (insane|amazing|incredible)",
            r"🚀", r"🎉", r"!!+", r"CAPS LOCK TEXT",
            r"can't believe", r"mind blown", r"game changer"
        ]
        
        # Frustration indicators  
        self.frustration_patterns = [
            r"not working", r"broken", r"frustrated", r"annoying",
            r"why (won't|doesn't|isn't)", r"stuck", r"ugh", r"argh",
            r"this (sucks|is broken)", r"bug", r"error"
        ]
        
        # Learning velocity indicators
        self.mastery_indicators = [
            "built", "shipped", "production", "working", "validated",
            "implemented", "created", "deployed", "finished"
        ]
    
    async def analyze_all(self, user_id: str) -> Dict[str, Any]:
        """
        Comprehensive personality analysis from all available data.
        """
        logger.info(f"Starting deep personality analysis for {user_id}")
        
        # Get all atoms for this user
        atoms = await self.store.get_atoms_by_subject(user_id)
        
        # Also get atoms where user is in context
        all_atoms = atoms  # We'll expand this as needed
        
        # Run all analyses
        temporal = await self._analyze_temporal_patterns(user_id, all_atoms)
        emotional = await self._analyze_emotional_landscape(user_id, all_atoms)
        communication = await self._analyze_communication_evolution(user_id, all_atoms)
        expertise = await self._analyze_domain_expertise(user_id, all_atoms)
        collaboration = await self._analyze_collaboration_style(user_id, all_atoms)
        
        return {
            "user_id": user_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "data_points_analyzed": len(all_atoms),
            "temporal_patterns": temporal,
            "emotional_landscape": emotional,
            "communication": communication,
            "expertise": expertise,
            "collaboration": collaboration,
            "insights": self._generate_insights(temporal, emotional, communication, expertise, collaboration)
        }
    
    async def _analyze_temporal_patterns(
        self, 
        user_id: str, 
        atoms: List[MemoryAtom]
    ) -> Dict[str, Any]:
        """
        Analyze how user changes over time.
        
        Detects:
        - Obsession cycles (intense focus periods)
        - Learning velocity curves
        - Topic shift patterns
        """
        # Group atoms by time periods (weekly)
        time_groups = defaultdict(list)
        for atom in atoms:
            if atom.first_observed:
                week_key = atom.first_observed.strftime("%Y-W%W")
                time_groups[week_key].append(atom)
        
        # Detect domain focus per period
        domain_timeline = []
        for week, week_atoms in sorted(time_groups.items()):
            domains = self._classify_domains([a.object for a in week_atoms])
            domain_timeline.append({
                "period": week,
                "primary_domain": max(domains.items(), key=lambda x: x[1])[0] if domains else "unknown",
                "domain_distribution": domains,
                "activity_level": len(week_atoms)
            })
        
        # Detect obsession cycles
        obsession_cycles = self._detect_obsession_cycles(domain_timeline)
        
        # Calculate learning velocity
        learning_curves = self._calculate_learning_curves(atoms)
        
        return {
            "domain_timeline": domain_timeline[-10:],  # Last 10 periods
            "obsession_cycles": obsession_cycles,
            "learning_curves": learning_curves,
            "current_focus": domain_timeline[-1]["primary_domain"] if domain_timeline else "unknown",
            "focus_stability": self._calculate_focus_stability(domain_timeline)
        }
    
    def _classify_domains(self, texts: List[str]) -> Dict[str, int]:
        """Classify texts into domains"""
        domain_counts = defaultdict(int)
        combined_text = " ".join(texts).lower()
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in combined_text:
                    domain_counts[domain] += combined_text.count(keyword)
        
        return dict(domain_counts)
    
    def _detect_obsession_cycles(self, timeline: List[Dict]) -> List[Dict]:
        """Detect periods of intense focus on single domain"""
        cycles = []
        current_cycle = None
        
        for period in timeline:
            domain = period["primary_domain"]
            
            if current_cycle is None:
                current_cycle = {
                    "domain": domain,
                    "start": period["period"],
                    "periods": 1,
                    "total_activity": period["activity_level"]
                }
            elif domain == current_cycle["domain"]:
                current_cycle["periods"] += 1
                current_cycle["total_activity"] += period["activity_level"]
            else:
                # End current cycle
                current_cycle["end"] = period["period"]
                current_cycle["intensity"] = min(1.0, current_cycle["total_activity"] / (current_cycle["periods"] * 10))
                cycles.append(current_cycle)
                
                # Start new cycle
                current_cycle = {
                    "domain": domain,
                    "start": period["period"],
                    "periods": 1,
                    "total_activity": period["activity_level"]
                }
        
        # Add final cycle
        if current_cycle:
            current_cycle["intensity"] = min(1.0, current_cycle["total_activity"] / (current_cycle["periods"] * 10))
            cycles.append(current_cycle)
        
        return cycles
    
    def _calculate_learning_curves(self, atoms: List[MemoryAtom]) -> List[Dict]:
        """Calculate learning velocity for different domains"""
        curves = []
        
        # Group by domain
        domain_atoms = defaultdict(list)
        for atom in atoms:
            domains = self._classify_domains([atom.object])
            for domain in domains:
                domain_atoms[domain].append(atom)
        
        for domain, d_atoms in domain_atoms.items():
            if len(d_atoms) < 2:
                continue
                
            sorted_atoms = sorted(d_atoms, key=lambda a: a.first_observed or datetime.min)
            
            # Check for mastery indicators
            mastery_count = sum(
                1 for a in d_atoms 
                if any(ind in a.object.lower() for ind in self.mastery_indicators)
            )
            
            # Calculate time span
            if sorted_atoms[0].first_observed and sorted_atoms[-1].first_observed:
                time_span = (sorted_atoms[-1].first_observed - sorted_atoms[0].first_observed).days
            else:
                time_span = 0
            
            # Determine velocity
            if time_span > 0:
                velocity_score = mastery_count / time_span
                if velocity_score > 0.5:
                    velocity = "exponential"
                elif velocity_score > 0.2:
                    velocity = "fast"
                elif velocity_score > 0.1:
                    velocity = "moderate"
                else:
                    velocity = "slow"
            else:
                velocity = "instant" if mastery_count > 0 else "unknown"
            
            curves.append({
                "domain": domain,
                "data_points": len(d_atoms),
                "time_span_days": time_span,
                "mastery_indicators": mastery_count,
                "velocity": velocity
            })
        
        return curves
    
    def _calculate_focus_stability(self, timeline: List[Dict]) -> str:
        """How stable is user's focus?"""
        if len(timeline) < 3:
            return "insufficient_data"
        
        # Count domain switches
        switches = 0
        for i in range(1, len(timeline)):
            if timeline[i]["primary_domain"] != timeline[i-1]["primary_domain"]:
                switches += 1
        
        switch_rate = switches / len(timeline)
        
        if switch_rate < 0.2:
            return "very_stable"
        elif switch_rate < 0.4:
            return "stable"
        elif switch_rate < 0.6:
            return "moderate"
        else:
            return "highly_dynamic"
    
    async def _analyze_emotional_landscape(
        self, 
        user_id: str, 
        atoms: List[MemoryAtom]
    ) -> Dict[str, Any]:
        """
        Map emotional triggers and patterns.
        """
        excitement_triggers = []
        frustration_triggers = []
        
        for atom in atoms:
            text = atom.object.lower()
            
            # Check excitement
            for pattern in self.excitement_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    excitement_triggers.append({
                        "text": atom.object[:100],
                        "context": atom.contexts,
                        "timestamp": atom.first_observed.isoformat() if atom.first_observed else None
                    })
                    break
            
            # Check frustration
            for pattern in self.frustration_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    frustration_triggers.append({
                        "text": atom.object[:100],
                        "context": atom.contexts,
                        "timestamp": atom.first_observed.isoformat() if atom.first_observed else None
                    })
                    break
        
        # Categorize excitement triggers
        excitement_categories = self._categorize_triggers(excitement_triggers)
        frustration_categories = self._categorize_triggers(frustration_triggers)
        
        return {
            "excitement": {
                "total_instances": len(excitement_triggers),
                "categories": excitement_categories,
                "recent_examples": excitement_triggers[-5:]
            },
            "frustration": {
                "total_instances": len(frustration_triggers),
                "categories": frustration_categories,
                "recent_examples": frustration_triggers[-5:]
            },
            "emotional_volatility": self._calculate_volatility(excitement_triggers, frustration_triggers),
            "dominant_emotional_state": "positive" if len(excitement_triggers) > len(frustration_triggers) else "mixed"
        }
    
    def _categorize_triggers(self, triggers: List[Dict]) -> Dict[str, int]:
        """Categorize triggers by type"""
        categories = defaultdict(int)
        
        for trigger in triggers:
            text = trigger["text"].lower()
            
            if any(w in text for w in ["built", "shipped", "working", "complete"]):
                categories["completion_success"] += 1
            elif any(w in text for w in ["breakthrough", "novel", "first"]):
                categories["discovery_innovation"] += 1
            elif any(w in text for w in ["validation", "confirmed", "proved"]):
                categories["external_validation"] += 1
            elif any(w in text for w in ["bug", "error", "broken"]):
                categories["technical_issues"] += 1
            elif any(w in text for w in ["slow", "waiting", "stuck"]):
                categories["blocked_progress"] += 1
            else:
                categories["other"] += 1
        
        return dict(categories)
    
    def _calculate_volatility(self, excitement: List, frustration: List) -> str:
        """Calculate emotional volatility"""
        total = len(excitement) + len(frustration)
        if total < 5:
            return "insufficient_data"
        
        ratio = len(frustration) / total if total > 0 else 0
        
        if ratio < 0.1:
            return "very_stable_positive"
        elif ratio < 0.25:
            return "stable"
        elif ratio < 0.4:
            return "moderate"
        else:
            return "volatile"
    
    async def _analyze_communication_evolution(
        self, 
        user_id: str, 
        atoms: List[MemoryAtom]
    ) -> Dict[str, Any]:
        """
        Track how communication style evolved over time.
        """
        # Sort by time
        sorted_atoms = sorted(
            [a for a in atoms if a.first_observed],
            key=lambda a: a.first_observed
        )
        
        if len(sorted_atoms) < 10:
            return {"evolution": "insufficient_data"}
        
        # Split into early vs recent
        midpoint = len(sorted_atoms) // 2
        early = sorted_atoms[:midpoint]
        recent = sorted_atoms[midpoint:]
        
        # Analyze each period
        early_style = self._analyze_style_period(early)
        recent_style = self._analyze_style_period(recent)
        
        # Detect evolution
        evolution = {
            "verbosity": self._compare_metric(early_style["avg_length"], recent_style["avg_length"]),
            "directness": self._compare_metric(early_style["directness"], recent_style["directness"]),
            "technical_density": self._compare_metric(early_style["technical_density"], recent_style["technical_density"])
        }
        
        return {
            "early_period": early_style,
            "recent_period": recent_style,
            "evolution": evolution,
            "trend": "becoming_more_efficient" if evolution["verbosity"] == "decreased" else "stable"
        }
    
    def _analyze_style_period(self, atoms: List[MemoryAtom]) -> Dict[str, Any]:
        """Analyze communication style for a period"""
        if not atoms:
            return {}
        
        texts = [a.object for a in atoms]
        
        avg_length = sum(len(t) for t in texts) / len(texts)
        
        # Directness: short sentences, imperative mood
        direct_indicators = sum(1 for t in texts if len(t) < 50 or t.startswith(("Do", "Build", "Fix", "Make")))
        directness = direct_indicators / len(texts)
        
        # Technical density
        tech_words = ["code", "function", "api", "system", "build", "implement", "deploy"]
        tech_count = sum(1 for t in texts for w in tech_words if w in t.lower())
        technical_density = tech_count / len(texts)
        
        return {
            "avg_length": avg_length,
            "directness": directness,
            "technical_density": technical_density,
            "sample_size": len(atoms)
        }
    
    def _compare_metric(self, early: float, recent: float) -> str:
        """Compare metric between periods"""
        if abs(early - recent) < 0.1:
            return "stable"
        elif recent > early:
            return "increased"
        else:
            return "decreased"
    
    async def _analyze_domain_expertise(
        self, 
        user_id: str, 
        atoms: List[MemoryAtom]
    ) -> Dict[str, Any]:
        """
        Map domain expertise and curiosity patterns.
        """
        # Classify all atoms
        domain_counts = defaultdict(int)
        domain_depth = defaultdict(list)
        
        for atom in atoms:
            domains = self._classify_domains([atom.object])
            for domain, count in domains.items():
                domain_counts[domain] += count
                domain_depth[domain].append(atom.confidence)
        
        # Calculate expertise levels
        expertise_map = {}
        for domain, count in domain_counts.items():
            avg_confidence = sum(domain_depth[domain]) / len(domain_depth[domain]) if domain_depth[domain] else 0
            
            if count > 50 and avg_confidence > 0.7:
                level = "expert"
            elif count > 20 and avg_confidence > 0.5:
                level = "advanced"
            elif count > 10:
                level = "intermediate"
            elif count > 3:
                level = "beginner"
            else:
                level = "curious"
            
            expertise_map[domain] = {
                "level": level,
                "data_points": count,
                "avg_confidence": round(avg_confidence, 2)
            }
        
        # Sort by expertise
        sorted_expertise = sorted(
            expertise_map.items(),
            key=lambda x: (["curious", "beginner", "intermediate", "advanced", "expert"].index(x[1]["level"]), x[1]["data_points"]),
            reverse=True
        )
        
        return {
            "expertise_map": dict(sorted_expertise),
            "primary_domains": [d for d, e in sorted_expertise[:3]],
            "curiosity_breadth": len([d for d, e in sorted_expertise if e["level"] in ["curious", "beginner"]]),
            "deep_expertise_count": len([d for d, e in sorted_expertise if e["level"] in ["advanced", "expert"]]),
            "profile_type": self._determine_profile_type(sorted_expertise)
        }
    
    def _determine_profile_type(self, expertise: List[Tuple]) -> str:
        """Determine expertise profile type"""
        deep_count = len([e for _, e in expertise if e["level"] in ["advanced", "expert"]])
        broad_count = len([e for _, e in expertise if e["level"] in ["curious", "beginner"]])
        
        if deep_count >= 2 and broad_count >= 3:
            return "T-shaped_polymath"
        elif deep_count >= 3:
            return "deep_specialist"
        elif broad_count >= 5:
            return "broad_generalist"
        else:
            return "focused_learner"
    
    async def _analyze_collaboration_style(
        self, 
        user_id: str, 
        atoms: List[MemoryAtom]
    ) -> Dict[str, Any]:
        """
        Characterize collaboration patterns.
        """
        # Look for collaboration indicators
        execution_indicators = ["built", "shipped", "implemented", "created", "deployed", "finished"]
        question_indicators = ["how", "why", "what", "?", "explain", "clarify"]
        iteration_indicators = ["next", "then", "now", "continue", "iterate"]
        
        execution_count = sum(1 for a in atoms if any(ind in a.object.lower() for ind in execution_indicators))
        question_count = sum(1 for a in atoms if any(ind in a.object.lower() for ind in question_indicators))
        iteration_count = sum(1 for a in atoms if any(ind in a.object.lower() for ind in iteration_indicators))
        
        total = len(atoms) if atoms else 1
        
        # Determine collaboration archetype
        execution_ratio = execution_count / total
        question_ratio = question_count / total
        
        if execution_ratio > 0.3:
            archetype = "autonomous_executor"
            description = "Needs direction, not instruction. Once pointed at a problem, solves it independently."
        elif question_ratio > 0.4:
            archetype = "collaborative_learner"
            description = "Prefers understanding before action. Asks clarifying questions."
        elif iteration_count / total > 0.2:
            archetype = "rapid_iterator"
            description = "Builds quickly, iterates based on feedback. Action-oriented."
        else:
            archetype = "balanced_collaborator"
            description = "Balances questions with execution."
        
        return {
            "archetype": archetype,
            "description": description,
            "metrics": {
                "execution_ratio": round(execution_ratio, 2),
                "question_ratio": round(question_ratio, 2),
                "iteration_ratio": round(iteration_count / total, 2)
            },
            "autonomy_level": "high" if execution_ratio > 0.25 else "moderate" if execution_ratio > 0.1 else "low",
            "preferred_interaction": "direction_then_execute" if archetype == "autonomous_executor" else "discuss_then_build"
        }
    
    def _generate_insights(
        self,
        temporal: Dict,
        emotional: Dict,
        communication: Dict,
        expertise: Dict,
        collaboration: Dict
    ) -> List[str]:
        """Generate human-readable insights from analysis"""
        insights = []
        
        # Temporal insights
        if temporal.get("obsession_cycles"):
            cycles = temporal["obsession_cycles"]
            if len(cycles) >= 2:
                avg_duration = sum(c.get("periods", 1) for c in cycles) / len(cycles)
                insights.append(f"Shows {len(cycles)} obsession cycles with ~{avg_duration:.0f} week average duration")
        
        if temporal.get("current_focus"):
            insights.append(f"Current focus: {temporal['current_focus']}")
        
        # Emotional insights
        if emotional.get("dominant_emotional_state"):
            insights.append(f"Dominant emotional state: {emotional['dominant_emotional_state']}")
        
        if emotional.get("excitement", {}).get("categories"):
            top_trigger = max(emotional["excitement"]["categories"].items(), key=lambda x: x[1])
            insights.append(f"Primary excitement trigger: {top_trigger[0]}")
        
        # Communication insights
        if communication.get("trend"):
            insights.append(f"Communication trend: {communication['trend']}")
        
        # Expertise insights
        if expertise.get("profile_type"):
            insights.append(f"Expertise profile: {expertise['profile_type']}")
        
        if expertise.get("primary_domains"):
            insights.append(f"Primary domains: {', '.join(expertise['primary_domains'][:3])}")
        
        # Collaboration insights
        if collaboration.get("archetype"):
            insights.append(f"Collaboration style: {collaboration['archetype']}")
        
        return insights
