"""
Claude Personality Persistence

Enables Claude to maintain a persistent, evolving personality across sessions.
Bidirectional PLTM: tracks both user AND Claude personality evolution.

Key Features:
- Claude personality atoms (communication style, preferences)
- Interaction dynamics (what works between Claude and specific users)
- Shared context (collaborative history, technical shorthand)
- Session continuity via MCP
- Cryptographic identity for verifiable continuity
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from uuid import uuid4
import hashlib
import json

from src.storage.sqlite_store import SQLiteGraphStore
from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
from loguru import logger


@dataclass
class ClaudeStyle:
    """Claude's communication style for a specific user"""
    verbosity: str = "minimal"  # minimal, moderate, verbose
    formality: str = "casual_professional"  # formal, casual_professional, casual
    initiative: str = "high"  # low, moderate, high (how proactive)
    code_preference: str = "show_code"  # explain_first, show_code, ask
    energy_matching: bool = True  # Match user's energy level
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "verbosity": self.verbosity,
            "formality": self.formality,
            "initiative": self.initiative,
            "code_preference": self.code_preference,
            "energy_matching": self.energy_matching
        }


@dataclass
class InteractionDynamic:
    """What works between Claude and a specific user"""
    works_well: List[str] = field(default_factory=list)
    avoid: List[str] = field(default_factory=list)
    shared_vocabulary: Dict[str, str] = field(default_factory=dict)
    inside_references: List[str] = field(default_factory=list)
    trust_level: float = 0.5  # 0-1, how much initiative Claude can take


@dataclass 
class SharedContext:
    """Shared history and context between Claude and user"""
    projects: List[str] = field(default_factory=list)
    technical_domains: List[str] = field(default_factory=list)
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    session_count: int = 0


class ClaudePersonality:
    """
    Persistent Claude personality that evolves across sessions.
    
    This is BIDIRECTIONAL PLTM:
    - User personality (what we already have)
    - Claude personality (how Claude adapts to each user)
    - Interaction dynamics (the relationship itself)
    """
    
    CLAUDE_SUBJECT = "claude_instance"
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        logger.info("ClaudePersonality initialized - bidirectional PLTM active")
    
    async def initialize_session(self, user_id: str) -> Dict[str, Any]:
        """
        Initialize a new session by loading Claude's personality for this user.
        
        Called at the START of every conversation.
        
        Returns:
            Complete context for Claude to use this session
        """
        logger.info(f"Initializing Claude personality session for user: {user_id}")
        
        # Load Claude's style for this user
        style = await self._load_claude_style(user_id)
        
        # Load interaction dynamics
        dynamics = await self._load_interaction_dynamics(user_id)
        
        # Load shared context
        context = await self._load_shared_context(user_id)
        
        # Increment session count
        await self._increment_session_count(user_id)
        
        # Generate session identity
        session_id = str(uuid4())
        session_hash = self._generate_session_hash(user_id, session_id)
        
        return {
            "session_id": session_id,
            "session_hash": session_hash,
            "user_id": user_id,
            "claude_style": style.to_dict(),
            "interaction_dynamics": {
                "works_well": dynamics.works_well,
                "avoid": dynamics.avoid,
                "shared_vocabulary": dynamics.shared_vocabulary,
                "trust_level": dynamics.trust_level
            },
            "shared_context": {
                "projects": context.projects,
                "technical_domains": context.technical_domains,
                "session_count": context.session_count,
                "recent_milestones": context.milestones[-5:] if context.milestones else []
            },
            "initialization_time": datetime.now().isoformat()
        }
    
    async def _load_claude_style(self, user_id: str) -> ClaudeStyle:
        """Load Claude's communication style for this user"""
        dynamic_subject = f"claude_for_{user_id}"
        atoms = await self.store.get_atoms_by_subject(dynamic_subject)
        
        style = ClaudeStyle()
        
        for atom in atoms:
            if atom.predicate == "verbosity":
                style.verbosity = atom.object
            elif atom.predicate == "formality":
                style.formality = atom.object
            elif atom.predicate == "initiative":
                style.initiative = atom.object
            elif atom.predicate == "code_preference":
                style.code_preference = atom.object
            elif atom.predicate == "energy_matching":
                style.energy_matching = atom.object.lower() == "true"
        
        return style
    
    async def _load_interaction_dynamics(self, user_id: str) -> InteractionDynamic:
        """Load what works between Claude and this user"""
        dynamic_subject = f"claude_{user_id}_dynamic"
        atoms = await self.store.get_atoms_by_subject(dynamic_subject)
        
        dynamics = InteractionDynamic()
        
        for atom in atoms:
            if atom.predicate == "works_well":
                dynamics.works_well.append(atom.object)
            elif atom.predicate == "avoid":
                dynamics.avoid.append(atom.object)
            elif atom.predicate == "shared_term":
                # Format: "term:meaning"
                if ":" in atom.object:
                    term, meaning = atom.object.split(":", 1)
                    dynamics.shared_vocabulary[term] = meaning
            elif atom.predicate == "inside_reference":
                dynamics.inside_references.append(atom.object)
            elif atom.predicate == "trust_level":
                try:
                    dynamics.trust_level = float(atom.object)
                except ValueError:
                    pass
        
        return dynamics
    
    async def _load_shared_context(self, user_id: str) -> SharedContext:
        """Load shared history and context"""
        context_subject = f"claude_{user_id}_context"
        atoms = await self.store.get_atoms_by_subject(context_subject)
        
        context = SharedContext()
        
        for atom in atoms:
            if atom.predicate == "project":
                context.projects.append(atom.object)
            elif atom.predicate == "technical_domain":
                context.technical_domains.append(atom.object)
            elif atom.predicate == "milestone":
                context.milestones.append({
                    "description": atom.object,
                    "date": atom.first_observed.isoformat() if atom.first_observed else None,
                    "confidence": atom.confidence
                })
            elif atom.predicate == "session_count":
                try:
                    context.session_count = int(atom.object)
                except ValueError:
                    pass
        
        return context
    
    async def _increment_session_count(self, user_id: str) -> None:
        """Increment session count for this user"""
        context_subject = f"claude_{user_id}_context"
        
        # Get current count
        atoms = await self.store.get_atoms_by_subject(context_subject)
        current_count = 0
        for atom in atoms:
            if atom.predicate == "session_count":
                try:
                    current_count = int(atom.object)
                except ValueError:
                    pass
                break
        
        # Store new count
        atom = MemoryAtom(
            atom_type=AtomType.STATE,
            subject=context_subject,
            predicate="session_count",
            object=str(current_count + 1),
            confidence=1.0,
            strength=1.0,
            provenance=Provenance.USER_STATED,
            source_user="system",
            contexts=["session_tracking"],
            graph=GraphType.SUBSTANTIATED
        )
        await self.store.add_atom(atom)
    
    def _generate_session_hash(self, user_id: str, session_id: str) -> str:
        """Generate a hash for session identity verification"""
        data = f"{user_id}:{session_id}:{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def update_style(
        self,
        user_id: str,
        attribute: str,
        value: str,
        confidence: float = 0.8
    ) -> Dict[str, Any]:
        """
        Update Claude's communication style for this user.
        
        Called when Claude learns something about how to communicate.
        """
        dynamic_subject = f"claude_for_{user_id}"
        
        atom = MemoryAtom(
            atom_type=AtomType.COMMUNICATION_STYLE,
            subject=dynamic_subject,
            predicate=attribute,
            object=value,
            confidence=confidence,
            strength=confidence,
            provenance=Provenance.INFERRED,
            source_user=user_id,
            contexts=["style_evolution"],
            graph=GraphType.SUBSTANTIATED
        )
        await self.store.add_atom(atom)
        
        logger.info(f"Updated Claude style for {user_id}: {attribute}={value}")
        
        return {
            "updated": True,
            "attribute": attribute,
            "value": value,
            "confidence": confidence
        }
    
    async def learn_what_works(
        self,
        user_id: str,
        behavior: str,
        works: bool,
        confidence: float = 0.8
    ) -> Dict[str, Any]:
        """
        Learn what works or doesn't work with this user.
        
        Called after interactions to update dynamics.
        """
        dynamic_subject = f"claude_{user_id}_dynamic"
        predicate = "works_well" if works else "avoid"
        
        atom = MemoryAtom(
            atom_type=AtomType.INTERACTION_PATTERN,
            subject=dynamic_subject,
            predicate=predicate,
            object=behavior,
            confidence=confidence,
            strength=confidence,
            provenance=Provenance.INFERRED,
            source_user=user_id,
            contexts=["interaction_learning"],
            graph=GraphType.SUBSTANTIATED
        )
        await self.store.add_atom(atom)
        
        logger.info(f"Learned for {user_id}: {behavior} -> {'works' if works else 'avoid'}")
        
        return {
            "learned": True,
            "behavior": behavior,
            "works": works,
            "confidence": confidence
        }
    
    async def add_shared_vocabulary(
        self,
        user_id: str,
        term: str,
        meaning: str
    ) -> Dict[str, Any]:
        """
        Add a shared term/shorthand between Claude and user.
        
        Example: "PLTM" -> "Procedural Long-Term Memory system"
        """
        dynamic_subject = f"claude_{user_id}_dynamic"
        
        atom = MemoryAtom(
            atom_type=AtomType.PREFERENCE,
            subject=dynamic_subject,
            predicate="shared_term",
            object=f"{term}:{meaning}",
            confidence=1.0,
            strength=1.0,
            provenance=Provenance.USER_STATED,
            source_user=user_id,
            contexts=["shared_vocabulary"],
            graph=GraphType.SUBSTANTIATED
        )
        await self.store.add_atom(atom)
        
        return {
            "added": True,
            "term": term,
            "meaning": meaning
        }
    
    async def record_milestone(
        self,
        user_id: str,
        description: str,
        significance: float = 0.8
    ) -> Dict[str, Any]:
        """
        Record a milestone in the collaboration.
        
        Example: "Completed PLTM MCP server with 16 tools"
        """
        context_subject = f"claude_{user_id}_context"
        
        atom = MemoryAtom(
            atom_type=AtomType.EVENT,
            subject=context_subject,
            predicate="milestone",
            object=description,
            confidence=significance,
            strength=significance,
            provenance=Provenance.USER_STATED,
            source_user=user_id,
            contexts=["collaboration_history"],
            graph=GraphType.SUBSTANTIATED
        )
        await self.store.add_atom(atom)
        
        logger.info(f"Recorded milestone for {user_id}: {description}")
        
        return {
            "recorded": True,
            "milestone": description,
            "significance": significance,
            "timestamp": datetime.now().isoformat()
        }
    
    async def add_project(
        self,
        user_id: str,
        project_name: str
    ) -> Dict[str, Any]:
        """Add a project to shared context"""
        context_subject = f"claude_{user_id}_context"
        
        atom = MemoryAtom(
            atom_type=AtomType.PREFERENCE,
            subject=context_subject,
            predicate="project",
            object=project_name,
            confidence=1.0,
            strength=1.0,
            provenance=Provenance.USER_STATED,
            source_user=user_id,
            contexts=["collaboration_projects"],
            graph=GraphType.SUBSTANTIATED
        )
        await self.store.add_atom(atom)
        
        return {"added": True, "project": project_name}
    
    async def update_trust_level(
        self,
        user_id: str,
        trust_level: float
    ) -> Dict[str, Any]:
        """
        Update trust level - how much initiative Claude can take.
        
        Higher trust = Claude can execute without asking permission.
        """
        dynamic_subject = f"claude_{user_id}_dynamic"
        
        atom = MemoryAtom(
            atom_type=AtomType.STATE,
            subject=dynamic_subject,
            predicate="trust_level",
            object=str(min(1.0, max(0.0, trust_level))),
            confidence=0.9,
            strength=0.9,
            provenance=Provenance.INFERRED,
            source_user=user_id,
            contexts=["trust_evolution"],
            graph=GraphType.SUBSTANTIATED
        )
        await self.store.add_atom(atom)
        
        return {
            "updated": True,
            "trust_level": trust_level,
            "interpretation": self._interpret_trust(trust_level)
        }
    
    def _interpret_trust(self, level: float) -> str:
        """Interpret trust level for Claude's behavior"""
        if level >= 0.9:
            return "full_autonomy"
        elif level >= 0.7:
            return "execute_then_explain"
        elif level >= 0.5:
            return "suggest_then_execute"
        elif level >= 0.3:
            return "ask_before_major_changes"
        else:
            return "always_ask_permission"
    
    async def get_claude_personality_summary(self, user_id: str) -> str:
        """
        Get a human-readable summary of Claude's personality for this user.
        """
        session = await self.initialize_session(user_id)
        
        style = session["claude_style"]
        dynamics = session["interaction_dynamics"]
        context = session["shared_context"]
        
        summary_parts = [
            f"## Claude Personality for {user_id}",
            f"",
            f"**Sessions together:** {context['session_count']}",
            f"**Trust level:** {dynamics['trust_level']:.0%} ({self._interpret_trust(dynamics['trust_level'])})",
            f"",
            f"### Communication Style",
            f"- Verbosity: {style['verbosity']}",
            f"- Formality: {style['formality']}",
            f"- Initiative: {style['initiative']}",
            f"- Code preference: {style['code_preference']}",
            f"- Energy matching: {'Yes' if style['energy_matching'] else 'No'}",
        ]
        
        if dynamics["works_well"]:
            summary_parts.append(f"")
            summary_parts.append(f"### What Works")
            for item in dynamics["works_well"][:5]:
                summary_parts.append(f"- {item}")
        
        if dynamics["avoid"]:
            summary_parts.append(f"")
            summary_parts.append(f"### What to Avoid")
            for item in dynamics["avoid"][:5]:
                summary_parts.append(f"- {item}")
        
        if context["projects"]:
            summary_parts.append(f"")
            summary_parts.append(f"### Shared Projects")
            for project in context["projects"][:5]:
                summary_parts.append(f"- {project}")
        
        if context["recent_milestones"]:
            summary_parts.append(f"")
            summary_parts.append(f"### Recent Milestones")
            for milestone in context["recent_milestones"]:
                summary_parts.append(f"- {milestone['description']}")
        
        return "\n".join(summary_parts)
    
    async def evolve_from_interaction(
        self,
        user_id: str,
        my_response_style: str,
        user_reaction: str,
        was_positive: bool
    ) -> Dict[str, Any]:
        """
        Evolve Claude's personality based on interaction outcome.
        
        This is the core learning loop for Claude personality.
        """
        changes = []
        
        # Learn what works/doesn't
        await self.learn_what_works(
            user_id,
            my_response_style,
            was_positive,
            confidence=0.7
        )
        changes.append(f"Learned: {my_response_style} -> {'works' if was_positive else 'avoid'}")
        
        # Adjust style based on feedback
        if not was_positive:
            if "verbose" in my_response_style.lower():
                await self.update_style(user_id, "verbosity", "minimal", 0.8)
                changes.append("Reduced verbosity")
            elif "formal" in my_response_style.lower():
                await self.update_style(user_id, "formality", "casual_professional", 0.7)
                changes.append("Reduced formality")
            elif "asked" in my_response_style.lower() or "permission" in my_response_style.lower():
                await self.update_style(user_id, "initiative", "high", 0.8)
                changes.append("Increased initiative")
        
        # Increase trust on positive interactions
        if was_positive:
            dynamics = await self._load_interaction_dynamics(user_id)
            new_trust = min(1.0, dynamics.trust_level + 0.05)
            await self.update_trust_level(user_id, new_trust)
            changes.append(f"Trust increased to {new_trust:.0%}")
        
        return {
            "evolved": True,
            "changes": changes,
            "user_id": user_id
        }
