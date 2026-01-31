"""Improved ontology with semantic clarity and type-specific rules"""

from typing import Optional
from src.core.models import AtomType, MemoryAtom


class Ontology:
    """
    Ontology wrapper providing access to type-specific rules and validation.
    """
    
    def __init__(self):
        self.rules = ONTOLOGY_RULES
    
    def get_rules(self, atom_type: AtomType) -> dict:
        """Get rules for a specific atom type"""
        return self.rules.get(atom_type, {})
    
    def is_exclusive_predicate(self, predicate: str) -> bool:
        """Check if a predicate is exclusive (only one value allowed)"""
        for atom_type, rules in self.rules.items():
            if predicate in rules.get("allowed_predicates", []):
                return rules.get("exclusive", False)
        return False


# Improved ontology with granular types and type-specific rules
ONTOLOGY_RULES: dict[AtomType, dict] = {
    AtomType.ENTITY: {
        "allowed_predicates": [
            "is", "named", "called",
            "has_property", "characterized_by",
            "allergic_to",  # Medical attributes
        ],
        "description": "Identity and attributes",
        "decay_rate": 0.01,  # Very slow (identity rarely changes)
        "exclusive": False,  # Can have multiple attributes (e.g., multiple allergies)
        "examples": [
            "[User] [is] [software engineer]",
            "[User] [named] [Alby]",
            "[User] [allergic_to] [peanuts]"
        ],
    },
    
    AtomType.AFFILIATION: {
        "allowed_predicates": [
            "works_at", "works_for", "employed_by",
            "studies_at", "member_of", "part_of",
            "works_in",  # Location-based work
            "lives_in",  # Residential location
        ],
        "description": "Organizational affiliations and locations",
        "decay_rate": 0.03,  # Slow (jobs/schools change infrequently)
        "exclusive": True,   # Usually one job/school at a time
        "temporal": True,    # Track start/end dates
        "examples": [
            "[User] [works_at] [Anthropic]",
            "[User] [studies_at] [Oxford]",
            "[User] [lives_in] [Seattle]"
        ],
    },
    
    AtomType.SOCIAL: {
        "allowed_predicates": [
            "knows", "friends_with", "colleagues_with",
            "reports_to", "manages", "mentors",
            "works_with",
        ],
        "description": "Social relationships",
        "decay_rate": 0.05,  # Medium (relationships evolve)
        "exclusive": False,  # Can know many people
        "examples": [
            "[User] [knows] [Sarah]",
            "[User] [reports_to] [Manager_X]"
        ],
    },
    
    AtomType.PREFERENCE: {
        "allowed_predicates": [
            "likes", "dislikes", "loves", "hates",
            "prefers", "avoids", "enjoys",
            "neutral", "liked_past", "prefers_over",
        ],
        "description": "Preferences and tastes",
        "decay_rate": 0.08,  # Medium-fast (preferences change)
        "exclusive": False,  # Can like multiple things
        "contextual": True,  # Can like X in context A, hate in context B
        "opposite_pairs": [
            ("likes", "dislikes"),
            ("loves", "hates"),
            ("prefers", "avoids"),
        ],
        "examples": [
            "[User] [likes] [jazz music]",
            "[User] [prefers] [Python over JavaScript]"
        ],
    },
    
    AtomType.PERSONALITY_TRAIT: {
        "allowed_predicates": [
            "has_trait", "exhibits", "characterized_by",
            "personality_is", "tends_to_be",
        ],
        "description": "Stable personality traits",
        "decay_rate": 0.02,  # Very stable (personality changes slowly)
        "exclusive": False,  # Can have multiple traits
        "examples": [
            "[User] [has_trait] [humorous]",
            "[User] [exhibits] [directness]",
            "[User] [tends_to_be] [analytical]"
        ],
    },
    
    AtomType.COMMUNICATION_STYLE: {
        "allowed_predicates": [
            "prefers_style", "dislikes_style", "responds_well_to",
            "communication_preference", "style_is",
        ],
        "description": "Communication style preferences",
        "decay_rate": 0.05,  # Somewhat stable
        "exclusive": False,  # Can have multiple style preferences
        "examples": [
            "[User] [prefers_style] [concise responses]",
            "[User] [responds_well_to] [technical depth]",
            "[User] [dislikes_style] [verbose explanations]"
        ],
    },
    
    AtomType.INTERACTION_PATTERN: {
        "allowed_predicates": [
            "communication_style_is", "prefers_formality",
            "communication_includes", "interaction_style",
        ],
        "description": "Behavioral interaction patterns",
        "decay_rate": 0.08,  # Changes slowly
        "exclusive": False,  # Can have multiple patterns
        "examples": [
            "[User] [communication_style_is] [direct and to-the-point]",
            "[User] [prefers_formality] [casual]",
            "[User] [communication_includes] [humor]"
        ],
    },
    
    AtomType.BELIEF: {
        "allowed_predicates": [
            "thinks", "believes", "assumes", "expects",
            "trusts", "distrusts", "doubts",
            "supports", "opposes",
            "agrees", "disagrees",
            "accepts", "rejects",
        ],
        "description": "Beliefs and opinions",
        "decay_rate": 0.1,   # Fast (opinions change)
        "exclusive": False,
        "opposite_pairs": [
            ("trusts", "distrusts"),
            ("supports", "opposes"),
            ("agrees", "disagrees"),
            ("accepts", "rejects"),
        ],
        "examples": [
            "[User] [thinks] [Python is the best language]",
            "[User] [trusts] [the system]"
        ],
    },
    
    AtomType.SKILL: {
        "allowed_predicates": [
            "can_do", "proficient_in", "expert_at",
            "learning", "started_learning", "mastered",
            "uses", "does", "drives",  # Action-based skills
            "will_learn",  # Future skill acquisition
        ],
        "description": "Skills and capabilities",
        "decay_rate": 0.02,  # Slow (skills persist)
        "exclusive": False,
        "progressive": True,  # Skills have levels (learning → proficient → expert)
        "progression_sequence": ["learning", "proficient_in", "expert_at", "mastered"],
        "examples": [
            "[User] [proficient_in] [Python]",
            "[User] [learning] [Rust]"
        ],
    },
    
    AtomType.EVENT: {
        "allowed_predicates": [
            "completed", "started", "finished",
            "failed", "attempted", "decided",
            "happened", "studied",
            "worked_at_year", "works_at_year",  # Temporal events
        ],
        "description": "Time-bound events",
        "decay_rate": 0.06,  # Medium (events fade from relevance)
        "temporal": True,    # Must have timestamp
        "immutable": True,   # Events don't change, just get archived
        "examples": [
            "[User] [completed] [Q3 presentation]",
            "[User] [started] [new job at Meta]"
        ],
    },
    
    AtomType.STATE: {
        "allowed_predicates": [
            "currently", "temporarily", "status_is",
            "mood_is", "feeling",
            "current_mood", "status", "condition",
        ],
        "description": "Volatile current states",
        "decay_rate": 0.5,   # Very fast (states are volatile)
        "exclusive": True,   # Usually one state at a time
        "temporal": True,    # Always 'right now'
        "examples": [
            "[User] [currently] [tired]",
            "[User] [status_is] [available]"
        ],
    },
    
    AtomType.HYPOTHESIS: {
        "allowed_predicates": [
            "might", "possibly", "maybe", "likely",
            "probably", "uncertain_about",
        ],
        "description": "Unverified hypotheses",
        "decay_rate": 0.15,  # Fast (unverified claims fade)
        "confidence_max": 0.7,  # Hypotheses never become fully certain
        "examples": [
            "[User] [might] [prefer async communication]",
            "[User] [probably] [likes coffee]"
        ],
    },
    
    AtomType.INVARIANT: {
        "allowed_predicates": [
            "always", "never", "must", "cannot",
            "refuses_to", "insists_on",
        ],
        "description": "Stated rules and invariants",
        "decay_rate": 0.0,   # Never decay (stated rules)
        "confidence_min": 0.9,  # Must be high confidence to be invariant
        "examples": [
            "[User] [always] [double-checks financial data]",
            "[User] [never] [works on weekends]"
        ],
    },
    
    # Legacy RELATION type (backward compatibility)
    AtomType.RELATION: {
        "allowed_predicates": [
            # Keep all existing predicates for backward compatibility
            "works_with", "works_at", "works_in", "reports_to", "knows",
            "likes", "dislikes", "prefers", "uses", "wants", "avoids",
            "supports", "opposes", "agrees", "disagrees", "trusts",
            "distrusts", "accepts", "rejects", "drives", "does",
            "studied", "neutral", "liked_past", "will_learn",
            "started_learning", "worked_at_year", "works_at_year",
            "prefers_over",
        ],
        "description": "Legacy relation type (being phased out)",
        "decay_rate": 0.05,  # Generic decay rate
        "deprecated": True,  # Mark as deprecated
    },
}


# Predicate relationships (for conflict detection)
PREDICATE_RELATIONSHIPS = {
    "opposites": [
        ("likes", "dislikes"),
        ("loves", "hates"),
        ("prefers", "avoids"),
        ("trusts", "distrusts"),
        ("supports", "opposes"),
        ("agrees", "disagrees"),
        ("accepts", "rejects"),
    ],
    
    "exclusive_groups": [
        ["works_at", "works_for", "employed_by"],  # Can only work at one place
        ["studies_at", "enrolled_in"],             # Can only study at one place
        ["is", "type_of"],                          # Identity is singular
    ],
    
    "progressive_sequences": [
        ["learning", "proficient_in", "expert_at", "mastered"],  # Skill progression
        ["started", "in_progress", "completed"],                  # Task progression
    ],
}


def validate_atom(atom: MemoryAtom) -> tuple[bool, str]:
    """
    Enhanced validation with semantic rules.

    Returns:
        (is_valid, reason)
    """
    rules = ONTOLOGY_RULES.get(atom.atom_type)

    if not rules:
        return False, f"Unknown atom type: {atom.atom_type}"

    allowed_predicates = rules["allowed_predicates"]

    # Check if predicate is allowed for this atom type
    if atom.predicate not in allowed_predicates:
        return (
            False,
            f"Predicate '{atom.predicate}' not allowed for {atom.atom_type}. "
            f"Allowed: {', '.join(allowed_predicates)}",
        )

    # Basic semantic validation
    if not atom.subject or not atom.predicate or not atom.object:
        return False, "Subject, predicate, and object must all be non-empty"
    
    # Type-specific validation
    if rules.get("immutable") and hasattr(atom, "assertion_count") and atom.assertion_count > 1:
        return False, f"Events are immutable - cannot update {atom.id}"
    
    if rules.get("temporal") and not atom.first_observed:
        return False, f"Temporal atoms must have timestamp"
    
    if rules.get("confidence_max") and atom.confidence > rules["confidence_max"]:
        return False, f"Confidence too high for {atom.atom_type} (max: {rules['confidence_max']})"
    
    if rules.get("confidence_min") and atom.confidence < rules["confidence_min"]:
        return False, f"Confidence too low for {atom.atom_type} (min: {rules['confidence_min']})"

    return True, "Valid"


def get_allowed_predicates(atom_type: AtomType) -> list[str]:
    """Get list of allowed predicates for an atom type"""
    rules = ONTOLOGY_RULES.get(atom_type)
    if not rules:
        return []
    return rules["allowed_predicates"]


def get_decay_rate(atom_type: AtomType) -> float:
    """Get decay rate from ontology (for Ebbinghaus curves)"""
    rules = ONTOLOGY_RULES.get(atom_type)
    return rules.get("decay_rate", 0.05) if rules else 0.05


def is_exclusive_predicate(predicate: str) -> bool:
    """Check if predicate is exclusive (only one value allowed)"""
    for group in PREDICATE_RELATIONSHIPS["exclusive_groups"]:
        if predicate in group:
            return True
    
    # Check type-specific exclusive flag
    for atom_type, rules in ONTOLOGY_RULES.items():
        if predicate in rules["allowed_predicates"] and rules.get("exclusive"):
            return True
    
    return False


def get_opposite_predicate(predicate: str) -> Optional[str]:
    """Find opposite predicate if exists"""
    for pred1, pred2 in PREDICATE_RELATIONSHIPS["opposites"]:
        if predicate == pred1:
            return pred2
        if predicate == pred2:
            return pred1
    return None


def is_contextual_type(atom_type: AtomType) -> bool:
    """Check if atom type supports contextual coexistence"""
    rules = ONTOLOGY_RULES.get(atom_type)
    return rules.get("contextual", False) if rules else False


def is_progressive_type(atom_type: AtomType) -> bool:
    """Check if atom type has progressive sequences (e.g., skill levels)"""
    rules = ONTOLOGY_RULES.get(atom_type)
    return rules.get("progressive", False) if rules else False


def get_progression_sequence(atom_type: AtomType) -> Optional[list[str]]:
    """Get progression sequence for atom type if it exists"""
    rules = ONTOLOGY_RULES.get(atom_type)
    return rules.get("progression_sequence") if rules else None
