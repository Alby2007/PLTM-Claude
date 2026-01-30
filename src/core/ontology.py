"""Ontology validation rules for memory atoms"""

from src.core.models import AtomType, MemoryAtom


# Ontology rules define what predicates are allowed for each atom type
ONTOLOGY_RULES: dict[AtomType, dict[str, list[str]]] = {
    AtomType.ENTITY: {
        "allowed_predicates": ["is", "has", "located_at", "named", "type_of"],
        "description": "Core categories: people, organizations, concepts",
    },
    AtomType.RELATION: {
        "allowed_predicates": [
            "works_with",
            "works_at",
            "works_in",
            "reports_to",
            "knows",
            "likes",
            "dislikes",
            "prefers",
            "uses",
            "wants",
            "avoids",
            "supports",
            "opposes",
            "agrees",
            "disagrees",
            "trusts",
            "distrusts",
            "accepts",
            "rejects",
            "drives",
            "does",
            "studied",
            "neutral",
            "liked_past",
            "will_learn",
            "started_learning",
            "worked_at_year",
            "works_at_year",
            "prefers_over",
        ],
        "description": "Relationships between entities",
    },
    AtomType.STATE: {
        "allowed_predicates": ["current_mood", "status", "condition"],
        "description": "Volatile state information",
    },
    AtomType.EVENT: {
        "allowed_predicates": ["completed", "started", "failed", "decided", "studied"],
        "description": "Time-bound events",
    },
    AtomType.HYPOTHESIS: {
        "allowed_predicates": ["might", "possibly", "likely"],
        "description": "Tentative explanatory models",
    },
    AtomType.INVARIANT: {
        "allowed_predicates": ["always", "never", "must"],
        "description": "Assumed true until proven otherwise",
    },
}


def validate_atom(atom: MemoryAtom) -> tuple[bool, str]:
    """
    Check if atom follows ontology rules.

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

    return True, "Valid"


def get_allowed_predicates(atom_type: AtomType) -> list[str]:
    """Get list of allowed predicates for an atom type"""
    rules = ONTOLOGY_RULES.get(atom_type)
    if not rules:
        return []
    return rules["allowed_predicates"]
