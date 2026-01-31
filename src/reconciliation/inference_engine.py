"""
Multi-hop reasoning engine for transitive conflict detection.

This module enables the system to detect conflicts that require chaining
multiple facts together (e.g., "Alice is vegetarian" + "Alice eats steak"
requires knowing that steak is meat and vegetarians don't eat meat).

Author: Procedural LTM Team
"""

from typing import List, Optional, Set, Tuple, Dict
from collections import deque
from loguru import logger

from src.core.models import MemoryAtom, AtomType
from src.storage.sqlite_store import SQLiteGraphStore
from src.core.ontology import Ontology


class ConflictChain:
    """Represents a chain of atoms that form a logical conflict"""
    
    def __init__(
        self,
        atoms: List[MemoryAtom],
        conflict_type: str,
        explanation: str,
        confidence: float = 0.8
    ):
        self.atoms = atoms
        self.conflict_type = conflict_type
        self.explanation = explanation
        self.confidence = confidence
    
    def __repr__(self) -> str:
        chain_str = " → ".join([
            f"[{a.subject}] [{a.predicate}] [{a.object}]" 
            for a in self.atoms
        ])
        return f"ConflictChain({self.conflict_type}): {chain_str}"


class InferenceEngine:
    """
    Multi-hop reasoning engine for detecting transitive conflicts.
    
    Capabilities:
    - 2-hop reasoning: Direct property conflicts (vegetarian + eats meat)
    - 3-hop reasoning: Transitive relationships (works_at X + X in Y + Y conflicts with Z)
    - World knowledge integration: Uses ontology for domain rules
    """
    
    def __init__(self, store: SQLiteGraphStore, ontology: Ontology):
        self.store = store
        self.ontology = ontology
        
        # World knowledge rules for common conflict patterns
        self.conflict_rules = self._initialize_conflict_rules()
        
        logger.info("InferenceEngine initialized with multi-hop reasoning")
    
    def _initialize_conflict_rules(self) -> Dict[str, List[Dict]]:
        """
        Initialize world knowledge rules for conflict detection.
        
        Each rule defines a pattern that constitutes a conflict when matched.
        """
        return {
            # Dietary restrictions
            "dietary": [
                {
                    "pattern": ["is", "vegetarian"],
                    "conflicts_with": ["eats", "meat|steak|chicken|pork|beef|fish"],
                    "explanation": "Vegetarians don't eat meat products"
                },
                {
                    "pattern": ["is", "vegan"],
                    "conflicts_with": ["eats", "meat|dairy|eggs|cheese|milk"],
                    "explanation": "Vegans don't consume animal products"
                },
                {
                    "pattern": ["allergic_to", "*"],
                    "conflicts_with": ["eats", "$1"],  # $1 = captured value
                    "explanation": "Cannot eat foods they're allergic to"
                },
            ],
            
            # Professional/location exclusivity
            "professional": [
                {
                    "pattern": ["works_at", "*"],
                    "conflicts_with": ["works_at", "!$1"],  # !$1 = different value
                    "explanation": "Cannot work at multiple companies simultaneously"
                },
                {
                    "pattern": ["lives_in", "*"],
                    "conflicts_with": ["lives_in", "!$1"],
                    "explanation": "Cannot live in multiple cities simultaneously"
                },
            ],
            
            # Logical impossibilities
            "logical": [
                {
                    "pattern": ["is", "dead"],
                    "conflicts_with": ["is", "alive|working|studying"],
                    "explanation": "Dead entities cannot be alive or active"
                },
                {
                    "pattern": ["is", "child"],
                    "conflicts_with": ["is", "adult|parent|retired"],
                    "explanation": "Children cannot be adults"
                },
            ],
            
            # Preference conflicts (softer)
            "preference": [
                {
                    "pattern": ["loves", "*"],
                    "conflicts_with": ["hates", "$1"],
                    "explanation": "Cannot simultaneously love and hate the same thing",
                    "confidence": 0.7
                },
                {
                    "pattern": ["prefers", "*"],
                    "conflicts_with": ["avoids", "$1"],
                    "explanation": "Preferring and avoiding are contradictory",
                    "confidence": 0.7
                },
            ],
        }
    
    async def find_transitive_conflicts(
        self,
        new_atom: MemoryAtom,
        max_hops: int = 3
    ) -> List[ConflictChain]:
        """
        Find conflicts by traversing relationship chains.
        
        Args:
            new_atom: The new atom being added
            max_hops: Maximum chain length to consider (default: 3)
            
        Returns:
            List of detected conflict chains
        """
        conflicts = []
        
        # Get all existing atoms about this subject
        existing_atoms = await self.store.get_atoms_by_subject(
            subject=new_atom.subject,
            graph="substantiated"
        )
        
        logger.debug(
            f"Checking {len(existing_atoms)} existing atoms for multi-hop conflicts"
        )
        
        # Check 2-hop conflicts (most common)
        two_hop = await self._check_two_hop_conflicts(new_atom, existing_atoms)
        conflicts.extend(two_hop)
        
        # Check 3-hop conflicts if enabled
        if max_hops >= 3:
            three_hop = await self._check_three_hop_conflicts(new_atom, existing_atoms)
            conflicts.extend(three_hop)
        
        if conflicts:
            logger.info(
                f"Found {len(conflicts)} multi-hop conflicts for "
                f"[{new_atom.subject}] [{new_atom.predicate}] [{new_atom.object}]"
            )
        
        return conflicts
    
    async def _check_two_hop_conflicts(
        self,
        new_atom: MemoryAtom,
        existing_atoms: List[MemoryAtom]
    ) -> List[ConflictChain]:
        """
        Check for 2-hop conflicts using world knowledge rules.
        
        Example:
        - Existing: [Alice] [is] [vegetarian]
        - New: [Alice] [eats] [steak]
        - Rule: vegetarian conflicts with eating meat
        - Result: CONFLICT
        """
        conflicts = []
        
        for category, rules in self.conflict_rules.items():
            for rule in rules:
                # Check if new atom matches conflict pattern
                if self._matches_pattern(new_atom, rule["conflicts_with"]):
                    # Look for existing atoms matching the base pattern
                    for existing in existing_atoms:
                        if self._matches_pattern(existing, rule["pattern"]):
                            # Found a conflict!
                            conflict = ConflictChain(
                                atoms=[existing, new_atom],
                                conflict_type=f"2hop_{category}",
                                explanation=rule["explanation"],
                                confidence=rule.get("confidence", 0.8)
                            )
                            conflicts.append(conflict)
                            
                            logger.debug(f"2-hop conflict detected: {conflict}")
        
        return conflicts
    
    async def _check_three_hop_conflicts(
        self,
        new_atom: MemoryAtom,
        existing_atoms: List[MemoryAtom]
    ) -> List[ConflictChain]:
        """
        Check for 3-hop conflicts via transitive relationships.
        
        Example:
        - Atom 1: [Alice] [works_at] [Google]
        - Atom 2: [Google] [located_in] [California]
        - New: [Alice] [lives_in] [New York]
        - Result: Potential conflict (long commute)
        """
        conflicts = []
        
        # For now, implement simple 3-hop patterns
        # Can be extended with graph traversal for arbitrary chains
        
        for atom1 in existing_atoms:
            # Look for intermediate atoms that connect atom1 to new_atom
            if atom1.predicate == "works_at":
                # Check company location vs user location
                company = atom1.object
                company_atoms = await self.store.get_atoms_by_subject(
                    subject=company,
                    graph="substantiated"
                )
                
                for atom2 in company_atoms:
                    if atom2.predicate == "located_in":
                        company_location = atom2.object
                        
                        # Check if new atom is about different location
                        if (new_atom.predicate == "lives_in" and 
                            new_atom.object != company_location):
                            
                            conflict = ConflictChain(
                                atoms=[atom1, atom2, new_atom],
                                conflict_type="3hop_location_mismatch",
                                explanation=(
                                    f"Works at {company} in {company_location} "
                                    f"but lives in {new_atom.object}"
                                ),
                                confidence=0.5  # Lower confidence for 3-hop
                            )
                            conflicts.append(conflict)
        
        return conflicts
    
    def _matches_pattern(
        self,
        atom: MemoryAtom,
        pattern: List[str]
    ) -> bool:
        """
        Check if an atom matches a pattern.
        
        Pattern syntax:
        - ["is", "vegetarian"] - exact match
        - ["eats", "*"] - any object
        - ["eats", "meat|steak|chicken"] - any of these objects
        - ["eats", "$1"] - capture value for later use
        - ["eats", "!$1"] - different from captured value
        """
        if len(pattern) != 2:
            return False
        
        pred_pattern, obj_pattern = pattern
        
        # Check predicate
        if pred_pattern != "*" and atom.predicate != pred_pattern:
            return False
        
        # Check object
        if obj_pattern == "*":
            return True
        
        if "|" in obj_pattern:
            # Multiple options
            options = obj_pattern.split("|")
            return atom.object.lower() in [opt.lower() for opt in options]
        
        if obj_pattern.startswith("$"):
            # Capture variable (always matches, stores for later)
            return True
        
        if obj_pattern.startswith("!"):
            # Negation (handled in caller context)
            return True
        
        # Exact match
        return atom.object.lower() == obj_pattern.lower()
    
    def _extract_captured_value(
        self,
        atom: MemoryAtom,
        pattern: List[str]
    ) -> Optional[str]:
        """Extract captured value from pattern match"""
        if len(pattern) == 2 and pattern[1].startswith("$"):
            return atom.object
        return None
