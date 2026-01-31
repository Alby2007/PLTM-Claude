"""
Cross-Domain Synthesis

Discovers meta-patterns across ALL learned knowledge:
- Patterns that appear in multiple domains
- Transfer learning opportunities
- Novel insights from cross-domain connections
- Universal principles underlying specific knowledge
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re

from src.storage.sqlite_store import SQLiteGraphStore
from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
from loguru import logger


@dataclass
class MetaPattern:
    """A pattern that appears across multiple domains"""
    pattern_id: str
    name: str
    description: str
    domains: List[str]
    occurrences: int
    confidence: float
    examples: List[Dict[str, Any]]
    discovered_at: datetime = field(default_factory=datetime.now)


@dataclass
class CrossDomainInsight:
    """An insight derived from cross-domain analysis"""
    insight_id: str
    source_domains: List[str]
    insight: str
    supporting_evidence: List[str]
    novelty_score: float  # How novel/unexpected is this insight
    confidence: float
    discovered_at: datetime = field(default_factory=datetime.now)


@dataclass
class TransferOpportunity:
    """An opportunity to transfer knowledge between domains"""
    from_domain: str
    to_domain: str
    transferable_concept: str
    potential_application: str
    confidence: float


class CrossDomainSynthesizer:
    """
    Synthesizes knowledge across domains to discover meta-patterns.
    
    This is where AGI-level insights emerge:
    - Patterns in physics that apply to economics
    - Software patterns that mirror biological systems
    - Mathematical structures underlying multiple fields
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self.discovered_patterns: List[MetaPattern] = []
        self.discovered_insights: List[CrossDomainInsight] = []
        
        # Known universal patterns to look for
        self.universal_pattern_templates = {
            "feedback_loop": {
                "indicators": ["feedback", "loop", "cycle", "recursive", "self-reinforcing"],
                "domains": ["control_systems", "biology", "economics", "psychology", "ecology"]
            },
            "emergence": {
                "indicators": ["emergent", "self-organizing", "bottom-up", "collective", "swarm"],
                "domains": ["complexity", "biology", "ai_ml", "sociology", "physics"]
            },
            "optimization": {
                "indicators": ["optimize", "minimize", "maximize", "efficient", "optimal"],
                "domains": ["mathematics", "computer_science", "economics", "biology", "engineering"]
            },
            "network_effects": {
                "indicators": ["network", "connected", "graph", "nodes", "edges", "topology"],
                "domains": ["computer_science", "sociology", "biology", "economics", "physics"]
            },
            "information_flow": {
                "indicators": ["information", "signal", "communication", "transmission", "encoding"],
                "domains": ["computer_science", "biology", "physics", "neuroscience", "economics"]
            },
            "equilibrium": {
                "indicators": ["equilibrium", "balance", "stable", "steady state", "homeostasis"],
                "domains": ["physics", "economics", "biology", "chemistry", "game_theory"]
            },
            "hierarchy": {
                "indicators": ["hierarchy", "layers", "levels", "abstraction", "composition"],
                "domains": ["computer_science", "biology", "organization", "linguistics", "physics"]
            },
            "evolution": {
                "indicators": ["evolution", "adaptation", "selection", "mutation", "fitness"],
                "domains": ["biology", "ai_ml", "economics", "technology", "culture"]
            },
        }
        
        logger.info("CrossDomainSynthesizer initialized")
    
    async def synthesize_all(self) -> Dict[str, Any]:
        """
        Run full cross-domain synthesis on all learned knowledge.
        
        Returns:
            Discovered patterns, insights, and transfer opportunities
        """
        logger.info("Starting cross-domain synthesis")
        
        # Get all atoms grouped by domain
        domain_atoms = await self._get_atoms_by_domain()
        
        # Discover meta-patterns
        patterns = await self._discover_meta_patterns(domain_atoms)
        
        # Generate cross-domain insights
        insights = await self._generate_insights(domain_atoms, patterns)
        
        # Find transfer opportunities
        transfers = await self._find_transfer_opportunities(domain_atoms, patterns)
        
        # Store discoveries
        await self._store_discoveries(patterns, insights)
        
        return {
            "ok": True,
            "domains": len(domain_atoms),
            "patterns": len(patterns),
            "insights": len(insights),
            "transfers": len(transfers),
            "top_patterns": [p.name for p in patterns[:5]],
            "top_insights": [i.insight[:80] for i in insights[:3]]
        }
    
    async def _get_atoms_by_domain(self) -> Dict[str, List[MemoryAtom]]:
        """Group all atoms by their domain context"""
        domain_atoms = defaultdict(list)
        
        # Get atoms from various subjects that indicate learned content
        # This is simplified - in production would have better domain indexing
        
        # Query atoms with domain contexts
        all_subjects = ["source:", "paper:", "repo:", "concept:"]
        
        for subject_prefix in all_subjects:
            atoms = await self.store.get_atoms_by_subject(subject_prefix)
            for atom in atoms:
                # Extract domain from contexts
                for ctx in atom.contexts:
                    if ctx in self.universal_pattern_templates.get("feedback_loop", {}).get("domains", []):
                        domain_atoms[ctx].append(atom)
                    elif "_" not in ctx and len(ctx) > 3:
                        # Likely a domain name
                        domain_atoms[ctx].append(atom)
        
        return dict(domain_atoms)
    
    async def _discover_meta_patterns(
        self, 
        domain_atoms: Dict[str, List[MemoryAtom]]
    ) -> List[MetaPattern]:
        """Discover patterns that appear across multiple domains"""
        patterns = []
        
        for pattern_name, template in self.universal_pattern_templates.items():
            indicators = template["indicators"]
            expected_domains = template["domains"]
            
            # Find occurrences across domains
            found_in_domains = []
            total_occurrences = 0
            examples = []
            
            for domain, atoms in domain_atoms.items():
                domain_occurrences = 0
                for atom in atoms:
                    text = f"{atom.subject} {atom.predicate} {atom.object}".lower()
                    if any(ind in text for ind in indicators):
                        domain_occurrences += 1
                        if len(examples) < 5:
                            examples.append({
                                "domain": domain,
                                "subject": atom.subject,
                                "predicate": atom.predicate,
                                "object": atom.object[:100]
                            })
                
                if domain_occurrences > 0:
                    found_in_domains.append(domain)
                    total_occurrences += domain_occurrences
            
            # Pattern is significant if found in 2+ domains
            if len(found_in_domains) >= 2:
                confidence = min(0.95, len(found_in_domains) / len(expected_domains) * 0.8 + 0.2)
                
                pattern = MetaPattern(
                    pattern_id=f"pattern_{pattern_name}_{datetime.now().strftime('%Y%m%d')}",
                    name=pattern_name,
                    description=f"Universal pattern '{pattern_name}' found across {len(found_in_domains)} domains",
                    domains=found_in_domains,
                    occurrences=total_occurrences,
                    confidence=confidence,
                    examples=examples
                )
                patterns.append(pattern)
                self.discovered_patterns.append(pattern)
        
        return patterns
    
    async def _generate_insights(
        self,
        domain_atoms: Dict[str, List[MemoryAtom]],
        patterns: List[MetaPattern]
    ) -> List[CrossDomainInsight]:
        """Generate insights from cross-domain analysis"""
        insights = []
        
        # Insight 1: Pattern convergence
        for pattern in patterns:
            if len(pattern.domains) >= 3:
                insight = CrossDomainInsight(
                    insight_id=f"insight_convergence_{pattern.name}",
                    source_domains=pattern.domains,
                    insight=f"The '{pattern.name}' pattern appears independently in {', '.join(pattern.domains)}, suggesting a fundamental principle",
                    supporting_evidence=[f"{e['domain']}: {e['subject']} {e['predicate']}" for e in pattern.examples],
                    novelty_score=0.7,
                    confidence=pattern.confidence
                )
                insights.append(insight)
                self.discovered_insights.append(insight)
        
        # Insight 2: Domain overlap analysis
        domain_pairs = []
        domains = list(domain_atoms.keys())
        for i, d1 in enumerate(domains):
            for d2 in domains[i+1:]:
                # Find shared concepts
                d1_concepts = set(a.subject for a in domain_atoms[d1])
                d2_concepts = set(a.subject for a in domain_atoms[d2])
                shared = d1_concepts & d2_concepts
                
                if len(shared) > 2:
                    insight = CrossDomainInsight(
                        insight_id=f"insight_overlap_{d1}_{d2}",
                        source_domains=[d1, d2],
                        insight=f"Domains '{d1}' and '{d2}' share {len(shared)} concepts, indicating potential for knowledge transfer",
                        supporting_evidence=list(shared)[:5],
                        novelty_score=0.6,
                        confidence=0.7
                    )
                    insights.append(insight)
        
        # Insight 3: Unexpected connections
        # Look for concepts that appear in very different domains
        concept_domains = defaultdict(set)
        for domain, atoms in domain_atoms.items():
            for atom in atoms:
                concept_domains[atom.subject].add(domain)
        
        for concept, domains_set in concept_domains.items():
            if len(domains_set) >= 3:
                insight = CrossDomainInsight(
                    insight_id=f"insight_bridge_{concept[:20]}",
                    source_domains=list(domains_set),
                    insight=f"Concept '{concept}' bridges {len(domains_set)} domains: {', '.join(list(domains_set)[:5])}",
                    supporting_evidence=[f"Found in {d}" for d in list(domains_set)[:5]],
                    novelty_score=0.8,
                    confidence=0.65
                )
                insights.append(insight)
        
        return insights
    
    async def _find_transfer_opportunities(
        self,
        domain_atoms: Dict[str, List[MemoryAtom]],
        patterns: List[MetaPattern]
    ) -> List[TransferOpportunity]:
        """Find opportunities to transfer knowledge between domains"""
        opportunities = []
        
        # For each pattern, suggest transfers to domains where it's not yet applied
        for pattern in patterns:
            expected_domains = self.universal_pattern_templates.get(pattern.name, {}).get("domains", [])
            missing_domains = set(expected_domains) - set(pattern.domains)
            
            for missing in missing_domains:
                if missing in domain_atoms:  # We have knowledge about this domain
                    opp = TransferOpportunity(
                        from_domain=pattern.domains[0],  # Source domain
                        to_domain=missing,
                        transferable_concept=pattern.name,
                        potential_application=f"Apply '{pattern.name}' principles from {pattern.domains[0]} to {missing}",
                        confidence=0.6
                    )
                    opportunities.append(opp)
        
        # Look for technique transfers
        # If a technique works well in one domain, suggest for similar domains
        technique_atoms = [
            a for atoms in domain_atoms.values() 
            for a in atoms 
            if a.atom_type == AtomType.SKILL
        ]
        
        for atom in technique_atoms[:20]:  # Limit for performance
            source_domain = atom.contexts[0] if atom.contexts else "unknown"
            
            # Find similar domains
            for domain in domain_atoms.keys():
                if domain != source_domain:
                    opp = TransferOpportunity(
                        from_domain=source_domain,
                        to_domain=domain,
                        transferable_concept=atom.subject,
                        potential_application=f"Apply '{atom.subject}' technique to {domain}",
                        confidence=0.5
                    )
                    opportunities.append(opp)
        
        return opportunities[:50]  # Limit results
    
    async def _store_discoveries(
        self,
        patterns: List[MetaPattern],
        insights: List[CrossDomainInsight]
    ) -> None:
        """Store discovered patterns and insights as atoms"""
        
        # Store patterns
        for pattern in patterns:
            atom = MemoryAtom(
                atom_type=AtomType.INVARIANT,
                subject=f"meta_pattern:{pattern.name}",
                predicate="discovered_in",
                object=", ".join(pattern.domains),
                confidence=pattern.confidence,
                strength=pattern.confidence,
                provenance=Provenance.INFERRED,
                source_user="cross_domain_synthesis",
                contexts=["meta_pattern", "cross_domain"] + pattern.domains,
                graph=GraphType.SUBSTANTIATED
            )
            await self.store.add_atom(atom)
        
        # Store insights
        for insight in insights[:20]:  # Limit storage
            atom = MemoryAtom(
                atom_type=AtomType.HYPOTHESIS,
                subject=f"insight:{insight.insight_id}",
                predicate="cross_domain_insight",
                object=insight.insight[:300],
                confidence=insight.confidence,
                strength=insight.novelty_score,
                provenance=Provenance.INFERRED,
                source_user="cross_domain_synthesis",
                contexts=["insight", "cross_domain"] + insight.source_domains,
                graph=GraphType.SUBSTANTIATED
            )
            await self.store.add_atom(atom)
        
        logger.info(f"Stored {len(patterns)} patterns and {min(len(insights), 20)} insights")
    
    async def get_patterns_for_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Get all meta-patterns relevant to a specific domain"""
        relevant = [
            {
                "name": p.name,
                "description": p.description,
                "other_domains": [d for d in p.domains if d != domain],
                "confidence": p.confidence,
                "examples": [e for e in p.examples if e.get("domain") == domain]
            }
            for p in self.discovered_patterns
            if domain in p.domains
        ]
        return relevant
    
    async def get_transfer_suggestions(
        self, 
        from_domain: str, 
        to_domain: str
    ) -> List[Dict[str, Any]]:
        """Get specific transfer suggestions between two domains"""
        # Run synthesis if not done
        if not self.discovered_patterns:
            await self.synthesize_all()
        
        suggestions = []
        
        # Find patterns in from_domain that could apply to to_domain
        for pattern in self.discovered_patterns:
            if from_domain in pattern.domains:
                expected = self.universal_pattern_templates.get(pattern.name, {}).get("domains", [])
                if to_domain in expected or to_domain in pattern.domains:
                    suggestions.append({
                        "pattern": pattern.name,
                        "description": f"Apply '{pattern.name}' from {from_domain} to {to_domain}",
                        "confidence": pattern.confidence,
                        "examples_in_source": [
                            e for e in pattern.examples 
                            if e.get("domain") == from_domain
                        ][:3]
                    })
        
        return suggestions
    
    async def query_universal_principles(self) -> List[Dict[str, Any]]:
        """Get all discovered universal principles"""
        return [
            {
                "principle": p.name,
                "domains_found": p.domains,
                "occurrences": p.occurrences,
                "confidence": p.confidence,
                "description": p.description
            }
            for p in self.discovered_patterns
            if len(p.domains) >= 3  # Universal = found in 3+ domains
        ]
