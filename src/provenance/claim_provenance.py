"""
Claim Provenance System

Every claim in PLTM must be traceable to its source.
No more "universal principles" without citations.

This transforms PLTM from "cool demo" to "solid research result".
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

from loguru import logger


class SourceType(str, Enum):
    """Types of provenance sources"""
    ARXIV = "arxiv"           # Research papers
    GITHUB = "github"         # Code repositories
    WIKIPEDIA = "wikipedia"   # Encyclopedia
    DOI = "doi"               # Academic papers
    URL = "url"               # General web
    BOOK = "book"             # Published books
    CONVERSATION = "conversation"  # User interaction
    INTERNAL = "internal"     # Claude's prior knowledge (must be marked!)


@dataclass
class ClaimProvenance:
    """
    Provenance for a single claim.
    
    Every fact, principle, or pattern MUST have this attached.
    """
    provenance_id: str
    source_type: SourceType
    source_url: str                    # Full URL or identifier
    source_title: str                  # Paper title, repo name, etc.
    quoted_span: str                   # Exact text supporting the claim
    page_or_section: Optional[str]     # "p.3, §2.1" or "line 45-67"
    accessed_at: datetime
    content_hash: str                  # SHA256 of quoted_span for verification
    confidence: float                  # How directly does source support claim?
    
    # For academic sources
    authors: Optional[List[str]] = None
    publication_date: Optional[str] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    
    # For code sources
    commit_sha: Optional[str] = None
    file_path: Optional[str] = None
    line_range: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Compact dict for storage"""
        return {
            "id": self.provenance_id,
            "type": self.source_type.value,
            "url": self.source_url,
            "title": self.source_title[:100],
            "quote": self.quoted_span[:500],
            "loc": self.page_or_section,
            "at": self.accessed_at.isoformat(),
            "hash": self.content_hash[:16],
            "conf": self.confidence,
            "arxiv": self.arxiv_id,
            "doi": self.doi,
            "sha": self.commit_sha
        }
    
    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute content hash for verification"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    @classmethod
    def from_arxiv(
        cls,
        arxiv_id: str,
        title: str,
        authors: List[str],
        quoted_span: str,
        page_or_section: str,
        confidence: float = 0.8
    ) -> "ClaimProvenance":
        """Create provenance from arXiv paper"""
        return cls(
            provenance_id=f"prov_arxiv_{arxiv_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            source_type=SourceType.ARXIV,
            source_url=f"https://arxiv.org/abs/{arxiv_id}",
            source_title=title,
            quoted_span=quoted_span,
            page_or_section=page_or_section,
            accessed_at=datetime.now(),
            content_hash=cls.compute_hash(quoted_span),
            confidence=confidence,
            authors=authors,
            arxiv_id=arxiv_id
        )
    
    @classmethod
    def from_github(
        cls,
        repo: str,
        file_path: str,
        commit_sha: str,
        quoted_span: str,
        line_range: str,
        confidence: float = 0.9
    ) -> "ClaimProvenance":
        """Create provenance from GitHub code"""
        return cls(
            provenance_id=f"prov_gh_{repo.replace('/', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            source_type=SourceType.GITHUB,
            source_url=f"https://github.com/{repo}/blob/{commit_sha}/{file_path}#L{line_range}",
            source_title=f"{repo}/{file_path}",
            quoted_span=quoted_span,
            page_or_section=f"lines {line_range}",
            accessed_at=datetime.now(),
            content_hash=cls.compute_hash(quoted_span),
            confidence=confidence,
            commit_sha=commit_sha,
            file_path=file_path,
            line_range=line_range
        )
    
    @classmethod
    def from_internal(
        cls,
        claim: str,
        reasoning: str,
        confidence: float = 0.5  # Lower confidence for internal claims
    ) -> "ClaimProvenance":
        """
        Create provenance for Claude's internal knowledge.
        
        IMPORTANT: This marks claims that come from Claude's training,
        not from cited sources. These should be treated with skepticism
        and ideally replaced with proper citations.
        """
        return cls(
            provenance_id=f"prov_internal_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            source_type=SourceType.INTERNAL,
            source_url="claude:internal_knowledge",
            source_title="Claude's Prior Knowledge (UNVERIFIED)",
            quoted_span=reasoning,
            page_or_section=None,
            accessed_at=datetime.now(),
            content_hash=cls.compute_hash(reasoning),
            confidence=confidence
        )


@dataclass
class ProvenanceChain:
    """
    Chain of provenance for derived claims.
    
    When claim C is derived from claims A and B,
    we track the full derivation chain.
    """
    chain_id: str
    claim: str
    direct_sources: List[ClaimProvenance]
    derivation_method: str  # "synthesis", "inference", "aggregation"
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.chain_id,
            "claim": self.claim[:200],
            "sources": [s.provenance_id for s in self.direct_sources],
            "method": self.derivation_method,
            "conf": self.confidence
        }


class ProvenanceStore:
    """
    Store and retrieve provenance records.
    
    Integrates with SQLite for persistence.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.provenances: Dict[str, ClaimProvenance] = {}
        self.chains: Dict[str, ProvenanceChain] = {}
        self.claim_to_provenance: Dict[str, List[str]] = {}  # claim_id -> [provenance_ids]
        
        logger.info("ProvenanceStore initialized")
    
    def add_provenance(self, claim_id: str, provenance: ClaimProvenance) -> str:
        """Add provenance for a claim"""
        self.provenances[provenance.provenance_id] = provenance
        
        if claim_id not in self.claim_to_provenance:
            self.claim_to_provenance[claim_id] = []
        self.claim_to_provenance[claim_id].append(provenance.provenance_id)
        
        logger.debug(f"Added provenance {provenance.provenance_id} for claim {claim_id}")
        return provenance.provenance_id
    
    def get_provenance(self, claim_id: str) -> List[ClaimProvenance]:
        """Get all provenance for a claim"""
        prov_ids = self.claim_to_provenance.get(claim_id, [])
        return [self.provenances[pid] for pid in prov_ids if pid in self.provenances]
    
    def add_chain(self, chain: ProvenanceChain) -> str:
        """Add a derivation chain"""
        self.chains[chain.chain_id] = chain
        return chain.chain_id
    
    def verify_provenance(self, provenance_id: str, quoted_span: str) -> bool:
        """Verify that quoted span matches stored hash"""
        if provenance_id not in self.provenances:
            return False
        
        prov = self.provenances[provenance_id]
        computed_hash = ClaimProvenance.compute_hash(quoted_span)
        return computed_hash == prov.content_hash
    
    def get_unverified_claims(self) -> List[str]:
        """Get claims that only have INTERNAL provenance (need real citations)"""
        unverified = []
        for claim_id, prov_ids in self.claim_to_provenance.items():
            provs = [self.provenances[pid] for pid in prov_ids if pid in self.provenances]
            if all(p.source_type == SourceType.INTERNAL for p in provs):
                unverified.append(claim_id)
        return unverified
    
    def get_stats(self) -> Dict[str, Any]:
        """Get provenance statistics"""
        by_type = {}
        for prov in self.provenances.values():
            t = prov.source_type.value
            by_type[t] = by_type.get(t, 0) + 1
        
        unverified = len(self.get_unverified_claims())
        
        return {
            "total": len(self.provenances),
            "by_type": by_type,
            "chains": len(self.chains),
            "unverified": unverified
        }


# Example: Real provenance for "attention mechanism" principle
ATTENTION_PROVENANCE = ClaimProvenance.from_arxiv(
    arxiv_id="1706.03762",
    title="Attention Is All You Need",
    authors=["Vaswani", "Shazeer", "Parmar", "Uszkoreit", "Jones", "Gomez", "Kaiser", "Polosukhin"],
    quoted_span="An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.",
    page_or_section="§3.2",
    confidence=0.95
)

# Example: Real provenance for "information integration" principle (IIT)
IIT_PROVENANCE = ClaimProvenance.from_arxiv(
    arxiv_id="1405.7089",
    title="From the Phenomenology to the Mechanisms of Consciousness: Integrated Information Theory 3.0",
    authors=["Oizumi", "Albantakis", "Tononi"],
    quoted_span="Integrated information (Φ) is defined as the information generated by a complex of elements, above and beyond the information generated by its parts.",
    page_or_section="Abstract",
    confidence=0.9
)

# Example: Internal claim that NEEDS citation
INTERNAL_EXAMPLE = ClaimProvenance.from_internal(
    claim="Systems learn best at the edge of chaos",
    reasoning="This is a common claim in complexity theory but I don't have a specific citation",
    confidence=0.4  # Low confidence because uncited
)
