"""
ArXiv Paper Ingestion with Real Provenance

Fetches papers from arXiv, extracts key claims, and stores them
with REAL citations (URL, authors, quoted spans).

This fixes the "no provenance" problem identified in the experiment.
Every claim is traceable to its source.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
import hashlib
import xml.etree.ElementTree as ET

from loguru import logger

# ArXiv API base URL
ARXIV_API_URL = "http://export.arxiv.org/api/query"


@dataclass
class ArxivPaper:
    """Metadata for an arXiv paper"""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str
    updated: str
    pdf_url: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.arxiv_id,
            "title": self.title[:100],
            "authors": self.authors[:5],
            "abstract": self.abstract[:200],
            "categories": self.categories,
            "published": self.published
        }


@dataclass
class ExtractedClaim:
    """A claim extracted from a paper with provenance"""
    claim_id: str
    claim_text: str
    claim_type: str  # definition, finding, method, hypothesis
    quoted_span: str
    confidence: float
    
    # Provenance
    arxiv_id: str
    paper_title: str
    authors: List[str]
    section: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.claim_id,
            "claim": self.claim_text[:100],
            "type": self.claim_type,
            "quote": self.quoted_span[:150],
            "conf": self.confidence,
            "arxiv": self.arxiv_id,
            "authors": self.authors[:3]
        }
    
    def get_provenance_dict(self) -> Dict[str, Any]:
        """Get provenance info for storage"""
        return {
            "source_type": "arxiv",
            "source_url": f"https://arxiv.org/abs/{self.arxiv_id}",
            "source_title": self.paper_title,
            "quoted_span": self.quoted_span,
            "authors": ", ".join(self.authors[:5]),
            "arxiv_id": self.arxiv_id,
            "confidence": self.confidence
        }


class ArxivIngestion:
    """
    Ingest papers from arXiv with real provenance tracking.
    
    Features:
    - Fetch paper metadata via arXiv API
    - Extract key claims from abstract
    - Store with full citation info
    - Track provenance for every claim
    """
    
    def __init__(self, store):
        self.store = store
        self._claim_counter = 0
        self.ingestion_history: List[Dict[str, Any]] = []
        
        logger.info("ArxivIngestion initialized")
    
    async def fetch_paper(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Fetch paper metadata from arXiv API.
        
        Args:
            arxiv_id: ArXiv ID (e.g., "1706.03762" or "2401.12345")
        
        Returns:
            ArxivPaper with metadata, or None if not found
        """
        import urllib.request
        import urllib.parse
        
        # Clean arxiv_id
        arxiv_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")
        
        # Build query URL
        query = f"id_list={arxiv_id}"
        url = f"{ARXIV_API_URL}?{query}"
        
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                xml_data = response.read().decode('utf-8')
            
            # Parse XML
            root = ET.fromstring(xml_data)
            
            # Namespace handling
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            entry = root.find('atom:entry', ns)
            if entry is None:
                logger.warning(f"Paper not found: {arxiv_id}")
                return None
            
            # Extract fields
            title = entry.find('atom:title', ns)
            title_text = title.text.strip().replace('\n', ' ') if title is not None else ""
            
            abstract = entry.find('atom:summary', ns)
            abstract_text = abstract.text.strip().replace('\n', ' ') if abstract is not None else ""
            
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)
            
            categories = []
            for cat in entry.findall('arxiv:primary_category', ns):
                term = cat.get('term')
                if term:
                    categories.append(term)
            for cat in entry.findall('atom:category', ns):
                term = cat.get('term')
                if term and term not in categories:
                    categories.append(term)
            
            published = entry.find('atom:published', ns)
            published_text = published.text[:10] if published is not None else ""
            
            updated = entry.find('atom:updated', ns)
            updated_text = updated.text[:10] if updated is not None else ""
            
            # Get PDF link
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            paper = ArxivPaper(
                arxiv_id=arxiv_id,
                title=title_text,
                authors=authors,
                abstract=abstract_text,
                categories=categories,
                published=published_text,
                updated=updated_text,
                pdf_url=pdf_url
            )
            
            logger.info(f"Fetched paper: {arxiv_id} - {title_text[:50]}")
            return paper
            
        except Exception as e:
            logger.error(f"Error fetching {arxiv_id}: {e}")
            return None
    
    def extract_claims(self, paper: ArxivPaper) -> List[ExtractedClaim]:
        """
        Extract key claims from paper abstract.
        
        Uses pattern matching to identify:
        - Definitions ("X is defined as...")
        - Findings ("We show that...", "Results indicate...")
        - Methods ("We propose...", "Our approach...")
        - Hypotheses ("We hypothesize...", "It is conjectured...")
        """
        claims = []
        abstract = paper.abstract
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', abstract)
        
        # Patterns for different claim types
        patterns = {
            "definition": [
                r"(?:is|are)\s+defined\s+as",
                r"(?:we|I)\s+define",
                r"refers?\s+to",
                r"(?:is|are)\s+(?:a|an|the)\s+\w+\s+(?:that|which|where)"
            ],
            "finding": [
                r"(?:we|our\s+results?)\s+show",
                r"(?:we|I)\s+(?:find|found|demonstrate)",
                r"results?\s+(?:indicate|suggest|show)",
                r"(?:we|I)\s+observe",
                r"empirically",
                r"experiments?\s+(?:show|demonstrate|reveal)"
            ],
            "method": [
                r"(?:we|I)\s+propose",
                r"(?:we|I)\s+introduce",
                r"(?:we|I)\s+present",
                r"our\s+(?:approach|method|algorithm|framework)",
                r"(?:we|I)\s+develop"
            ],
            "hypothesis": [
                r"(?:we|I)\s+hypothesize",
                r"(?:we|I)\s+conjecture",
                r"(?:it\s+is|we)\s+(?:hypothesized|conjectured)",
                r"(?:we|I)\s+speculate"
            ]
        }
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Check each pattern type
            for claim_type, type_patterns in patterns.items():
                for pattern in type_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        self._claim_counter += 1
                        
                        # Generate claim summary
                        claim_text = self._summarize_claim(sentence, claim_type)
                        
                        claim = ExtractedClaim(
                            claim_id=f"claim_{paper.arxiv_id}_{self._claim_counter}",
                            claim_text=claim_text,
                            claim_type=claim_type,
                            quoted_span=sentence,
                            confidence=0.7,  # Base confidence for pattern match
                            arxiv_id=paper.arxiv_id,
                            paper_title=paper.title,
                            authors=paper.authors,
                            section="abstract"
                        )
                        claims.append(claim)
                        break  # Only match one type per sentence
        
        # If no patterns matched, extract key sentences as general claims
        if not claims and sentences:
            # Take first and last substantive sentences
            for i, sentence in enumerate(sentences):
                if len(sentence) > 50 and i in [0, len(sentences)-1]:
                    self._claim_counter += 1
                    claim = ExtractedClaim(
                        claim_id=f"claim_{paper.arxiv_id}_{self._claim_counter}",
                        claim_text=self._summarize_claim(sentence, "general"),
                        claim_type="general",
                        quoted_span=sentence,
                        confidence=0.5,
                        arxiv_id=paper.arxiv_id,
                        paper_title=paper.title,
                        authors=paper.authors,
                        section="abstract"
                    )
                    claims.append(claim)
        
        logger.info(f"Extracted {len(claims)} claims from {paper.arxiv_id}")
        return claims
    
    def _summarize_claim(self, sentence: str, claim_type: str) -> str:
        """Generate a short summary of the claim"""
        # Truncate and clean
        summary = sentence[:150]
        if len(sentence) > 150:
            summary = summary.rsplit(' ', 1)[0] + "..."
        return summary
    
    async def ingest_paper(
        self,
        arxiv_id: str,
        user_id: str = "pltm_knowledge"
    ) -> Dict[str, Any]:
        """
        Ingest a paper: fetch, extract claims, store with provenance.
        
        Args:
            arxiv_id: ArXiv paper ID
            user_id: User/subject to store claims under
        
        Returns:
            Ingestion result with paper info and claims stored
        """
        # Fetch paper
        paper = await self.fetch_paper(arxiv_id)
        if not paper:
            return {"ok": False, "error": f"Paper not found: {arxiv_id}"}
        
        # Extract claims
        claims = self.extract_claims(paper)
        
        # Store claims with provenance
        stored_count = 0
        for claim in claims:
            try:
                # Store as atom
                atom_id = hashlib.md5(
                    f"{claim.arxiv_id}:{claim.claim_text}".encode()
                ).hexdigest()[:16]
                
                await self.store._conn.execute(
                    """INSERT OR REPLACE INTO atoms 
                       (id, atom_type, graph, subject, predicate, object, metadata, confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        atom_id,
                        "fact",
                        "substantiated",
                        user_id,
                        f"learned_from_{claim.claim_type}",
                        claim.claim_text,
                        "{}",
                        claim.confidence
                    )
                )
                
                # Store provenance
                prov = claim.get_provenance_dict()
                prov_id = f"prov_{atom_id}"
                content_hash = hashlib.sha256(claim.quoted_span.encode()).hexdigest()
                
                await self.store._conn.execute(
                    """INSERT OR REPLACE INTO provenance
                       (id, claim_id, source_type, source_url, source_title, quoted_span,
                        accessed_at, content_hash, confidence, authors, arxiv_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        prov_id,
                        atom_id,
                        prov["source_type"],
                        prov["source_url"],
                        prov["source_title"],
                        prov["quoted_span"],
                        int(datetime.now().timestamp()),
                        content_hash,
                        prov["confidence"],
                        prov["authors"],
                        prov["arxiv_id"]
                    )
                )
                
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Error storing claim {claim.claim_id}: {e}")
        
        await self.store._conn.commit()
        
        result = {
            "ok": True,
            "arxiv_id": arxiv_id,
            "title": paper.title[:80],
            "authors": paper.authors[:3],
            "claims_extracted": len(claims),
            "claims_stored": stored_count,
            "categories": paper.categories[:3]
        }
        
        self.ingestion_history.append({
            "arxiv_id": arxiv_id,
            "claims": stored_count,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Ingested {arxiv_id}: {stored_count} claims with provenance")
        return result
    
    async def search_arxiv(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers matching query.
        
        Args:
            query: Search query (title, abstract, author)
            max_results: Maximum papers to return
        
        Returns:
            List of paper summaries
        """
        import urllib.request
        import urllib.parse
        
        # Build search URL
        search_query = urllib.parse.quote(query)
        url = f"{ARXIV_API_URL}?search_query=all:{search_query}&max_results={max_results}"
        
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                xml_data = response.read().decode('utf-8')
            
            root = ET.fromstring(xml_data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            results = []
            for entry in root.findall('atom:entry', ns):
                # Extract ID from URL
                id_elem = entry.find('atom:id', ns)
                if id_elem is not None:
                    arxiv_id = id_elem.text.split('/')[-1]
                else:
                    continue
                
                title = entry.find('atom:title', ns)
                title_text = title.text.strip().replace('\n', ' ')[:100] if title is not None else ""
                
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns)
                    if name is not None:
                        authors.append(name.text)
                
                results.append({
                    "id": arxiv_id,
                    "title": title_text,
                    "authors": authors[:3]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    def get_ingestion_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent ingestion history"""
        return self.ingestion_history[-last_n:]
