"""
PLTM Provenance System

Every claim must have verifiable provenance:
- Source URL (arxiv, github, wikipedia)
- Quoted span from source
- Content hash for verification
- Timestamp of access
"""

from src.provenance.claim_provenance import (
    ClaimProvenance,
    SourceType,
    ProvenanceStore,
    ProvenanceChain
)
