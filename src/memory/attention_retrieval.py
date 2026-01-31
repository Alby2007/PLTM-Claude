"""
Attention-Weighted Memory Retrieval

Universal Principle #3: Selective Amplification (Attention)

Weight memories by relevance to current context.
Most relevant memories get amplified, irrelevant ones suppressed.

Benefits:
- Context-dependent truth
- Efficient retrieval
- Mirrors transformer attention mechanism
- Enables nuanced understanding
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
from collections import OrderedDict
import math
import re
import hashlib
import numpy as np

from src.storage.sqlite_store import SQLiteGraphStore
from src.core.models import MemoryAtom, AtomType
from loguru import logger


@dataclass
class AttentionScore:
    """Attention score for a memory"""
    memory_id: str
    raw_score: float
    normalized_score: float
    relevance_factors: Dict[str, float]


@dataclass
class AttentionContext:
    """Context for attention computation"""
    query: str
    keywords: List[str]
    domain: Optional[str]
    recency_weight: float = 0.3
    semantic_weight: float = 0.5
    confidence_weight: float = 0.2


class AttentionMemoryRetrieval:
    """
    Transformer-inspired attention mechanism for memory retrieval.
    
    Key insight: Not all memories are equally relevant.
    Weight by context similarity, recency, and confidence.
    
    Like self-attention in transformers:
    - Query = current context
    - Keys = memory subjects/predicates
    - Values = memory content
    - Attention weights = relevance scores
    """
    
    # Configuration
    CACHE_SIZE = 100  # Max cached query results
    CACHE_TTL_SECONDS = 300  # Cache expires after 5 minutes
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self._query_cache: OrderedDict[str, Tuple[datetime, List[Any]]] = OrderedDict()
        
        logger.info("AttentionMemoryRetrieval initialized")
    
    def _cache_key(self, user_id: str, query: str, domain: Optional[str]) -> str:
        """Generate cache key for query"""
        content = f"{user_id}:{query}:{domain or ''}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached(self, key: str) -> Optional[List[Any]]:
        """Get cached result if valid"""
        if key in self._query_cache:
            cached_time, result = self._query_cache[key]
            if (datetime.now() - cached_time).seconds < self.CACHE_TTL_SECONDS:
                self._query_cache.move_to_end(key)  # LRU update
                return result
            else:
                del self._query_cache[key]  # Expired
        return None
    
    def _set_cached(self, key: str, result: List[Any]) -> None:
        """Cache result with LRU eviction"""
        self._query_cache[key] = (datetime.now(), result)
        self._query_cache.move_to_end(key)
        
        # Evict oldest if over limit
        while len(self._query_cache) > self.CACHE_SIZE:
            self._query_cache.popitem(last=False)
    
    def clear_cache(self) -> int:
        """Clear attention cache. Returns count cleared."""
        count = len(self._query_cache)
        self._query_cache.clear()
        return count
    
    async def retrieve_with_attention(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        domain: Optional[str] = None,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Retrieve memories weighted by attention to query.
        
        Args:
            user_id: User to retrieve for
            query: Current context/question
            top_k: Number of memories to return
            domain: Optional domain filter
            temperature: Softmax temperature (higher = more uniform)
        
        Returns:
            Top-k memories with attention scores
        """
        # Check cache first
        cache_key = self._cache_key(user_id, query, domain)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return {"n": len(cached), "top": cached[:5], "c": 1}  # c=1 means cached
        
        # Build attention context
        context = AttentionContext(
            query=query,
            keywords=self._extract_keywords(query),
            domain=domain
        )
        
        # Get all relevant memories
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        
        if not all_atoms:
            return {"found": 0, "memories": []}
        
        # Compute attention scores
        scored_memories = []
        for atom in all_atoms:
            score = self._compute_attention_score(atom, context)
            scored_memories.append((atom, score))
        
        # Apply softmax normalization
        scores = [s[1] for s in scored_memories]
        normalized = self._softmax(scores, temperature)
        
        # Combine with normalized scores
        for i, (atom, raw_score) in enumerate(scored_memories):
            scored_memories[i] = (atom, raw_score, normalized[i])
        
        # Sort by normalized score
        scored_memories.sort(key=lambda x: x[2], reverse=True)
        
        # Take top-k
        top_memories = scored_memories[:top_k]
        
        # Build result and cache it
        result = [{"p": m[0].predicate, "o": m[0].object[:40], "a": round(m[2], 3)} for m in top_memories]
        self._set_cached(cache_key, result)
        
        return {
            "n": len(top_memories),
            "top": result[:5]
        }
    
    def _compute_attention_score(
        self,
        atom: MemoryAtom,
        context: AttentionContext
    ) -> float:
        """
        Compute attention score for a single memory.
        
        Score = semantic_similarity * recency * confidence
        """
        # Semantic similarity (keyword overlap)
        semantic_score = self._semantic_similarity(atom, context)
        
        # Recency score (exponential decay)
        recency_score = self._recency_score(atom)
        
        # Confidence score
        confidence_score = atom.confidence
        
        # Weighted combination
        total = (
            context.semantic_weight * semantic_score +
            context.recency_weight * recency_score +
            context.confidence_weight * confidence_score
        )
        
        return total
    
    def _semantic_similarity(
        self,
        atom: MemoryAtom,
        context: AttentionContext
    ) -> float:
        """Compute semantic similarity via keyword overlap"""
        atom_text = f"{atom.subject} {atom.predicate} {atom.object}".lower()
        atom_words = set(atom_text.split())
        
        query_words = set(context.keywords)
        
        if not query_words:
            return 0.5  # Neutral if no keywords
        
        overlap = len(atom_words & query_words)
        similarity = overlap / len(query_words)
        
        # Boost for exact phrase match
        if context.query.lower() in atom_text:
            similarity = min(1.0, similarity + 0.3)
        
        # Boost for domain match
        if context.domain and context.domain in atom.contexts:
            similarity = min(1.0, similarity + 0.2)
        
        return similarity
    
    def _recency_score(self, atom: MemoryAtom) -> float:
        """Compute recency score with exponential decay"""
        if not atom.first_observed:
            return 0.5
        
        age_days = (datetime.now() - atom.first_observed).days
        
        # Exponential decay with half-life of 30 days
        half_life = 30
        decay = math.exp(-0.693 * age_days / half_life)
        
        return decay
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        # Remove common words
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "what", "which", "who", "whom", "this", "that", "these",
            "those", "am", "and", "but", "if", "or", "because", "until",
            "while", "about", "against", "between", "into", "through"
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords
    
    def _softmax(self, scores: List[float], temperature: float = 1.0) -> List[float]:
        """Apply softmax normalization"""
        if not scores:
            return []
        
        # Scale by temperature
        scaled = [s / temperature for s in scores]
        
        # Subtract max for numerical stability
        max_score = max(scaled)
        exp_scores = [math.exp(s - max_score) for s in scaled]
        
        # Normalize
        total = sum(exp_scores)
        if total == 0:
            return [1.0 / len(scores)] * len(scores)
        
        return [e / total for e in exp_scores]
    
    def _entropy(self, distribution: List[float]) -> float:
        """Compute entropy of attention distribution"""
        entropy = 0.0
        for p in distribution:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    async def multi_head_attention(
        self,
        user_id: str,
        query: str,
        num_heads: int = 4,
        top_k_per_head: int = 5
    ) -> Dict[str, Any]:
        """
        Multi-head attention - different aspects of the query.
        
        Like transformer multi-head attention:
        - Each head focuses on different aspects
        - Results are combined for richer retrieval
        """
        heads_results = []
        
        # Head 1: Semantic focus
        semantic_ctx = AttentionContext(
            query=query,
            keywords=self._extract_keywords(query),
            domain=None,
            semantic_weight=0.8,
            recency_weight=0.1,
            confidence_weight=0.1
        )
        
        # Head 2: Recency focus
        recency_ctx = AttentionContext(
            query=query,
            keywords=self._extract_keywords(query),
            domain=None,
            semantic_weight=0.2,
            recency_weight=0.7,
            confidence_weight=0.1
        )
        
        # Head 3: Confidence focus
        confidence_ctx = AttentionContext(
            query=query,
            keywords=self._extract_keywords(query),
            domain=None,
            semantic_weight=0.2,
            recency_weight=0.1,
            confidence_weight=0.7
        )
        
        # Head 4: Balanced
        balanced_ctx = AttentionContext(
            query=query,
            keywords=self._extract_keywords(query),
            domain=None,
            semantic_weight=0.4,
            recency_weight=0.3,
            confidence_weight=0.3
        )
        
        contexts = [semantic_ctx, recency_ctx, confidence_ctx, balanced_ctx][:num_heads]
        head_names = ["semantic", "recency", "confidence", "balanced"][:num_heads]
        
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        
        for ctx, name in zip(contexts, head_names):
            scored = []
            for atom in all_atoms:
                score = self._compute_attention_score(atom, ctx)
                scored.append((atom, score))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:top_k_per_head]
            
            heads_results.append({
                "head": name,
                "memories": [
                    {
                        "predicate": m[0].predicate,
                        "object": m[0].object[:50],
                        "score": round(m[1], 3)
                    }
                    for m in top
                ]
            })
        
        return {"heads": len(heads_results), "top": [h["memories"][0] if h["memories"] else {} for h in heads_results]}
    
    async def cross_attention(
        self,
        user_id: str,
        source_context: str,
        target_context: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Cross-attention between two contexts.
        
        Find memories that bridge two different contexts.
        Useful for finding connections between domains.
        """
        source_keywords = set(self._extract_keywords(source_context))
        target_keywords = set(self._extract_keywords(target_context))
        
        all_atoms = await self.store.get_atoms_by_subject(user_id)
        
        bridging_memories = []
        
        for atom in all_atoms:
            atom_text = f"{atom.subject} {atom.predicate} {atom.object}".lower()
            atom_words = set(atom_text.split())
            
            source_overlap = len(atom_words & source_keywords)
            target_overlap = len(atom_words & target_keywords)
            
            # Memory bridges if it relates to both contexts
            if source_overlap > 0 and target_overlap > 0:
                bridge_score = (source_overlap + target_overlap) * atom.confidence
                bridging_memories.append((atom, bridge_score, source_overlap, target_overlap))
        
        bridging_memories.sort(key=lambda x: x[1], reverse=True)
        top = bridging_memories[:top_k]
        
        return {"n": len(top), "bridges": [{"p": m[0].predicate, "s": round(m[1], 2)} for m in top[:5]]}

    async def mmr_retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        lambda_param: float = 0.6,
        min_dissim: float = 0.25,
        max_candidates: int = 100  # Limit to avoid O(n²) blowup
    ) -> Dict[str, Any]:
        """
        Maximal Marginal Relevance (MMR) retrieval.
        
        Per Carbonell & Goldstein (1998):
        MMR(d_i) = λ·rel(d_i, q) - (1-λ)·max(sim(d_i, d_j) for d_j in S)
        
        Args:
            user_id: User to retrieve for
            query: Current context/question
            top_k: Number of diverse memories to return
            lambda_param: Relevance weight (0.6 = balanced, higher = more relevant)
            min_dissim: Hard floor for minimum pairwise dissimilarity
            max_candidates: Max atoms to consider (limits O(n²) cost)
        
        Returns:
            Top-k diverse memories with MMR scores and diversity metrics
        """
        # Get relevant memories with limit
        all_atoms = await self.store.get_substantiated_atoms(subject=user_id, limit=max_candidates)
        
        if not all_atoms:
            return {"n": 0, "memories": [], "mean_dissim": 0.0}
        
        # Build attention context for relevance scoring
        context = AttentionContext(
            query=query,
            keywords=self._extract_keywords(query),
            domain=None
        )
        
        # Compute relevance scores and embeddings
        relevance_scores = []
        embeddings = []
        
        for atom in all_atoms:
            # Relevance score from attention mechanism
            rel_score = self._compute_attention_score(atom, context)
            relevance_scores.append(rel_score)
            
            # Simple bag-of-words embedding for similarity
            emb = self._text_to_embedding(f"{atom.subject} {atom.predicate} {atom.object}")
            embeddings.append(emb)
        
        relevance_scores = np.array(relevance_scores)
        embeddings = np.array(embeddings)
        
        # MMR selection
        selected_indices, metrics = self._mmr_select(
            relevance_scores, embeddings, top_k, lambda_param, min_dissim
        )
        
        # Build result
        selected_memories = []
        for idx in selected_indices:
            atom = all_atoms[idx]
            selected_memories.append({
                "p": atom.predicate,
                "o": atom.object[:50],
                "rel": round(relevance_scores[idx], 3),
                "conf": round(atom.confidence, 2)
            })
        
        return {
            "n": len(selected_memories),
            "memories": selected_memories[:5],
            "mean_dissim": round(metrics["mean_dissim"], 3),
            "lambda": lambda_param
        }
    
    def _text_to_embedding(self, text: str, dim: int = 64) -> np.ndarray:
        """
        Simple bag-of-words embedding.
        
        Uses hash-based feature extraction for speed.
        In production, would use sentence-transformers.
        """
        words = text.lower().split()
        embedding = np.zeros(dim)
        
        for word in words:
            # Hash word to get feature index
            h = hash(word) % dim
            embedding[h] += 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _mmr_select(
        self,
        relevance_scores: np.ndarray,
        embeddings: np.ndarray,
        k: int,
        lambda_param: float,
        min_dissim: float
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Greedy MMR selection with O(n*k) complexity.
        
        Returns:
            (selected_indices, metrics_dict)
        """
        import time
        start_time = time.time()
        max_time = 2.0  # Max 2 seconds
        
        n_items = len(relevance_scores)
        if n_items == 0:
            return [], {"mean_dissim": 0.0}
        if k >= n_items:
            return list(range(n_items)), {"mean_dissim": 0.0}
        
        selected = []
        remaining = list(range(n_items))  # Use list for faster iteration
        
        # Pre-sort by relevance for faster first pick
        remaining.sort(key=lambda i: relevance_scores[i], reverse=True)
        
        for _ in range(k):
            # Timeout check
            if time.time() - start_time > max_time:
                logger.warning(f"MMR timeout after {len(selected)} selections")
                break
            
            best_idx = None
            best_score = float('-inf')
            
            # Only check top candidates (optimization)
            candidates = remaining[:min(50, len(remaining))]
            
            for idx in candidates:
                rel = relevance_scores[idx]
                
                if len(selected) == 0:
                    mmr_score = rel
                else:
                    # Compute max similarity to selected (vectorized would be faster)
                    max_sim = max(
                        self._cosine_similarity(embeddings[idx], embeddings[sel_idx])
                        for sel_idx in selected
                    )
                    mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim
                
                # Skip if too similar to already selected
                if selected and min_dissim > 0:
                    min_dissim_val = min(
                        1.0 - self._cosine_similarity(embeddings[idx], embeddings[sel_idx])
                        for sel_idx in selected
                    )
                    if min_dissim_val < min_dissim:
                        continue
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is None:
                break
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        mean_dissim = self._compute_mean_dissimilarity(embeddings, selected) if len(selected) > 1 else 0.0
        
        return selected, {"mean_dissim": mean_dissim, "n_selected": len(selected)}
    
    def _compute_mean_dissimilarity(self, embeddings: np.ndarray, indices: List[int]) -> float:
        """Compute mean pairwise dissimilarity of selected items"""
        if len(indices) < 2:
            return 0.0
        
        dissims = []
        for i, idx1 in enumerate(indices):
            for idx2 in indices[i+1:]:
                sim = self._cosine_similarity(embeddings[idx1], embeddings[idx2])
                dissim = 1.0 - sim
                dissims.append(dissim)
        
        return float(np.mean(dissims)) if dissims else 0.0
