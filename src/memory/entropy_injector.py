"""
Entropy Injector for PLTM

Deliberately activates distant knowledge clusters to break
the "conceptual neighborhood" trap identified in the experiment.

Problem: Even "diverse" domains share neighborhoods → flat entropy
Solution: Force activation of semantically distant memories

Methods:
1. Random domain sampling - pick from least-accessed domains
2. Antipodal activation - find memories maximally distant from current context
3. Temporal diversity - mix old and new memories
4. Cross-domain bridging - activate memories that span multiple domains
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import random
import math

from loguru import logger


@dataclass
class EntropyInjection:
    """Result of an entropy injection"""
    injection_id: str
    method: str
    memories_activated: int
    domains_touched: int
    estimated_entropy_gain: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.injection_id,
            "method": self.method,
            "n": self.memories_activated,
            "domains": self.domains_touched,
            "entropy_gain": round(self.estimated_entropy_gain, 3)
        }


class EntropyInjector:
    """
    Entropy injection system for breaking conceptual neighborhoods.
    
    When the system is over-integrated (high I, low H), this forces
    diversity by activating distant memory clusters.
    """
    
    def __init__(self, store):
        self.store = store
        self._injection_counter = 0
        self.injection_history: List[EntropyInjection] = []
        
        logger.info("EntropyInjector initialized")
    
    async def inject_random_domains(
        self,
        user_id: str,
        n_domains: int = 3,
        memories_per_domain: int = 2
    ) -> EntropyInjection:
        """
        Inject entropy by sampling from random/least-accessed domains.
        
        Finds domains the user has knowledge in but rarely accesses,
        then activates memories from those domains.
        """
        self._injection_counter += 1
        
        # Get all domains for user
        cursor = await self.store._conn.execute(
            """SELECT DISTINCT predicate, COUNT(*) as cnt 
               FROM atoms WHERE subject = ? 
               GROUP BY predicate ORDER BY cnt ASC LIMIT 20""",
            (user_id,)
        )
        rows = await cursor.fetchall()
        
        if not rows:
            return EntropyInjection(
                injection_id=f"inj_{self._injection_counter}",
                method="random_domains",
                memories_activated=0,
                domains_touched=0,
                estimated_entropy_gain=0.0,
                timestamp=datetime.now()
            )
        
        # Sample from least-accessed domains
        domains = [row[0] for row in rows]
        selected_domains = random.sample(domains, min(n_domains, len(domains)))
        
        activated = []
        for domain in selected_domains:
            cursor = await self.store._conn.execute(
                """SELECT id, predicate, object FROM atoms 
                   WHERE subject = ? AND predicate = ?
                   ORDER BY RANDOM() LIMIT ?""",
                (user_id, domain, memories_per_domain)
            )
            memories = await cursor.fetchall()
            activated.extend(memories)
        
        # Estimate entropy gain (more domains = more entropy)
        entropy_gain = 0.1 * len(selected_domains) * (1 + math.log(len(activated) + 1))
        
        injection = EntropyInjection(
            injection_id=f"inj_{self._injection_counter}",
            method="random_domains",
            memories_activated=len(activated),
            domains_touched=len(selected_domains),
            estimated_entropy_gain=entropy_gain,
            timestamp=datetime.now()
        )
        
        self.injection_history.append(injection)
        logger.info(f"Entropy injection: {len(activated)} memories from {len(selected_domains)} domains")
        
        return injection
    
    async def inject_antipodal(
        self,
        user_id: str,
        current_context: str,
        n_memories: int = 5
    ) -> Tuple[EntropyInjection, List[Dict[str, Any]]]:
        """
        Inject entropy by finding memories maximally distant from current context.
        
        Uses simple keyword non-overlap as distance metric.
        """
        self._injection_counter += 1
        
        context_words = set(current_context.lower().split())
        
        # Get candidate memories
        cursor = await self.store._conn.execute(
            """SELECT id, predicate, object FROM atoms 
               WHERE subject = ? AND graph = 'substantiated'
               LIMIT 100""",
            (user_id,)
        )
        rows = await cursor.fetchall()
        
        if not rows:
            return EntropyInjection(
                injection_id=f"inj_{self._injection_counter}",
                method="antipodal",
                memories_activated=0,
                domains_touched=0,
                estimated_entropy_gain=0.0,
                timestamp=datetime.now()
            ), []
        
        # Score by distance (non-overlap)
        scored = []
        for row in rows:
            memory_text = f"{row[1]} {row[2]}".lower()
            memory_words = set(memory_text.split())
            
            # Distance = 1 - (overlap / union)
            overlap = len(context_words & memory_words)
            union = len(context_words | memory_words)
            distance = 1 - (overlap / union) if union > 0 else 1.0
            
            scored.append((row, distance))
        
        # Sort by distance (most distant first)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N most distant
        selected = scored[:n_memories]
        
        # Count unique domains
        domains = set(row[0][1] for row in selected)
        
        # Estimate entropy gain
        avg_distance = sum(s[1] for s in selected) / len(selected) if selected else 0
        entropy_gain = 0.15 * len(selected) * avg_distance
        
        injection = EntropyInjection(
            injection_id=f"inj_{self._injection_counter}",
            method="antipodal",
            memories_activated=len(selected),
            domains_touched=len(domains),
            estimated_entropy_gain=entropy_gain,
            timestamp=datetime.now()
        )
        
        self.injection_history.append(injection)
        
        memories = [{"p": s[0][1], "o": s[0][2][:40], "dist": round(s[1], 2)} for s in selected]
        
        return injection, memories
    
    async def inject_temporal_diversity(
        self,
        user_id: str,
        n_old: int = 3,
        n_recent: int = 2
    ) -> Tuple[EntropyInjection, List[Dict[str, Any]]]:
        """
        Inject entropy by mixing old and recent memories.
        
        Temporal diversity prevents recency bias from dominating.
        """
        self._injection_counter += 1
        
        # Get oldest memories
        cursor = await self.store._conn.execute(
            """SELECT id, predicate, object, first_observed FROM atoms 
               WHERE subject = ? AND graph = 'substantiated'
               ORDER BY first_observed ASC LIMIT ?""",
            (user_id, n_old)
        )
        old_memories = await cursor.fetchall()
        
        # Get most recent memories
        cursor = await self.store._conn.execute(
            """SELECT id, predicate, object, first_observed FROM atoms 
               WHERE subject = ? AND graph = 'substantiated'
               ORDER BY first_observed DESC LIMIT ?""",
            (user_id, n_recent)
        )
        recent_memories = await cursor.fetchall()
        
        all_memories = old_memories + recent_memories
        domains = set(m[1] for m in all_memories)
        
        # Entropy gain from temporal spread
        entropy_gain = 0.12 * len(all_memories) * (1 + 0.5 * len(domains))
        
        injection = EntropyInjection(
            injection_id=f"inj_{self._injection_counter}",
            method="temporal_diversity",
            memories_activated=len(all_memories),
            domains_touched=len(domains),
            estimated_entropy_gain=entropy_gain,
            timestamp=datetime.now()
        )
        
        self.injection_history.append(injection)
        
        memories = [{"p": m[1], "o": m[2][:40], "age": "old" if m in old_memories else "recent"} 
                    for m in all_memories]
        
        return injection, memories
    
    async def get_entropy_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get entropy-related statistics for a user.
        
        Helps diagnose if entropy injection is needed.
        """
        # Count by domain
        cursor = await self.store._conn.execute(
            """SELECT predicate, COUNT(*) FROM atoms 
               WHERE subject = ? GROUP BY predicate""",
            (user_id,)
        )
        domain_counts = await cursor.fetchall()
        
        if not domain_counts:
            return {"domains": 0, "total": 0, "entropy": 0.0, "needs_injection": False}
        
        total = sum(c[1] for c in domain_counts)
        
        # Compute Shannon entropy of domain distribution
        entropy = 0.0
        for _, count in domain_counts:
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Max entropy for this number of domains
        max_entropy = math.log2(len(domain_counts)) if len(domain_counts) > 1 else 1.0
        
        # Normalized entropy (0-1)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Diagnose if injection needed
        needs_injection = normalized_entropy < 0.6 or len(domain_counts) < 5
        
        return {
            "domains": len(domain_counts),
            "total": total,
            "entropy": round(entropy, 3),
            "normalized": round(normalized_entropy, 3),
            "max_entropy": round(max_entropy, 3),
            "needs_injection": needs_injection,
            "top_domains": [{"d": d[0], "n": d[1]} for d in sorted(domain_counts, key=lambda x: x[1], reverse=True)[:5]]
        }
    
    def get_injection_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent injection history"""
        return [inj.to_dict() for inj in self.injection_history[-last_n:]]
