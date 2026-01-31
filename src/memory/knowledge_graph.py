"""
Knowledge Network Graph

Universal Principle #4: Network Effects (Connectivity)

Track connections between concepts. More connections = higher value.
Knowledge compounds exponentially through network effects.

Benefits:
- Emergent insights from connections
- Knowledge value increases with connectivity
- Enables cross-domain synthesis
- Mirrors neural network structure
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math

from src.storage.sqlite_store import SQLiteGraphStore
from src.core.models import MemoryAtom, AtomType, Provenance, GraphType
from loguru import logger


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph"""
    node_id: str
    concept: str
    domain: str
    connections: int = 0
    value_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeEdge:
    """An edge connecting two knowledge nodes"""
    from_node: str
    to_node: str
    relationship: str
    strength: float
    bidirectional: bool = False


class KnowledgeNetworkGraph:
    """
    Knowledge graph with network effects.
    
    Key insight: Knowledge value increases with connections.
    A fact connected to 100 other facts is more valuable than
    an isolated fact - it enables more inferences.
    
    Like Metcalfe's Law: Value ∝ n² (connections)
    """
    
    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        
        logger.info("KnowledgeNetworkGraph initialized - network effects enabled")
    
    async def add_concept(
        self,
        concept: str,
        domain: str,
        related_concepts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Add a concept to the knowledge graph.
        
        Automatically creates edges to related concepts.
        """
        node_id = self._concept_to_id(concept)
        
        if node_id not in self.nodes:
            self.nodes[node_id] = KnowledgeNode(
                node_id=node_id,
                concept=concept,
                domain=domain
            )
        
        edges_created = 0
        
        if related_concepts:
            for related in related_concepts:
                related_id = self._concept_to_id(related)
                
                # Create related node if doesn't exist
                if related_id not in self.nodes:
                    self.nodes[related_id] = KnowledgeNode(
                        node_id=related_id,
                        concept=related,
                        domain=domain
                    )
                
                # Create edge
                if related_id not in self.adjacency[node_id]:
                    self.edges.append(KnowledgeEdge(
                        from_node=node_id,
                        to_node=related_id,
                        relationship="related_to",
                        strength=0.5,
                        bidirectional=True
                    ))
                    self.adjacency[node_id].add(related_id)
                    self.adjacency[related_id].add(node_id)
                    edges_created += 1
                    
                    # Update connection counts
                    self.nodes[node_id].connections += 1
                    self.nodes[related_id].connections += 1
        
        # Recalculate value scores
        self._update_value_scores()
        
        return {"id": node_id, "edges": edges_created, "conn": self.nodes[node_id].connections}
    
    def _concept_to_id(self, concept: str) -> str:
        """Convert concept to node ID"""
        return concept.lower().replace(" ", "_")
    
    def _update_value_scores(self):
        """
        Update value scores based on network effects.
        
        Value = log(1 + connections) * centrality_bonus
        
        This implements Metcalfe's Law: more connections = more value
        """
        if not self.nodes:
            return
        
        max_connections = max(n.connections for n in self.nodes.values()) or 1
        
        for node in self.nodes.values():
            # Base value from connections (logarithmic to prevent explosion)
            connection_value = math.log(1 + node.connections)
            
            # Centrality bonus (normalized)
            centrality = node.connections / max_connections
            
            # Combined score
            node.value_score = connection_value * (1 + centrality)
    
    async def find_path(
        self,
        from_concept: str,
        to_concept: str,
        max_depth: int = 6
    ) -> Dict[str, Any]:
        """
        Find path between two concepts using bidirectional BFS.
        
        Bidirectional search is O(b^(d/2)) vs O(b^d) for standard BFS.
        Much faster for sparse graphs.
        """
        from_id = self._concept_to_id(from_concept)
        to_id = self._concept_to_id(to_concept)
        
        if from_id not in self.nodes or to_id not in self.nodes:
            return {"found": False, "reason": "Concept not in graph"}
        
        if from_id == to_id:
            return {"path": [from_concept], "len": 0}
        
        # Bidirectional BFS
        forward_visited = {from_id: [from_id]}
        backward_visited = {to_id: [to_id]}
        forward_queue = [from_id]
        backward_queue = [to_id]
        
        for depth in range(max_depth // 2 + 1):
            # Expand forward
            if forward_queue:
                next_forward = []
                for current in forward_queue:
                    for neighbor in self.adjacency[current]:
                        if neighbor in backward_visited:
                            # Found meeting point!
                            path = forward_visited[current] + backward_visited[neighbor][::-1]
                            return {"path": [self.nodes[n].concept for n in path], "len": len(path) - 1}
                        if neighbor not in forward_visited:
                            forward_visited[neighbor] = forward_visited[current] + [neighbor]
                            next_forward.append(neighbor)
                forward_queue = next_forward
            
            # Expand backward
            if backward_queue:
                next_backward = []
                for current in backward_queue:
                    for neighbor in self.adjacency[current]:
                        if neighbor in forward_visited:
                            # Found meeting point!
                            path = forward_visited[neighbor] + backward_visited[current][::-1]
                            return {"path": [self.nodes[n].concept for n in path], "len": len(path) - 1}
                        if neighbor not in backward_visited:
                            backward_visited[neighbor] = backward_visited[current] + [neighbor]
                            next_backward.append(neighbor)
                backward_queue = next_backward
            
            if not forward_queue and not backward_queue:
                break
        
        return {"found": False, "reason": "No path within depth limit"}
    
    async def get_most_connected(self, top_k: int = 10) -> Dict[str, Any]:
        """
        Get most connected concepts (highest network value).
        
        These are the "hub" concepts that connect many others.
        """
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.value_score,
            reverse=True
        )[:top_k]
        
        return {"top": [{"c": n.concept, "conn": n.connections} for n in sorted_nodes], "nodes": len(self.nodes)}
    
    async def get_neighborhood(
        self,
        concept: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get neighborhood of a concept up to given depth.
        
        Returns all concepts within N hops.
        """
        node_id = self._concept_to_id(concept)
        
        if node_id not in self.nodes:
            return {"found": False}
        
        visited = {node_id}
        current_level = {node_id}
        levels = [{concept}]
        
        for d in range(depth):
            next_level = set()
            for node in current_level:
                for neighbor in self.adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
            
            if next_level:
                levels.append({self.nodes[n].concept for n in next_level})
            current_level = next_level
        
        return {"center": concept, "total": len(visited), "d1": list(levels[1]) if len(levels) > 1 else []}
    
    async def find_bridges(self, top_k: int = 10) -> Dict[str, Any]:
        """
        Find bridge concepts that connect different domains.
        
        These are the most valuable for cross-domain insights.
        """
        bridges = []
        
        for node_id, node in self.nodes.items():
            neighbor_domains = set()
            for neighbor_id in self.adjacency[node_id]:
                neighbor = self.nodes.get(neighbor_id)
                if neighbor:
                    neighbor_domains.add(neighbor.domain)
            
            # Bridge if connected to multiple domains
            if len(neighbor_domains) > 1:
                bridge_score = len(neighbor_domains) * node.connections
                bridges.append((node, neighbor_domains, bridge_score))
        
        bridges.sort(key=lambda x: x[2], reverse=True)
        
        return {"bridges": [{"c": b[0].concept, "domains": len(b[1])} for b in bridges[:top_k]]}
    
    async def compute_pagerank(
        self,
        damping: float = 0.85,
        iterations: int = 20
    ) -> Dict[str, Any]:
        """
        Compute PageRank for all concepts.
        
        Higher PageRank = more important concept in the network.
        """
        if not self.nodes:
            return {"nodes": 0}
        
        n = len(self.nodes)
        node_ids = list(self.nodes.keys())
        
        # Initialize
        ranks = {nid: 1.0 / n for nid in node_ids}
        
        for _ in range(iterations):
            new_ranks = {}
            for node_id in node_ids:
                # Sum of incoming ranks
                incoming = 0.0
                for other_id in node_ids:
                    if node_id in self.adjacency[other_id]:
                        out_degree = len(self.adjacency[other_id])
                        if out_degree > 0:
                            incoming += ranks[other_id] / out_degree
                
                new_ranks[node_id] = (1 - damping) / n + damping * incoming
            
            ranks = new_ranks
        
        # Sort by rank
        sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
        
        return {"top": [{"c": self.nodes[nid].concept, "pr": round(rank, 4)} for nid, rank in sorted_ranks[:10]]}
    
    async def suggest_connections(
        self,
        concept: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Suggest new connections for a concept.
        
        Based on:
        - Common neighbors (triadic closure)
        - Same domain
        - Similar connection patterns
        """
        node_id = self._concept_to_id(concept)
        
        if node_id not in self.nodes:
            return {"found": False}
        
        node = self.nodes[node_id]
        current_neighbors = self.adjacency[node_id]
        
        suggestions = []
        
        for other_id, other_node in self.nodes.items():
            if other_id == node_id or other_id in current_neighbors:
                continue
            
            score = 0.0
            reasons = []
            
            # Common neighbors (triadic closure)
            common = len(current_neighbors & self.adjacency[other_id])
            if common > 0:
                score += common * 0.5
                reasons.append(f"{common} common connections")
            
            # Same domain
            if other_node.domain == node.domain:
                score += 0.3
                reasons.append("same domain")
            
            # Similar connectivity
            conn_diff = abs(other_node.connections - node.connections)
            if conn_diff < 3:
                score += 0.2
                reasons.append("similar connectivity")
            
            if score > 0:
                suggestions.append((other_node.concept, score, reasons))
        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return {"suggestions": [{"to": s[0], "s": round(s[1], 2)} for s in suggestions[:top_k]]}
    
    async def get_network_stats(self) -> Dict[str, Any]:
        """Get overall network statistics"""
        if not self.nodes:
            return {"nodes": 0, "edges": 0}
        
        connections = [n.connections for n in self.nodes.values()]
        
        return {"n": len(self.nodes), "e": len(self.edges), "d": len(set(n.domain for n in self.nodes.values()))}
