"""
Φ (Phi) Integration Measurement

Calculates integrated information metrics for domains based on stored knowledge atoms.
Uses network connectivity and mutual information as proxy for Tononi's Φ.

Architecture:
- Builds a dependency graph from atoms in a domain
- Measures integration = how much the system's parts are interdependent
- Tracks Φ over time via snapshots stored in SQLite
"""

import json
import math
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class PhiResult:
    domain: str
    phi: float  # 0.0 - 1.0
    n_nodes: int
    n_edges: int
    density: float
    components: int  # disconnected subgraphs
    max_component_size: int
    vulnerability: float  # how much Φ drops if strongest node removed
    timestamp: float = field(default_factory=time.time)
    sub_scores: Dict[str, float] = field(default_factory=dict)


class PhiIntegrationCalculator:
    """Calculate Φ integration metrics from knowledge atoms."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent.parent.parent / "data" / "pltm_mcp.db"
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create phi_snapshots table for timeseries tracking"""
        conn = sqlite3.connect(str(self.db_path), timeout=2)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS phi_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                phi REAL NOT NULL,
                n_nodes INTEGER,
                n_edges INTEGER,
                density REAL,
                components INTEGER,
                vulnerability REAL,
                sub_scores TEXT,
                timestamp REAL NOT NULL,
                UNIQUE(domain, timestamp)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_phi_domain ON phi_snapshots(domain)")
        conn.commit()
        conn.close()
    
    def calculate(self, domain: str, atoms: list) -> PhiResult:
        """
        Calculate Φ for a domain from its atoms.
        
        Φ proxy = weighted combination of:
        - Connectivity density (how interconnected)
        - Integration ratio (largest component / total nodes)
        - Mutual information proxy (shared predicates across subjects)
        - Vulnerability (1 - Φ_after_removal / Φ_before)
        """
        if not atoms:
            return PhiResult(domain=domain, phi=0.0, n_nodes=0, n_edges=0,
                           density=0.0, components=0, max_component_size=0,
                           vulnerability=0.0, sub_scores={})
        
        # Build adjacency graph: subjects and objects are nodes, predicates are edges
        nodes = set()
        edges = []
        adj = defaultdict(set)
        
        for atom in atoms:
            s = atom.subject.lower().strip()
            o = atom.object.lower().strip()[:60]  # Truncate long objects
            nodes.add(s)
            nodes.add(o)
            edges.append((s, o, atom.predicate, atom.confidence))
            adj[s].add(o)
            adj[o].add(s)
        
        n_nodes = len(nodes)
        n_edges = len(edges)
        
        if n_nodes < 2:
            return PhiResult(domain=domain, phi=0.1, n_nodes=n_nodes, n_edges=n_edges,
                           density=0.0, components=1, max_component_size=n_nodes,
                           vulnerability=0.0, sub_scores={"connectivity": 0.1})
        
        # 1. Connectivity density
        max_edges = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0
        connectivity_score = min(1.0, density * 5)  # Scale up since real graphs are sparse
        
        # 2. Connected components (integration)
        components = self._find_components(nodes, adj)
        max_component = max(len(c) for c in components) if components else 0
        integration_ratio = max_component / n_nodes if n_nodes > 0 else 0
        
        # 3. Mutual information proxy: shared predicates across different subject pairs
        predicate_subjects = defaultdict(set)
        for atom in atoms:
            predicate_subjects[atom.predicate.lower()].add(atom.subject.lower())
        
        shared_predicates = sum(1 for p, subs in predicate_subjects.items() if len(subs) > 1)
        total_predicates = len(predicate_subjects)
        mi_score = shared_predicates / total_predicates if total_predicates > 0 else 0
        
        # 4. Vulnerability: how much integration drops if most-connected node removed
        if n_nodes > 2:
            degree = {n: len(adj[n]) for n in nodes}
            hub = max(degree, key=degree.get)
            reduced_nodes = nodes - {hub}
            reduced_adj = defaultdict(set)
            for n in reduced_nodes:
                reduced_adj[n] = adj[n] - {hub}
            reduced_components = self._find_components(reduced_nodes, reduced_adj)
            reduced_max = max(len(c) for c in reduced_components) if reduced_components else 0
            reduced_integration = reduced_max / len(reduced_nodes) if reduced_nodes else 0
            vulnerability = max(0, integration_ratio - reduced_integration)
        else:
            vulnerability = 0.0
        
        # Composite Φ
        phi = (
            0.30 * connectivity_score +
            0.35 * integration_ratio +
            0.20 * mi_score +
            0.15 * (1.0 - vulnerability)  # Lower vulnerability = more robust integration
        )
        phi = round(min(1.0, max(0.0, phi)), 3)
        
        sub_scores = {
            "connectivity": round(connectivity_score, 3),
            "integration": round(integration_ratio, 3),
            "mutual_info": round(mi_score, 3),
            "robustness": round(1.0 - vulnerability, 3),
        }
        
        result = PhiResult(
            domain=domain, phi=phi, n_nodes=n_nodes, n_edges=n_edges,
            density=round(density, 4), components=len(components),
            max_component_size=max_component, vulnerability=round(vulnerability, 3),
            sub_scores=sub_scores,
        )
        
        # Persist snapshot
        self._save_snapshot(result)
        
        return result
    
    def _find_components(self, nodes, adj) -> List[set]:
        """BFS to find connected components"""
        visited = set()
        components = []
        for start in nodes:
            if start in visited:
                continue
            component = set()
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                for neighbor in adj.get(node, set()):
                    if neighbor not in visited and neighbor in nodes:
                        queue.append(neighbor)
            components.append(component)
        return components
    
    def _save_snapshot(self, result: PhiResult):
        """Persist Φ snapshot for timeseries"""
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=2)
            conn.execute(
                "INSERT OR REPLACE INTO phi_snapshots (domain, phi, n_nodes, n_edges, density, components, vulnerability, sub_scores, timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
                (result.domain, result.phi, result.n_nodes, result.n_edges,
                 result.density, result.components, result.vulnerability,
                 json.dumps(result.sub_scores), result.timestamp)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to save phi snapshot: {e}")
    
    def get_timeseries(self, domain: str, limit: int = 50) -> List[Dict]:
        """Get Φ history for a domain"""
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=2)
            cursor = conn.execute(
                "SELECT phi, n_nodes, n_edges, density, components, vulnerability, sub_scores, timestamp FROM phi_snapshots WHERE domain = ? ORDER BY timestamp DESC LIMIT ?",
                (domain, limit)
            )
            rows = cursor.fetchall()
            conn.close()
            return [
                {"phi": r[0], "nodes": r[1], "edges": r[2], "density": r[3],
                 "components": r[4], "vulnerability": r[5],
                 "sub_scores": json.loads(r[6]) if r[6] else {},
                 "ts": r[7]}
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"Failed to get phi timeseries: {e}")
            return []
    
    def to_compact(self, result: PhiResult) -> Dict:
        """Token-efficient serialization"""
        return {
            "domain": result.domain,
            "phi": result.phi,
            "nodes": result.n_nodes,
            "edges": result.n_edges,
            "density": result.density,
            "components": result.components,
            "max_component": result.max_component_size,
            "vulnerability": result.vulnerability,
            "sub": result.sub_scores,
        }
