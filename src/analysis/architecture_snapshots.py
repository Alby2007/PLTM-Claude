"""
Architecture Snapshotting & A/B Evaluation

Captures versioned snapshots of the PLTM system state (tool inventory,
memory counts, key metrics) to enable evidence-based comparison of
architecture changes rather than narrative-driven "this feels better."
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger


class ArchitectureSnapshotter:
    """
    Takes and compares snapshots of the PLTM system state.
    
    Each snapshot captures:
    - Tool inventory (count + names)
    - Data counts (atoms, typed memories, tables)
    - Key metrics (accuracy, confabulation rate, Φ stats, etc.)
    - Config hash for change detection
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent.parent.parent / "data" / "pltm_mcp.db"
        self._ensure_tables()

    def _ensure_tables(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS architecture_snapshots (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                timestamp REAL NOT NULL,
                tool_count INTEGER DEFAULT 0,
                tool_list TEXT DEFAULT '[]',
                atom_count INTEGER DEFAULT 0,
                typed_memory_count INTEGER DEFAULT 0,
                table_count INTEGER DEFAULT 0,
                config_hash TEXT DEFAULT '',
                metrics TEXT DEFAULT '{}',
                notes TEXT DEFAULT ''
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_arch_snap_ts ON architecture_snapshots(timestamp DESC)")
        conn.commit()
        conn.close()

    def take_snapshot(
        self,
        label: str,
        tool_names: Optional[List[str]] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Capture the current system state as a snapshot.
        
        Args:
            label: Human-readable name (e.g., "pre-ΦRMS", "v2.1")
            tool_names: List of registered MCP tool names (passed from server)
            notes: Optional description of what changed
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA busy_timeout = 5000")
        now = time.time()
        snap_id = str(uuid4())

        tool_list = sorted(tool_names or [])
        tool_count = len(tool_list)
        config_hash = hashlib.sha256(json.dumps(tool_list).encode()).hexdigest()[:16]

        # Gather counts
        atom_count = self._safe_count(conn, "SELECT COUNT(*) FROM atoms")
        typed_memory_count = self._safe_count(conn, "SELECT COUNT(*) FROM typed_memories")
        table_count = self._safe_count(
            conn, "SELECT COUNT(*) FROM sqlite_master WHERE type='table'")

        # Gather metrics
        metrics = self._gather_metrics(conn)

        conn.execute("""
            INSERT INTO architecture_snapshots
            (id, label, timestamp, tool_count, tool_list, atom_count,
             typed_memory_count, table_count, config_hash, metrics, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (snap_id, label, now, tool_count, json.dumps(tool_list),
              atom_count, typed_memory_count, table_count, config_hash,
              json.dumps(metrics, default=str), notes))
        conn.commit()
        conn.close()

        result = {
            "id": snap_id,
            "label": label,
            "timestamp": now,
            "tool_count": tool_count,
            "atom_count": atom_count,
            "typed_memory_count": typed_memory_count,
            "table_count": table_count,
            "config_hash": config_hash,
            "metrics": metrics,
            "notes": notes,
        }
        logger.info(f"Architecture snapshot '{label}' taken: {snap_id}")
        return result

    def list_snapshots(self, limit: int = 20) -> Dict[str, Any]:
        """List recent snapshots."""
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("""
            SELECT id, label, timestamp, tool_count, atom_count,
                   typed_memory_count, table_count, config_hash, notes
            FROM architecture_snapshots
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()
        conn.close()

        snapshots = [{
            "id": r[0], "label": r[1], "timestamp": r[2],
            "tool_count": r[3], "atom_count": r[4],
            "typed_memory_count": r[5], "table_count": r[6],
            "config_hash": r[7], "notes": r[8],
        } for r in rows]

        return {"count": len(snapshots), "snapshots": snapshots}

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the most recent snapshot."""
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute("""
            SELECT id, label, timestamp, tool_count, tool_list, atom_count,
                   typed_memory_count, table_count, config_hash, metrics, notes
            FROM architecture_snapshots
            ORDER BY timestamp DESC LIMIT 1
        """).fetchone()
        conn.close()

        if not row:
            return None
        return self._row_to_dict(row)

    def compare(self, snapshot_id_a: str, snapshot_id_b: str) -> Dict[str, Any]:
        """
        Compare two snapshots and produce a structured diff.
        
        Returns deltas for counts, metrics, and tool inventory changes.
        """
        conn = sqlite3.connect(str(self.db_path))
        a = conn.execute(
            "SELECT * FROM architecture_snapshots WHERE id = ?", (snapshot_id_a,)
        ).fetchone()
        b = conn.execute(
            "SELECT * FROM architecture_snapshots WHERE id = ?", (snapshot_id_b,)
        ).fetchone()
        conn.close()

        if not a or not b:
            missing = []
            if not a:
                missing.append(snapshot_id_a)
            if not b:
                missing.append(snapshot_id_b)
            return {"error": f"Snapshot(s) not found: {', '.join(missing)}"}

        snap_a = self._row_to_full_dict(a)
        snap_b = self._row_to_full_dict(b)

        # Tool diff
        tools_a = set(json.loads(snap_a["tool_list"]))
        tools_b = set(json.loads(snap_b["tool_list"]))
        tools_added = sorted(tools_b - tools_a)
        tools_removed = sorted(tools_a - tools_b)

        # Metric deltas
        metrics_a = json.loads(snap_a["metrics"]) if isinstance(snap_a["metrics"], str) else snap_a["metrics"]
        metrics_b = json.loads(snap_b["metrics"]) if isinstance(snap_b["metrics"], str) else snap_b["metrics"]
        metric_deltas = {}
        all_keys = set(list(metrics_a.keys()) + list(metrics_b.keys()))
        for key in sorted(all_keys):
            val_a = metrics_a.get(key)
            val_b = metrics_b.get(key)
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                delta = val_b - val_a
                metric_deltas[key] = {
                    "before": round(val_a, 4),
                    "after": round(val_b, 4),
                    "delta": round(delta, 4),
                    "direction": "improved" if delta > 0 else ("declined" if delta < 0 else "unchanged"),
                }
            else:
                metric_deltas[key] = {"before": val_a, "after": val_b}

        # Count deltas
        count_deltas = {}
        for field in ["tool_count", "atom_count", "typed_memory_count", "table_count"]:
            va = snap_a[field]
            vb = snap_b[field]
            count_deltas[field] = {"before": va, "after": vb, "delta": vb - va}

        return {
            "snapshot_a": {"id": snap_a["id"], "label": snap_a["label"], "timestamp": snap_a["timestamp"]},
            "snapshot_b": {"id": snap_b["id"], "label": snap_b["label"], "timestamp": snap_b["timestamp"]},
            "config_changed": snap_a["config_hash"] != snap_b["config_hash"],
            "count_deltas": count_deltas,
            "metric_deltas": metric_deltas,
            "tools_added": tools_added,
            "tools_removed": tools_removed,
            "tools_added_count": len(tools_added),
            "tools_removed_count": len(tools_removed),
        }

    # ========== Internal helpers ==========

    def _gather_metrics(self, conn) -> Dict[str, Any]:
        """Collect key system metrics from the database."""
        metrics = {}

        # Claim accuracy
        try:
            row = conn.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                       SUM(CASE WHEN actual_truth IS NOT NULL THEN 1 ELSE 0 END) as resolved
                FROM prediction_book
            """).fetchone()
            if row and row[2] and row[2] > 0:
                metrics["claim_accuracy"] = round(row[1] / row[2], 4)
            metrics["total_claims"] = row[0] if row else 0
            metrics["resolved_claims"] = row[2] if row else 0
        except Exception:
            pass

        # Confabulation rate
        try:
            row = conn.execute("SELECT COUNT(*) FROM confabulation_log").fetchone()
            metrics["confabulation_count"] = row[0] if row else 0
        except Exception:
            pass

        # Average confidence
        try:
            row = conn.execute(
                "SELECT AVG(CAST(json_extract(metadata, '$.confidence') AS REAL)) FROM atoms"
            ).fetchone()
            if row and row[0]:
                metrics["avg_atom_confidence"] = round(row[0], 4)
        except Exception:
            pass

        # Phi stats
        try:
            row = conn.execute("""
                SELECT AVG(phi_score), MIN(phi_score), MAX(phi_score), COUNT(*)
                FROM phi_memory_scores
            """).fetchone()
            if row and row[3] > 0:
                metrics["phi_avg"] = round(row[0], 4) if row[0] else 0
                metrics["phi_min"] = round(row[1], 4) if row[1] else 0
                metrics["phi_max"] = round(row[2], 4) if row[2] else 0
                metrics["phi_scored_count"] = row[3]
        except Exception:
            pass

        # Memory type distribution
        try:
            rows = conn.execute(
                "SELECT memory_type, COUNT(*) FROM typed_memories GROUP BY memory_type"
            ).fetchall()
            metrics["memory_types"] = {r[0]: r[1] for r in rows}
        except Exception:
            pass

        # Latest personality snapshot
        try:
            row = conn.execute("""
                SELECT avg_verbosity, confabulation_rate, verification_rate,
                       intellectual_honesty, overall_accuracy
                FROM personality_snapshot
                ORDER BY timestamp DESC LIMIT 1
            """).fetchone()
            if row:
                metrics["personality"] = {
                    "verbosity": row[0],
                    "confabulation_rate": row[1],
                    "verification_rate": row[2],
                    "intellectual_honesty": row[3],
                    "overall_accuracy": row[4],
                }
        except Exception:
            pass

        # Tool error rate (if tool_invocations exists)
        try:
            row = conn.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as errors
                FROM tool_invocations
                WHERE timestamp > ?
            """, (time.time() - 86400 * 7,)).fetchone()
            if row and row[0] > 0:
                metrics["tool_error_rate_7d"] = round(row[1] / row[0], 4)
                metrics["tool_calls_7d"] = row[0]
        except Exception:
            pass

        return metrics

    @staticmethod
    def _safe_count(conn, query: str) -> int:
        try:
            return conn.execute(query).fetchone()[0]
        except Exception:
            return 0

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        return {
            "id": row[0], "label": row[1], "timestamp": row[2],
            "tool_count": row[3], "tool_list": json.loads(row[4]),
            "atom_count": row[5], "typed_memory_count": row[6],
            "table_count": row[7], "config_hash": row[8],
            "metrics": json.loads(row[9]) if row[9] else {},
            "notes": row[10],
        }

    @staticmethod
    def _row_to_full_dict(row) -> Dict[str, Any]:
        return {
            "id": row[0], "label": row[1], "timestamp": row[2],
            "tool_count": row[3], "tool_list": row[4],
            "atom_count": row[5], "typed_memory_count": row[6],
            "table_count": row[7], "config_hash": row[8],
            "metrics": row[9], "notes": row[10],
        }
