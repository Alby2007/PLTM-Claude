"""
Tool Usage Analytics System

Logs every MCP tool invocation and provides analytics for:
- Usage frequency and patterns
- Error rates and performance
- Redundancy detection (unused, co-occurring, always-sequential tools)
- Consolidation proposals for tool inventory evolution
"""

import hashlib
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class ToolAnalytics:
    """
    Instruments MCP tool calls and analyzes usage patterns.
    
    Designed to be called from the call_tool wrapper in pltm_server.py.
    Uses synchronous sqlite3 (not aiosqlite) for minimal overhead in the
    hot path — logging a row should be <1ms.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent.parent.parent / "data" / "pltm_mcp.db"
        self._current_sequences: Dict[str, List[str]] = {}  # session_id → [tool_names]
        self._ensure_tables()

    def _ensure_tables(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_invocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                session_id TEXT DEFAULT '',
                timestamp REAL NOT NULL,
                duration_ms REAL DEFAULT 0,
                success INTEGER DEFAULT 1,
                error_type TEXT DEFAULT NULL,
                args_hash TEXT DEFAULT '',
                result_size INTEGER DEFAULT 0
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_inv_name ON tool_invocations(tool_name)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_inv_ts ON tool_invocations(timestamp)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_inv_session ON tool_invocations(session_id)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sequence TEXT NOT NULL,
                length INTEGER DEFAULT 0,
                timestamp REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    # ========== LOGGING (hot path — must be fast) ==========

    def log_invocation(
        self,
        tool_name: str,
        session_id: str = "",
        duration_ms: float = 0,
        success: bool = True,
        error_type: Optional[str] = None,
        args_hash: str = "",
        result_size: int = 0,
    ):
        """Log a single tool invocation. Called from call_tool wrapper."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                """INSERT INTO tool_invocations
                   (tool_name, session_id, timestamp, duration_ms, success, error_type, args_hash, result_size)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (tool_name, session_id, time.time(), duration_ms,
                 1 if success else 0, error_type, args_hash, result_size),
            )
            conn.commit()
            conn.close()

            # Track sequence
            if session_id:
                self._current_sequences.setdefault(session_id, []).append(tool_name)
        except Exception as e:
            logger.debug(f"ToolAnalytics.log_invocation failed: {e}")

    def end_session(self, session_id: str):
        """Flush the accumulated tool sequence for a session."""
        seq = self._current_sequences.pop(session_id, [])
        if not seq:
            return
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                "INSERT INTO tool_sequences (session_id, sequence, length, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, json.dumps(seq), len(seq), time.time()),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"ToolAnalytics.end_session failed: {e}")

    @staticmethod
    def hash_args(args: Dict[str, Any]) -> str:
        """Privacy-safe hash of tool arguments."""
        raw = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ========== ANALYTICS ==========

    def get_usage_stats(self, days: int = 30) -> Dict[str, Any]:
        """Per-tool usage statistics over the last N days."""
        cutoff = time.time() - (days * 86400)
        conn = sqlite3.connect(str(self.db_path))

        rows = conn.execute("""
            SELECT tool_name,
                   COUNT(*) as calls,
                   AVG(duration_ms) as avg_duration,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate,
                   MAX(timestamp) as last_used,
                   AVG(result_size) as avg_result_size
            FROM tool_invocations
            WHERE timestamp > ?
            GROUP BY tool_name
            ORDER BY calls DESC
        """, (cutoff,)).fetchall()

        total_calls = sum(r[1] for r in rows)
        tools = []
        for r in rows:
            tools.append({
                "tool": r[0],
                "calls": r[1],
                "avg_duration_ms": round(r[2], 1) if r[2] else 0,
                "success_rate": round(r[3], 3) if r[3] else 0,
                "last_used": r[4],
                "avg_result_size": round(r[5], 0) if r[5] else 0,
                "pct_of_total": round(r[1] / max(1, total_calls) * 100, 1),
            })

        conn.close()
        return {
            "period_days": days,
            "total_calls": total_calls,
            "unique_tools_used": len(tools),
            "tools": tools,
        }

    def get_redundancy_report(self, days: int = 30) -> Dict[str, Any]:
        """Identify unused, error-prone, and co-occurring tools."""
        cutoff = time.time() - (days * 86400)
        conn = sqlite3.connect(str(self.db_path))

        # Get all invocations in period
        rows = conn.execute("""
            SELECT tool_name, success, session_id
            FROM tool_invocations
            WHERE timestamp > ?
            ORDER BY session_id, timestamp
        """, (cutoff,)).fetchall()

        # Build per-tool stats
        tool_calls = Counter()
        tool_errors = Counter()
        session_tools: Dict[str, List[str]] = defaultdict(list)

        for r in rows:
            tool_calls[r[0]] += 1
            if not r[1]:
                tool_errors[r[0]] += 1
            if r[2]:
                session_tools[r[2]].append(r[0])

        # Co-occurrence: tools that appear together in >50% of sessions
        pair_counts = Counter()
        for sid, tools in session_tools.items():
            unique = set(tools)
            for t1 in unique:
                for t2 in unique:
                    if t1 < t2:
                        pair_counts[(t1, t2)] += 1

        num_sessions = len(session_tools)
        co_occurring = []
        for (t1, t2), count in pair_counts.most_common(20):
            if num_sessions > 0 and count / num_sessions > 0.5:
                co_occurring.append({
                    "tools": [t1, t2],
                    "co_occurrence_rate": round(count / num_sessions, 2),
                    "sessions": count,
                })

        # High error rate tools
        error_prone = []
        for tool, calls in tool_calls.items():
            errors = tool_errors.get(tool, 0)
            if calls >= 3 and errors / calls > 0.2:
                error_prone.append({
                    "tool": tool,
                    "calls": calls,
                    "errors": errors,
                    "error_rate": round(errors / calls, 2),
                })
        error_prone.sort(key=lambda x: x["error_rate"], reverse=True)

        # Never-called tools (need tool list from caller — return tools with 0 calls)
        used_tools = set(tool_calls.keys())

        conn.close()
        return {
            "period_days": days,
            "tools_used": len(used_tools),
            "co_occurring_pairs": co_occurring[:10],
            "error_prone": error_prone[:10],
            "used_tool_names": sorted(used_tools),
        }

    def get_sequence_patterns(self, min_support: int = 3) -> Dict[str, Any]:
        """Find frequent tool call subsequences."""
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute(
            "SELECT sequence FROM tool_sequences ORDER BY timestamp DESC LIMIT 200"
        ).fetchall()
        conn.close()

        if not rows:
            return {"patterns": [], "sequences_analyzed": 0}

        # Extract bigrams and trigrams
        bigrams = Counter()
        trigrams = Counter()

        for (seq_json,) in rows:
            try:
                seq = json.loads(seq_json)
            except Exception:
                continue
            for i in range(len(seq) - 1):
                bigrams[(seq[i], seq[i + 1])] += 1
            for i in range(len(seq) - 2):
                trigrams[(seq[i], seq[i + 1], seq[i + 2])] += 1

        patterns = []
        for gram, count in bigrams.most_common(20):
            if count >= min_support:
                patterns.append({
                    "sequence": list(gram),
                    "occurrences": count,
                    "type": "bigram",
                })
        for gram, count in trigrams.most_common(10):
            if count >= min_support:
                patterns.append({
                    "sequence": list(gram),
                    "occurrences": count,
                    "type": "trigram",
                })

        patterns.sort(key=lambda x: x["occurrences"], reverse=True)

        return {
            "sequences_analyzed": len(rows),
            "patterns": patterns[:20],
        }

    def propose_consolidation(self, all_tool_names: Optional[List[str]] = None, days: int = 30) -> Dict[str, Any]:
        """
        Generate actionable proposals for tool inventory evolution.
        
        Proposals:
        - Deprecate: tools with 0 calls in the period
        - Combine: tools always called sequentially
        - Investigate: tools with high error rates
        - Optimize: slowest tools
        """
        stats = self.get_usage_stats(days)
        redundancy = self.get_redundancy_report(days)
        patterns = self.get_sequence_patterns()

        proposals = []

        # 1. Deprecation candidates: never used
        used_names = set(t["tool"] for t in stats["tools"])
        if all_tool_names:
            never_used = sorted(set(all_tool_names) - used_names)
            if never_used:
                proposals.append({
                    "type": "deprecate",
                    "severity": "low",
                    "description": f"{len(never_used)} tools never called in {days}d",
                    "tools": never_used[:20],
                    "action": "Consider removing or documenting why these exist",
                })

        # 2. Combine candidates: sequential patterns
        for pat in patterns.get("patterns", []):
            if pat["occurrences"] >= 5 and pat["type"] == "bigram":
                proposals.append({
                    "type": "combine",
                    "severity": "medium",
                    "description": f"'{pat['sequence'][0]}' → '{pat['sequence'][1]}' called sequentially {pat['occurrences']} times",
                    "tools": pat["sequence"],
                    "action": "Consider combining into a single tool or adding a convenience wrapper",
                })

        # 3. Error-prone tools
        for ep in redundancy.get("error_prone", []):
            proposals.append({
                "type": "investigate",
                "severity": "high",
                "description": f"'{ep['tool']}' has {ep['error_rate']*100:.0f}% error rate ({ep['errors']}/{ep['calls']})",
                "tools": [ep["tool"]],
                "action": "Fix error handling or improve input validation",
            })

        # 4. Slowest tools
        slow_tools = sorted(stats["tools"], key=lambda t: t["avg_duration_ms"], reverse=True)
        for t in slow_tools[:3]:
            if t["avg_duration_ms"] > 5000 and t["calls"] >= 3:
                proposals.append({
                    "type": "optimize",
                    "severity": "medium",
                    "description": f"'{t['tool']}' averages {t['avg_duration_ms']:.0f}ms per call",
                    "tools": [t["tool"]],
                    "action": "Profile and optimize, or add caching",
                })

        # 5. Highly co-occurring tools
        for co in redundancy.get("co_occurring_pairs", []):
            if co["co_occurrence_rate"] > 0.8:
                proposals.append({
                    "type": "combine",
                    "severity": "low",
                    "description": f"'{co['tools'][0]}' and '{co['tools'][1]}' co-occur in {co['co_occurrence_rate']*100:.0f}% of sessions",
                    "tools": co["tools"],
                    "action": "Consider merging or creating a combined convenience tool",
                })

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        proposals.sort(key=lambda p: severity_order.get(p["severity"], 3))

        return {
            "period_days": days,
            "total_proposals": len(proposals),
            "proposals": proposals,
            "summary": {
                "deprecate": sum(1 for p in proposals if p["type"] == "deprecate"),
                "combine": sum(1 for p in proposals if p["type"] == "combine"),
                "investigate": sum(1 for p in proposals if p["type"] == "investigate"),
                "optimize": sum(1 for p in proposals if p["type"] == "optimize"),
            },
        }
