"""
Session Continuity System

Enables seamless cross-conversation resumption via three components:
1. WorkingMemoryCompressor — snapshots the "active working set" at session end
2. TrajectoryEncoder — encodes conversation direction from tool sequences
3. HandoffProtocol — orchestrates a single token-budgeted handoff block

One tool call at session start → Claude resumes as if the conversation never ended.
"""

import json
import math
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger


# ── Token estimation ──────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)


# ── Tool → Topic mapping ─────────────────────────────────────────────────────

TOOL_TOPICS = {
    # Memory formation
    "store_episodic": "memory_formation",
    "store_semantic": "knowledge_capture",
    "store_belief": "belief_formation",
    "store_procedural": "rule_learning",
    "store_memory_atom": "memory_formation",
    # Retrieval
    "recall_memories": "retrieval",
    "search_memories": "retrieval",
    "semantic_search": "retrieval",
    "what_do_i_know_about": "retrieval",
    "get_relevant_context": "retrieval",
    "contextual_retrieve": "retrieval",
    "attention_retrieve": "retrieval",
    "attention_multihead": "retrieval",
    "mmr_retrieve": "retrieval",
    "find_similar_memories": "retrieval",
    # Memory management
    "phi_score_memories": "memory_optimization",
    "phi_prune": "memory_optimization",
    "phi_consolidate": "memory_consolidation",
    "phi_build_context": "context_building",
    "apply_memory_decay": "memory_maintenance",
    "consolidate_memories": "memory_consolidation",
    "consolidate_typed_memories": "memory_consolidation",
    "auto_prune_memories": "memory_maintenance",
    "correct_memory": "memory_correction",
    "forget_memory": "memory_correction",
    # Epistemic
    "log_claim": "reasoning",
    "resolve_claim": "verification",
    "check_before_claiming": "epistemic_hygiene",
    "calibrate_confidence_live": "epistemic_hygiene",
    "belief_auto_check": "epistemic_hygiene",
    # Goals
    "create_goal": "planning",
    "update_goal": "execution",
    "get_goals": "planning",
    # Self-modeling
    "learn_communication_style": "self_reflection",
    "track_curiosity_spike": "self_reflection",
    "evolve_self_model": "self_evolution",
    "self_profile": "self_reflection",
    "bootstrap_self_model": "self_evolution",
    # Learning
    "learn_from_url": "learning",
    "learn_from_paper": "learning",
    "learn_from_code": "learning",
    "learn_from_conversation": "learning",
    "ingest_arxiv": "learning",
    "cross_domain_synthesis": "synthesis",
    # Session
    "auto_init_session": "session_management",
    "end_session": "session_management",
    "session_handoff": "session_management",
    # Personality
    "query_personality": "personality",
    "detect_mood": "personality",
    "get_claude_personality": "personality",
    "evolve_claude_personality": "personality",
    # Analysis
    "tool_usage_stats": "meta_analysis",
    "snapshot_architecture": "meta_analysis",
    "compare_architectures": "meta_analysis",
    # Data
    "query_pltm_sql": "data_access",
}


def _topic_for_tool(tool_name: str) -> str:
    """Map a tool name to a high-level topic label."""
    return TOOL_TOPICS.get(tool_name, "general")


# ── Table creation ────────────────────────────────────────────────────────────

def _ensure_tables(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS working_memory_snapshots (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            active_memories TEXT DEFAULT '[]',
            active_tools TEXT DEFAULT '[]',
            open_questions TEXT DEFAULT '[]',
            key_facts TEXT DEFAULT '[]',
            token_count INTEGER DEFAULT 0,
            compressed_block TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_wm_session
            ON working_memory_snapshots(session_id);
        CREATE INDEX IF NOT EXISTS idx_wm_ts
            ON working_memory_snapshots(timestamp DESC);

        CREATE TABLE IF NOT EXISTS trajectory_snapshots (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            topic_flow TEXT DEFAULT '[]',
            decision_points TEXT DEFAULT '[]',
            open_threads TEXT DEFAULT '[]',
            reasoning_chains TEXT DEFAULT '[]',
            momentum TEXT DEFAULT '{}',
            compressed_block TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_traj_session
            ON trajectory_snapshots(session_id);
        CREATE INDEX IF NOT EXISTS idx_traj_ts
            ON trajectory_snapshots(timestamp DESC);
    """)
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Component 1: Working Memory Compressor
# ═══════════════════════════════════════════════════════════════════════════════

class WorkingMemoryCompressor:
    """
    Captures the 'active working set' at session end — memories, tools, state
    that were *in use*, not everything in long-term storage.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        _ensure_tables(db_path)

    async def capture(
        self,
        session_id: str,
        user_id: str,
        token_budget: int = 500,
    ) -> Dict[str, Any]:
        """
        Capture current working memory state.

        1. Get memories accessed in last 4h
        2. Get tools called this session
        3. Get active goals + blockers
        4. Get recent confabulation patterns
        5. Get conversation_state items updated recently
        6. Rank by recency × access_count
        7. Greedy-pack into token_budget
        8. Render compressed text block
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        now = time.time()
        cutoff_4h = now - (4 * 3600)

        # 1. Recently accessed typed memories
        active_memories = []
        try:
            rows = conn.execute(
                """SELECT id, memory_type, content, confidence, strength,
                          last_accessed, access_count, tags
                   FROM typed_memories
                   WHERE user_id = ? AND last_accessed > ?
                   ORDER BY last_accessed DESC LIMIT 20""",
                (user_id, cutoff_4h),
            ).fetchall()
            for r in rows:
                active_memories.append({
                    "id": r["id"],
                    "type": r["memory_type"],
                    "content": r["content"][:120],
                    "confidence": round(r["confidence"], 2),
                    "strength": round(r["strength"], 2),
                    "access_count": r["access_count"],
                })
        except Exception:
            pass  # Table may not exist yet

        # 2. Tools called this session
        active_tools = []
        try:
            rows = conn.execute(
                """SELECT tool_name, COUNT(*) as cnt
                   FROM tool_invocations
                   WHERE session_id = ?
                   GROUP BY tool_name
                   ORDER BY cnt DESC""",
                (session_id,),
            ).fetchall()
            active_tools = [
                {"tool": r["tool_name"], "calls": r["cnt"]} for r in rows
            ]
        except Exception:
            pass

        # 3. Active goals
        goals = []
        try:
            rows = conn.execute(
                """SELECT goal_id, title, status, priority, progress, blockers
                   FROM goals
                   WHERE status IN ('active', 'in_progress')
                   ORDER BY priority LIMIT 5"""
            ).fetchall()
            for r in rows:
                blockers = json.loads(r["blockers"] or "[]")
                goals.append({
                    "title": r["title"][:80],
                    "progress": round(r["progress"] * 100),
                    "priority": r["priority"],
                    "blockers": blockers[:2],
                })
        except Exception:
            pass

        # 4. Recent confabulation patterns
        confab_patterns = []
        try:
            rows = conn.execute(
                """SELECT failure_mode, prevention_strategy, claim_text
                   FROM confabulation_log
                   ORDER BY timestamp DESC LIMIT 3"""
            ).fetchall()
            confab_patterns = [
                {"mode": r["failure_mode"], "prevention": (r["prevention_strategy"] or "")[:60]}
                for r in rows
            ]
        except Exception:
            pass

        # 5. Recent conversation_state
        state_items = []
        try:
            rows = conn.execute(
                """SELECT key, value, category
                   FROM conversation_state
                   WHERE updated_at > ?
                   ORDER BY updated_at DESC LIMIT 10""",
                (cutoff_4h,),
            ).fetchall()
            for r in rows:
                val = r["value"]
                if len(val) > 80:
                    val = val[:77] + "..."
                state_items.append({"key": r["key"], "value": val})
        except Exception:
            pass

        # 6. Open questions (unresolved claims)
        open_questions = []
        try:
            rows = conn.execute(
                """SELECT claim, domain, felt_confidence
                   FROM prediction_book
                   WHERE actual_truth IS NULL
                   ORDER BY timestamp DESC LIMIT 5"""
            ).fetchall()
            open_questions = [
                {"claim": r["claim"][:80], "domain": r["domain"]}
                for r in rows
            ]
        except Exception:
            pass

        conn.close()

        # 7. Render compressed block
        block = self._render_block(
            active_memories, active_tools, goals,
            confab_patterns, state_items, open_questions,
            token_budget,
        )
        token_count = _estimate_tokens(block)

        # 8. Store snapshot
        snap_id = uuid4().hex[:12]
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT INTO working_memory_snapshots
               (id, session_id, timestamp, active_memories, active_tools,
                open_questions, key_facts, token_count, compressed_block)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (snap_id, session_id, now,
             json.dumps(active_memories), json.dumps(active_tools),
             json.dumps(open_questions), json.dumps(state_items),
             token_count, block),
        )
        conn.commit()
        conn.close()

        return {
            "ok": True,
            "snapshot_id": snap_id,
            "session_id": session_id,
            "active_memories": len(active_memories),
            "active_tools": len(active_tools),
            "goals": len(goals),
            "open_questions": len(open_questions),
            "token_count": token_count,
            "compressed_block": block,
        }

    def get_latest(self, user_id: str = "") -> Optional[Dict]:
        """Get most recent working memory snapshot."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """SELECT * FROM working_memory_snapshots
               ORDER BY timestamp DESC LIMIT 1"""
        ).fetchone()
        conn.close()
        if not row:
            return None
        return {
            "snapshot_id": row["id"],
            "session_id": row["session_id"],
            "timestamp": row["timestamp"],
            "age_hours": round((time.time() - row["timestamp"]) / 3600, 1),
            "active_memories": json.loads(row["active_memories"]),
            "active_tools": json.loads(row["active_tools"]),
            "open_questions": json.loads(row["open_questions"]),
            "key_facts": json.loads(row["key_facts"]),
            "token_count": row["token_count"],
            "compressed_block": row["compressed_block"],
        }

    def _render_block(
        self,
        memories: List[Dict],
        tools: List[Dict],
        goals: List[Dict],
        confabs: List[Dict],
        state: List[Dict],
        open_qs: List[Dict],
        budget: int,
    ) -> str:
        """Render items into a compact text block within token budget."""
        lines = []

        # Goals (highest priority)
        if goals:
            goal_parts = []
            for g in goals[:3]:
                part = f"{g['title']} ({g['progress']}%)"
                if g.get("blockers"):
                    part += f" [blocked: {g['blockers'][0]}]"
                goal_parts.append(part)
            lines.append("DOING: " + " | ".join(goal_parts))

        # Tools used
        if tools:
            tool_parts = [f"{t['tool']}×{t['calls']}" for t in tools[:8]]
            lines.append("TOOLS: " + ", ".join(tool_parts))

        # Key facts from memories
        if memories:
            fact_parts = [m["content"][:60] for m in memories[:5]]
            lines.append("FACTS: " + " | ".join(fact_parts))

        # Confabulation warnings
        if confabs:
            avoid_parts = [f"{c['mode']}: {c['prevention']}" for c in confabs[:2]]
            lines.append("AVOID: " + " | ".join(avoid_parts))

        # Open questions
        if open_qs:
            q_parts = [q["claim"][:50] for q in open_qs[:3]]
            lines.append("OPEN: " + " | ".join(q_parts))

        # State
        if state:
            s_parts = [f"{s['key']}={s['value'][:30]}" for s in state[:5]]
            lines.append("STATE: " + ", ".join(s_parts))

        block = "\n".join(lines)

        # Trim to budget
        while _estimate_tokens(block) > budget and lines:
            lines.pop()
            block = "\n".join(lines)

        return block


# ═══════════════════════════════════════════════════════════════════════════════
# Component 2: Trajectory Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class TrajectoryEncoder:
    """
    Encodes the *shape* of the conversation — topic flow, decisions,
    open threads, and momentum — from tool call sequences.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        _ensure_tables(db_path)

    def encode(self, session_id: str, user_id: str = "claude") -> Dict[str, Any]:
        """
        Encode the conversation trajectory for a session.

        1. Get tool sequence → infer topic flow
        2. Get goal_log actions → decision points
        3. Get unresolved claims → open threads
        4. Infer momentum from last few tool calls
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        now = time.time()

        # 1. Tool sequence → topic flow
        tool_rows = conn.execute(
            """SELECT tool_name FROM tool_invocations
               WHERE session_id = ?
               ORDER BY timestamp""",
            (session_id,),
        ).fetchall()
        tool_sequence = [r["tool_name"] for r in tool_rows]
        topic_flow = self._infer_topics(tool_sequence)

        # 2. Goal log actions this session → decision points
        decision_points = []
        try:
            # Get session start time
            session_start = conn.execute(
                """SELECT MIN(timestamp) FROM tool_invocations
                   WHERE session_id = ?""",
                (session_id,),
            ).fetchone()[0] or (now - 3600)

            goal_rows = conn.execute(
                """SELECT g.title, gl.action, gl.details
                   FROM goal_log gl
                   JOIN goals g ON gl.goal_id = g.goal_id
                   WHERE gl.timestamp > ?
                   ORDER BY gl.timestamp""",
                (session_start,),
            ).fetchall()
            for r in goal_rows:
                decision_points.append({
                    "goal": r["title"][:60],
                    "action": r["action"],
                    "details": (r["details"] or "")[:80],
                })
        except Exception:
            pass

        # 3. Unresolved claims → open threads
        open_threads = []
        try:
            rows = conn.execute(
                """SELECT claim, domain, felt_confidence
                   FROM prediction_book
                   WHERE actual_truth IS NULL
                   ORDER BY timestamp DESC LIMIT 5"""
            ).fetchall()
            for r in rows:
                open_threads.append({
                    "thread": r["claim"][:80],
                    "domain": r["domain"],
                    "priority": "high" if r["felt_confidence"] > 0.7 else "medium",
                })
        except Exception:
            pass

        # 4. Momentum — last few tools indicate direction
        momentum = self._infer_momentum(tool_sequence)

        # Render compressed block
        block = self._render_block(topic_flow, decision_points, open_threads, momentum)

        conn.close()

        # Store snapshot
        snap_id = uuid4().hex[:12]
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT INTO trajectory_snapshots
               (id, session_id, timestamp, topic_flow, decision_points,
                open_threads, reasoning_chains, momentum, compressed_block)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (snap_id, session_id, now,
             json.dumps(topic_flow), json.dumps(decision_points),
             json.dumps(open_threads), "[]",
             json.dumps(momentum), block),
        )
        conn.commit()
        conn.close()

        return {
            "ok": True,
            "snapshot_id": snap_id,
            "session_id": session_id,
            "topic_flow": topic_flow,
            "decision_points": decision_points,
            "open_threads": open_threads,
            "momentum": momentum,
            "compressed_block": block,
        }

    def get_latest(self) -> Optional[Dict]:
        """Get most recent trajectory snapshot."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """SELECT * FROM trajectory_snapshots
               ORDER BY timestamp DESC LIMIT 1"""
        ).fetchone()
        conn.close()
        if not row:
            return None
        return {
            "snapshot_id": row["id"],
            "session_id": row["session_id"],
            "timestamp": row["timestamp"],
            "age_hours": round((time.time() - row["timestamp"]) / 3600, 1),
            "topic_flow": json.loads(row["topic_flow"]),
            "decision_points": json.loads(row["decision_points"]),
            "open_threads": json.loads(row["open_threads"]),
            "momentum": json.loads(row["momentum"]),
            "compressed_block": row["compressed_block"],
        }

    def _infer_topics(self, tool_sequence: List[str]) -> List[str]:
        """Deduplicate consecutive topics from tool sequence."""
        if not tool_sequence:
            return []
        topics = []
        prev = None
        for tool in tool_sequence:
            topic = _topic_for_tool(tool)
            if topic != prev:
                topics.append(topic)
                prev = topic
        return topics

    def _infer_momentum(self, tool_sequence: List[str]) -> Dict[str, Any]:
        """Infer conversation direction from the last N tool calls."""
        if not tool_sequence:
            return {"direction": "unknown", "confidence": 0.0, "next_likely_action": "start fresh"}

        last_n = tool_sequence[-5:]
        recent_topics = [_topic_for_tool(t) for t in last_n]
        topic_counts = Counter(recent_topics)
        dominant = topic_counts.most_common(1)[0]

        # Suggest next action based on dominant recent topic
        next_actions = {
            "memory_formation": "Likely storing more knowledge",
            "retrieval": "Likely needs more information",
            "memory_optimization": "Continuing memory management",
            "memory_consolidation": "Finishing consolidation pass",
            "epistemic_hygiene": "Verifying claims",
            "reasoning": "Building reasoning chain",
            "verification": "Checking previous claims",
            "planning": "Defining next steps",
            "execution": "Working on active goals",
            "self_reflection": "Analyzing own patterns",
            "learning": "Ingesting new knowledge",
            "synthesis": "Connecting cross-domain ideas",
            "context_building": "Preparing context for conversation",
            "personality": "Adjusting interaction style",
            "meta_analysis": "Reviewing system performance",
        }

        return {
            "direction": dominant[0],
            "confidence": round(dominant[1] / len(last_n), 2),
            "next_likely_action": next_actions.get(dominant[0], "Continue current work"),
            "last_tool": tool_sequence[-1] if tool_sequence else None,
        }

    def _render_block(
        self,
        topic_flow: List[str],
        decisions: List[Dict],
        open_threads: List[Dict],
        momentum: Dict,
    ) -> str:
        lines = []

        if topic_flow:
            # Show last 6 topic transitions
            flow_str = " → ".join(topic_flow[-6:])
            lines.append(f"FLOW: {flow_str}")

        if decisions:
            for d in decisions[:3]:
                lines.append(f"DECIDED: {d['action']} on '{d['goal']}' — {d['details']}")

        if open_threads:
            thread_parts = [t["thread"][:40] for t in open_threads[:3]]
            lines.append("OPEN: " + " | ".join(thread_parts))

        if momentum.get("direction") and momentum["direction"] != "unknown":
            lines.append(f"MOMENTUM: {momentum['next_likely_action']} (direction: {momentum['direction']})")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Component 3: Handoff Protocol
# ═══════════════════════════════════════════════════════════════════════════════

class HandoffProtocol:
    """
    Orchestrates all continuity components into a single token-budgeted
    handoff block for session resumption.

    Replaces calling auto_init_session + get_session_bridge + phi_build_context.
    """

    # Default budget allocation (proportional)
    BUDGET_RATIOS = {
        "identity": 0.10,
        "working_memory": 0.27,
        "trajectory": 0.13,
        "goals": 0.13,
        "warnings": 0.07,
        "phi_context": 0.30,
    }

    def __init__(
        self,
        db_path: Path,
        working_memory: WorkingMemoryCompressor,
        trajectory_encoder: TrajectoryEncoder,
    ):
        self.db_path = db_path
        self.wm = working_memory
        self.te = trajectory_encoder

    async def generate_handoff(
        self,
        user_id: str = "claude",
        token_budget: int = 1500,
        include_sections: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete handoff block for session resumption.

        Budget allocation adapts: if a section has no data, its tokens
        are redistributed to other sections.
        """
        sections = include_sections or list(self.BUDGET_RATIOS.keys())
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        # Gather raw data for each section
        raw = {}

        if "identity" in sections:
            raw["identity"] = self._gather_identity(conn, user_id)

        if "working_memory" in sections:
            wm_snap = self.wm.get_latest(user_id)
            raw["working_memory"] = wm_snap["compressed_block"] if wm_snap else ""

        if "trajectory" in sections:
            traj_snap = self.te.get_latest()
            raw["trajectory"] = traj_snap["compressed_block"] if traj_snap else ""

        if "goals" in sections:
            raw["goals"] = self._gather_goals(conn)

        if "warnings" in sections:
            raw["warnings"] = self._gather_warnings(conn, user_id)

        if "phi_context" in sections:
            raw["phi_context"] = self._gather_phi_context(conn, user_id)

        conn.close()

        # Redistribute budget from empty sections
        active_sections = {k: v for k, v in raw.items() if v}
        empty_sections = {k: v for k, v in raw.items() if not v}
        if empty_sections and active_sections:
            freed = sum(self.BUDGET_RATIOS.get(k, 0) for k in empty_sections)
            bonus = freed / len(active_sections)
        else:
            bonus = 0

        # Render each section within its budget
        rendered = {}
        for section in sections:
            if section not in active_sections:
                continue
            section_budget = int(token_budget * (self.BUDGET_RATIOS.get(section, 0.1) + bonus))
            text = raw[section]
            # Trim to budget
            while _estimate_tokens(text) > section_budget and len(text) > 20:
                text = text[:int(len(text) * 0.85)]
                # Trim to last newline for cleanliness
                nl = text.rfind("\n")
                if nl > len(text) // 2:
                    text = text[:nl]
            rendered[section] = text

        # Assemble final handoff block
        block_parts = []
        if "identity" in rendered:
            block_parts.append(f"<identity>{rendered['identity']}</identity>")
        if "working_memory" in rendered:
            block_parts.append(f"<working_memory>\n{rendered['working_memory']}\n</working_memory>")
        if "trajectory" in rendered:
            block_parts.append(f"<trajectory>\n{rendered['trajectory']}\n</trajectory>")
        if "goals" in rendered:
            block_parts.append(f"<goals>\n{rendered['goals']}\n</goals>")
        if "warnings" in rendered:
            block_parts.append(f"<warnings>\n{rendered['warnings']}\n</warnings>")
        if "phi_context" in rendered:
            block_parts.append(f"<phi_context>\n{rendered['phi_context']}\n</phi_context>")

        handoff_block = f"<session_handoff budget=\"{token_budget}\">\n" + "\n".join(block_parts) + "\n</session_handoff>"
        total_tokens = _estimate_tokens(handoff_block)

        return {
            "ok": True,
            "handoff_block": handoff_block,
            "token_count": total_tokens,
            "token_budget": token_budget,
            "sections_included": list(rendered.keys()),
            "sections_empty": list(empty_sections.keys()) if empty_sections else [],
        }

    async def end_and_capture(
        self,
        session_id: str,
        user_id: str = "claude",
        summary: str = "",
    ) -> Dict[str, Any]:
        """
        End-of-session capture: snapshot working memory + encode trajectory.
        Should be called before/alongside end_session.
        """
        wm_result = await self.wm.capture(session_id, user_id)
        traj_result = self.te.encode(session_id, user_id)

        return {
            "ok": True,
            "session_id": session_id,
            "working_memory": {
                "snapshot_id": wm_result["snapshot_id"],
                "active_memories": wm_result["active_memories"],
                "active_tools": wm_result["active_tools"],
                "token_count": wm_result["token_count"],
            },
            "trajectory": {
                "snapshot_id": traj_result["snapshot_id"],
                "topic_flow": traj_result["topic_flow"],
                "momentum": traj_result["momentum"],
            },
            "summary": summary,
        }

    # ── Section gatherers ─────────────────────────────────────────────────

    def _gather_identity(self, conn: sqlite3.Connection, user_id: str) -> str:
        """One-line personality summary from latest snapshot."""
        try:
            row = conn.execute(
                """SELECT avg_verbosity, avg_hedging, dominant_tone,
                          confabulation_rate, verification_rate,
                          intellectual_honesty, overall_accuracy,
                          avg_engagement, pushback_rate
                   FROM personality_snapshot
                   ORDER BY timestamp DESC LIMIT 1"""
            ).fetchone()
            if not row:
                return ""

            parts = []
            # Communication style
            v = row["avg_verbosity"] or 0
            h = row["avg_hedging"] or 0
            tone = row["dominant_tone"] or "neutral"
            if v > 0.6:
                parts.append("verbose")
            elif v < 0.3:
                parts.append("concise")
            if h > 0.3:
                parts.append("hedging")
            elif h < 0.1:
                parts.append("direct")
            parts.append(f"{tone} tone")

            # Reasoning
            honesty = row["intellectual_honesty"] or 0
            accuracy = row["overall_accuracy"] or 0
            confab = row["confabulation_rate"] or 0
            parts.append(f"honesty:{honesty:.2f}")
            parts.append(f"accuracy:{accuracy:.0%}")
            if confab > 0.2:
                parts.append(f"confab:{confab:.0%}⚠")

            # Values
            pushback = row["pushback_rate"] or 0
            if pushback > 0.5:
                parts.append("strong boundaries")

            return " | ".join(parts)
        except Exception:
            return ""

    def _gather_goals(self, conn: sqlite3.Connection) -> str:
        """Serialize active goals compactly."""
        try:
            rows = conn.execute(
                """SELECT title, priority, progress, blockers
                   FROM goals
                   WHERE status IN ('active', 'in_progress')
                   ORDER BY priority LIMIT 5"""
            ).fetchall()
            if not rows:
                return ""
            lines = []
            for i, r in enumerate(rows, 1):
                prio = r["priority"].upper()[:3] if r["priority"] else "MED"
                progress = round((r["progress"] or 0) * 100)
                blockers = json.loads(r["blockers"] or "[]")
                line = f"{i}. [{prio}] {r['title'][:60]} ({progress}%)"
                if blockers:
                    line += f" — blocked: {blockers[0][:40]}"
                lines.append(line)
            return "\n".join(lines)
        except Exception:
            return ""

    def _gather_warnings(self, conn: sqlite3.Connection, user_id: str) -> str:
        """Calibration warnings + confabulation patterns."""
        warnings = []
        try:
            # Overconfident domains
            rows = conn.execute(
                """SELECT domain, overconfidence_ratio, accuracy_ratio, total_claims
                   FROM calibration_cache
                   WHERE overconfidence_ratio > 0.3 AND total_claims >= 3
                   ORDER BY overconfidence_ratio DESC LIMIT 3"""
            ).fetchall()
            for r in rows:
                warnings.append(
                    f"⚠ Overconfident in '{r['domain']}' ({r['overconfidence_ratio']:.0%} overconf, {r['accuracy_ratio']:.0%} accuracy)"
                )

            # Recent confabulation patterns
            rows = conn.execute(
                """SELECT failure_mode, prevention_strategy
                   FROM confabulation_log
                   ORDER BY timestamp DESC LIMIT 2"""
            ).fetchall()
            for r in rows:
                prevention = (r["prevention_strategy"] or "verify first")[:60]
                warnings.append(f"⚠ Confab pattern: {r['failure_mode']} — {prevention}")
        except Exception:
            pass

        return "\n".join(warnings)

    def _gather_phi_context(self, conn: sqlite3.Connection, user_id: str) -> str:
        """Get high-value memories for context injection."""
        try:
            rows = conn.execute(
                """SELECT memory_type, content, confidence, strength, tags
                   FROM typed_memories
                   WHERE user_id = ?
                   ORDER BY strength * confidence DESC
                   LIMIT 8""",
                (user_id,),
            ).fetchall()
            if not rows:
                return ""
            lines = []
            type_labels = {
                "semantic": "FACT",
                "belief": "BELIEF",
                "procedural": "RULE",
                "episodic": "EVENT",
            }
            for r in rows:
                label = type_labels.get(r["memory_type"], "MEM")
                conf = r["confidence"] or 0
                content = r["content"][:100]
                if label == "BELIEF":
                    lines.append(f"[{label} conf={conf:.0%}] {content}")
                else:
                    lines.append(f"[{label}] {content}")
            return "\n".join(lines)
        except Exception:
            return ""
