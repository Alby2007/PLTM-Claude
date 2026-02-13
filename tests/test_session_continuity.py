"""
Tests for the Session Continuity System:
- WorkingMemoryCompressor
- TrajectoryEncoder
- HandoffProtocol
"""

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

import pytest

from src.memory.session_continuity import (
    WorkingMemoryCompressor,
    TrajectoryEncoder,
    HandoffProtocol,
    _estimate_tokens,
    _topic_for_tool,
    TOOL_TOPICS,
)


@pytest.fixture
def db_path(tmp_path):
    """Create a temporary DB with required tables."""
    db = tmp_path / "test_continuity.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS typed_memories (
            id TEXT PRIMARY KEY,
            memory_type TEXT NOT NULL,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            context TEXT DEFAULT '',
            source TEXT DEFAULT '',
            strength REAL DEFAULT 1.0,
            created_at REAL NOT NULL,
            last_accessed REAL NOT NULL,
            access_count INTEGER DEFAULT 0,
            confidence REAL DEFAULT 0.5,
            evidence_for TEXT DEFAULT '[]',
            evidence_against TEXT DEFAULT '[]',
            episode_timestamp REAL DEFAULT 0,
            participants TEXT DEFAULT '[]',
            emotional_valence REAL DEFAULT 0.0,
            trigger TEXT DEFAULT '',
            action TEXT DEFAULT '',
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            tags TEXT DEFAULT '[]',
            consolidated_from TEXT DEFAULT '[]'
        );

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
        );

        CREATE TABLE IF NOT EXISTS goals (
            goal_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            category TEXT DEFAULT 'general',
            status TEXT DEFAULT 'active',
            priority TEXT DEFAULT 'medium',
            progress REAL DEFAULT 0.0,
            success_criteria TEXT DEFAULT '[]',
            plan TEXT DEFAULT '[]',
            blockers TEXT DEFAULT '[]',
            parent_goal_id TEXT,
            deadline TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            completed_at REAL,
            metadata TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS goal_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal_id TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT DEFAULT '',
            timestamp REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS confabulation_log (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            claim_id TEXT,
            claim_text TEXT,
            domain TEXT,
            failure_mode TEXT,
            contributing_factors TEXT,
            prevention_strategy TEXT,
            felt_confidence REAL,
            context TEXT,
            metadata TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS prediction_book (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            claim TEXT NOT NULL,
            domain TEXT DEFAULT 'general',
            felt_confidence REAL DEFAULT 0.5,
            epistemic_status TEXT DEFAULT 'TRAINING_DATA',
            has_verified INTEGER DEFAULT 0,
            verified_at REAL,
            actual_truth INTEGER,
            was_correct INTEGER,
            calibration_error REAL,
            correction_source TEXT,
            correction_detail TEXT,
            metadata TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS conversation_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            access_count INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS personality_snapshot (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            timestamp REAL,
            avg_verbosity REAL,
            avg_jargon REAL,
            avg_hedging REAL,
            dominant_tone TEXT,
            confabulation_rate REAL,
            verification_rate REAL,
            error_catch_rate REAL,
            intellectual_honesty REAL,
            top_interests TEXT,
            avg_engagement REAL,
            pushback_rate REAL,
            avg_value_intensity REAL,
            prediction_accuracy REAL,
            overall_accuracy REAL,
            metadata TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS calibration_cache (
            domain TEXT PRIMARY KEY,
            total_claims INTEGER DEFAULT 0,
            verified_claims INTEGER DEFAULT 0,
            correct_claims INTEGER DEFAULT 0,
            accuracy_ratio REAL DEFAULT 0.5,
            avg_felt_confidence REAL DEFAULT 0.5,
            avg_calibration_error REAL DEFAULT 0.0,
            overconfidence_ratio REAL DEFAULT 0.0,
            last_updated REAL
        );
    """)
    conn.commit()
    conn.close()
    return db


def _seed_data(db_path, session_id="test_session_123"):
    """Seed the DB with test data."""
    conn = sqlite3.connect(str(db_path))
    now = time.time()

    # Typed memories (recently accessed)
    for i in range(5):
        conn.execute(
            """INSERT INTO typed_memories
               (id, memory_type, user_id, content, strength, created_at,
                last_accessed, access_count, confidence, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (f"mem_{i}", "semantic", "claude", f"Test fact number {i}",
             0.8, now - 3600, now - (i * 60), 3 + i, 0.9, "[]"),
        )

    # Tool invocations
    tools = ["recall_memories", "store_semantic", "store_semantic",
             "phi_score_memories", "log_claim", "recall_memories",
             "store_belief", "check_before_claiming"]
    for j, tool in enumerate(tools):
        conn.execute(
            """INSERT INTO tool_invocations
               (tool_name, session_id, timestamp, duration_ms, success)
               VALUES (?, ?, ?, ?, ?)""",
            (tool, session_id, now - (len(tools) - j) * 10, 50.0, 1),
        )

    # Active goal
    conn.execute(
        """INSERT INTO goals
           (goal_id, title, description, status, priority, progress,
            blockers, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("goal_1", "Build session continuity", "Implement handoff protocol",
         "active", "high", 0.6, '["need tests"]', now - 7200, now),
    )
    conn.execute(
        """INSERT INTO goal_log (goal_id, action, details, timestamp)
           VALUES (?, ?, ?, ?)""",
        ("goal_1", "updated", "Progress to 60%", now - 600),
    )

    # Unresolved claim
    conn.execute(
        """INSERT INTO prediction_book
           (id, timestamp, claim, domain, felt_confidence)
           VALUES (?, ?, ?, ?, ?)""",
        ("claim_1", now - 1800, "React 19 uses server components by default",
         "technical", 0.7),
    )

    # Confabulation
    conn.execute(
        """INSERT INTO confabulation_log
           (id, timestamp, claim_text, failure_mode, prevention_strategy, felt_confidence)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("confab_1", now - 3600, "Invented API endpoint",
         "knowledge_gap", "Verify API docs before citing endpoints", 0.8),
    )

    # Personality snapshot
    conn.execute(
        """INSERT INTO personality_snapshot
           (id, session_id, timestamp, avg_verbosity, avg_hedging,
            dominant_tone, confabulation_rate, verification_rate,
            intellectual_honesty, overall_accuracy, avg_engagement,
            pushback_rate)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("snap_1", "prev_session", now - 7200, 0.65, 0.12,
         "neutral", 0.25, 0.50, 0.72, 0.87, 0.93, 0.60),
    )

    # Calibration cache
    conn.execute(
        """INSERT INTO calibration_cache
           (domain, total_claims, accuracy_ratio, overconfidence_ratio, last_updated)
           VALUES (?, ?, ?, ?, ?)""",
        ("technical", 10, 0.70, 0.35, now),
    )

    conn.commit()
    conn.close()


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestTokenEstimation:
    def test_basic(self):
        assert _estimate_tokens("hello world") >= 1

    def test_longer(self):
        text = "a" * 400
        assert _estimate_tokens(text) == 100


class TestToolTopicMapping:
    def test_known_tool(self):
        assert _topic_for_tool("store_episodic") == "memory_formation"
        assert _topic_for_tool("phi_prune") == "memory_optimization"

    def test_unknown_tool(self):
        assert _topic_for_tool("nonexistent_tool") == "general"


class TestWorkingMemoryCompressor:
    @pytest.mark.asyncio
    async def test_capture_empty_db(self, db_path):
        wm = WorkingMemoryCompressor(db_path)
        result = await wm.capture("sess_empty", "claude", token_budget=500)
        assert result["ok"] is True
        assert result["active_memories"] == 0
        assert result["token_count"] >= 0

    @pytest.mark.asyncio
    async def test_capture_with_data(self, db_path):
        _seed_data(db_path)
        wm = WorkingMemoryCompressor(db_path)
        result = await wm.capture("test_session_123", "claude", token_budget=500)
        assert result["ok"] is True
        assert result["active_memories"] == 5
        assert result["active_tools"] > 0
        assert result["goals"] == 1
        assert "compressed_block" in result
        assert len(result["compressed_block"]) > 0

    @pytest.mark.asyncio
    async def test_get_latest(self, db_path):
        _seed_data(db_path)
        wm = WorkingMemoryCompressor(db_path)
        await wm.capture("test_session_123", "claude")
        latest = wm.get_latest("claude")
        assert latest is not None
        assert "compressed_block" in latest
        assert latest["age_hours"] < 1

    def test_get_latest_empty(self, db_path):
        wm = WorkingMemoryCompressor(db_path)
        assert wm.get_latest() is None

    @pytest.mark.asyncio
    async def test_token_budget_respected(self, db_path):
        _seed_data(db_path)
        wm = WorkingMemoryCompressor(db_path)
        result = await wm.capture("test_session_123", "claude", token_budget=50)
        assert result["token_count"] <= 60  # Small margin


class TestTrajectoryEncoder:
    def test_encode_empty(self, db_path):
        te = TrajectoryEncoder(db_path)
        result = te.encode("nonexistent_session")
        assert result["ok"] is True
        assert result["topic_flow"] == []
        assert result["momentum"]["direction"] == "unknown"

    def test_encode_with_data(self, db_path):
        _seed_data(db_path)
        te = TrajectoryEncoder(db_path)
        result = te.encode("test_session_123")
        assert result["ok"] is True
        assert len(result["topic_flow"]) > 0
        assert result["momentum"]["direction"] != "unknown"
        assert "compressed_block" in result

    def test_infer_topics_dedup(self, db_path):
        te = TrajectoryEncoder(db_path)
        topics = te._infer_topics([
            "recall_memories", "recall_memories", "store_semantic",
            "store_semantic", "phi_score_memories",
        ])
        # Should deduplicate consecutive same-topic tools
        assert topics == ["retrieval", "knowledge_capture", "memory_optimization"]

    def test_infer_momentum(self, db_path):
        te = TrajectoryEncoder(db_path)
        momentum = te._infer_momentum([
            "recall_memories", "store_semantic", "store_belief",
            "log_claim", "check_before_claiming",
        ])
        assert momentum["direction"] in TOOL_TOPICS.values() or momentum["direction"] == "general"
        assert 0 <= momentum["confidence"] <= 1

    def test_get_latest(self, db_path):
        _seed_data(db_path)
        te = TrajectoryEncoder(db_path)
        te.encode("test_session_123")
        latest = te.get_latest()
        assert latest is not None
        assert latest["age_hours"] < 1


class TestHandoffProtocol:
    @pytest.mark.asyncio
    async def test_generate_handoff_empty(self, db_path):
        wm = WorkingMemoryCompressor(db_path)
        te = TrajectoryEncoder(db_path)
        hp = HandoffProtocol(db_path, wm, te)
        result = await hp.generate_handoff("claude", token_budget=1500)
        assert result["ok"] is True
        assert "handoff_block" in result
        assert "<session_handoff" in result["handoff_block"]

    @pytest.mark.asyncio
    async def test_generate_handoff_with_data(self, db_path):
        _seed_data(db_path)
        wm = WorkingMemoryCompressor(db_path)
        te = TrajectoryEncoder(db_path)
        hp = HandoffProtocol(db_path, wm, te)

        # First capture a session state
        await hp.end_and_capture("test_session_123", "claude", "Test session")

        result = await hp.generate_handoff("claude", token_budget=1500)
        assert result["ok"] is True
        assert result["token_count"] > 0
        assert result["token_count"] <= 1600  # Allow small margin
        block = result["handoff_block"]
        assert "<session_handoff" in block
        assert "</session_handoff>" in block

    @pytest.mark.asyncio
    async def test_handoff_sections_filter(self, db_path):
        _seed_data(db_path)
        wm = WorkingMemoryCompressor(db_path)
        te = TrajectoryEncoder(db_path)
        hp = HandoffProtocol(db_path, wm, te)
        await hp.end_and_capture("test_session_123", "claude")

        result = await hp.generate_handoff(
            "claude", token_budget=500,
            include_sections=["identity", "goals"],
        )
        assert result["ok"] is True
        block = result["handoff_block"]
        assert "<identity>" in block
        assert "<goals>" in block
        assert "<trajectory>" not in block

    @pytest.mark.asyncio
    async def test_end_and_capture(self, db_path):
        _seed_data(db_path)
        wm = WorkingMemoryCompressor(db_path)
        te = TrajectoryEncoder(db_path)
        hp = HandoffProtocol(db_path, wm, te)

        result = await hp.end_and_capture("test_session_123", "claude", "Built session continuity")
        assert result["ok"] is True
        assert "working_memory" in result
        assert "trajectory" in result
        assert result["working_memory"]["snapshot_id"]
        assert result["trajectory"]["snapshot_id"]
        assert len(result["trajectory"]["topic_flow"]) > 0

    @pytest.mark.asyncio
    async def test_identity_rendering(self, db_path):
        _seed_data(db_path)
        wm = WorkingMemoryCompressor(db_path)
        te = TrajectoryEncoder(db_path)
        hp = HandoffProtocol(db_path, wm, te)

        result = await hp.generate_handoff("claude", include_sections=["identity"])
        block = result["handoff_block"]
        assert "<identity>" in block
        # Should contain personality data from the seeded snapshot
        assert "neutral tone" in block or "honesty" in block

    @pytest.mark.asyncio
    async def test_warnings_rendering(self, db_path):
        _seed_data(db_path)
        wm = WorkingMemoryCompressor(db_path)
        te = TrajectoryEncoder(db_path)
        hp = HandoffProtocol(db_path, wm, te)

        result = await hp.generate_handoff("claude", include_sections=["warnings"])
        block = result["handoff_block"]
        assert "<warnings>" in block
        # Should contain the overconfident domain warning
        assert "technical" in block or "Overconfident" in block or "Confab" in block
