"""
Tests for Improvement Loop systems:
1. ToolAnalytics — tool usage logging and analytics
2. ArchitectureSnapshotter — versioned system state snapshots
"""

import json
import os
import sqlite3
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.tool_analytics import ToolAnalytics
from src.analysis.architecture_snapshots import ArchitectureSnapshotter


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_loops.db"


@pytest.fixture
def analytics(db_path):
    return ToolAnalytics(db_path)


@pytest.fixture
def snapshotter(db_path):
    # Create prerequisite tables that snapshotter reads from
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS atoms (
            id TEXT PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT,
            metadata TEXT DEFAULT '{}'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS typed_memories (
            id TEXT PRIMARY KEY, memory_type TEXT, user_id TEXT,
            content TEXT, strength REAL DEFAULT 1.0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_book (
            id TEXT PRIMARY KEY, claim TEXT, was_correct INTEGER,
            actual_truth TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS confabulation_log (
            id TEXT PRIMARY KEY, timestamp REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS personality_snapshot (
            id TEXT PRIMARY KEY, session_id TEXT, timestamp REAL,
            avg_verbosity REAL, confabulation_rate REAL,
            verification_rate REAL, intellectual_honesty REAL,
            overall_accuracy REAL
        )
    """)
    conn.commit()
    conn.close()
    return ArchitectureSnapshotter(db_path)


# ============================================================
# ToolAnalytics Tests
# ============================================================

class TestToolAnalytics:
    def test_log_invocation(self, analytics, db_path):
        analytics.log_invocation("store_memory_atom", "sess1", 15.5, True)

        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT * FROM tool_invocations").fetchone()
        conn.close()
        assert row is not None
        assert row[1] == "store_memory_atom"  # tool_name
        assert row[2] == "sess1"  # session_id
        assert row[4] == 15.5  # duration_ms
        assert row[5] == 1  # success

    def test_log_invocation_error(self, analytics, db_path):
        analytics.log_invocation("bad_tool", "sess1", 30000, False, "timeout")

        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT * FROM tool_invocations").fetchone()
        conn.close()
        assert row[5] == 0  # success = false
        assert row[6] == "timeout"  # error_type

    def test_end_session_flushes_sequence(self, analytics, db_path):
        analytics.log_invocation("tool_a", "sess1", 10, True)
        analytics.log_invocation("tool_b", "sess1", 20, True)
        analytics.log_invocation("tool_c", "sess1", 30, True)
        analytics.end_session("sess1")

        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT sequence, length FROM tool_sequences").fetchone()
        conn.close()
        seq = json.loads(row[0])
        assert seq == ["tool_a", "tool_b", "tool_c"]
        assert row[1] == 3

    def test_end_session_empty(self, analytics, db_path):
        analytics.end_session("nonexistent")
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM tool_sequences").fetchone()[0]
        conn.close()
        assert count == 0

    def test_hash_args(self):
        h1 = ToolAnalytics.hash_args({"user_id": "test", "limit": 10})
        h2 = ToolAnalytics.hash_args({"limit": 10, "user_id": "test"})
        assert h1 == h2  # Order-independent
        assert len(h1) == 16

    def test_get_usage_stats(self, analytics):
        for i in range(5):
            analytics.log_invocation("tool_a", "s1", 10 + i, True)
        for i in range(3):
            analytics.log_invocation("tool_b", "s1", 20, True)
        analytics.log_invocation("tool_b", "s1", 100, False, "ValueError")

        stats = analytics.get_usage_stats(days=1)
        assert stats["total_calls"] == 9
        assert stats["unique_tools_used"] == 2

        tool_a = next(t for t in stats["tools"] if t["tool"] == "tool_a")
        assert tool_a["calls"] == 5
        assert tool_a["success_rate"] == 1.0

        tool_b = next(t for t in stats["tools"] if t["tool"] == "tool_b")
        assert tool_b["calls"] == 4
        assert tool_b["success_rate"] == 0.75

    def test_get_usage_stats_empty(self, analytics):
        stats = analytics.get_usage_stats(days=1)
        assert stats["total_calls"] == 0
        assert stats["unique_tools_used"] == 0

    def test_get_redundancy_report(self, analytics):
        # Simulate 3 sessions with co-occurring tools
        for sid in ["s1", "s2", "s3"]:
            analytics.log_invocation("init_session", sid, 10, True)
            analytics.log_invocation("recall_memories", sid, 20, True)
        # Add an error-prone tool
        for _ in range(5):
            analytics.log_invocation("bad_tool", "s1", 50, False, "Error")

        report = analytics.get_redundancy_report(days=1)
        assert report["tools_used"] == 3
        assert len(report["error_prone"]) >= 1
        assert report["error_prone"][0]["tool"] == "bad_tool"

    def test_get_sequence_patterns(self, analytics, db_path):
        # Insert some sequences directly
        conn = sqlite3.connect(str(db_path))
        for i in range(5):
            conn.execute(
                "INSERT INTO tool_sequences (session_id, sequence, length, timestamp) VALUES (?, ?, ?, ?)",
                (f"s{i}", json.dumps(["init", "recall", "store"]), 3, time.time()))
        conn.commit()
        conn.close()

        patterns = analytics.get_sequence_patterns(min_support=3)
        assert patterns["sequences_analyzed"] == 5
        assert len(patterns["patterns"]) > 0
        # "init" → "recall" should be a frequent bigram
        bigrams = [p for p in patterns["patterns"] if p["type"] == "bigram"]
        assert any(p["sequence"] == ["init", "recall"] for p in bigrams)

    def test_propose_consolidation(self, analytics):
        # Log some usage
        for _ in range(10):
            analytics.log_invocation("popular_tool", "s1", 10, True)
        for _ in range(5):
            analytics.log_invocation("error_tool", "s1", 100, False, "Err")

        all_tools = ["popular_tool", "error_tool", "never_used_tool"]
        proposals = analytics.propose_consolidation(all_tool_names=all_tools, days=1)

        assert proposals["total_proposals"] > 0
        types = [p["type"] for p in proposals["proposals"]]
        assert "deprecate" in types  # never_used_tool
        assert "investigate" in types  # error_tool


# ============================================================
# ArchitectureSnapshotter Tests
# ============================================================

class TestArchitectureSnapshotter:
    def test_take_snapshot(self, snapshotter):
        result = snapshotter.take_snapshot(
            label="test-v1",
            tool_names=["tool_a", "tool_b", "tool_c"],
            notes="Initial snapshot")

        assert result["label"] == "test-v1"
        assert result["tool_count"] == 3
        assert result["notes"] == "Initial snapshot"
        assert "id" in result
        assert "metrics" in result

    def test_list_snapshots(self, snapshotter):
        snapshotter.take_snapshot("v1", ["a", "b"])
        snapshotter.take_snapshot("v2", ["a", "b", "c"])

        result = snapshotter.list_snapshots()
        assert result["count"] == 2
        assert result["snapshots"][0]["label"] == "v2"  # Most recent first
        assert result["snapshots"][1]["label"] == "v1"

    def test_get_latest(self, snapshotter):
        snapshotter.take_snapshot("v1", ["a"])
        snapshotter.take_snapshot("v2", ["a", "b"])

        latest = snapshotter.get_latest()
        assert latest is not None
        assert latest["label"] == "v2"
        assert latest["tool_count"] == 2

    def test_get_latest_empty(self, snapshotter):
        assert snapshotter.get_latest() is None

    def test_compare_snapshots(self, snapshotter):
        s1 = snapshotter.take_snapshot("before", ["tool_a", "tool_b"])
        # Add some data between snapshots
        conn = sqlite3.connect(str(snapshotter.db_path))
        conn.execute("INSERT INTO atoms (id, subject, predicate, object) VALUES ('a1', 's', 'p', 'o')")
        conn.commit()
        conn.close()
        s2 = snapshotter.take_snapshot("after", ["tool_a", "tool_b", "tool_c"])

        diff = snapshotter.compare(s1["id"], s2["id"])
        assert diff["config_changed"] is True
        assert diff["tools_added"] == ["tool_c"]
        assert diff["tools_removed"] == []
        assert diff["tools_added_count"] == 1
        assert diff["count_deltas"]["tool_count"]["delta"] == 1
        assert diff["count_deltas"]["atom_count"]["delta"] == 1

    def test_compare_missing_snapshot(self, snapshotter):
        s1 = snapshotter.take_snapshot("v1", ["a"])
        diff = snapshotter.compare(s1["id"], "nonexistent")
        assert "error" in diff

    def test_config_hash_deterministic(self, snapshotter):
        s1 = snapshotter.take_snapshot("v1", ["b", "a", "c"])
        s2 = snapshotter.take_snapshot("v2", ["a", "c", "b"])
        # Same tools in different order → same hash (sorted internally)
        assert s1["config_hash"] == s2["config_hash"]

    def test_metrics_capture(self, snapshotter, db_path):
        # Add some test data for metrics
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO prediction_book (id, claim, was_correct, actual_truth) VALUES ('c1', 'test', 1, 'true')")
        conn.execute("INSERT INTO confabulation_log (id, timestamp) VALUES ('cf1', ?)", (time.time(),))
        conn.commit()
        conn.close()

        result = snapshotter.take_snapshot("with-data", ["tool_a"])
        metrics = result["metrics"]
        assert metrics.get("total_claims") == 1
        assert metrics.get("confabulation_count") == 1
