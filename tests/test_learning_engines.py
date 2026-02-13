"""
Tests for Autonomous Learning Engine and Active Learning Engine.
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import aiosqlite


# ============================================================
# Autonomous Learning Engine Tests
# ============================================================

class TestAutonomousLearningEngine:
    """Tests for AutonomousLearningEngine."""

    @pytest.fixture
    async def engine(self, tmp_path):
        from src.learning.autonomous_learning import AutonomousLearningEngine
        db_path = tmp_path / "test.db"
        eng = AutonomousLearningEngine(db_path)
        await eng.connect()
        yield eng
        await eng.close()

    @pytest.mark.asyncio
    async def test_tables_created(self, engine):
        """Tables should be created on connect."""
        cursor = await engine._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [r[0] for r in await cursor.fetchall()]
        assert "learning_schedules" in tables
        assert "learning_runs" in tables
        assert "learning_accumulation" in tables

    @pytest.mark.asyncio
    async def test_default_schedules_seeded(self, engine):
        """Default schedules should be seeded on first connect."""
        result = await engine.get_schedules()
        assert result["n"] >= 5
        task_names = [s["task"] for s in result["schedules"]]
        assert "arxiv_consciousness" in task_names
        assert "github_trending" in task_names
        assert "consolidation" in task_names
        assert "phi_measurement" in task_names

    @pytest.mark.asyncio
    async def test_update_schedule(self, engine):
        """Should update schedule interval and enabled state."""
        result = await engine.update_schedule("arxiv_consciousness", interval_hours=48.0)
        assert result["ok"]

        schedules = await engine.get_schedules()
        arxiv = next(s for s in schedules["schedules"] if s["task"] == "arxiv_consciousness")
        assert arxiv["h"] == 48.0

    @pytest.mark.asyncio
    async def test_update_schedule_disable(self, engine):
        """Should disable a schedule."""
        result = await engine.update_schedule("news_feed", enabled=False)
        assert result["ok"]

        schedules = await engine.get_schedules()
        news = next(s for s in schedules["schedules"] if s["task"] == "news_feed")
        assert news["on"] is False

    @pytest.mark.asyncio
    async def test_run_due_tasks_none_due(self, engine):
        """No tasks should be due immediately after seeding (last_run=0 means all due)."""
        # All tasks have last_run_at=0, so they're all due
        result = await engine.run_due_tasks()
        assert result["ok"]
        # They'll fail because no store/network, but should attempt
        assert result["checked"] >= 5

    @pytest.mark.asyncio
    async def test_run_task_unknown(self, engine):
        """Unknown task should complete with zero items."""
        result = await engine.run_task("nonexistent_task")
        assert result["fetched"] == 0
        assert result["stored"] == 0

    @pytest.mark.asyncio
    async def test_run_phi_measurement_no_store(self, engine):
        """Phi measurement without store should handle gracefully."""
        result = await engine.run_task("phi_measurement")
        assert result["task"] == "phi_measurement"
        # Should complete even without store
        assert result["status"] in ("completed", "failed")

    @pytest.mark.asyncio
    async def test_learning_digest_empty(self, engine):
        """Digest with no runs should return empty but valid structure."""
        result = await engine.get_learning_digest(since_hours=24)
        assert result["runs"] == 0
        assert result["items_learned"] == 0
        assert isinstance(result["phi_trajectory"], list)
        assert isinstance(result["velocity"], dict)

    @pytest.mark.asyncio
    async def test_phi_history_empty(self, engine):
        """Phi history with no measurements should return empty."""
        result = await engine.get_phi_history()
        assert result["n"] == 0
        assert result["trend"] == "stable"

    @pytest.mark.asyncio
    async def test_run_persists(self, engine):
        """Running a task should persist a learning_runs record."""
        await engine.run_task("phi_measurement")

        cursor = await engine._conn.execute("SELECT COUNT(*) FROM learning_runs")
        count = (await cursor.fetchone())[0]
        assert count >= 1

    @pytest.mark.asyncio
    async def test_schedule_updates_after_run(self, engine):
        """Schedule should update last_run_at and run_count after a run."""
        await engine.run_task("phi_measurement")

        cursor = await engine._conn.execute(
            "SELECT run_count, last_run_at FROM learning_schedules WHERE task='phi_measurement'"
        )
        row = await cursor.fetchone()
        assert row[0] >= 1  # run_count
        assert row[1] > 0  # last_run_at

    @pytest.mark.asyncio
    async def test_run_history(self, engine):
        """Should return run history."""
        await engine.run_task("phi_measurement")
        result = await engine.get_run_history(limit=5)
        assert result["n"] >= 1
        assert result["runs"][0]["task"] == "phi_measurement"


# ============================================================
# Active Learning Engine Tests
# ============================================================

class TestActiveLearningEngine:
    """Tests for ActiveLearningEngine."""

    @pytest.fixture
    async def engine(self, tmp_path):
        from src.learning.active_learning import ActiveLearningEngine
        db_path = tmp_path / "test.db"
        eng = ActiveLearningEngine(db_path)
        await eng.connect()
        yield eng
        await eng.close()

    @pytest.mark.asyncio
    async def test_tables_created(self, engine):
        """Tables should be created on connect."""
        cursor = await engine._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [r[0] for r in await cursor.fetchall()]
        assert "active_searches" in tables
        assert "hypotheses" in tables
        assert "evidence_log" in tables

    # ── Hypothesis Lifecycle ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_propose_hypothesis(self, engine):
        """Should create a hypothesis."""
        result = await engine.propose_hypothesis(
            statement="Integrated Information Theory explains consciousness",
            domains=["consciousness", "neuroscience"],
            prior_confidence=0.6,
        )
        assert result["ok"]
        assert result["confidence"] == 0.6
        assert result["status"] == "proposed"
        assert result["id"].startswith("hyp_")

    @pytest.mark.asyncio
    async def test_submit_evidence_for(self, engine):
        """Submitting supporting evidence should increase confidence."""
        hyp = await engine.propose_hypothesis(
            statement="Test hypothesis", prior_confidence=0.5
        )
        hyp_id = hyp["id"]

        result = await engine.submit_evidence(
            hypothesis_id=hyp_id,
            evidence="Found supporting paper",
            direction="for",
            strength=0.8,
        )
        assert result["ok"]
        # Confidence should increase from 0.5
        new_conf = float(result["confidence"].split("→")[1])
        assert new_conf > 0.5

    @pytest.mark.asyncio
    async def test_submit_evidence_against(self, engine):
        """Submitting counter-evidence should decrease confidence."""
        hyp = await engine.propose_hypothesis(
            statement="Test hypothesis", prior_confidence=0.5
        )
        hyp_id = hyp["id"]

        result = await engine.submit_evidence(
            hypothesis_id=hyp_id,
            evidence="Found contradicting evidence",
            direction="against",
            strength=0.8,
        )
        assert result["ok"]
        new_conf = float(result["confidence"].split("→")[1])
        assert new_conf < 0.5

    @pytest.mark.asyncio
    async def test_hypothesis_auto_confirm(self, engine):
        """Hypothesis should auto-confirm with enough strong evidence."""
        hyp = await engine.propose_hypothesis(
            statement="Test hypothesis", prior_confidence=0.7
        )
        hyp_id = hyp["id"]

        # Submit many strong supporting evidence to push past 0.90 threshold
        for i in range(15):
            result = await engine.submit_evidence(
                hypothesis_id=hyp_id,
                evidence=f"Strong evidence {i}",
                direction="for",
                strength=0.9,
            )
            if result["status"] == "confirmed":
                break

        assert result["status"] == "confirmed"
        assert result["resolution"]

    @pytest.mark.asyncio
    async def test_hypothesis_auto_refute(self, engine):
        """Hypothesis should auto-refute with enough counter-evidence."""
        hyp = await engine.propose_hypothesis(
            statement="Test hypothesis", prior_confidence=0.3
        )
        hyp_id = hyp["id"]

        for i in range(15):
            result = await engine.submit_evidence(
                hypothesis_id=hyp_id,
                evidence=f"Counter evidence {i}",
                direction="against",
                strength=0.9,
            )
            if result["status"] == "refuted":
                break

        assert result["status"] == "refuted"

    @pytest.mark.asyncio
    async def test_get_active_hypotheses(self, engine):
        """Should return active hypotheses."""
        await engine.propose_hypothesis(statement="Hyp 1")
        await engine.propose_hypothesis(statement="Hyp 2")

        result = await engine.get_active_hypotheses()
        assert result["ok"]
        assert result["n"] == 2

    @pytest.mark.asyncio
    async def test_get_hypothesis_detail(self, engine):
        """Should return full hypothesis details."""
        hyp = await engine.propose_hypothesis(
            statement="Detailed hypothesis",
            domains=["physics"],
            verification_strategy="Look for papers on X",
        )

        result = await engine.get_hypothesis(hyp["id"])
        assert result["ok"]
        assert result["statement"] == "Detailed hypothesis"
        assert result["strategy"] == "Look for papers on X"

    @pytest.mark.asyncio
    async def test_resolve_hypothesis_manually(self, engine):
        """Should manually resolve a hypothesis."""
        hyp = await engine.propose_hypothesis(statement="Manual resolve test")

        result = await engine.resolve_hypothesis(
            hyp["id"], status="confirmed", resolution="Confirmed by experiment"
        )
        assert result["ok"]
        assert result["status"] == "confirmed"

        # Should no longer appear in active
        active = await engine.get_active_hypotheses()
        assert active["n"] == 0

    @pytest.mark.asyncio
    async def test_evidence_on_resolved_fails(self, engine):
        """Can't submit evidence on resolved hypothesis."""
        hyp = await engine.propose_hypothesis(statement="Resolved test")
        await engine.resolve_hypothesis(hyp["id"], status="confirmed")

        result = await engine.submit_evidence(
            hypothesis_id=hyp["id"],
            evidence="Late evidence",
            direction="for",
        )
        assert not result["ok"]

    # ── Search Pipeline ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_search_history_empty(self, engine):
        """Search history should be empty initially."""
        result = await engine.get_search_history()
        assert result["ok"]
        assert result["n"] == 0

    @pytest.mark.asyncio
    async def test_extract_facts(self, engine):
        """Should extract facts from search results."""
        from src.learning.active_learning import SearchResult
        result = SearchResult(
            query="test",
            source="arxiv",
            title="Test Paper",
            content="Integrated Information Theory is a mathematical framework for consciousness. "
                    "IIT proposes that consciousness arises from integrated information. "
                    "The theory shows that phi measures the amount of integrated information.",
            url="https://arxiv.org/abs/test",
            relevance=0.8,
        )
        facts = await engine._extract_facts(result)
        assert len(facts) >= 1

    @pytest.mark.asyncio
    async def test_verify_fact_new(self, engine):
        """Fact without store should be marked as new."""
        from src.learning.active_learning import VerifiedFact
        fact = VerifiedFact(
            fact_id="test",
            subject="IIT",
            predicate="is_a",
            object="theory of consciousness",
            confidence=0.8,
            source_url="https://test.com",
            verification_status="pending",
        )
        result = await engine._verify_fact(fact)
        assert result.verification_status == "new"
