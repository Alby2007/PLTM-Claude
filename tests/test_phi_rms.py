"""
Tests for the Φ-Optimized Resource Management System (ΦRMS).
Tests PhiMemoryScorer, CriticalityPruner, PhiConsolidator, PhiContextBuilder.
"""

import asyncio
import os
import sys
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory.memory_types import TypedMemory, TypedMemoryStore, MemoryType
from src.memory.phi_rms import (
    PhiMemoryScorer,
    CriticalityPruner,
    PhiConsolidator,
    PhiContextBuilder,
    _classify_domains,
    _estimate_tokens,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_phi.db")


@pytest.fixture
async def store(db_path):
    s = TypedMemoryStore(db_path)
    await s.connect()
    yield s
    await s.close()


@pytest.fixture
async def scorer(db_path, store):
    s = PhiMemoryScorer(store, embedding_store=None, db_path=db_path)
    await s.connect()
    yield s
    await s.close()


def make_mem(
    mem_type=MemoryType.SEMANTIC, content="test fact",
    user_id="test_user", tags=None, **kw
):
    now = time.time()
    return TypedMemory(
        id=kw.get("id", str(uuid4())),
        memory_type=mem_type,
        user_id=user_id,
        content=content,
        context=kw.get("context", ""),
        source=kw.get("source", "test"),
        strength=kw.get("strength", 1.0),
        created_at=now,
        last_accessed=now,
        access_count=kw.get("access_count", 1),
        confidence=kw.get("confidence", 0.8),
        evidence_for=kw.get("evidence_for", []),
        evidence_against=kw.get("evidence_against", []),
        episode_timestamp=kw.get("episode_timestamp", 0),
        participants=kw.get("participants", []),
        emotional_valence=kw.get("emotional_valence", 0),
        trigger=kw.get("trigger", ""),
        action=kw.get("action", ""),
        success_count=kw.get("success_count", 0),
        failure_count=kw.get("failure_count", 0),
        consolidated_from=kw.get("consolidated_from", []),
        consolidation_count=kw.get("consolidation_count", 0),
        tags=tags or [],
    )


# ============================================================
# Helper function tests
# ============================================================

class TestHelpers:
    def test_classify_domains_technical(self):
        domains = _classify_domains("python react api", ["technical"])
        assert "technical" in domains

    def test_classify_domains_multiple(self):
        domains = _classify_domains("python project deploy", ["work"])
        assert "technical" in domains
        assert "work" in domains

    def test_classify_domains_empty(self):
        domains = _classify_domains("", [])
        assert isinstance(domains, list)

    def test_estimate_tokens_basic(self):
        mem = make_mem(content="hello world")
        tokens = _estimate_tokens(mem)
        assert tokens >= 1
        assert tokens == max(1, (len("hello world") + len("")) // 4)

    def test_estimate_tokens_procedural(self):
        mem = make_mem(
            mem_type=MemoryType.PROCEDURAL,
            content="rule",
            trigger="when user says X",
            action="do Y",
        )
        tokens = _estimate_tokens(mem)
        expected = max(1, (len("rule") + len("") + len("when user says X") + len("do Y")) // 4)
        assert tokens == expected


# ============================================================
# PhiMemoryScorer tests
# ============================================================

class TestPhiMemoryScorer:
    @pytest.mark.asyncio
    async def test_score_memory_basic(self, store, scorer):
        mem = make_mem(tags=["python", "work"])
        await store.store(mem, bypass_jury=True)

        result = await scorer.score_memory(mem)
        assert "phi_score" in result
        assert "graph_contribution" in result
        assert "domain_bridging" in result
        assert "semantic_uniqueness" in result
        assert "consolidation_potential" in result
        assert "token_cost" in result
        assert 0.0 <= result["phi_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_score_all_empty(self, store, scorer):
        result = await scorer.score_all("nonexistent_user")
        assert result["total"] == 0
        assert result["scored"] == 0

    @pytest.mark.asyncio
    async def test_score_all_multiple(self, store, scorer):
        for i in range(5):
            mem = make_mem(
                content=f"fact number {i}",
                tags=["python", "test"] if i % 2 == 0 else ["work", "deploy"],
            )
            await store.store(mem, bypass_jury=True)

        result = await scorer.score_all("test_user")
        assert result["total"] == 5
        assert result["scored"] == 5
        assert result["avg_phi"] > 0
        assert "by_type" in result

    @pytest.mark.asyncio
    async def test_get_scores(self, store, scorer):
        mem = make_mem(tags=["python"])
        await store.store(mem, bypass_jury=True)
        await scorer.score_memory(mem)

        scores = await scorer.get_scores("test_user")
        assert len(scores) == 1
        assert scores[0]["memory_id"] == mem.id

    @pytest.mark.asyncio
    async def test_get_score_single(self, store, scorer):
        mem = make_mem(tags=["python"])
        await store.store(mem, bypass_jury=True)
        await scorer.score_memory(mem)

        score = await scorer.get_score(mem.id)
        assert score is not None
        assert score["memory_id"] == mem.id

    @pytest.mark.asyncio
    async def test_get_score_missing(self, scorer):
        score = await scorer.get_score("nonexistent")
        assert score is None

    @pytest.mark.asyncio
    async def test_graph_contribution(self, store, scorer):
        # Create memories with overlapping tags
        m1 = make_mem(content="fact 1", tags=["python", "api"])
        m2 = make_mem(content="fact 2", tags=["python", "react"])
        m3 = make_mem(content="fact 3", tags=["unrelated"])
        for m in [m1, m2, m3]:
            await store.store(m, bypass_jury=True)

        all_mems = [m1, m2, m3]
        # m1 shares "python" with m2 → graph_contribution > 0
        gc = scorer._graph_contribution(m1, all_mems)
        assert gc > 0
        # m3 shares nothing → graph_contribution = 0
        gc3 = scorer._graph_contribution(m3, all_mems)
        assert gc3 == 0

    @pytest.mark.asyncio
    async def test_domain_bridging(self, scorer):
        mem = make_mem(content="python project deploy", tags=["technical", "work"])
        db = scorer._domain_bridging(mem)
        assert db > 0  # Should detect technical + work domains

    @pytest.mark.asyncio
    async def test_semantic_uniqueness_no_embeddings(self, scorer):
        mem = make_mem()
        su = await scorer._semantic_uniqueness(mem)
        assert su == 0.5  # Fallback when no embedding store

    @pytest.mark.asyncio
    async def test_consolidation_potential_episodic(self, store, scorer):
        mems = []
        for i in range(6):
            m = make_mem(
                mem_type=MemoryType.EPISODIC,
                content=f"event {i}",
                tags=["meeting"],
            )
            mems.append(m)
            await store.store(m, bypass_jury=True)

        # First episodic has 5 siblings with shared tag
        cp = scorer._consolidation_potential(mems[0], mems)
        assert cp == 1.0  # 5 siblings / 5 = 1.0

    @pytest.mark.asyncio
    async def test_consolidation_potential_belief(self, scorer):
        mem = make_mem(
            mem_type=MemoryType.BELIEF,
            evidence_for=["a", "b", "c"],
        )
        cp = scorer._consolidation_potential(mem, [])
        assert cp == 0.6  # 3/5


# ============================================================
# CriticalityPruner tests
# ============================================================

class TestCriticalityPruner:
    @pytest.mark.asyncio
    async def test_prune_empty(self, store, scorer):
        mock_crit = AsyncMock()
        mock_crit.get_criticality_state = AsyncMock(return_value={"i": 0.5, "r": 1.0, "z": "critical"})
        pruner = CriticalityPruner(scorer, mock_crit, store)

        result = await pruner.prune("nonexistent_user")
        assert result["pruned"] == 0
        assert result["tokens_freed"] == 0

    @pytest.mark.asyncio
    async def test_simulate_prune(self, store, scorer):
        for i in range(5):
            mem = make_mem(content=f"low value fact {i}", tags=[])
            await store.store(mem, bypass_jury=True)

        mock_crit = AsyncMock()
        mock_crit.get_criticality_state = AsyncMock(return_value={"i": 0.5, "r": 1.0, "z": "critical"})
        pruner = CriticalityPruner(scorer, mock_crit, store)

        result = await pruner.simulate_prune("test_user")
        assert result["dry_run"] is True
        # Memories should still exist after dry run
        remaining = await store.query("test_user", min_strength=0.0, limit=100)
        assert len(remaining) == 5

    @pytest.mark.asyncio
    async def test_prune_protects_procedures(self, store, scorer):
        proc = make_mem(
            mem_type=MemoryType.PROCEDURAL,
            content="important rule",
            trigger="when X",
            action="do Y",
            success_count=5,
            tags=["rule"],
        )
        await store.store(proc, bypass_jury=True)

        mock_crit = AsyncMock()
        mock_crit.get_criticality_state = AsyncMock(return_value={"i": 0.5, "r": 1.0, "z": "critical"})
        pruner = CriticalityPruner(scorer, mock_crit, store)

        result = await pruner.prune("test_user")
        assert result["protected"] >= 1
        # Procedure should still exist
        remaining = await store.get(proc.id)
        assert remaining is not None

    @pytest.mark.asyncio
    async def test_prune_respects_max_removals(self, store, scorer):
        for i in range(10):
            mem = make_mem(content=f"disposable {i}", tags=[])
            await store.store(mem, bypass_jury=True)

        mock_crit = AsyncMock()
        mock_crit.get_criticality_state = AsyncMock(return_value={"i": 0.5, "r": 1.0, "z": "critical"})
        pruner = CriticalityPruner(scorer, mock_crit, store)

        result = await pruner.prune("test_user", max_removals=3)
        assert result["pruned"] <= 3


# ============================================================
# PhiConsolidator tests
# ============================================================

class TestPhiConsolidator:
    @pytest.mark.asyncio
    async def test_consolidate_empty(self, store, scorer):
        consolidator = PhiConsolidator(store, embedding_store=None, scorer=scorer)
        result = await consolidator.consolidate("test_user")
        assert result["clusters_found"] == 0
        assert result["promoted"] == 0

    @pytest.mark.asyncio
    async def test_consolidate_tag_based(self, store, scorer):
        # Create 4 episodic memories with same tag → should cluster
        for i in range(4):
            mem = make_mem(
                mem_type=MemoryType.EPISODIC,
                content=f"meeting about project alpha, discussion {i}",
                tags=["meeting", "alpha"],
            )
            await store.store(mem, bypass_jury=True)

        consolidator = PhiConsolidator(store, embedding_store=None, scorer=scorer)
        result = await consolidator.consolidate("test_user", min_cluster_size=3)
        assert result["clusters_found"] >= 1
        # Should have promoted at least one semantic memory
        assert result["promoted"] >= 1 or result["reinforced"] >= 1

    @pytest.mark.asyncio
    async def test_consolidate_too_few(self, store, scorer):
        # Only 2 episodes → below min_cluster_size=3
        for i in range(2):
            mem = make_mem(
                mem_type=MemoryType.EPISODIC,
                content=f"event {i}",
                tags=["rare"],
            )
            await store.store(mem, bypass_jury=True)

        consolidator = PhiConsolidator(store, embedding_store=None, scorer=scorer)
        result = await consolidator.consolidate("test_user", min_cluster_size=3)
        assert result["clusters_found"] == 0


# ============================================================
# PhiContextBuilder tests
# ============================================================

class TestPhiContextBuilder:
    @pytest.mark.asyncio
    async def test_build_context_empty(self, store, scorer):
        builder = PhiContextBuilder(scorer, store, embedding_store=None)
        result = await builder.build_context("test_user", ["hello"])
        assert result["memories_selected"] == 0
        assert result["prompt_block"] == ""

    @pytest.mark.asyncio
    async def test_build_context_with_procedures(self, store, scorer):
        proc = make_mem(
            mem_type=MemoryType.PROCEDURAL,
            content="be concise",
            trigger="when user asks",
            action="respond briefly",
            tags=["communication"],
            strength=0.8,
            success_count=3,
        )
        await store.store(proc, bypass_jury=True)
        await scorer.score_memory(proc)

        builder = PhiContextBuilder(scorer, store, embedding_store=None)
        result = await builder.build_context("test_user", ["how are you"])
        # Should include the procedural memory as fallback
        assert result["memories_selected"] >= 1
        assert "<phi_context>" in result["prompt_block"]

    @pytest.mark.asyncio
    async def test_build_context_respects_budget(self, store, scorer):
        # Create many memories
        for i in range(20):
            mem = make_mem(
                content=f"This is a fairly long fact about topic number {i} with extra detail " * 3,
                tags=["test"],
                strength=0.5,
            )
            await store.store(mem, bypass_jury=True)
            await scorer.score_memory(mem)

        builder = PhiContextBuilder(scorer, store, embedding_store=None)
        result = await builder.build_context("test_user", ["test"], token_budget=100)
        # Should not exceed budget (except for critical memories)
        assert result["total_tokens"] <= 200  # Allow some slack for critical additions

    @pytest.mark.asyncio
    async def test_format_line(self, scorer, store):
        builder = PhiContextBuilder(scorer, store)

        sem = make_mem(mem_type=MemoryType.SEMANTIC, content="Python is great")
        line = builder._format_line(sem)
        assert "[FACT]" in line
        assert "Python is great" in line

        belief = make_mem(mem_type=MemoryType.BELIEF, content="User likes Python", confidence=0.9)
        line = builder._format_line(belief)
        assert "[BELIEF" in line

        proc = make_mem(
            mem_type=MemoryType.PROCEDURAL,
            trigger="user says hi",
            action="greet warmly",
        )
        line = builder._format_line(proc)
        assert "[RULE]" in line
        assert "greet warmly" in line
