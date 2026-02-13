"""
Tests for Hierarchical Memory Index, Predictive Prefetch, and Auto Contradiction Resolver.
"""

import json
import time
from pathlib import Path

import pytest
import aiosqlite


# ============================================================
# Hierarchical Memory Index Tests
# ============================================================

class TestHierarchicalMemoryIndex:

    @pytest.fixture
    async def index(self, tmp_path):
        from src.memory.hierarchical_memory import HierarchicalMemoryIndex
        db_path = tmp_path / "test.db"
        idx = HierarchicalMemoryIndex(db_path)
        await idx.connect()
        yield idx
        await idx.close()

    @pytest.mark.asyncio
    async def test_tables_created(self, index):
        cursor = await index._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [r[0] for r in await cursor.fetchall()]
        assert "memory_index" in tables

    @pytest.mark.asyncio
    async def test_index_atom_consciousness(self, index):
        result = await index.index_atom(
            "atom1", "researcher", "studies", "integrated information theory of consciousness",
            contexts=["neuroscience"]
        )
        assert result["domain"] == "consciousness"

    @pytest.mark.asyncio
    async def test_index_atom_technical(self, index):
        result = await index.index_atom(
            "atom2", "user", "prefers", "python for backend development",
            contexts=["coding"]
        )
        assert result["domain"] == "technical"

    @pytest.mark.asyncio
    async def test_index_atom_ai_ml(self, index):
        result = await index.index_atom(
            "atom3", "paper", "proposes", "new transformer architecture for language model",
            contexts=["ai"]
        )
        assert result["domain"] == "ai_ml"

    @pytest.mark.asyncio
    async def test_query_by_domain(self, index):
        await index.index_atom("a1", "x", "y", "consciousness qualia", [])
        await index.index_atom("a2", "x", "y", "python code algorithm", [])
        await index.index_atom("a3", "x", "y", "consciousness neural correlates", [])

        results = await index.query_by_domain("consciousness")
        assert len(results) == 2
        assert "a1" in results
        assert "a3" in results

    @pytest.mark.asyncio
    async def test_query_by_subtopic(self, index):
        await index.index_atom("a1", "x", "y", "integrated information phi tononi", [])
        await index.index_atom("a2", "x", "y", "global workspace baars broadcasting", [])

        iit = await index.query_by_subtopic("consciousness", "iit")
        gw = await index.query_by_subtopic("consciousness", "global_workspace")
        assert "a1" in iit
        assert "a2" in gw

    @pytest.mark.asyncio
    async def test_get_tree(self, index):
        await index.index_atom("a1", "x", "y", "consciousness qualia", [])
        await index.index_atom("a2", "x", "y", "python code", [])
        await index.index_atom("a3", "x", "y", "transformer language model", [])

        tree = await index.get_tree()
        assert tree["total_indexed"] == 3
        assert tree["domains"] >= 2

    @pytest.mark.asyncio
    async def test_smart_retrieve(self, index):
        await index.index_atom("a1", "x", "y", "consciousness qualia hard problem", [])
        await index.index_atom("a2", "x", "y", "python debugging algorithm", [])

        result = await index.smart_retrieve("what is consciousness")
        assert result["domain"] == "consciousness"
        assert "a1" in result["atom_ids"]

    @pytest.mark.asyncio
    async def test_classify_domain_multiword(self, index):
        """Multi-word phrases should match."""
        domain = index._classify_domain("integrated information theory")
        assert domain == "consciousness"

    @pytest.mark.asyncio
    async def test_classify_domain_general_fallback(self, index):
        domain = index._classify_domain("xyzzy foobar baz")
        assert domain == "general"

    @pytest.mark.asyncio
    async def test_extract_keywords(self, index):
        kws = index._extract_keywords("The quick brown fox jumps over the lazy dog")
        assert "quick" in kws
        assert "brown" in kws
        assert "the" not in kws  # stop word


# ============================================================
# Predictive Prefetch Engine Tests
# ============================================================

class TestPredictivePrefetchEngine:

    @pytest.fixture
    async def engine(self, tmp_path):
        from src.memory.hierarchical_memory import (
            HierarchicalMemoryIndex, PredictivePrefetchEngine,
        )
        db_path = tmp_path / "test.db"
        idx = HierarchicalMemoryIndex(db_path)
        await idx.connect()
        eng = PredictivePrefetchEngine(idx)
        yield eng
        await idx.close()

    def test_record_tool_call(self, engine):
        engine.record_tool_call("search_and_learn")
        assert "learning" in engine._recent_domains

    def test_record_query(self, engine):
        engine.record_query("consciousness and integrated information")
        assert "consciousness" in engine._recent_domains

    def test_predict_empty(self, engine):
        preds = engine.predict_next_domains()
        assert preds == [("general", 0.5)]

    def test_predict_after_tools(self, engine):
        engine.record_tool_call("search_and_learn")
        engine.record_tool_call("propose_hypothesis")
        engine.record_tool_call("ingest_arxiv")

        preds = engine.predict_next_domains()
        # Learning should be top prediction
        assert preds[0][0] == "learning"

    def test_predict_momentum(self, engine):
        # Same domain 3x → momentum boost
        for _ in range(3):
            engine.record_tool_call("search_and_learn")

        preds = engine.predict_next_domains()
        assert preds[0][0] == "learning"
        assert preds[0][1] > 0.5  # Should have high score

    def test_predict_adjacency(self, engine):
        engine.record_query("consciousness neural correlates")
        preds = engine.predict_next_domains(top_k=5)
        domains = [p[0] for p in preds]
        # Adjacent domains should appear
        assert "ai_ml" in domains or "complexity" in domains

    def test_stats_initial(self, engine):
        stats = engine.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_trajectory_limit(self, engine):
        for i in range(30):
            engine.record_tool_call("search_and_learn")
        assert len(engine._recent_tools) == engine.MAX_TRAJECTORY

    @pytest.mark.asyncio
    async def test_prefetch_empty(self, engine):
        engine.record_query("consciousness")
        result = await engine.prefetch()
        # No store set, so no atoms loaded, but predictions should work
        assert "predictions" in result


# ============================================================
# Auto Contradiction Resolver Tests
# ============================================================

class TestAutoContradictionResolver:

    @pytest.fixture
    async def resolver(self, tmp_path):
        from src.memory.hierarchical_memory import AutoContradictionResolver
        db_path = tmp_path / "test.db"

        # Create a minimal atoms table for testing
        conn = await aiosqlite.connect(str(db_path))
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS atoms (
                id TEXT PRIMARY KEY,
                atom_type TEXT NOT NULL,
                graph TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                confidence REAL DEFAULT 0.5,
                first_observed INTEGER DEFAULT 0,
                last_accessed INTEGER DEFAULT 0
            )
        """)
        await conn.commit()
        await conn.close()

        res = AutoContradictionResolver(db_path)
        await res.connect()
        yield res
        await res.close()

    async def _insert_atom(self, resolver, atom_id, subject, predicate, obj,
                           confidence=0.5, first_observed=None):
        if first_observed is None:
            first_observed = int(time.time())
        await resolver._conn.execute(
            """INSERT INTO atoms (id, atom_type, graph, subject, predicate, object, metadata, confidence, first_observed)
               VALUES (?, 'fact', 'substantiated', ?, ?, ?, '{}', ?, ?)""",
            (atom_id, subject, predicate, obj, confidence, first_observed),
        )
        await resolver._conn.commit()

    @pytest.mark.asyncio
    async def test_tables_created(self, resolver):
        cursor = await resolver._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [r[0] for r in await cursor.fetchall()]
        assert "contradiction_log" in tables

    @pytest.mark.asyncio
    async def test_no_conflicts(self, resolver):
        await self._insert_atom(resolver, "a1", "alice", "likes", "cats")
        await self._insert_atom(resolver, "a2", "bob", "likes", "dogs")
        result = await resolver.scan_and_resolve()
        assert result["conflicts_found"] == 0

    @pytest.mark.asyncio
    async def test_confidence_gap_resolution(self, resolver):
        """Higher confidence should win when gap > 0.3."""
        await self._insert_atom(resolver, "a1", "alice", "lives_in", "London", confidence=0.9)
        await self._insert_atom(resolver, "a2", "alice", "lives_in", "Paris", confidence=0.3)

        result = await resolver.scan_and_resolve()
        assert result["conflicts_found"] == 1
        assert result["auto_resolved"] == 1
        assert result["needs_review"] == 0

    @pytest.mark.asyncio
    async def test_recency_resolution(self, resolver):
        """Newer atom should win when age gap > 30 days."""
        old_time = int(time.time()) - (60 * 86400)  # 60 days ago
        new_time = int(time.time())

        await self._insert_atom(resolver, "a1", "alice", "works_at", "OldCorp",
                                confidence=0.5, first_observed=old_time)
        await self._insert_atom(resolver, "a2", "alice", "works_at", "NewCorp",
                                confidence=0.5, first_observed=new_time)

        result = await resolver.scan_and_resolve()
        assert result["conflicts_found"] == 1
        assert result["auto_resolved"] == 1

    @pytest.mark.asyncio
    async def test_needs_review(self, resolver):
        """Similar confidence + similar age → needs review."""
        now = int(time.time())
        await self._insert_atom(resolver, "a1", "alice", "favorite_color", "blue",
                                confidence=0.6, first_observed=now)
        await self._insert_atom(resolver, "a2", "alice", "favorite_color", "green",
                                confidence=0.5, first_observed=now - 86400)

        result = await resolver.scan_and_resolve()
        assert result["conflicts_found"] == 1
        assert result["needs_review"] == 1

    @pytest.mark.asyncio
    async def test_get_unresolved(self, resolver):
        now = int(time.time())
        await self._insert_atom(resolver, "a1", "x", "y", "val1", confidence=0.5, first_observed=now)
        await self._insert_atom(resolver, "a2", "x", "y", "val2", confidence=0.5, first_observed=now)
        await resolver.scan_and_resolve()

        unresolved = await resolver.get_unresolved()
        assert unresolved["ok"]
        assert unresolved["n"] >= 1

    @pytest.mark.asyncio
    async def test_resolution_stats(self, resolver):
        now = int(time.time())
        await self._insert_atom(resolver, "a1", "x", "y", "v1", confidence=0.9, first_observed=now)
        await self._insert_atom(resolver, "a2", "x", "y", "v2", confidence=0.2, first_observed=now)
        await resolver.scan_and_resolve()

        stats = await resolver.get_resolution_stats()
        assert stats["total"] >= 1
        assert stats["auto_resolved"] >= 1

    @pytest.mark.asyncio
    async def test_no_duplicate_resolution(self, resolver):
        """Same conflict shouldn't be resolved twice."""
        await self._insert_atom(resolver, "a1", "x", "y", "v1", confidence=0.9)
        await self._insert_atom(resolver, "a2", "x", "y", "v2", confidence=0.2)

        r1 = await resolver.scan_and_resolve()
        r2 = await resolver.scan_and_resolve()
        # Second scan should find the conflict but skip it (already logged)
        assert r1["auto_resolved"] == 1
        assert r2["auto_resolved"] == 0

    @pytest.mark.asyncio
    async def test_loser_moved_to_historical(self, resolver):
        """Auto-resolved loser should be moved to historical graph."""
        await self._insert_atom(resolver, "a1", "x", "y", "winner", confidence=0.9)
        await self._insert_atom(resolver, "a2", "x", "y", "loser", confidence=0.2)
        await resolver.scan_and_resolve()

        cursor = await resolver._conn.execute(
            "SELECT graph FROM atoms WHERE id = 'a2'"
        )
        row = await cursor.fetchone()
        assert row[0] == "historical"
