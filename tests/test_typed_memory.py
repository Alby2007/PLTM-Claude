"""
Unit tests for the TypedMemory system.
Tests store, query, decay, consolidation, jury, and search.
"""

import asyncio
import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory.memory_types import TypedMemory, TypedMemoryStore, MemoryType


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
async def store(db_path):
    s = TypedMemoryStore(db_path)
    await s.connect()
    yield s
    await s.close()


def make_mem(mem_type=MemoryType.SEMANTIC, content="test fact", user_id="test_user", **kw):
    now = time.time()
    return TypedMemory(
        id="",
        memory_type=mem_type,
        user_id=user_id,
        content=content,
        context=kw.get("context", ""),
        source=kw.get("source", "test"),
        strength=kw.get("strength", 1.0),
        created_at=now,
        last_accessed=now,
        confidence=kw.get("confidence", 0.8),
        episode_timestamp=now if mem_type == MemoryType.EPISODIC else 0,
        emotional_valence=kw.get("emotional_valence", 0.0),
        trigger=kw.get("trigger", ""),
        action=kw.get("action", ""),
        tags=kw.get("tags", []),
    )


# ===== STORE TESTS =====

@pytest.mark.asyncio
async def test_store_and_get(store):
    mem = make_mem(content="Python is a programming language")
    mid = await store.store(mem, bypass_jury=True)
    assert mid != ""
    
    retrieved = await store.get(mid)
    assert retrieved is not None
    assert retrieved.content == "Python is a programming language"
    assert retrieved.memory_type == MemoryType.SEMANTIC


@pytest.mark.asyncio
async def test_store_all_types(store):
    types = [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.BELIEF, MemoryType.PROCEDURAL]
    for mt in types:
        mem = make_mem(mem_type=mt, content=f"test {mt.value}")
        mid = await store.store(mem, bypass_jury=True)
        assert mid != "", f"Failed to store {mt.value}"
        got = await store.get(mid)
        assert got.memory_type == mt


@pytest.mark.asyncio
async def test_store_returns_empty_on_jury_reject(db_path):
    """Jury should reject obviously bad content."""
    from src.memory.memory_jury import MemoryJury
    jury = MemoryJury(enable_meta_judge=False, db_path=db_path)
    s = TypedMemoryStore(db_path, jury=jury)
    await s.connect()
    
    # Very short content should be rejected
    mem = make_mem(content="x")
    mid = await s.store(mem)
    # Either rejected (empty) or quarantined (non-empty but weakened)
    assert mid == "" or mem.strength < 1.0
    await s.close()


# ===== QUERY TESTS =====

@pytest.mark.asyncio
async def test_query_by_type(store):
    await store.store(make_mem(MemoryType.SEMANTIC, "fact one"), bypass_jury=True)
    await store.store(make_mem(MemoryType.EPISODIC, "event one"), bypass_jury=True)
    await store.store(make_mem(MemoryType.SEMANTIC, "fact two"), bypass_jury=True)
    
    semantics = await store.query("test_user", MemoryType.SEMANTIC)
    assert len(semantics) == 2
    
    episodics = await store.query("test_user", MemoryType.EPISODIC)
    assert len(episodics) == 1


@pytest.mark.asyncio
async def test_query_by_tags(store):
    await store.store(make_mem(content="python tip", tags=["python", "coding"]), bypass_jury=True)
    await store.store(make_mem(content="rust tip", tags=["rust", "coding"]), bypass_jury=True)
    await store.store(make_mem(content="cooking recipe", tags=["cooking"]), bypass_jury=True)
    
    coding = await store.query("test_user", tags=["coding"])
    assert len(coding) == 2
    
    python = await store.query("test_user", tags=["python"])
    assert len(python) == 1


@pytest.mark.asyncio
async def test_query_min_strength(store):
    await store.store(make_mem(content="strong", strength=0.9), bypass_jury=True)
    await store.store(make_mem(content="weak", strength=0.05), bypass_jury=True)
    
    strong = await store.query("test_user", min_strength=0.5)
    assert len(strong) == 1
    assert strong[0].content == "strong"


# ===== DECAY TESTS =====

def test_decay_curve():
    """Episodic memories should decay faster than semantic."""
    now = time.time()
    ep = TypedMemory(
        id="ep1", memory_type=MemoryType.EPISODIC, user_id="u",
        content="event", context="", source="", strength=1.0,
        created_at=now - 172800, last_accessed=now - 172800,  # 2 days ago
        confidence=0.8, episode_timestamp=now - 172800,
    )
    sem = TypedMemory(
        id="sem1", memory_type=MemoryType.SEMANTIC, user_id="u",
        content="fact", context="", source="", strength=1.0,
        created_at=now - 172800, last_accessed=now - 172800,
        confidence=0.8,
    )
    
    ep_str = ep.current_strength()
    sem_str = sem.current_strength()
    
    assert ep_str < sem_str, "Episodic should decay faster than semantic"
    assert ep_str < 1.0, "Episodic should have decayed"
    assert sem_str < 1.0, "Semantic should have decayed some"


# ===== STATS TESTS =====

@pytest.mark.asyncio
async def test_get_stats(store):
    await store.store(make_mem(MemoryType.SEMANTIC, "s1"), bypass_jury=True)
    await store.store(make_mem(MemoryType.SEMANTIC, "s2"), bypass_jury=True)
    await store.store(make_mem(MemoryType.BELIEF, "b1", confidence=0.6), bypass_jury=True)
    
    stats = await store.get_stats("test_user")
    assert stats["total"] == 3
    assert "semantic" in stats
    assert stats["semantic"]["count"] == 2
    assert "belief" in stats
    assert stats["belief"]["count"] == 1


# ===== SEARCH TESTS =====

@pytest.mark.asyncio
async def test_fts_search(store):
    await store.store(make_mem(content="Python is great for data science"), bypass_jury=True)
    await store.store(make_mem(content="Rust is fast for systems programming"), bypass_jury=True)
    
    results = await store.search("test_user", "Python")
    assert len(results) >= 1
    assert any("Python" in r.content for r in results)


# ===== BELIEF UPDATE TESTS =====

@pytest.mark.asyncio
async def test_update_belief(store):
    mem = make_mem(MemoryType.BELIEF, "AI will surpass humans", confidence=0.5)
    mid = await store.store(mem, bypass_jury=True)
    
    updated = await store.update_belief(mid, "for", "evidence1", 0.1)
    assert updated is not None
    assert updated.confidence > 0.5
    assert len(updated.evidence_for) == 1


# ===== PROCEDURAL TESTS =====

@pytest.mark.asyncio
async def test_record_procedure_outcome(store):
    mem = make_mem(MemoryType.PROCEDURAL, "when X do Y", trigger="X", action="Y")
    mid = await store.store(mem, bypass_jury=True)
    
    proc = await store.record_procedure_outcome(mid, success=True)
    assert proc is not None
    assert proc.success_count == 1
    
    proc = await store.record_procedure_outcome(mid, success=False)
    assert proc.failure_count == 1
    assert proc.success_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
