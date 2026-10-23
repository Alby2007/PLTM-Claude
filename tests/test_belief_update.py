"""
Regression tests for the update_belief / strength-decay bug report.

- update_belief("against") must LOWER confidence even without evidence_id
- Editing (update_belief/correct_memory) must not decay stored strength
- get()/query() must not mutate mem.strength to the decayed floor
- Exact-duplicate QualityJudge rejects stay REJECT in consensus
"""

import time

import pytest

from src.memory.memory_jury import MemoryJury, QualityJudge, ConsensusJudge, Verdict
from src.memory.memory_types import MemoryType, TypedMemory, TypedMemoryStore


def make(mt, **kw):
    now = time.time()
    return TypedMemory(
        id="", memory_type=mt, user_id="t", content="water is wet",
        context="", source="t", strength=kw.get("strength", 1.0),
        created_at=now, last_accessed=kw.get("last_accessed", now),
        confidence=kw.get("confidence", 0.9),
        episode_timestamp=0, participants=[], emotional_valence=0.0,
        trigger="", action="", success_count=0, failure_count=0,
        consolidated_from=[], consolidation_count=0, tags=[],
        evidence_for=[], evidence_against=[])


@pytest.fixture
async def store(tmp_path):
    s = TypedMemoryStore(str(tmp_path / "t.db"))
    await s.connect()
    yield s


class TestUpdateBelief:
    @pytest.mark.asyncio
    async def test_against_lowers_confidence_without_evidence_id(self, store):
        mid = await store.store(make(MemoryType.BELIEF, confidence=0.9))
        b = await store.update_belief(mid, evidence_type="against",
                                    evidence_id="", confidence_delta=0.5)
        assert b.confidence == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_against_records_evidence_with_id(self, store):
        mid = await store.store(make(MemoryType.BELIEF, confidence=0.9))
        b = await store.update_belief(mid, evidence_type="against",
                                    evidence_id="mem-ev-1", confidence_delta=0.5)
        assert b.confidence == pytest.approx(0.4)
        assert b.evidence_against == ["mem-ev-1"]

    @pytest.mark.asyncio
    async def test_for_raises_confidence(self, store):
        mid = await store.store(make(MemoryType.BELIEF, confidence=0.5))
        b = await store.update_belief(mid, evidence_type="for",
                                    evidence_id="mem-ev-2", confidence_delta=0.3)
        assert b.confidence == pytest.approx(0.8)
        assert b.evidence_for == ["mem-ev-2"]

    @pytest.mark.asyncio
    async def test_update_belief_preserves_strength(self, store):
        mid = await store.store(make(MemoryType.BELIEF, strength=1.0))
        await store._conn.execute(
            "UPDATE typed_memories SET last_accessed=? WHERE id=?",
            (time.time() - 60 * 86400, mid))
        await store._conn.commit()
        await store.update_belief(mid, evidence_type="for",
                                  evidence_id="e", confidence_delta=0.1)
        cur = await store._conn.execute(
            "SELECT strength FROM typed_memories WHERE id=?", (mid,))
        assert (await cur.fetchone())[0] == pytest.approx(1.0)


class TestStrengthDurability:
    @pytest.mark.asyncio
    async def test_get_does_not_mutate_strength(self, store):
        mid = await store.store(make(MemoryType.BELIEF, strength=1.0,
                                     last_accessed=time.time() - 60 * 86400))
        mem = await store.get(mid)
        assert mem.strength == pytest.approx(1.0)          # stored value
        assert mem.current_strength() < 1.0                # decayed view still available

    @pytest.mark.asyncio
    async def test_correct_memory_preserves_strength(self, store):
        mid = await store.store(make(MemoryType.BELIEF, strength=1.0,
                                     last_accessed=time.time() - 60 * 86400))
        await store.correct_memory(mid, "water is wet at 20C", reason="precision")
        cur = await store._conn.execute(
            "SELECT strength FROM typed_memories WHERE id=?", (mid,))
        assert (await cur.fetchone())[0] == pytest.approx(1.0)


class TestExactDuplicateConsensus:
    def test_exact_duplicate_rejects_not_quarantined(self):
        jury = MemoryJury(enable_meta_judge=False)
        # QualityJudge REJECTs exact duplicates; consensus must keep REJECT
        # so the duplicate is not stored at all.
        d = jury.deliberate("the sky is blue",
                            existing_contents=["the sky is blue"])
        assert d.verdict == Verdict.REJECT

    def test_other_quality_reject_still_quarantines(self):
        # A too-generic memory still hits the "any reject -> quarantine" path.
        q = QualityJudge().evaluate("ok")
        c = ConsensusJudge()
        d = c.aggregate([q])
        assert q.verdict == Verdict.REJECT
        assert d.verdict == Verdict.QUARANTINE
