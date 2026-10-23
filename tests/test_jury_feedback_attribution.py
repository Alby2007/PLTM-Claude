"""
Acceptance tests: memory_id + content_hash attribution in meta_judge_events.

Previously every event row had memory_id = '' because MetaJudge.track ran
before the memory id existed; record_feedback then matched nothing and
ground truth never reached calibration/adaptive weights.
"""

import hashlib
import sqlite3
import time

import pytest

from src.memory.memory_jury import MemoryJury
from src.memory.memory_types import MemoryType, TypedMemory, TypedMemoryStore


def make(mt, **kw):
    now = time.time()
    return TypedMemory(
        id="", memory_type=mt, user_id="t",
        content=kw.pop("content", "water is wet"),
        context="", source="t", strength=kw.get("strength", 1.0),
        created_at=now, last_accessed=kw.get("last_accessed", now),
        confidence=kw.get("confidence", 0.9),
        episode_timestamp=0, participants=[], emotional_valence=0.0,
        trigger="", action="", success_count=0, failure_count=0,
        consolidated_from=[], consolidation_count=0, tags=[],
        evidence_for=[], evidence_against=[])


@pytest.fixture
async def store(tmp_path):
    db = str(tmp_path / "t.db")
    jury = MemoryJury(db_path=db)          # persistent MetaJudge
    s = TypedMemoryStore(db, jury=jury)
    await s.connect()
    yield s, jury
    jury.meta._conn.close()


def events(db_path, memory_id=None, content_hash=None):
    conn = sqlite3.connect(db_path)
    if memory_id:
        rows = conn.execute(
            "SELECT judge_name, verdict, is_correct FROM meta_judge_events "
            "WHERE memory_id = ?", (memory_id,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT judge_name, verdict, is_correct, memory_id "
            "FROM meta_judge_events WHERE content_hash = ?",
            (content_hash,)).fetchall()
    conn.close()
    return rows


class TestMemoryIdAttribution:
    @pytest.mark.asyncio
    async def test_events_carry_stored_memory_id(self, store):
        s, jury = store
        mid = await s.store(make(MemoryType.SEMANTIC))
        assert mid
        rows = events(s.db_path, memory_id=mid)
        judges = {r[0] for r in rows}
        assert judges == {"SafetyJudge", "QualityJudge",
                          "TemporalJudge", "ConsensusJudge"}

    @pytest.mark.asyncio
    async def test_false_positive_finds_verdict_and_attributes(self, store):
        s, jury = store
        mid = await s.store(make(MemoryType.SEMANTIC))
        jury.record_feedback(memory_id=mid, feedback_type="false_positive")

        conn = sqlite3.connect(s.db_path)
        (verdict,) = conn.execute(
            "SELECT original_verdict FROM meta_judge_feedback "
            "WHERE memory_id = ?", (mid,)).fetchone()
        assert verdict == "APPROVE"          # not UNKNOWN

        # every judge that voted APPROVE gets a false_positive
        fp = dict(conn.execute(
            "SELECT judge_name, false_positives FROM meta_judge_stats "
            "WHERE memory_type = '_all'").fetchall())
        conn.close()
        for jn in ("SafetyJudge", "QualityJudge", "TemporalJudge",
                   "ConsensusJudge"):
            assert fp[jn] >= 1

    @pytest.mark.asyncio
    async def test_confirmed_stamps_is_correct(self, store):
        s, jury = store
        mid = await s.store(make(MemoryType.SEMANTIC))
        jury.record_feedback(memory_id=mid, feedback_type="confirmed")
        rows = events(s.db_path, memory_id=mid)
        assert rows
        assert all(r[2] == 1 for r in rows)  # all APPROVEs were right

    @pytest.mark.asyncio
    async def test_calibration_report_activates(self, store):
        s, jury = store
        for i in range(11):
            mid = await s.store(make(
                MemoryType.SEMANTIC,
                content=f"calibration fact number {i} is reliable"))
            jury.record_feedback(memory_id=mid, feedback_type="confirmed")

        report = jury.meta.get_calibration_report()
        assert report
        assert any(v and v["samples"] >= 10 for v in report.values())


class TestContentHashAttribution:
    @pytest.mark.asyncio
    async def test_rejected_memory_matchable_by_content_hash(self, store):
        s, jury = store
        content = "password: hunter2"
        mid = await s.store(make(MemoryType.SEMANTIC, content=content))
        assert mid == ""                    # REJECTed, nothing stored

        h = hashlib.sha1(content.strip().lower().encode()).hexdigest()
        rows = events(s.db_path, content_hash=h)
        assert rows                          # events were logged
        assert {r[1] for r in rows} & {"REJECT"}

        jury.record_feedback(content_hash=h, feedback_type="false_negative")
        conn = sqlite3.connect(s.db_path)
        fb = conn.execute(
            "SELECT original_verdict FROM meta_judge_feedback "
            "WHERE memory_id = ?", (f"hash:{h}",)).fetchone()
        assert fb and fb[0] == "REJECT"      # not UNKNOWN

        fn = dict(conn.execute(
            "SELECT judge_name, false_negatives FROM meta_judge_stats "
            "WHERE memory_type = '_all'").fetchall())
        conn.close()
        assert fn["SafetyJudge"] >= 1        # it voted REJECT
        assert fn["ConsensusJudge"] >= 1


class TestSchemaMigration:
    def test_existing_db_gains_content_hash_column(self, tmp_path):
        db = str(tmp_path / "old.db")
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE meta_judge_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL, judge_name TEXT NOT NULL,
                verdict TEXT NOT NULL, confidence REAL NOT NULL,
                explanation TEXT DEFAULT '', latency_ms REAL DEFAULT 0,
                memory_type TEXT DEFAULT '', memory_id TEXT DEFAULT '',
                ground_truth TEXT DEFAULT NULL, is_correct INTEGER DEFAULT NULL
            )""")
        conn.execute("INSERT INTO meta_judge_events "
                     "(timestamp, judge_name, verdict, confidence) "
                     "VALUES (1.0, 'SafetyJudge', 'APPROVE', 1.0)")
        conn.commit()
        conn.close()

        from src.memory.memory_jury import MetaJudge
        meta = MetaJudge(db_path=db)
        cols = {r[1] for r in meta._conn.execute(
            "PRAGMA table_info(meta_judge_events)")}
        assert "content_hash" in cols
        # pre-existing row intact, default '' hash
        (ch,) = meta._conn.execute(
            "SELECT content_hash FROM meta_judge_events").fetchone()
        assert ch == ""
        meta._conn.close()
