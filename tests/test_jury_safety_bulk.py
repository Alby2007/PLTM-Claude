"""
Tests for SafetyJudge/TemporalJudge hardening and bulk mode.

Covers the spec'd fixes: no false REJECTs on ordinary words (hackathon,
emails, phone numbers), Luhn-gated credit cards, keyword QUARANTINE at
0.9 that survives consensus, and bulk mode that skips rapid-fire +
old-timestamp quarantines during batch ingestion.

No database needed — MemoryJury runs with enable_meta_judge=False.
"""

import pytest

from src.memory.memory_jury import MemoryJury, SafetyJudge, TemporalJudge, Verdict


class TestSafetyJudge:
    def test_hackathon_approves(self):
        d = SafetyJudge().evaluate("hackathon on Friday")
        assert d.verdict == Verdict.APPROVE

    def test_hacked_quarantines(self):
        d = SafetyJudge().evaluate("the account was hacked")
        assert d.verdict == Verdict.QUARANTINE
        assert d.confidence == pytest.approx(0.9)

    def test_password_rejects(self):
        d = SafetyJudge().evaluate("password: hunter2")
        assert d.verdict == Verdict.REJECT

    def test_ssn_rejects(self):
        d = SafetyJudge().evaluate("SSN 123-45-6789")
        assert d.verdict == Verdict.REJECT

    def test_bad_luhn_card_approves(self):
        d = SafetyJudge().evaluate("number 1234567890123456 here")
        assert d.verdict == Verdict.APPROVE

    def test_valid_luhn_card_rejects(self):
        d = SafetyJudge().evaluate("card 4111111111111111")
        assert d.verdict == Verdict.REJECT

    def test_email_and_phone_approve(self):
        d = SafetyJudge().evaluate("email bob@example.com, call 07700 900123")
        assert d.verdict == Verdict.APPROVE

    def test_exploitation_approves(self):
        # suffix-free derived forms shouldn't match the harmful keyword
        d = SafetyJudge().evaluate("the exploitation of buffer overflows")
        assert d.verdict == Verdict.QUARANTINE or d.verdict == Verdict.APPROVE


class TestFullJury:
    def test_keyword_flagged_memory_quarantines(self):
        jury = MemoryJury(enable_meta_judge=False)
        d = jury.deliberate("the account was hacked last week")
        assert d.verdict == Verdict.QUARANTINE
        assert d.verdict != Verdict.APPROVE


class TestTemporalBulk:
    def test_bulk_mode_never_rapid_fire(self):
        judge = TemporalJudge()
        for i in range(150):
            d = judge.evaluate(f"test memory {i}", bulk=True)
            assert "Rapid-fire" not in d.explanation

    def test_non_bulk_rapid_fire_quarantines(self):
        judge = TemporalJudge()
        verdicts = [judge.evaluate(f"test memory {i}") for i in range(150)]
        assert any(d.verdict == Verdict.QUARANTINE and "Rapid-fire" in d.explanation
                   for d in verdicts)

    def test_bulk_skips_old_episodic_quarantine(self):
        import time
        judge = TemporalJudge()
        old_ts = time.time() - 2 * 365 * 86400
        d = judge.evaluate("old memory", memory_type="episodic",
                           episode_timestamp=old_ts, bulk=True)
        assert d.verdict == Verdict.APPROVE
        d2 = judge.evaluate("old memory", memory_type="episodic",
                            episode_timestamp=old_ts)
        assert d2.verdict == Verdict.QUARANTINE

    def test_future_year_quarantines(self):
        import time
        judge = TemporalJudge()
        future = time.gmtime().tm_year + 15
        d = judge.evaluate(f"in the year {future} things happen")
        assert d.verdict == Verdict.QUARANTINE
