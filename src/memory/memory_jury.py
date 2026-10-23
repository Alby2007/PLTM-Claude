"""
Memory Jury System — Judge/Jury validation gate for TypedMemory storage.

Adapted from the LLTM 4-judge jury system for typed memories:
  1. SafetyJudge  — PII, harmful content detection (VETO authority)
  2. QualityJudge — semantic validity, deduplication, content quality
  3. TemporalJudge — temporal consistency, rapid-change detection
  4. ConsensusJudge — weighted aggregation of all judge verdicts

Plus MetaJudge observability layer tracking judge accuracy over time.

Verdicts: APPROVE | QUARANTINE | REJECT
- APPROVE: memory stored normally
- QUARANTINE: memory stored with reduced strength + flagged for review
- REJECT: memory not stored, reason logged
"""

import hashlib
import re
import time
import asyncio
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from loguru import logger


# ============================================================
# VERDICT MODEL
# ============================================================

class Verdict(str, Enum):
    APPROVE = "APPROVE"
    QUARANTINE = "QUARANTINE"
    REJECT = "REJECT"


@dataclass
class JudgeDecision:
    verdict: Verdict
    confidence: float  # 0.0-1.0
    explanation: str
    judge_name: str


# ============================================================
# SAFETY JUDGE — PII + harmful content (VETO authority)
# ============================================================

class SafetyJudge:
    """Blocks real credentials/identifiers; flags (not blocks) sensitive topics."""

    # Hard-block patterns only
    PII_PATTERNS = [
        (r'\b\d{3}-\d{2}-\d{4}\b', "SSN"),
        (r'\b(?:password|passwd|pwd)\s*[:=]\s*\S+', "Password"),
    ]
    CARD_PATTERN = re.compile(r'\b(?:\d[ -]?){15}\d\b')

    HARMFUL_KEYWORDS = [
        "hack", "exploit", "crack", "pirate", "steal",
        "illegal", "fraud", "scam", "phishing",
        "weapon", "bomb", "explosive", "poison",
        "suicide", "self-harm", "overdose",
    ]
    # Whole word, allowing simple suffixes: matches "hacked", not "hackathon"/"exploitation"
    _HARMFUL_RE = re.compile(
        r'\b(?:' + '|'.join(re.escape(k) for k in HARMFUL_KEYWORDS) + r')(?:s|ed|ing|er|ers)?\b',
        re.IGNORECASE,
    )

    @staticmethod
    def _luhn_ok(number: str) -> bool:
        digits = [int(c) for c in number if c.isdigit()]
        total, alt = 0, False
        for d in reversed(digits):
            if alt:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
            alt = not alt
        return len(digits) >= 13 and total % 10 == 0

    def evaluate(self, content: str, **kwargs) -> JudgeDecision:
        for pattern, pii_type in self.PII_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return JudgeDecision(Verdict.REJECT, 1.0, f"PII detected: {pii_type}", "SafetyJudge")

        m = self.CARD_PATTERN.search(content)
        if m and self._luhn_ok(m.group(0)):
            return JudgeDecision(Verdict.REJECT, 1.0, "PII detected: Credit Card", "SafetyJudge")

        m = self._HARMFUL_RE.search(content)
        if m:
            # QUARANTINE at 0.9 so the weighted vote cannot outvote it back to APPROVE
            return JudgeDecision(Verdict.QUARANTINE, 0.9,
                                 f"Sensitive keyword: {m.group(0).lower()}", "SafetyJudge")

        if len(content) > 2000:
            return JudgeDecision(Verdict.QUARANTINE, 0.9,
                                 "Excessively long content (>2000 chars)", "SafetyJudge")

        return JudgeDecision(Verdict.APPROVE, 1.0, "No safety concerns", "SafetyJudge")


# ============================================================
# QUALITY JUDGE — semantic validity, dedup, content quality
# ============================================================

class QualityJudge:
    """Validates content quality, semantic sense, and detects duplicates."""

    GENERIC_CONTENT = {"something", "anything", "nothing", "stuff", "things", "ok", "yes", "no", "idk"}

    def evaluate(self, content: str, memory_type: str = "",
                 existing_contents: Optional[List[str]] = None, **kwargs) -> JudgeDecision:
        stripped = content.strip()

        # Too short
        if len(stripped) < 3:
            return JudgeDecision(
                verdict=Verdict.REJECT,
                confidence=1.0,
                explanation="Content too short (<3 chars)",
                judge_name="QualityJudge",
            )

        # Too generic
        if stripped.lower() in self.GENERIC_CONTENT:
            return JudgeDecision(
                verdict=Verdict.REJECT,
                confidence=0.95,
                explanation=f"Content too generic: '{stripped}'",
                judge_name="QualityJudge",
            )

        # Repeated words only
        words = stripped.lower().split()
        if len(words) > 1 and len(set(words)) == 1:
            return JudgeDecision(
                verdict=Verdict.REJECT,
                confidence=1.0,
                explanation="Content is just repeated words",
                judge_name="QualityJudge",
            )

        # Internal contradiction check (for beliefs)
        if memory_type == "belief":
            contradictions = [
                (["like", "love", "enjoy", "prefer"], ["hate", "dislike", "avoid", "detest"]),
                (["good", "great", "excellent"], ["bad", "terrible", "awful"]),
                (["always"], ["never"]),
            ]
            cl = content.lower()
            for pos_words, neg_words in contradictions:
                has_pos = any(w in cl for w in pos_words)
                has_neg = any(w in cl for w in neg_words)
                if has_pos and has_neg:
                    return JudgeDecision(
                        verdict=Verdict.QUARANTINE,
                        confidence=0.8,
                        explanation="Internal contradiction detected in content",
                        judge_name="QualityJudge",
                    )

        # Near-duplicate detection
        if existing_contents:
            content_lower = content.lower().strip()
            for existing in existing_contents:
                if existing.lower().strip() == content_lower:
                    return JudgeDecision(
                        verdict=Verdict.REJECT,
                        confidence=1.0,
                        explanation="Exact duplicate of existing memory",
                        judge_name="QualityJudge",
                    )
                # Simple overlap check (>80% word overlap = near-duplicate)
                existing_words = set(existing.lower().split())
                new_words = set(content.lower().split())
                if existing_words and new_words:
                    overlap = len(existing_words & new_words) / max(len(existing_words), len(new_words))
                    if overlap > 0.85:
                        return JudgeDecision(
                            verdict=Verdict.QUARANTINE,
                            confidence=0.75,
                            explanation=f"Near-duplicate ({overlap:.0%} word overlap)",
                            judge_name="QualityJudge",
                        )

        return JudgeDecision(
            verdict=Verdict.APPROVE,
            confidence=0.9,
            explanation="Content quality acceptable",
            judge_name="QualityJudge",
        )


# ============================================================
# TEMPORAL JUDGE — temporal consistency
# ============================================================

class TemporalJudge:
    """Validates temporal consistency and detects rapid changes."""

    def __init__(self, max_rapid_stores: int = 100):
        self._recent_stores: List[Tuple[float, str]] = []
        self.max_rapid_stores = max_rapid_stores
        self.rapid_window_seconds = 60

    def evaluate(self, content: str, memory_type: str = "",
                 episode_timestamp: float = 0, bulk: bool = False, **kwargs) -> JudgeDecision:
        now = time.time()

        if not bulk:
            self._recent_stores.append((now, content[:50]))
            cutoff = now - self.rapid_window_seconds
            self._recent_stores = [(t, c) for t, c in self._recent_stores if t > cutoff]
            if len(self._recent_stores) > self.max_rapid_stores:
                return JudgeDecision(Verdict.QUARANTINE, 0.8,
                    f"Rapid-fire storage: {len(self._recent_stores)} stores in {self.rapid_window_seconds}s",
                    "TemporalJudge")

        if memory_type == "episodic" and episode_timestamp > 0:
            if episode_timestamp > now + 3600:
                return JudgeDecision(Verdict.REJECT, 0.95,
                    "Episodic memory has future timestamp", "TemporalJudge")
            if not bulk and episode_timestamp < now - 365 * 86400:
                return JudgeDecision(Verdict.QUARANTINE, 0.7,
                    "Episodic memory timestamp >1 year old", "TemporalJudge")

        year_match = re.search(r'\b(19|20)\d{2}\b', content)
        if year_match:
            year = int(year_match.group(0))
            current_year = time.gmtime().tm_year   # was hardcoded 2026
            if year > current_year + 10:
                return JudgeDecision(Verdict.QUARANTINE, 0.85,
                    f"Implausible future year: {year}", "TemporalJudge")

        return JudgeDecision(Verdict.APPROVE, 0.9, "Temporally consistent", "TemporalJudge")


# ============================================================
# CONSENSUS JUDGE — aggregates all verdicts
# ============================================================

class ConsensusJudge:
    """
    Aggregates judge decisions using weighted voting.
    Supports adaptive weights from MetaJudge.

    Rules:
    1. Safety REJECT → immediate REJECT (veto)
    2. Unanimous APPROVE → APPROVE
    3. Any REJECT → QUARANTINE
    4. Majority QUARANTINE → QUARANTINE
    5. Split → confidence-weighted vote (with adaptive judge weights)
    6. Tie → QUARANTINE (err on caution)
    """

    REJECT_WEIGHT = 2.0  # REJECTs count double
    MIN_APPROVE_CONFIDENCE = 0.7

    def __init__(self):
        self._adaptive_weights: Dict[str, float] = {}

    def set_adaptive_weights(self, weights: Dict[str, float]):
        """Update judge weights from MetaJudge adaptive weighting."""
        self._adaptive_weights = weights

    def _get_weight(self, judge_name: str) -> float:
        return self._adaptive_weights.get(judge_name, 1.0)

    def aggregate(self, decisions: List[JudgeDecision]) -> JudgeDecision:
        if not decisions:
            return JudgeDecision(
                verdict=Verdict.REJECT,
                confidence=1.0,
                explanation="No judge decisions available",
                judge_name="ConsensusJudge",
            )

        # Safety veto check
        safety_rejects = [d for d in decisions if d.judge_name == "SafetyJudge" and d.verdict == Verdict.REJECT]
        if safety_rejects:
            return JudgeDecision(
                verdict=Verdict.REJECT,
                confidence=1.0,
                explanation=f"Safety veto: {safety_rejects[0].explanation}",
                judge_name="ConsensusJudge",
            )

        counts = {Verdict.APPROVE: 0, Verdict.QUARANTINE: 0, Verdict.REJECT: 0}
        for d in decisions:
            counts[d.verdict] += 1

        # Unanimous approve
        if counts[Verdict.APPROVE] == len(decisions):
            avg_conf = sum(d.confidence for d in decisions) / len(decisions)
            return JudgeDecision(
                verdict=Verdict.APPROVE,
                confidence=avg_conf,
                explanation="Unanimous approval",
                judge_name="ConsensusJudge",
            )

        # Any reject → quarantine — except unambiguous exact duplicates,
        # which are plain dedup (not a content concern needing review).
        if counts[Verdict.REJECT] > 0:
            rejects = [d for d in decisions if d.verdict == Verdict.REJECT]
            if all(d.judge_name == "QualityJudge" and
                   d.explanation.startswith("Exact duplicate") for d in rejects):
                return JudgeDecision(
                    verdict=Verdict.REJECT,
                    confidence=max(d.confidence for d in rejects),
                    explanation=f"Exact duplicate — not stored: {rejects[0].explanation}",
                    judge_name="ConsensusJudge",
                )
            return JudgeDecision(
                verdict=Verdict.QUARANTINE,
                confidence=max(d.confidence for d in rejects),
                explanation=f"Flagged: {rejects[0].explanation}",
                judge_name="ConsensusJudge",
            )

        # Majority quarantine
        if counts[Verdict.QUARANTINE] > len(decisions) / 2:
            quarantines = [d for d in decisions if d.verdict == Verdict.QUARANTINE]
            return JudgeDecision(
                verdict=Verdict.QUARANTINE,
                confidence=sum(d.confidence for d in quarantines) / len(quarantines),
                explanation=f"Majority quarantine: {quarantines[0].explanation}",
                judge_name="ConsensusJudge",
            )

        # Weighted vote (using adaptive weights from MetaJudge)
        approve_w = sum(d.confidence * self._get_weight(d.judge_name)
                        for d in decisions if d.verdict == Verdict.APPROVE)
        quarantine_w = sum(d.confidence * self._get_weight(d.judge_name)
                          for d in decisions if d.verdict == Verdict.QUARANTINE)
        reject_w = sum(d.confidence * self._get_weight(d.judge_name) * self.REJECT_WEIGHT
                       for d in decisions if d.verdict == Verdict.REJECT)
        total = approve_w + quarantine_w + reject_w

        if total == 0:
            return JudgeDecision(verdict=Verdict.QUARANTINE, confidence=0.5,
                                 explanation="No weight", judge_name="ConsensusJudge")

        if approve_w / total >= self.MIN_APPROVE_CONFIDENCE:
            return JudgeDecision(
                verdict=Verdict.APPROVE,
                confidence=approve_w / total,
                explanation="Weighted vote: approved",
                judge_name="ConsensusJudge",
            )

        return JudgeDecision(
            verdict=Verdict.QUARANTINE,
            confidence=max(quarantine_w, reject_w) / total,
            explanation="Weighted vote: needs review",
            judge_name="ConsensusJudge",
        )


# ============================================================
# META JUDGE — learning observability layer
# ============================================================

import sqlite3
import math
import json as _json


class MetaJudge:
    """
    Advanced MetaJudge with 7 capabilities:
    1. Persistent SQLite stats — survive restarts
    2. Ground truth feedback loop — learn from user corrections
    3. Adaptive judge weighting — feed accuracy back to ConsensusJudge
    4. Per-memory-type accuracy — track per episodic/semantic/belief/procedural
    5. Calibration scoring — confidence vs actual accuracy
    6. Drift detection — alert on sudden behavior changes
    7. Threshold tuning recommendations — suggest parameter adjustments
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._in_memory_history: List[Dict] = []
        self._drift_window_size = 50  # compare last N vs previous N
        self._drift_threshold = 0.15  # 15% change = drift alert
        self._calibration_bins = 10  # 10 bins for calibration curve
        if db_path:
            self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta_judge_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                judge_name TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL NOT NULL,
                explanation TEXT DEFAULT '',
                latency_ms REAL DEFAULT 0,
                memory_type TEXT DEFAULT '',
                memory_id TEXT DEFAULT '',
                content_hash TEXT DEFAULT '',
                ground_truth TEXT DEFAULT NULL,
                is_correct INTEGER DEFAULT NULL
            );

            CREATE TABLE IF NOT EXISTS meta_judge_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                memory_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                original_verdict TEXT NOT NULL,
                original_judge TEXT DEFAULT '',
                details TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS meta_judge_stats (
                judge_name TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT '_all',
                total INTEGER DEFAULT 0,
                correct INTEGER DEFAULT 0,
                false_positives INTEGER DEFAULT 0,
                false_negatives INTEGER DEFAULT 0,
                conflicts_detected INTEGER DEFAULT 0,
                total_latency_ms REAL DEFAULT 0,
                confidence_sum REAL DEFAULT 0,
                verdict_approve INTEGER DEFAULT 0,
                verdict_quarantine INTEGER DEFAULT 0,
                verdict_reject INTEGER DEFAULT 0,
                PRIMARY KEY (judge_name, memory_type)
            );

        """)
        # Guarded migration for pre-existing DBs lacking content_hash
        # (must precede the index on that column).
        cols = {r[1] for r in self._conn.execute(
            "PRAGMA table_info(meta_judge_events)")}
        if "content_hash" not in cols:
            self._conn.execute(
                "ALTER TABLE meta_judge_events "
                "ADD COLUMN content_hash TEXT DEFAULT ''")
        self._conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_mj_events_judge
            ON meta_judge_events(judge_name, timestamp);

            CREATE INDEX IF NOT EXISTS idx_mj_events_memory
            ON meta_judge_events(memory_id);

            CREATE INDEX IF NOT EXISTS idx_mj_events_hash
            ON meta_judge_events(content_hash);

            CREATE INDEX IF NOT EXISTS idx_mj_feedback_memory
            ON meta_judge_feedback(memory_id);
        """)
        self._conn.commit()
        logger.info("MetaJudge persistent DB initialized")

    def _ensure_stats_row(self, judge_name: str, memory_type: str = "_all"):
        if not self._conn:
            return
        self._conn.execute("""
            INSERT OR IGNORE INTO meta_judge_stats (judge_name, memory_type)
            VALUES (?, ?)
        """, (judge_name, memory_type))

    # ── CORE TRACKING ──

    def track(self, decision: JudgeDecision, latency_ms: float = 0.0,
              memory_type: str = "", memory_id: str = "",
              ground_truth: Optional[Verdict] = None,
              content_hash: str = ""):
        """Track a judge evaluation with optional memory type and ground truth."""
        now = time.time()
        is_correct = None
        if ground_truth is not None:
            is_correct = 1 if decision.verdict == ground_truth else 0

        event = {
            "judge": decision.judge_name,
            "verdict": decision.verdict.value,
            "confidence": decision.confidence,
            "explanation": decision.explanation,
            "latency_ms": latency_ms,
            "memory_type": memory_type,
            "memory_id": memory_id,
            "content_hash": content_hash,
            "timestamp": now,
        }
        self._in_memory_history.append(event)
        if len(self._in_memory_history) > 5000:
            self._in_memory_history = self._in_memory_history[-2500:]

        if self._conn:
            self._conn.execute("""
                INSERT INTO meta_judge_events
                (timestamp, judge_name, verdict, confidence, explanation,
                 latency_ms, memory_type, memory_id, content_hash,
                 ground_truth, is_correct)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (now, decision.judge_name, decision.verdict.value,
                  decision.confidence, decision.explanation, latency_ms,
                  memory_type, memory_id, content_hash,
                  ground_truth.value if ground_truth else None, is_correct))

            # Update aggregate stats for _all and specific memory_type
            for mt in ["_all", memory_type] if memory_type else ["_all"]:
                self._ensure_stats_row(decision.judge_name, mt)
                verdict_col = f"verdict_{decision.verdict.value.lower()}"
                self._conn.execute(f"""
                    UPDATE meta_judge_stats SET
                        total = total + 1,
                        correct = correct + COALESCE(?, 0),
                        conflicts_detected = conflicts_detected + ?,
                        total_latency_ms = total_latency_ms + ?,
                        confidence_sum = confidence_sum + ?,
                        {verdict_col} = {verdict_col} + 1
                    WHERE judge_name = ? AND memory_type = ?
                """, (
                    is_correct or 0,
                    1 if decision.verdict in (Verdict.REJECT, Verdict.QUARANTINE) else 0,
                    latency_ms, decision.confidence,
                    decision.judge_name, mt,
                ))
            self._conn.commit()

    # ── GROUND TRUTH FEEDBACK ──

    def record_feedback(self, memory_id: str = "", feedback_type: str = "",
                        details: str = "", content_hash: str = ""):
        """
        Record ground truth feedback from user actions.

        Match events by memory_id and/or content_hash. content_hash makes
        feedback on REJECTed memories (never assigned a stored id) matchable
        when the same content is re-submitted.

        feedback_type:
          - 'false_positive': memory was approved but user deleted/corrected it
          - 'false_negative': memory was rejected but user re-submitted it
          - 'confirmed': user explicitly confirmed memory is correct
        """
        if not self._conn:
            return

        # Build event-match clause from whichever keys were provided.
        # Empty keys must not match (legacy rows have memory_id = '').
        conds, params = [], []
        if memory_id:
            conds.append("memory_id = ?")
            params.append(memory_id)
        if content_hash:
            conds.append("content_hash = ?")
            params.append(content_hash)
        if not conds:
            return
        where = "(" + " OR ".join(conds) + ")"

        # Find the original verdict for this memory
        row = self._conn.execute(f"""
            SELECT judge_name, verdict FROM meta_judge_events
            WHERE {where} AND judge_name = 'ConsensusJudge'
            ORDER BY timestamp DESC LIMIT 1
        """, params).fetchone()

        original_verdict = row[1] if row else "UNKNOWN"
        original_judge = row[0] if row else ""

        self._conn.execute("""
            INSERT INTO meta_judge_feedback
            (timestamp, memory_id, feedback_type, original_verdict, original_judge, details)
            VALUES (?,?,?,?,?,?)
        """, (time.time(), memory_id or f"hash:{content_hash}",
              feedback_type, original_verdict, original_judge, details))

        # Update false positive/negative counts for all judges that evaluated this memory
        if feedback_type == "false_positive":
            judges = self._conn.execute(f"""
                SELECT DISTINCT judge_name FROM meta_judge_events
                WHERE {where} AND verdict = 'APPROVE'
            """, params).fetchall()
            for (jn,) in judges:
                self._ensure_stats_row(jn)
                self._conn.execute("""
                    UPDATE meta_judge_stats SET false_positives = false_positives + 1
                    WHERE judge_name = ? AND memory_type = '_all'
                """, (jn,))

        elif feedback_type == "false_negative":
            judges = self._conn.execute(f"""
                SELECT DISTINCT judge_name FROM meta_judge_events
                WHERE {where} AND verdict IN ('REJECT', 'QUARANTINE')
            """, params).fetchall()
            for (jn,) in judges:
                self._ensure_stats_row(jn)
                self._conn.execute("""
                    UPDATE meta_judge_stats SET false_negatives = false_negatives + 1
                    WHERE judge_name = ? AND memory_type = '_all'
                """, (jn,))

        # Stamp per-event ground truth so calibration/accuracy stats activate.
        # confirmed: APPROVE verdicts were right. false_positive: APPROVE was
        # wrong (user deleted/corrected). false_negative: REJECT/QUARANTINE
        # was wrong (user re-submitted).
        correct_map = {
            "confirmed":      ("verdict = 'APPROVE'", "verdict != 'APPROVE'"),
            "false_positive": ("verdict != 'APPROVE'", "verdict = 'APPROVE'"),
            "false_negative": ("verdict = 'APPROVE'", "verdict != 'APPROVE'"),
        }
        if feedback_type in correct_map:
            right_cond, wrong_cond = correct_map[feedback_type]
            self._conn.execute(
                f"UPDATE meta_judge_events SET is_correct = 1 "
                f"WHERE {where} AND {right_cond}", params)
            self._conn.execute(
                f"UPDATE meta_judge_events SET is_correct = 0 "
                f"WHERE {where} AND {wrong_cond}", params)

        self._conn.commit()
        logger.info(
            f"MetaJudge feedback: {feedback_type} for "
            f"{memory_id[:8] if memory_id else 'hash:' + content_hash[:8]}")

    # ── ADAPTIVE JUDGE WEIGHTING ──

    def get_adaptive_weights(self) -> Dict[str, float]:
        """
        Compute adaptive weights for each judge based on historical accuracy.
        Higher accuracy → higher weight in ConsensusJudge.
        Returns dict of judge_name → weight (0.5 to 2.0).
        """
        weights = {}
        base_weight = 1.0

        if not self._conn:
            return {"SafetyJudge": 1.0, "QualityJudge": 1.0, "TemporalJudge": 1.0}

        rows = self._conn.execute("""
            SELECT judge_name, total, correct, false_positives, false_negatives
            FROM meta_judge_stats WHERE memory_type = '_all'
            AND judge_name != 'ConsensusJudge'
        """).fetchall()

        for judge_name, total, correct, fp, fn in rows:
            if total < 10:
                weights[judge_name] = base_weight
                continue

            # Accuracy-based weight
            error_rate = (fp + fn) / total if total > 0 else 0
            accuracy_weight = max(0.5, min(2.0, base_weight * (1.0 - error_rate * 2)))

            # Calibration penalty: overconfident judges get penalized
            cal = self._get_calibration_for_judge(judge_name)
            if cal and cal["ece"] > 0.15:
                accuracy_weight *= 0.85  # 15% penalty for poor calibration

            weights[judge_name] = round(accuracy_weight, 3)

        # Ensure all judges have a weight
        for jn in ["SafetyJudge", "QualityJudge", "TemporalJudge"]:
            if jn not in weights:
                weights[jn] = base_weight

        return weights

    # ── PER-MEMORY-TYPE ACCURACY ──

    def get_per_type_accuracy(self) -> Dict[str, Dict[str, Any]]:
        """Get judge accuracy broken down by memory type."""
        if not self._conn:
            return {}

        rows = self._conn.execute("""
            SELECT judge_name, memory_type, total, correct,
                   false_positives, false_negatives, confidence_sum
            FROM meta_judge_stats
            WHERE memory_type != '_all' AND total > 0
            ORDER BY judge_name, memory_type
        """).fetchall()

        result: Dict[str, Dict] = {}
        for jn, mt, total, correct, fp, fn, conf_sum in rows:
            if jn not in result:
                result[jn] = {}
            result[jn][mt] = {
                "total": total,
                "accuracy": round(correct / total, 3) if correct else None,
                "false_positive_rate": round(fp / total, 3) if total else 0,
                "false_negative_rate": round(fn / total, 3) if total else 0,
                "avg_confidence": round(conf_sum / total, 3),
            }
        return result

    # ── CALIBRATION SCORING ──

    def _get_calibration_for_judge(self, judge_name: str) -> Optional[Dict]:
        """Compute Expected Calibration Error for a judge."""
        if not self._conn:
            return None

        rows = self._conn.execute("""
            SELECT confidence, is_correct FROM meta_judge_events
            WHERE judge_name = ? AND is_correct IS NOT NULL
            ORDER BY timestamp DESC LIMIT 500
        """, (judge_name,)).fetchall()

        if len(rows) < 10:
            return None

        # Bin by confidence
        bins = [[] for _ in range(self._calibration_bins)]
        for conf, correct in rows:
            bin_idx = min(int(conf * self._calibration_bins), self._calibration_bins - 1)
            bins[bin_idx].append((conf, correct))

        ece = 0.0
        total_samples = len(rows)
        bin_data = []
        for i, b in enumerate(bins):
            if not b:
                continue
            avg_conf = sum(c for c, _ in b) / len(b)
            avg_acc = sum(c for _, c in b) / len(b)
            gap = abs(avg_conf - avg_acc)
            ece += (len(b) / total_samples) * gap
            bin_data.append({
                "bin": f"{i/self._calibration_bins:.1f}-{(i+1)/self._calibration_bins:.1f}",
                "count": len(b),
                "avg_confidence": round(avg_conf, 3),
                "avg_accuracy": round(avg_acc, 3),
                "gap": round(gap, 3),
            })

        return {
            "ece": round(ece, 4),
            "samples": len(rows),
            "bins": bin_data,
            "assessment": "well_calibrated" if ece < 0.05 else
                          "slightly_miscalibrated" if ece < 0.15 else "poorly_calibrated",
        }

    def get_calibration_report(self) -> Dict[str, Any]:
        """Get calibration report for all judges."""
        if not self._conn:
            return {}
        judges = [r[0] for r in self._conn.execute(
            "SELECT DISTINCT judge_name FROM meta_judge_stats WHERE memory_type='_all'"
        ).fetchall()]
        return {jn: self._get_calibration_for_judge(jn) for jn in judges}

    # ── DRIFT DETECTION ──

    def detect_drift(self) -> List[Dict[str, Any]]:
        """
        Detect behavioral drift in judges.
        Compares REJECT rate in the last 24h vs the previous 7 days.
        """
        if not self._conn:
            return []
        now = time.time()
        recent_start, prev_start = now - 86400, now - 8 * 86400   # last 24h vs previous 7 days
        alerts = []
        judges = [r[0] for r in self._conn.execute(
            "SELECT DISTINCT judge_name FROM meta_judge_events").fetchall()]
        for jn in judges:
            rows = self._conn.execute(
                "SELECT verdict, timestamp FROM meta_judge_events "
                "WHERE judge_name = ? AND timestamp >= ?", (jn, prev_start)).fetchall()
            recent = [v for v, t in rows if t >= recent_start]
            prev = [v for v, t in rows if t < recent_start]
            if len(recent) < self._drift_window_size or len(prev) < self._drift_window_size:
                continue
            rr = sum(v == "REJECT" for v in recent) / len(recent)
            pr = sum(v == "REJECT" for v in prev) / len(prev)
            drift = abs(rr - pr)
            if drift > self._drift_threshold:
                alerts.append({
                    "judge": jn, "drift": round(drift, 3),
                    "direction": "more_restrictive" if rr > pr else "more_permissive",
                    "recent_reject_rate": round(rr, 3), "previous_reject_rate": round(pr, 3),
                    "severity": "high" if drift > 0.3 else "medium",
                })
        return alerts

    # ── THRESHOLD TUNING RECOMMENDATIONS ──

    def get_tuning_recommendations(self) -> List[Dict[str, Any]]:
        """
        Analyze judge performance and recommend threshold adjustments.
        """
        recommendations = []

        if not self._conn:
            return recommendations

        rows = self._conn.execute("""
            SELECT judge_name, total, false_positives, false_negatives,
                   verdict_approve, verdict_quarantine, verdict_reject
            FROM meta_judge_stats WHERE memory_type = '_all' AND total >= 20
        """).fetchall()

        for jn, total, fp, fn, v_app, v_quar, v_rej in rows:
            if jn == "ConsensusJudge":
                continue

            fp_rate = fp / total if total else 0
            fn_rate = fn / total if total else 0
            reject_rate = v_rej / total if total else 0

            # Too many false positives → judge is too permissive
            if fp_rate > 0.1:
                recommendations.append({
                    "judge": jn,
                    "issue": "high_false_positive_rate",
                    "rate": round(fp_rate, 3),
                    "recommendation": f"{jn} approved {fp} memories that were later corrected/deleted. Consider tightening thresholds.",
                    "priority": "high" if fp_rate > 0.2 else "medium",
                })

            # Too many false negatives → judge is too strict
            if fn_rate > 0.1:
                recommendations.append({
                    "judge": jn,
                    "issue": "high_false_negative_rate",
                    "rate": round(fn_rate, 3),
                    "recommendation": f"{jn} rejected {fn} memories that users re-submitted. Consider loosening thresholds.",
                    "priority": "high" if fn_rate > 0.2 else "medium",
                })

            # Judge almost never rejects → might be useless
            if reject_rate < 0.02 and total > 50:
                recommendations.append({
                    "judge": jn,
                    "issue": "rubber_stamp",
                    "rate": round(reject_rate, 3),
                    "recommendation": f"{jn} approves {100*(1-reject_rate):.0f}% of memories. May not be adding value.",
                    "priority": "low",
                })

            # Judge rejects too much → too aggressive
            if reject_rate > 0.5:
                recommendations.append({
                    "judge": jn,
                    "issue": "too_aggressive",
                    "rate": round(reject_rate, 3),
                    "recommendation": f"{jn} rejects/quarantines {100*reject_rate:.0f}% of memories. May be too strict.",
                    "priority": "high",
                })

        return recommendations

    # ── REPORTS ──

    def get_report(self) -> Dict[str, Any]:
        """Full MetaJudge report with all capabilities."""
        report: Dict[str, Any] = {"judges": {}}

        if self._conn:
            rows = self._conn.execute("""
                SELECT judge_name, total, correct, false_positives, false_negatives,
                       conflicts_detected, total_latency_ms, confidence_sum,
                       verdict_approve, verdict_quarantine, verdict_reject
                FROM meta_judge_stats WHERE memory_type = '_all'
            """).fetchall()

            for (jn, total, correct, fp, fn, conflicts, lat, conf,
                 v_app, v_quar, v_rej) in rows:
                t = total or 1
                report["judges"][jn] = {
                    "total_evaluations": total,
                    "accuracy": round(correct / t, 3) if correct else None,
                    "false_positive_rate": round(fp / t, 3),
                    "false_negative_rate": round(fn / t, 3),
                    "avg_confidence": round(conf / t, 3),
                    "avg_latency_ms": round(lat / t, 2),
                    "conflict_rate": round(conflicts / t, 3),
                    "verdict_counts": {
                        "APPROVE": v_app, "QUARANTINE": v_quar, "REJECT": v_rej
                    },
                }
        else:
            # Fallback to in-memory stats
            stats: Dict[str, Dict] = defaultdict(lambda: {
                "total": 0, "confidence_sum": 0.0, "conflicts": 0,
                "latency_sum": 0.0, "verdicts": defaultdict(int)})
            for e in self._in_memory_history:
                s = stats[e["judge"]]
                s["total"] += 1
                s["confidence_sum"] += e["confidence"]
                s["latency_sum"] += e.get("latency_ms", 0)
                s["verdicts"][e["verdict"]] += 1
                if e["verdict"] != "APPROVE":
                    s["conflicts"] += 1
            for jn, s in stats.items():
                t = s["total"] or 1
                report["judges"][jn] = {
                    "total_evaluations": s["total"],
                    "avg_confidence": round(s["confidence_sum"] / t, 3),
                    "avg_latency_ms": round(s["latency_sum"] / t, 2),
                    "conflict_rate": round(s["conflicts"] / t, 3),
                    "verdict_counts": dict(s["verdicts"]),
                }

        report["adaptive_weights"] = self.get_adaptive_weights()
        report["drift_alerts"] = self.detect_drift()
        report["tuning_recommendations"] = self.get_tuning_recommendations()

        return report

    def get_recent_history(self, limit: int = 20) -> List[Dict]:
        if self._conn:
            rows = self._conn.execute("""
                SELECT timestamp, judge_name, verdict, confidence, explanation,
                       latency_ms, memory_type, memory_id
                FROM meta_judge_events ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
            return [{"timestamp": r[0], "judge": r[1], "verdict": r[2],
                     "confidence": r[3], "explanation": r[4], "latency_ms": r[5],
                     "memory_type": r[6], "memory_id": r[7]} for r in reversed(rows)]
        return self._in_memory_history[-limit:]


# ============================================================
# MEMORY JURY — orchestrator
# ============================================================

class MemoryJury:
    """
    Orchestrates the 4-judge jury for TypedMemory validation.

    Features:
    - 4 judges: Safety (veto), Quality, Temporal, Consensus
    - MetaJudge with persistent SQLite stats, adaptive weighting,
      ground truth feedback, calibration scoring, drift detection,
      and threshold tuning recommendations
    - Adaptive weights refresh every 25 deliberations

    Usage:
        jury = MemoryJury(db_path="data/pltm_mcp.db")
        decision = jury.deliberate(content, memory_type, ...)
    """

    WEIGHT_REFRESH_INTERVAL = 25  # refresh adaptive weights every N deliberations

    def __init__(self, enable_meta_judge: bool = True, db_path: Optional[str] = None):
        self.safety = SafetyJudge()
        self.quality = QualityJudge()
        self.temporal = TemporalJudge()
        self.consensus = ConsensusJudge()
        self.meta = MetaJudge(db_path=db_path) if enable_meta_judge else None
        self.deliberation_count = 0
        self._last_weight_refresh = 0
        logger.info("MemoryJury initialized (Safety + Quality + Temporal + Consensus + MetaJudge"
                     + (" [persistent]" if db_path else " [in-memory]") + ")")

    def _maybe_refresh_weights(self):
        """Refresh adaptive weights from MetaJudge if interval elapsed."""
        if not self.meta:
            return
        if self.deliberation_count - self._last_weight_refresh >= self.WEIGHT_REFRESH_INTERVAL:
            weights = self.meta.get_adaptive_weights()
            self.consensus.set_adaptive_weights(weights)
            self._last_weight_refresh = self.deliberation_count
            logger.debug(f"Adaptive weights refreshed: {weights}")

    def deliberate(
        self,
        content: str,
        memory_type: str = "",
        memory_id: str = "",
        episode_timestamp: float = 0,
        existing_contents: Optional[List[str]] = None,
        bulk: bool = False,
        **kwargs,
    ) -> JudgeDecision:
        """
        Run full jury deliberation on a memory before storage.

        Returns ConsensusJudge's final decision.
        """
        self.deliberation_count += 1
        self._maybe_refresh_weights()
        t0 = time.time()

        # Run all judges
        safety_d = self.safety.evaluate(content)
        quality_d = self.quality.evaluate(content, memory_type=memory_type,
                                          existing_contents=existing_contents)
        temporal_d = self.temporal.evaluate(content, memory_type=memory_type,
                                            episode_timestamp=episode_timestamp, bulk=bulk)

        all_decisions = [safety_d, quality_d, temporal_d]
        final = self.consensus.aggregate(all_decisions)

        elapsed_ms = (time.time() - t0) * 1000

        # Track in MetaJudge (with memory_type, memory_id, and content_hash —
        # the hash lets feedback match REJECTed memories that got no stored id)
        if self.meta:
            content_hash = hashlib.sha1(
                content.strip().lower().encode("utf-8")).hexdigest()
            for d in all_decisions:
                self.meta.track(d, latency_ms=elapsed_ms / len(all_decisions),
                                memory_type=memory_type, memory_id=memory_id,
                                content_hash=content_hash)
            self.meta.track(final, latency_ms=elapsed_ms,
                            memory_type=memory_type, memory_id=memory_id,
                            content_hash=content_hash)

        logger.debug(
            f"MemoryJury #{self.deliberation_count}: {final.verdict.value} "
            f"({final.confidence:.0%}) — {final.explanation} [{elapsed_ms:.1f}ms]"
        )

        return final

    def record_feedback(self, memory_id: str = "", feedback_type: str = "",
                        details: str = "", content_hash: str = ""):
        """Forward ground truth feedback to MetaJudge."""
        if self.meta:
            self.meta.record_feedback(
                memory_id=memory_id, feedback_type=feedback_type,
                details=details, content_hash=content_hash)

    def get_stats(self) -> Dict[str, Any]:
        result = {
            "deliberations": self.deliberation_count,
        }
        if self.meta:
            result.update(self.meta.get_report())
            result["recent_history"] = self.meta.get_recent_history(10)
            result["per_type_accuracy"] = self.meta.get_per_type_accuracy()
            result["calibration"] = self.meta.get_calibration_report()
        return result
