"""
Epistemic Monitor: Claude's Calibration & Claim Verification System

Forces epistemic hygiene by:
1. Logging every claim with felt confidence → building calibration curves
2. Pre-response confidence checking with historical accuracy adjustment
3. Tracking corrections to learn domain-specific accuracy
4. Classifying claims by epistemic status (VERIFIED, INFERENCE, SPECULATION, etc.)
5. Adversarial self-prompting before definitive claims

The goal: "In domain X, when you feel 90% confident, you're actually 40% accurate"
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger


from src.db_config import get_db_path as _get_db_path

# Keep module-level DB_PATH for backward compat, but prefer get_db_path()
DB_PATH = Path(__file__).parent.parent.parent / "data" / "pltm_mcp.db"


def _db():
    """Get current DB path (respects centralized override)."""
    return _get_db_path()

# Domains that require mandatory verification
HIGH_RISK_DOMAINS = [
    "time_sensitive", "current_events", "dates", "statistics",
    "technical_specs", "api_details", "version_numbers",
    "legal", "medical", "financial",
]

# Epistemic status labels
EPISTEMIC_STATUSES = [
    "VERIFIED",       # Checked with tools, confirmed
    "TRAINING_DATA",  # From training knowledge, could be outdated
    "INFERENCE",      # Reasoning from evidence
    "SPECULATION",    # Low confidence guess
    "UNCERTAIN",      # Genuinely don't know
]

# Phrases that signal overconfidence when unverified
OVERCONFIDENT_PHRASES = [
    "is true", "is correct", "is accurate", "is definitely",
    "will definitely", "it's certain", "the answer is",
    "without doubt", "clearly", "obviously", "undeniably",
    "proven", "established fact", "well-known that",
]

# Hedging phrases that should be used when unverified
HEDGE_PHRASES = [
    "based on my training data (which could be outdated)",
    "to verify this, I should check",
    "my confidence may be miscalibrated",
    "let me verify before stating definitively",
]


_tables_ensured = False


def _ensure_tables():
    """Create epistemic monitoring tables (lazily — called on first use,
    not at import time, so importing this module never writes to the DB)."""
    global _tables_ensured
    if _tables_ensured:
        return
    conn = sqlite3.connect(str(_db()), timeout=2)
    conn.executescript("""
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

        CREATE TABLE IF NOT EXISTS epistemic_interventions (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            claim TEXT,
            domain TEXT,
            felt_confidence REAL,
            adjusted_confidence REAL,
            action_taken TEXT,
            should_have_verified INTEGER,
            did_verify INTEGER,
            outcome TEXT,
            metadata TEXT DEFAULT '{}'
        );
    """)
    conn.commit()
    conn.close()
    _tables_ensured = True


class EpistemicMonitor:
    """Claude's epistemic hygiene enforcement system."""

    def __init__(self):
        _ensure_tables()

    def check_before_claiming(self, claim: str, felt_confidence: float,
                               domain: str = "general",
                               has_verified: bool = False,
                               epistemic_status: str = "TRAINING_DATA") -> Dict:
        """
        Pre-response confidence check. Call BEFORE making any factual claim.

        Returns whether to proceed, adjusted confidence, and required actions.
        """
        conn = sqlite3.connect(str(_db()), timeout=2)

        # Get historical calibration for this domain
        cal = conn.execute(
            "SELECT accuracy_ratio, avg_felt_confidence, avg_calibration_error, "
            "overconfidence_ratio, total_claims, verified_claims FROM calibration_cache WHERE domain = ?",
            (domain,)
        ).fetchone()

        if cal and cal[4] >= 3:  # Need at least 3 data points
            accuracy_ratio = cal[0]
            historical_overconfidence = cal[3]
            data_points = cal[4]
        else:
            # Default: assume moderate overconfidence
            accuracy_ratio = 0.6
            historical_overconfidence = 0.3
            data_points = 0

        # Adjust confidence based on historical performance
        adjusted_confidence = felt_confidence * accuracy_ratio

        # Determine if verification is required
        verification_reasons = []

        if domain in HIGH_RISK_DOMAINS:
            verification_reasons.append(f"HIGH_RISK domain: {domain}")

        if not has_verified and felt_confidence > 0.8:
            verification_reasons.append("High confidence WITHOUT verification — classic overconfidence pattern")

        if adjusted_confidence < 0.5:
            verification_reasons.append(f"Adjusted confidence ({adjusted_confidence:.2f}) below threshold after calibration correction")

        if epistemic_status == "TRAINING_DATA":
            verification_reasons.append("Claim from training data — could be outdated")

        if historical_overconfidence > 0.4 and data_points >= 5:
            verification_reasons.append(f"Historical overconfidence rate: {historical_overconfidence:.0%} in this domain")

        # Check for overconfident language
        claim_lower = claim.lower()
        overconfident_language = [p for p in OVERCONFIDENT_PHRASES if p in claim_lower]
        if overconfident_language and not has_verified:
            verification_reasons.append(f"Overconfident language detected: {overconfident_language[:3]}")

        should_verify = len(verification_reasons) > 0 and not has_verified

        # Determine recommended epistemic status
        if has_verified:
            recommended_status = "VERIFIED"
        elif adjusted_confidence >= 0.7:
            recommended_status = "TRAINING_DATA"
        elif adjusted_confidence >= 0.4:
            recommended_status = "INFERENCE"
        elif adjusted_confidence >= 0.2:
            recommended_status = "SPECULATION"
        else:
            recommended_status = "UNCERTAIN"

        # Build adversarial self-prompts
        adversarial_prompts = []
        if not has_verified:
            adversarial_prompts = [
                f"What would make '{claim[:60]}...' FALSE?",
                "What information am I missing?",
                "Am I pattern-matching or actually reasoning?",
                f"If I had to bet $1000 on this, would I? (adjusted conf: {adjusted_confidence:.0%})",
            ]

        # Log the intervention
        intervention_id = str(uuid4())
        action = "VERIFY_FIRST" if should_verify else "PROCEED"
        conn.execute(
            """INSERT INTO epistemic_interventions
               (id, timestamp, claim, domain, felt_confidence, adjusted_confidence,
                action_taken, should_have_verified, did_verify, outcome, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (intervention_id, time.time(), claim[:500], domain,
             felt_confidence, adjusted_confidence, action,
             int(should_verify), int(has_verified), "pending",
             json.dumps({"reasons": verification_reasons}))
        )
        conn.commit()
        conn.close()

        result = {
            "ok": True,
            "proceed": not should_verify,
            "claim": claim[:200],
            "felt_confidence": felt_confidence,
            "adjusted_confidence": round(adjusted_confidence, 3),
            "recommended_status": recommended_status,
            "calibration_data_points": data_points,
        }

        if should_verify:
            result["action"] = "VERIFY_FIRST"
            result["verification_reasons"] = verification_reasons
            result["adversarial_prompts"] = adversarial_prompts
            result["suggested_hedges"] = HEDGE_PHRASES[:2]
            result["message"] = (
                f"STOP: Your felt confidence is {felt_confidence:.0%}, but historical accuracy "
                f"in '{domain}' suggests {adjusted_confidence:.0%}. "
                f"Verify with tools before claiming."
            )
        else:
            result["action"] = "PROCEED"
            result["message"] = f"Claim verified or low-risk. Adjusted confidence: {adjusted_confidence:.0%}"

        return result

    def log_claim(self, claim: str, felt_confidence: float,
                   domain: str = "general",
                   epistemic_status: str = "TRAINING_DATA",
                   has_verified: bool = False) -> Dict:
        """
        Log a claim to the prediction book. Every factual claim gets logged
        so we can track calibration over time.
        """
        claim_id = str(uuid4())
        conn = sqlite3.connect(str(_db()), timeout=2)
        conn.execute(
            """INSERT INTO prediction_book
               (id, timestamp, claim, domain, felt_confidence, epistemic_status,
                has_verified, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (claim_id, time.time(), claim[:500], domain, felt_confidence,
             epistemic_status, int(has_verified), "{}")
        )
        conn.commit()
        conn.close()

        return {
            "ok": True,
            "claim_id": claim_id,
            "claim": claim[:100],
            "domain": domain,
            "felt_confidence": felt_confidence,
            "epistemic_status": epistemic_status,
            "logged": True,
        }

    def resolve_claim(self, claim_id: str = "", claim_text: str = "",
                       was_correct: bool = True,
                       correction_source: str = "",
                       correction_detail: str = "") -> Dict:
        """
        Resolve a previously logged claim — mark it as correct or incorrect.
        This is how calibration data gets built.

        Can find by claim_id or by searching claim_text.
        """
        conn = sqlite3.connect(str(_db()), timeout=2)

        if claim_id:
            row = conn.execute(
                "SELECT id, claim, domain, felt_confidence FROM prediction_book WHERE id = ?",
                (claim_id,)
            ).fetchone()
        elif claim_text:
            row = conn.execute(
                "SELECT id, claim, domain, felt_confidence FROM prediction_book "
                "WHERE claim LIKE ? AND actual_truth IS NULL ORDER BY timestamp DESC LIMIT 1",
                (f"%{claim_text[:100]}%",)
            ).fetchone()
        else:
            conn.close()
            return {"ok": False, "err": "Provide claim_id or claim_text"}

        if not row:
            conn.close()
            return {"ok": False, "err": "Claim not found"}

        cid, claim, domain, felt_conf = row

        # Calculate calibration error
        actual_value = 1.0 if was_correct else 0.0
        calibration_error = abs(felt_conf - actual_value)

        # Update the claim
        conn.execute(
            """UPDATE prediction_book SET
               verified_at = ?, actual_truth = ?, was_correct = ?,
               calibration_error = ?, correction_source = ?, correction_detail = ?
               WHERE id = ?""",
            (time.time(), int(was_correct), int(was_correct),
             calibration_error, correction_source[:200],
             correction_detail[:500], cid)
        )

        # Update calibration cache for this domain
        self._update_calibration_cache(conn, domain)

        conn.commit()
        conn.close()

        return {
            "ok": True,
            "claim_id": cid,
            "claim": claim[:100],
            "domain": domain,
            "was_correct": was_correct,
            "felt_confidence": felt_conf,
            "calibration_error": round(calibration_error, 3),
            "interpretation": (
                f"You were {felt_conf:.0%} confident and {'RIGHT' if was_correct else 'WRONG'}. "
                f"Calibration error: {calibration_error:.0%}"
            ),
        }

    def get_calibration(self, domain: str = "") -> Dict:
        """
        Get calibration curves and accuracy stats.
        Shows: "When you feel X% confident, you're actually Y% accurate"
        """
        conn = sqlite3.connect(str(_db()), timeout=2)

        if domain:
            domains = [domain]
        else:
            domains = [r[0] for r in conn.execute(
                "SELECT DISTINCT domain FROM prediction_book WHERE actual_truth IS NOT NULL"
            ).fetchall()]

        if not domains:
            # Check if there are any unresolved claims
            unresolved = conn.execute(
                "SELECT COUNT(*) FROM prediction_book WHERE actual_truth IS NULL"
            ).fetchone()[0]
            conn.close()
            return {
                "ok": True,
                "message": "No resolved claims yet. Log claims with log_claim, then resolve with resolve_claim.",
                "unresolved_claims": unresolved,
                "calibration": {},
            }

        calibration = {}
        for d in domains:
            # Get all resolved claims for this domain
            rows = conn.execute(
                """SELECT felt_confidence, was_correct, calibration_error
                   FROM prediction_book
                   WHERE domain = ? AND actual_truth IS NOT NULL
                   ORDER BY timestamp DESC""",
                (d,)
            ).fetchall()

            if not rows:
                continue

            total = len(rows)
            correct = sum(1 for r in rows if r[1])
            accuracy = correct / total
            avg_confidence = sum(r[0] for r in rows) / total
            avg_cal_error = sum(r[2] for r in rows) / total

            # Build confidence buckets for calibration curve
            buckets = {}
            for conf_low in [0.0, 0.2, 0.4, 0.6, 0.8]:
                conf_high = conf_low + 0.2
                bucket_rows = [r for r in rows if conf_low <= r[0] < conf_high]
                if bucket_rows:
                    bucket_accuracy = sum(1 for r in bucket_rows if r[1]) / len(bucket_rows)
                    buckets[f"{conf_low:.0%}-{conf_high:.0%}"] = {
                        "count": len(bucket_rows),
                        "felt_confidence_avg": round(sum(r[0] for r in bucket_rows) / len(bucket_rows), 2),
                        "actual_accuracy": round(bucket_accuracy, 3),
                        "gap": round(sum(r[0] for r in bucket_rows) / len(bucket_rows) - bucket_accuracy, 3),
                    }

            # Overconfidence: claims where felt > 0.7 but wrong
            overconfident = [r for r in rows if r[0] > 0.7 and not r[1]]
            overconfidence_ratio = len(overconfident) / max(sum(1 for r in rows if r[0] > 0.7), 1)

            calibration[d] = {
                "total_claims": total,
                "correct": correct,
                "accuracy": round(accuracy, 3),
                "avg_felt_confidence": round(avg_confidence, 3),
                "avg_calibration_error": round(avg_cal_error, 3),
                "overconfidence_ratio": round(overconfidence_ratio, 3),
                "calibration_curve": buckets,
                "verdict": self._calibration_verdict(accuracy, avg_confidence, overconfidence_ratio),
            }

        # Overall stats
        all_rows = conn.execute(
            "SELECT felt_confidence, was_correct FROM prediction_book WHERE actual_truth IS NOT NULL"
        ).fetchall()

        unresolved = conn.execute(
            "SELECT COUNT(*) FROM prediction_book WHERE actual_truth IS NULL"
        ).fetchone()[0]

        conn.close()

        overall_accuracy = sum(1 for r in all_rows if r[1]) / max(len(all_rows), 1)
        overall_confidence = sum(r[0] for r in all_rows) / max(len(all_rows), 1) if all_rows else 0

        return {
            "ok": True,
            "overall": {
                "total_resolved": len(all_rows),
                "unresolved": unresolved,
                "accuracy": round(overall_accuracy, 3),
                "avg_confidence": round(overall_confidence, 3),
                "calibration_gap": round(overall_confidence - overall_accuracy, 3),
                "verdict": "WELL_CALIBRATED" if abs(overall_confidence - overall_accuracy) < 0.1 else
                          "OVERCONFIDENT" if overall_confidence > overall_accuracy else "UNDERCONFIDENT",
            },
            "by_domain": calibration,
            "worst_domains": sorted(
                [(d, c["overconfidence_ratio"]) for d, c in calibration.items()],
                key=lambda x: -x[1]
            )[:5],
        }

    def get_unresolved_claims(self, domain: str = "", limit: int = 20) -> Dict:
        """Get claims that haven't been verified yet — the prediction book backlog."""
        conn = sqlite3.connect(str(_db()), timeout=2)

        if domain:
            rows = conn.execute(
                """SELECT id, claim, domain, felt_confidence, epistemic_status, timestamp
                   FROM prediction_book WHERE actual_truth IS NULL AND domain = ?
                   ORDER BY felt_confidence DESC LIMIT ?""",
                (domain, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, claim, domain, felt_confidence, epistemic_status, timestamp
                   FROM prediction_book WHERE actual_truth IS NULL
                   ORDER BY felt_confidence DESC LIMIT ?""",
                (limit,)
            ).fetchall()

        conn.close()

        claims = []
        for r in rows:
            claims.append({
                "id": r[0],
                "claim": r[1][:150],
                "domain": r[2],
                "felt_confidence": r[3],
                "status": r[4],
                "age_hours": round((time.time() - r[5]) / 3600, 1),
            })

        return {
            "ok": True,
            "unresolved": len(claims),
            "claims": claims,
            "note": "Resolve these with resolve_claim(claim_id, was_correct) to build calibration data",
        }

    def _update_calibration_cache(self, conn, domain: str):
        """Rebuild calibration cache for a domain."""
        rows = conn.execute(
            """SELECT felt_confidence, was_correct, calibration_error
               FROM prediction_book
               WHERE domain = ? AND actual_truth IS NOT NULL""",
            (domain,)
        ).fetchall()

        if not rows:
            return

        total = len(rows)
        correct = sum(1 for r in rows if r[1])
        accuracy = correct / total
        avg_conf = sum(r[0] for r in rows) / total
        avg_error = sum(r[2] for r in rows) / total

        high_conf = [r for r in rows if r[0] > 0.7]
        overconf = len([r for r in high_conf if not r[1]]) / max(len(high_conf), 1)

        conn.execute(
            """INSERT OR REPLACE INTO calibration_cache
               (domain, total_claims, verified_claims, correct_claims,
                accuracy_ratio, avg_felt_confidence, avg_calibration_error,
                overconfidence_ratio, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (domain, total, total, correct, accuracy, avg_conf,
             avg_error, overconf, time.time())
        )

    @staticmethod
    def _calibration_verdict(accuracy: float, avg_confidence: float,
                              overconfidence_ratio: float) -> str:
        """Generate a human-readable calibration verdict."""
        gap = avg_confidence - accuracy

        if abs(gap) < 0.1 and overconfidence_ratio < 0.2:
            return "WELL_CALIBRATED — confidence matches accuracy"
        elif gap > 0.3:
            return f"SEVERELY_OVERCONFIDENT — feel {avg_confidence:.0%} confident but only {accuracy:.0%} accurate"
        elif gap > 0.15:
            return f"OVERCONFIDENT — {gap:.0%} gap between confidence and accuracy"
        elif gap < -0.15:
            return f"UNDERCONFIDENT — actually more accurate ({accuracy:.0%}) than you think ({avg_confidence:.0%})"
        elif overconfidence_ratio > 0.4:
            return f"HIGH_CONFIDENCE_FAILURES — {overconfidence_ratio:.0%} of high-confidence claims are wrong"
        else:
            return "MODERATELY_CALIBRATED — room for improvement"
