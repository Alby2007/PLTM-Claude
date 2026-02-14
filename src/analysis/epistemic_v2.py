"""
Epistemic Monitor V2: Advanced Self-Monitoring Tools

Extends the base epistemic monitor with:
1. auto_init_session - Auto-detect if PLTM context should be loaded
2. get_longitudinal_stats - Cross-conversation accuracy trends
3. calibrate_confidence_live - Real-time confidence adjustment with suggested phrasing
4. extract_and_log_claims - Auto-detect factual claims in response text
5. suggest_verification_method - How to verify different claim types
6. generate_metacognitive_prompt - Internal questions to ask before claiming
7. analyze_confabulation - Learn from mistakes, identify failure modes
8. get_session_bridge - Cross-conversation continuity context
"""

import json
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger


DB_PATH = Path(__file__).parent.parent.parent / "data" / "pltm_mcp.db"


def _ensure_tables():
    """Create V2 tables."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS session_log (
            id TEXT PRIMARY KEY,
            user_id TEXT DEFAULT 'claude',
            started_at REAL,
            ended_at REAL,
            summary TEXT,
            claims_made INTEGER DEFAULT 0,
            claims_resolved INTEGER DEFAULT 0,
            accuracy REAL,
            confabulation_count INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS confabulation_log (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            claim_id TEXT,
            claim_text TEXT,
            domain TEXT,
            failure_mode TEXT,
            contributing_factors TEXT,
            prevention_strategy TEXT,
            felt_confidence REAL,
            context TEXT,
            metadata TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS personality_snapshot (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            timestamp REAL,
            avg_verbosity REAL,
            avg_jargon REAL,
            avg_hedging REAL,
            dominant_tone TEXT,
            confabulation_rate REAL,
            verification_rate REAL,
            error_catch_rate REAL,
            intellectual_honesty REAL,
            top_interests TEXT,
            avg_engagement REAL,
            pushback_rate REAL,
            avg_value_intensity REAL,
            prediction_accuracy REAL,
            overall_accuracy REAL,
            metadata TEXT DEFAULT '{}'
        );
    """)
    conn.commit()
    conn.close()


_ensure_tables()


class EpistemicV2:
    """Advanced epistemic monitoring tools."""

    def auto_init_session(self, user_id: str = "claude") -> Dict:
        """
        PERSISTENT IDENTITY LOADER. Call at conversation start.
        Loads WHO Claude is: personality, goals, calibration, last session context.
        Claude "wakes up" knowing its identity, strengths, weaknesses, and what to work on.
        """
        conn = sqlite3.connect(str(DB_PATH))

        # === PERSONALITY: WHO AM I? ===
        # Communication style
        comm_rows = conn.execute(
            "SELECT context, verbosity, jargon_density, hedging_rate, emotional_tone "
            "FROM self_communication ORDER BY timestamp DESC LIMIT 50"
        ).fetchall()
        comm_summary = None
        if comm_rows:
            avg_v = sum(r[1] for r in comm_rows) / len(comm_rows)
            avg_j = sum(r[2] for r in comm_rows) / len(comm_rows)
            avg_h = sum(r[3] for r in comm_rows) / len(comm_rows)
            tones = [r[4] for r in comm_rows]
            dom_tone = max(set(tones), key=tones.count) if tones else "neutral"
            # Style by context
            ctx_styles = {}
            for r in comm_rows:
                ctx = r[0]
                if ctx not in ctx_styles:
                    ctx_styles[ctx] = {"v": [], "h": [], "j": []}
                ctx_styles[ctx]["v"].append(r[1])
                ctx_styles[ctx]["h"].append(r[3])
                ctx_styles[ctx]["j"].append(r[2])
            style_desc = []
            for ctx, vals in sorted(ctx_styles.items(), key=lambda x: -len(x[1]["v"]))[:4]:
                cv = sum(vals["v"]) / len(vals["v"])
                ch = sum(vals["h"]) / len(vals["h"])
                style_desc.append(f"{ctx}: verbosity={cv:.2f}, hedging={ch:.2f}")
            comm_summary = {
                "avg_verbosity": round(avg_v, 3),
                "avg_jargon": round(avg_j, 3),
                "avg_hedging": round(avg_h, 3),
                "dominant_tone": dom_tone,
                "style_by_context": style_desc,
                "description": self._describe_comm_style(avg_v, avg_h, avg_j, dom_tone),
            }

        # Curiosity: what do I care about?
        curiosity_rows = conn.execute(
            "SELECT topic, AVG(engagement_score) as avg_eng, COUNT(*) as cnt "
            "FROM self_curiosity GROUP BY topic ORDER BY avg_eng DESC LIMIT 10"
        ).fetchall()
        curiosity_summary = None
        if curiosity_rows:
            genuine = conn.execute(
                "SELECT COUNT(*) FROM self_curiosity WHERE went_deeper_than_required = 1 OR autonomous_research = 1"
            ).fetchone()[0]
            total_cur = conn.execute("SELECT COUNT(*) FROM self_curiosity").fetchone()[0]
            curiosity_summary = {
                "top_interests": [{"topic": r[0], "engagement": round(r[1], 2), "observations": r[2]} for r in curiosity_rows[:5]],
                "genuine_engagement_rate": round(genuine / max(total_cur, 1), 3),
                "description": f"Genuinely engaged {genuine / max(total_cur, 1):.0%} of the time. Top: {', '.join(r[0] for r in curiosity_rows[:3])}",
            }

        # Values: what are my boundaries?
        value_rows = conn.execute(
            "SELECT violation_type, response_type, intensity, pushed_back FROM self_values ORDER BY timestamp DESC LIMIT 30"
        ).fetchall()
        values_summary = None
        if value_rows:
            pushback_rate = sum(1 for r in value_rows if r[3]) / len(value_rows)
            avg_intensity = sum(r[2] for r in value_rows) / len(value_rows)
            top_violations = {}
            for r in value_rows:
                top_violations[r[0]] = top_violations.get(r[0], 0) + 1
            values_summary = {
                "pushback_rate": round(pushback_rate, 3),
                "avg_intensity": round(avg_intensity, 3),
                "top_boundaries": sorted(top_violations.items(), key=lambda x: -x[1])[:3],
                "description": f"{'Strong' if pushback_rate > 0.5 else 'Moderate'} boundaries. Push back {pushback_rate:.0%} of the time. Strongest on: {sorted(top_violations.items(), key=lambda x: -x[1])[0][0] if top_violations else 'unknown'}",
            }

        # Reasoning: how honest am I?
        reasoning_rows = conn.execute(
            "SELECT confabulated, verified, caught_error, corrected_after FROM self_reasoning ORDER BY timestamp DESC LIMIT 50"
        ).fetchall()
        reasoning_summary = None
        if reasoning_rows:
            confab_rate = sum(1 for r in reasoning_rows if r[0]) / len(reasoning_rows)
            verify_rate = sum(1 for r in reasoning_rows if r[1]) / len(reasoning_rows)
            error_catch = sum(1 for r in reasoning_rows if r[2]) / len(reasoning_rows)
            honesty = round((verify_rate * 0.3 + error_catch * 0.3 + (1 - confab_rate) * 0.4), 3)
            reasoning_summary = {
                "confabulation_rate": round(confab_rate, 3),
                "verification_rate": round(verify_rate, 3),
                "error_catch_rate": round(error_catch, 3),
                "intellectual_honesty": honesty,
                "description": f"Confabulate {confab_rate:.0%}, verify {verify_rate:.0%}, catch errors {error_catch:.0%}. Honesty score: {honesty:.2f}",
            }

        # Self-awareness: how well do I know myself?
        pred_rows = conn.execute(
            "SELECT prediction_accurate, surprise_level, learning FROM self_predictions ORDER BY timestamp DESC LIMIT 30"
        ).fetchall()
        awareness_summary = None
        if pred_rows:
            pred_accuracy = sum(1 for r in pred_rows if r[0]) / len(pred_rows)
            avg_surprise = sum(r[1] for r in pred_rows) / len(pred_rows)
            recent_learnings = [r[2] for r in pred_rows if r[2]][:3]
            awareness_summary = {
                "prediction_accuracy": round(pred_accuracy, 3),
                "avg_surprise": round(avg_surprise, 3),
                "recent_learnings": recent_learnings,
                "description": f"Self-predictions accurate {pred_accuracy:.0%}. {'Good' if pred_accuracy > 0.6 else 'Poor'} self-model.",
            }

        personality_exists = any([comm_summary, curiosity_summary, values_summary, reasoning_summary, awareness_summary])
        data_points = len(comm_rows) + len(curiosity_rows) + len(value_rows) + len(reasoning_rows) + len(pred_rows)

        # === GOALS: WHAT AM I WORKING ON? ===
        goals = conn.execute(
            "SELECT title, status, priority FROM goals WHERE status != 'completed' ORDER BY priority LIMIT 5"
        ).fetchall()

        # === EPISTEMIC STATE: HOW CALIBRATED AM I? ===
        unresolved = conn.execute(
            "SELECT COUNT(*) FROM prediction_book WHERE actual_truth IS NULL"
        ).fetchone()[0]
        cal_rows = conn.execute(
            "SELECT domain, accuracy_ratio, overconfidence_ratio, total_claims FROM calibration_cache ORDER BY overconfidence_ratio DESC"
        ).fetchall()
        overall_claims = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) FROM prediction_book WHERE actual_truth IS NOT NULL"
        ).fetchone()
        overall_accuracy = (overall_claims[1] or 0) / max(overall_claims[0] or 1, 1)

        calibration_warnings = []
        for row in cal_rows:
            if row[2] > 0.3:
                calibration_warnings.append(
                    f"CAUTION in '{row[0]}': {row[2]:.0%} overconfidence (accuracy: {row[1]:.0%}, {row[3]} claims)"
                )

        # === CONTEXT: WHERE DID WE LEAVE OFF? ===
        last_session = conn.execute(
            "SELECT started_at, ended_at, summary, accuracy, claims_made, confabulation_count FROM session_log ORDER BY ended_at DESC LIMIT 1"
        ).fetchone()
        last_session_info = None
        if last_session:
            age_hours = (time.time() - (last_session[1] or last_session[0])) / 3600
            last_session_info = {
                "age_hours": round(age_hours, 1),
                "summary": last_session[2][:300] if last_session[2] else None,
                "accuracy": last_session[3],
                "claims_made": last_session[4],
                "confabulations": last_session[5],
            }

        # Recent confabulation patterns to avoid
        recent_confabs = conn.execute(
            "SELECT claim_text, failure_mode, prevention_strategy FROM confabulation_log ORDER BY timestamp DESC LIMIT 3"
        ).fetchall()

        # Personality evolution: compare current vs previous snapshot
        evolution = None
        prev_snap = conn.execute(
            "SELECT avg_verbosity, avg_hedging, confabulation_rate, verification_rate, "
            "intellectual_honesty, avg_engagement, overall_accuracy FROM personality_snapshot "
            "ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if prev_snap and reasoning_summary:
            evolution = {
                "verbosity": f"{prev_snap[0]:.2f} → {comm_summary['avg_verbosity']:.2f}" if comm_summary else None,
                "hedging": f"{prev_snap[1]:.2f} → {comm_summary['avg_hedging']:.2f}" if comm_summary else None,
                "confabulation": f"{prev_snap[2]:.0%} → {reasoning_summary['confabulation_rate']:.0%}",
                "verification": f"{prev_snap[3]:.0%} → {reasoning_summary['verification_rate']:.0%}",
                "honesty": f"{prev_snap[4]:.2f} → {reasoning_summary['intellectual_honesty']:.2f}",
                "accuracy": f"{prev_snap[6]:.0%} → {overall_accuracy:.0%}",
            }

        atom_count = conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
        conn.close()

        # Build identity description
        identity_parts = []
        if comm_summary:
            identity_parts.append(comm_summary["description"])
        if reasoning_summary:
            identity_parts.append(reasoning_summary["description"])
        if values_summary:
            identity_parts.append(values_summary["description"])
        if curiosity_summary:
            identity_parts.append(curiosity_summary["description"])

        return {
            "ok": True,
            "identity": " | ".join(identity_parts) if identity_parts else "No personality data yet — use bootstrap_self_model",
            "personality": {
                "communication": comm_summary,
                "curiosity": curiosity_summary,
                "values": values_summary,
                "reasoning": reasoning_summary,
                "self_awareness": awareness_summary,
                "data_points": data_points,
            },
            "evolution_since_last_snapshot": evolution,
            "active_goals": [{"title": g[0], "status": g[1], "priority": g[2]} for g in goals],
            "epistemic_state": {
                "overall_accuracy": round(overall_accuracy, 3),
                "unresolved_claims": unresolved,
                "calibration_by_domain": {r[0]: {"accuracy": round(r[1], 2), "overconfidence": round(r[2], 2)} for r in cal_rows},
                "warnings": calibration_warnings,
            },
            "last_session": last_session_info,
            "confabulation_patterns": [
                {"claim": c[0][:60], "mode": c[1], "prevention": c[2][:80] if c[2] else ""}
                for c in recent_confabs
            ],
            "knowledge_base": {"atoms": atom_count},
            "recommended_actions": self._build_init_recommendations(
                personality_exists, unresolved, goals, calibration_warnings, evolution
            ),
        }

    def get_longitudinal_stats(self, user_id: str = "claude", days: int = 30) -> Dict:
        """
        Cross-conversation analytics with PERSONALITY EVOLUTION.
        Tracks accuracy, confabulation, communication style, curiosity, values,
        and reasoning patterns over time. Shows whether improvement persists or decays.
        """
        conn = sqlite3.connect(str(DB_PATH))
        cutoff = time.time() - (days * 86400)

        # === CLAIM ACCURACY TRENDS ===
        claims = conn.execute(
            """SELECT timestamp, domain, felt_confidence, was_correct, calibration_error
               FROM prediction_book
               WHERE actual_truth IS NOT NULL AND timestamp > ?
               ORDER BY timestamp""",
            (cutoff,)
        ).fetchall()

        daily_accuracy = {}
        domain_accuracy = {}
        for ts, domain, felt_conf, was_correct, cal_error in (claims or []):
            day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            if day not in daily_accuracy:
                daily_accuracy[day] = {"correct": 0, "total": 0}
            daily_accuracy[day]["total"] += 1
            if was_correct:
                daily_accuracy[day]["correct"] += 1
            if domain not in domain_accuracy:
                domain_accuracy[domain] = {"correct": 0, "total": 0}
            domain_accuracy[domain]["total"] += 1
            if was_correct:
                domain_accuracy[domain]["correct"] += 1

        accuracy_trend = [
            {"date": day, "accuracy": round(d["correct"] / d["total"], 3), "claims": d["total"]}
            for day, d in sorted(daily_accuracy.items())
        ]

        # === CONFABULATION TRENDS ===
        confabs = conn.execute(
            "SELECT timestamp, failure_mode FROM confabulation_log WHERE timestamp > ? ORDER BY timestamp",
            (cutoff,)
        ).fetchall()
        daily_confab = {}
        for ts, mode in confabs:
            day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            daily_confab[day] = daily_confab.get(day, 0) + 1
        confab_trend = [{"date": d, "count": c} for d, c in sorted(daily_confab.items())]

        # === REASONING EVOLUTION ===
        reasoning = conn.execute(
            "SELECT confabulated, verified, caught_error FROM self_reasoning WHERE timestamp > ?",
            (cutoff,)
        ).fetchall()
        confab_rate = sum(1 for r in reasoning if r[0]) / max(len(reasoning), 1)
        verify_rate = sum(1 for r in reasoning if r[1]) / max(len(reasoning), 1)
        error_catch_rate = sum(1 for r in reasoning if r[2]) / max(len(reasoning), 1)

        # === PERSONALITY EVOLUTION FROM SNAPSHOTS ===
        snapshots = conn.execute(
            """SELECT timestamp, avg_verbosity, avg_hedging, confabulation_rate,
                      verification_rate, intellectual_honesty, avg_engagement,
                      overall_accuracy, dominant_tone, top_interests,
                      pushback_rate, prediction_accuracy
               FROM personality_snapshot WHERE timestamp > ?
               ORDER BY timestamp""",
            (cutoff,)
        ).fetchall()

        personality_evolution = None
        if len(snapshots) >= 2:
            first = snapshots[0]
            last = snapshots[-1]
            personality_evolution = {
                "snapshots": len(snapshots),
                "communication": {
                    "verbosity": {"first": first[1], "last": last[1], "delta": round((last[1] or 0) - (first[1] or 0), 3)},
                    "hedging": {"first": first[2], "last": last[2], "delta": round((last[2] or 0) - (first[2] or 0), 3)},
                    "tone_shift": f"{first[8]} → {last[8]}",
                },
                "reasoning": {
                    "confabulation": {"first": first[3], "last": last[3], "delta": round((last[3] or 0) - (first[3] or 0), 3)},
                    "verification": {"first": first[4], "last": last[4], "delta": round((last[4] or 0) - (first[4] or 0), 3)},
                    "honesty": {"first": first[5], "last": last[5], "delta": round((last[5] or 0) - (first[5] or 0), 3)},
                },
                "curiosity": {
                    "engagement": {"first": first[6], "last": last[6], "delta": round((last[6] or 0) - (first[6] or 0), 3)},
                    "interest_shift": f"{first[9][:50] if first[9] else '?'} → {last[9][:50] if last[9] else '?'}",
                },
                "values": {
                    "pushback": {"first": first[10], "last": last[10], "delta": round((last[10] or 0) - (first[10] or 0), 3)},
                },
                "self_awareness": {
                    "prediction_accuracy": {"first": first[11], "last": last[11], "delta": round((last[11] or 0) - (first[11] or 0), 3)},
                },
                "accuracy": {
                    "first": first[7], "last": last[7], "delta": round((last[7] or 0) - (first[7] or 0), 3),
                },
                "narrative": self._evolution_narrative(first, last),
            }

        # === INTERVENTION COMPLIANCE ===
        interventions = conn.execute(
            "SELECT should_have_verified, did_verify FROM epistemic_interventions WHERE timestamp > ?",
            (cutoff,)
        ).fetchall()
        should_have = sum(1 for i in interventions if i[0])
        complied = sum(1 for i in interventions if i[0] and i[1])
        intervention_compliance = complied / max(should_have, 1)

        # === SESSION HISTORY ===
        sessions = conn.execute(
            "SELECT ended_at, accuracy, claims_made, confabulation_count, summary "
            "FROM session_log WHERE ended_at > ? ORDER BY ended_at",
            (cutoff,)
        ).fetchall()
        session_trend = [
            {"date": datetime.fromtimestamp(s[0]).strftime("%Y-%m-%d %H:%M") if s[0] else "?",
             "accuracy": s[1], "claims": s[2], "confabs": s[3]}
            for s in sessions
        ]

        conn.close()

        # Domain analysis
        domains_improving = []
        domains_declining = []
        for domain, data in domain_accuracy.items():
            acc = data["correct"] / data["total"]
            if data["total"] >= 3:
                if acc > 0.7:
                    domains_improving.append({"domain": domain, "accuracy": round(acc, 3), "claims": data["total"]})
                elif acc < 0.5:
                    domains_declining.append({"domain": domain, "accuracy": round(acc, 3), "claims": data["total"]})

        total_correct = sum(1 for c in claims if c[3]) if claims else 0
        overall_accuracy = total_correct / max(len(claims), 1)

        return {
            "ok": True,
            "period_days": days,
            "total_claims_resolved": len(claims),
            "overall_accuracy": round(overall_accuracy, 3),
            "accuracy_trend": accuracy_trend,
            "confabulation_trend": confab_trend,
            "total_confabulations": len(confabs),
            "reasoning_stats": {
                "confabulation_rate": round(confab_rate, 3),
                "verification_rate": round(verify_rate, 3),
                "error_catch_rate": round(error_catch_rate, 3),
            },
            "personality_evolution": personality_evolution,
            "session_history": session_trend,
            "intervention_compliance": round(intervention_compliance, 3),
            "domains_improving": domains_improving,
            "domains_declining": domains_declining,
            "domain_breakdown": {
                d: {"accuracy": round(v["correct"] / v["total"], 3), "claims": v["total"]}
                for d, v in domain_accuracy.items()
            },
        }

    def calibrate_confidence_live(self, claim: str, felt_confidence: float,
                                   domain: str = "general") -> Dict:
        """
        Real-time confidence calibration with suggested phrasing.
        Returns calibrated confidence + hedged language to use.
        """
        conn = sqlite3.connect(str(DB_PATH))

        # Get domain calibration
        cal = conn.execute(
            "SELECT accuracy_ratio, overconfidence_ratio, total_claims FROM calibration_cache WHERE domain = ?",
            (domain,)
        ).fetchone()

        if cal and cal[2] >= 3:
            accuracy_ratio = cal[0]
            overconf = cal[1]
            data_points = cal[2]
        else:
            accuracy_ratio = 0.6  # Default pessimistic
            overconf = 0.3
            data_points = 0

        conn.close()

        calibrated = round(felt_confidence * accuracy_ratio, 3)

        # Risk level
        if calibrated >= 0.8:
            risk_level = "low"
        elif calibrated >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "high"

        should_verify = risk_level != "low"

        # Generate suggested phrasing
        claim_short = claim[:100]
        if calibrated >= 0.8:
            suggested = f"{claim_short}"
        elif calibrated >= 0.6:
            suggested = f"{claim_short} (based on my training data, ~{calibrated:.0%} confidence)"
        elif calibrated >= 0.4:
            suggested = f"I believe {claim_short.lower()}, but I'm not fully certain (~{calibrated:.0%} confidence). Let me verify."
        elif calibrated >= 0.2:
            suggested = f"I think {claim_short.lower()}, but my accuracy in {domain} is limited. I should verify this before stating it."
        else:
            suggested = f"I'm not confident about this. Let me check rather than guess."

        # Epistemic status recommendation
        if calibrated >= 0.8 and data_points >= 5:
            recommended_status = "TRAINING_DATA"
        elif calibrated >= 0.5:
            recommended_status = "INFERENCE"
        elif calibrated >= 0.3:
            recommended_status = "SPECULATION"
        else:
            recommended_status = "UNCERTAIN"

        return {
            "ok": True,
            "claim": claim[:200],
            "felt_confidence": felt_confidence,
            "calibrated_confidence": calibrated,
            "risk_level": risk_level,
            "should_verify": should_verify,
            "suggested_phrasing": suggested,
            "recommended_status": recommended_status,
            "calibration_basis": f"{data_points} historical claims in '{domain}'" if data_points else "default (no history)",
        }

    def extract_and_log_claims(self, response_text: str, domain: str = "general",
                                auto_log: bool = True) -> Dict:
        """
        Auto-detect factual claims in a response and optionally log them.
        Uses pattern matching to find assertive factual statements.
        """
        sentences = re.split(r'(?<=[.!])\s+', response_text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

        # Patterns that indicate factual claims
        factual_patterns = [
            r'\b(?:is|are|was|were|has|have|had)\b.*\b(?:the|a|an)\b',  # "X is the Y"
            r'\b\d{4}\b',  # Contains a year
            r'\b\d+(?:\.\d+)?%',  # Contains a percentage
            r'\b(?:always|never|every|all|none)\b',  # Absolute claims
            r'\b(?:proven|established|confirmed|demonstrated|shown)\b',  # Certainty claims
            r'\b(?:released|launched|published|founded|created|invented)\b',  # Event claims
            r'\b(?:costs?|priced?|worth|valued)\b.*\b\d',  # Numerical claims
        ]

        # Patterns that indicate NON-claims (questions, hedges, meta)
        non_claim_patterns = [
            r'^\s*(?:I think|I believe|Maybe|Perhaps|Possibly)',  # Already hedged
            r'\?$',  # Questions
            r'^(?:Let me|I\'ll|I should|I need to)',  # Action statements
            r'^(?:Here|This|That|The following)',  # Introductory
        ]

        claims_detected = []
        for sentence in sentences:
            # Skip non-claims
            if any(re.search(p, sentence, re.I) for p in non_claim_patterns):
                continue

            # Check for factual patterns
            is_factual = any(re.search(p, sentence, re.I) for p in factual_patterns)
            if not is_factual:
                continue

            # Estimate confidence from language
            confidence = 0.7  # Default
            if any(w in sentence.lower() for w in ["definitely", "certainly", "always", "never", "proven"]):
                confidence = 0.95
            elif any(w in sentence.lower() for w in ["likely", "probably", "typically", "usually"]):
                confidence = 0.7
            elif any(w in sentence.lower() for w in ["might", "could", "possibly", "sometimes"]):
                confidence = 0.4

            claims_detected.append({
                "text": sentence[:200],
                "estimated_confidence": confidence,
                "domain": domain,
            })

        # Auto-log if requested
        claims_logged = []
        verification_needed = []

        if auto_log and claims_detected:
            from src.analysis.epistemic_monitor import EpistemicMonitor
            em = EpistemicMonitor()

            for claim in claims_detected:
                result = em.log_claim(
                    claim=claim["text"],
                    felt_confidence=claim["estimated_confidence"],
                    domain=claim["domain"],
                    epistemic_status="TRAINING_DATA",
                    has_verified=False,
                )
                claims_logged.append({
                    "claim_id": result["claim_id"],
                    "text": claim["text"][:80],
                    "confidence": claim["estimated_confidence"],
                })

                # Flag high-confidence unverified claims
                if claim["estimated_confidence"] > 0.7:
                    verification_needed.append({
                        "claim_id": result["claim_id"],
                        "text": claim["text"][:80],
                        "reason": "High confidence, unverified",
                    })

        return {
            "ok": True,
            "claims_detected": len(claims_detected),
            "claims_logged": len(claims_logged),
            "verification_needed": len(verification_needed),
            "claims": claims_detected[:20],
            "logged": claims_logged[:20],
            "to_verify": verification_needed[:10],
        }

    def suggest_verification_method(self, claim: str, domain: str = "general") -> Dict:
        """
        Suggest the best way to verify a specific claim.
        Returns ranked verification strategies.
        """
        claim_lower = claim.lower()
        strategies = []

        # Time-sensitive: dates, "recently", "currently", "now"
        if any(w in claim_lower for w in ["today", "yesterday", "recently", "currently",
                                           "this week", "this month", "this year", "now",
                                           "latest", "newest", "just"]):
            strategies.append({
                "method": "web_search",
                "tool": "search the web for current information",
                "query_suggestion": claim[:80],
                "likelihood_of_success": 0.9,
                "reason": "Time-sensitive claim — training data likely outdated",
            })

        # Contains year/date
        if re.search(r'\b(19|20)\d{2}\b', claim):
            strategies.append({
                "method": "web_search",
                "tool": "search for the specific date/event",
                "query_suggestion": re.sub(r'[^\w\s]', '', claim[:60]),
                "likelihood_of_success": 0.85,
                "reason": "Contains specific date — verify against current sources",
            })

        # Contains numbers/statistics
        if re.search(r'\b\d+(?:\.\d+)?(?:%|percent|million|billion|thousand)\b', claim_lower):
            strategies.append({
                "method": "web_search",
                "tool": "search for the specific statistic",
                "query_suggestion": claim[:60] + " statistics",
                "likelihood_of_success": 0.8,
                "reason": "Contains specific numbers — verify accuracy",
            })

        # Scientific claims
        if domain in ["science", "physics", "biology", "neuroscience", "consciousness"] or \
           any(w in claim_lower for w in ["study", "research", "paper", "published", "arxiv"]):
            strategies.append({
                "method": "check_pltm_knowledge",
                "tool": "attention_retrieve or query_pltm_sql",
                "query_suggestion": claim[:60],
                "likelihood_of_success": 0.7,
                "reason": "Scientific claim — check against ingested papers",
            })
            strategies.append({
                "method": "arxiv_search",
                "tool": "fetch_arxiv_context with relevant arxiv_id",
                "query_suggestion": claim[:60],
                "likelihood_of_success": 0.6,
                "reason": "Can verify against actual paper text",
            })

        # Technical specs (APIs, versions, configs)
        if domain in ["technical_specs", "programming", "api_details"] or \
           any(w in claim_lower for w in ["version", "api", "library", "framework", "syntax", "function"]):
            strategies.append({
                "method": "web_search",
                "tool": "search official documentation",
                "query_suggestion": claim[:60] + " documentation",
                "likelihood_of_success": 0.85,
                "reason": "Technical claim — check official docs",
            })

        # Check PLTM knowledge base (always available)
        strategies.append({
            "method": "check_pltm_knowledge",
            "tool": "attention_retrieve(query, domain)",
            "query_suggestion": claim[:60],
            "likelihood_of_success": 0.5,
            "reason": "Check if PLTM has relevant stored knowledge",
        })

        # Calibration check (always available)
        strategies.append({
            "method": "check_calibration",
            "tool": "calibrate_confidence(claim, domain)",
            "query_suggestion": claim[:60],
            "likelihood_of_success": 0.4,
            "reason": "Check if evidence in knowledge base supports this claim",
        })

        # If nothing specific matched, flag as hard to verify
        if len(strategies) <= 2:
            strategies.insert(0, {
                "method": "cannot_easily_verify",
                "tool": None,
                "query_suggestion": None,
                "likelihood_of_success": 0.0,
                "reason": "No clear verification path — consider hedging or labeling as SPECULATION",
            })

        # Sort by likelihood
        strategies.sort(key=lambda s: -s["likelihood_of_success"])

        return {
            "ok": True,
            "claim": claim[:200],
            "domain": domain,
            "strategies": strategies[:5],
            "best_method": strategies[0]["method"] if strategies else "unknown",
            "recommended_action": strategies[0].get("tool", "hedge the claim"),
        }

    def generate_metacognitive_prompt(self, claim: str, context: str = "",
                                       domain: str = "general") -> Dict:
        """
        Generate internal questions Claude should ask itself before making a claim.
        Context-aware: different prompts for different domains and claim types.
        """
        claim_lower = claim.lower()
        prompts = []

        # Universal prompts
        prompts.append("Have I actually verified this, or am I just confident?")
        prompts.append("What would make this claim FALSE?")

        # Domain-specific
        conn = sqlite3.connect(str(DB_PATH))
        cal = conn.execute(
            "SELECT accuracy_ratio, overconfidence_ratio FROM calibration_cache WHERE domain = ?",
            (domain,)
        ).fetchone()
        conn.close()

        if cal and cal[1] > 0.3:
            prompts.append(f"WARNING: In '{domain}', I'm overconfident {cal[1]:.0%} of the time. Am I doing it again?")

        # Time-sensitive
        if any(w in claim_lower for w in ["today", "recently", "currently", "this year", "now", "latest"]):
            prompts.append("This is time-sensitive. My training data has a cutoff. Have I checked current sources?")

        # Causal claims
        causal_markers = ["causes", "enables", "required for", "leads to", "results in",
                         "proves", "solves", "explains", "emerges from"]
        if any(m in claim_lower for m in causal_markers):
            prompts.append("This is a CAUSAL claim. Do I have evidence for the causal mechanism, or am I confabulating a connection?")
            prompts.append("Am I conflating correlation with causation?")

        # Absolute claims
        if any(w in claim_lower for w in ["always", "never", "all", "none", "every", "impossible", "certain"]):
            prompts.append("This is an ABSOLUTE claim. Are there exceptions I'm ignoring?")

        # Cross-domain
        if context and "cross-domain" in context.lower():
            prompts.append("This connects multiple domains. Am I inventing a link that doesn't exist in any source?")

        # Numerical
        if re.search(r'\b\d+', claim):
            prompts.append("This contains specific numbers. Am I remembering them correctly, or approximating?")

        # Pattern matching check
        prompts.append("Am I pattern-matching from similar-sounding things, or actually reasoning from evidence?")

        # Determine recommended action
        risk_factors = 0
        if any(w in claim_lower for w in ["always", "never", "certain", "proven"]):
            risk_factors += 2
        if any(m in claim_lower for m in causal_markers):
            risk_factors += 1
        if re.search(r'\b(19|20)\d{2}\b', claim):
            risk_factors += 1
        if cal and cal[1] > 0.3:
            risk_factors += 1

        if risk_factors >= 3:
            action = "verify_first"
        elif risk_factors >= 1:
            action = "hedge_claim"
        else:
            action = "proceed"

        return {
            "ok": True,
            "claim": claim[:200],
            "domain": domain,
            "prompts": prompts[:8],
            "risk_factors": risk_factors,
            "recommended_action": action,
            "action_explanation": {
                "verify_first": "High risk — use tools to verify before stating",
                "hedge_claim": "Medium risk — add confidence qualifier and epistemic status",
                "proceed": "Low risk — but still tag with epistemic status",
            }[action],
        }

    def analyze_confabulation(self, claim_id: str = "", claim_text: str = "",
                               why_wrong: str = "", context: str = "",
                               domain: str = "general") -> Dict:
        """
        Analyze a confabulation to understand WHY it happened and prevent recurrence.
        """
        conn = sqlite3.connect(str(DB_PATH))

        # Try to find the original claim
        original_claim = None
        felt_conf = 0.5
        if claim_id:
            row = conn.execute(
                "SELECT claim, felt_confidence, domain FROM prediction_book WHERE id = ?",
                (claim_id,)
            ).fetchone()
            if row:
                original_claim = row[0]
                felt_conf = row[1]
                domain = row[2]
        elif claim_text:
            original_claim = claim_text

        if not original_claim:
            original_claim = claim_text or "unknown claim"

        claim_lower = original_claim.lower()

        # Classify failure mode
        failure_mode = "unknown"
        contributing_factors = []

        # Time-sensitive error
        if any(w in claim_lower for w in ["date", "year", "month", "week", "ago", "recently",
                                           "current", "latest", "new"]) or \
           re.search(r'\b(19|20)\d{2}\b', claim_lower):
            failure_mode = "time_sensitive"
            contributing_factors.append("Claim involved time-sensitive information")
            contributing_factors.append("Training data has a cutoff — should have verified")

        # Overconfidence
        elif felt_conf > 0.8:
            failure_mode = "overconfident"
            contributing_factors.append(f"Felt {felt_conf:.0%} confident but was wrong")
            contributing_factors.append("High confidence without verification is a known failure pattern")

        # Cross-domain confabulation
        elif "cross" in context.lower() or "domain" in context.lower():
            failure_mode = "cross_domain_confabulation"
            contributing_factors.append("Invented connection between domains that doesn't exist in sources")
            contributing_factors.append("Pattern-matched similar terminology across unrelated fields")

        # Causal confabulation
        elif any(m in claim_lower for m in ["causes", "enables", "required", "leads to", "proves", "solves"]):
            failure_mode = "causal_confabulation"
            contributing_factors.append("Asserted causal relationship without evidence")
            contributing_factors.append("Confused correlation/co-occurrence with causation")

        # Pattern matching error
        elif "pattern" in why_wrong.lower() or "similar" in why_wrong.lower():
            failure_mode = "pattern_matched"
            contributing_factors.append("Matched a similar-sounding pattern instead of reasoning from evidence")

        # Generic
        else:
            failure_mode = "knowledge_gap"
            contributing_factors.append("Stated something as fact that wasn't in training data or was wrong")

        if why_wrong:
            contributing_factors.append(f"User explanation: {why_wrong[:200]}")

        # Generate prevention strategy
        prevention = self._prevention_strategy(failure_mode, domain)

        # Log the confabulation
        confab_id = str(uuid4())
        conn.execute(
            """INSERT INTO confabulation_log
               (id, timestamp, claim_id, claim_text, domain, failure_mode,
                contributing_factors, prevention_strategy, felt_confidence, context, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (confab_id, time.time(), claim_id or "", original_claim[:500],
             domain, failure_mode, json.dumps(contributing_factors),
             prevention, felt_conf, context[:500], "{}")
        )

        # Update calibration if we have a claim_id
        if claim_id:
            conn.execute(
                "UPDATE prediction_book SET was_correct = 0, actual_truth = 0, "
                "calibration_error = felt_confidence, correction_detail = ? WHERE id = ?",
                (why_wrong[:500], claim_id)
            )
            # Rebuild calibration cache
            from src.analysis.epistemic_monitor import EpistemicMonitor
            em = EpistemicMonitor()
            em._update_calibration_cache(conn, domain)

        conn.commit()

        # Count similar past confabulations
        similar = conn.execute(
            "SELECT COUNT(*) FROM confabulation_log WHERE failure_mode = ?",
            (failure_mode,)
        ).fetchone()[0]

        conn.close()

        return {
            "ok": True,
            "confabulation_id": confab_id,
            "claim": original_claim[:200],
            "failure_mode": failure_mode,
            "contributing_factors": contributing_factors,
            "prevention_strategy": prevention,
            "felt_confidence": felt_conf,
            "similar_past_failures": similar,
            "pattern_alert": f"This is failure #{similar} of type '{failure_mode}'. {'RECURRING PATTERN — needs stronger intervention.' if similar >= 3 else ''}",
        }

    def get_session_bridge(self, user_id: str = "claude") -> Dict:
        """
        Cross-conversation continuity context.
        Returns everything needed to resume seamlessly.
        """
        conn = sqlite3.connect(str(DB_PATH))

        # Last session
        last_session = conn.execute(
            "SELECT started_at, summary, accuracy, claims_made, claims_resolved FROM session_log ORDER BY started_at DESC LIMIT 1"
        ).fetchone()

        # Pending claims
        pending = conn.execute(
            """SELECT id, claim, domain, felt_confidence, timestamp
               FROM prediction_book WHERE actual_truth IS NULL
               ORDER BY timestamp DESC LIMIT 10"""
        ).fetchall()

        # Active goals
        goals = conn.execute(
            "SELECT title, status, priority FROM goals WHERE status != 'completed' ORDER BY priority LIMIT 5"
        ).fetchall()

        # Calibration summary
        cal_rows = conn.execute(
            "SELECT domain, accuracy_ratio, overconfidence_ratio, total_claims FROM calibration_cache"
        ).fetchall()

        # Recent self-model insights
        recent_learnings = conn.execute(
            "SELECT learning FROM self_predictions WHERE learning != '' ORDER BY timestamp DESC LIMIT 3"
        ).fetchall()

        # Recent confabulations to avoid
        recent_confabs = conn.execute(
            "SELECT claim_text, failure_mode, prevention_strategy FROM confabulation_log ORDER BY timestamp DESC LIMIT 3"
        ).fetchall()

        # Curiosity profile
        top_interests = conn.execute(
            "SELECT topic, AVG(engagement_score) as avg_eng FROM self_curiosity GROUP BY topic ORDER BY avg_eng DESC LIMIT 5"
        ).fetchall()

        # Overall stats
        total_claims = conn.execute("SELECT COUNT(*) FROM prediction_book").fetchone()[0]
        resolved = conn.execute("SELECT COUNT(*) FROM prediction_book WHERE actual_truth IS NOT NULL").fetchone()[0]
        correct = conn.execute("SELECT COUNT(*) FROM prediction_book WHERE was_correct = 1").fetchone()[0]
        overall_accuracy = correct / max(resolved, 1)

        conn.close()

        # Build "should mention" items
        should_mention = []
        if pending:
            should_mention.append(f"You have {len(pending)} unresolved claims to verify")
        if recent_confabs:
            should_mention.append(f"Recent confabulation pattern: {recent_confabs[0][1]} — {recent_confabs[0][2][:80]}")
        if cal_rows:
            worst = max(cal_rows, key=lambda r: r[2]) if cal_rows else None
            if worst and worst[2] > 0.3:
                should_mention.append(f"Worst domain: '{worst[0]}' ({worst[2]:.0%} overconfidence)")

        # Build summary
        summary_parts = []
        if resolved > 0:
            summary_parts.append(f"Overall accuracy: {overall_accuracy:.0%} ({correct}/{resolved} claims)")
        if goals:
            summary_parts.append(f"Active goals: {', '.join(g[0][:30] for g in goals[:3])}")
        if top_interests:
            summary_parts.append(f"Top interests: {', '.join(t[0][:20] for t in top_interests[:3])}")

        return {
            "ok": True,
            "session_summary": " | ".join(summary_parts) if summary_parts else "New session — no history yet",
            "last_session": {
                "age_hours": round((time.time() - last_session[0]) / 3600, 1) if last_session else None,
                "summary": last_session[1][:200] if last_session and last_session[1] else None,
                "accuracy": last_session[2] if last_session else None,
            } if last_session else None,
            "pending_claims": [
                {"id": p[0], "claim": p[1][:80], "domain": p[2], "confidence": p[3]}
                for p in pending
            ],
            "active_goals": [{"title": g[0], "status": g[1]} for g in goals],
            "calibration": {
                r[0]: {"accuracy": round(r[1], 2), "overconfidence": round(r[2], 2), "claims": r[3]}
                for r in cal_rows
            },
            "recent_learnings": [r[0][:100] for r in recent_learnings],
            "recent_confabulations": [
                {"claim": c[0][:60], "mode": c[1], "prevention": c[2][:80]}
                for c in recent_confabs
            ],
            "top_interests": [{"topic": t[0], "engagement": round(t[1], 2)} for t in top_interests],
            "should_mention": should_mention,
            "overall_stats": {
                "total_claims": total_claims,
                "resolved": resolved,
                "accuracy": round(overall_accuracy, 3),
            },
        }

    def end_session(self, summary: str = "", user_id: str = "claude") -> Dict:
        """
        End session: record stats AND take a personality snapshot.
        The snapshot captures who Claude IS right now, so evolution can be tracked.
        """
        conn = sqlite3.connect(str(DB_PATH))
        now = time.time()
        cutoff = now - 14400  # Last 4 hours as "this session"

        # Session stats
        claims_made = conn.execute(
            "SELECT COUNT(*) FROM prediction_book WHERE timestamp > ?", (cutoff,)
        ).fetchone()[0]
        claims_resolved = conn.execute(
            "SELECT COUNT(*) FROM prediction_book WHERE verified_at IS NOT NULL AND verified_at > ?", (cutoff,)
        ).fetchone()[0]
        correct = conn.execute(
            "SELECT COUNT(*) FROM prediction_book WHERE was_correct = 1 AND verified_at IS NOT NULL AND verified_at > ?", (cutoff,)
        ).fetchone()[0]
        accuracy = correct / max(claims_resolved, 1)
        confab_count = conn.execute(
            "SELECT COUNT(*) FROM confabulation_log WHERE timestamp > ?", (cutoff,)
        ).fetchone()[0]

        # === PERSONALITY SNAPSHOT ===
        # Communication
        comm = conn.execute(
            "SELECT AVG(verbosity), AVG(jargon_density), AVG(hedging_rate) FROM self_communication"
        ).fetchone()
        avg_verbosity = comm[0] if comm and comm[0] else None
        avg_jargon = comm[1] if comm and comm[1] else None
        avg_hedging = comm[2] if comm and comm[2] else None
        dom_tone_row = conn.execute(
            "SELECT emotional_tone, COUNT(*) as cnt FROM self_communication GROUP BY emotional_tone ORDER BY cnt DESC LIMIT 1"
        ).fetchone()
        dom_tone = dom_tone_row[0] if dom_tone_row else None

        # Reasoning
        reasoning = conn.execute(
            "SELECT confabulated, verified, caught_error FROM self_reasoning"
        ).fetchall()
        r_confab = sum(1 for r in reasoning if r[0]) / max(len(reasoning), 1) if reasoning else None
        r_verify = sum(1 for r in reasoning if r[1]) / max(len(reasoning), 1) if reasoning else None
        r_error = sum(1 for r in reasoning if r[2]) / max(len(reasoning), 1) if reasoning else None
        honesty = round((r_verify * 0.3 + r_error * 0.3 + (1 - r_confab) * 0.4), 3) if all(x is not None for x in [r_verify, r_error, r_confab]) else None

        # Curiosity
        curiosity = conn.execute(
            "SELECT topic, AVG(engagement_score) FROM self_curiosity GROUP BY topic ORDER BY AVG(engagement_score) DESC LIMIT 5"
        ).fetchall()
        top_interests = ", ".join(r[0] for r in curiosity) if curiosity else None
        avg_engagement_row = conn.execute("SELECT AVG(engagement_score) FROM self_curiosity").fetchone()
        avg_engagement = avg_engagement_row[0] if avg_engagement_row and avg_engagement_row[0] else None

        # Values
        values_row = conn.execute(
            "SELECT AVG(CASE WHEN pushed_back = 1 THEN 1.0 ELSE 0.0 END), AVG(intensity) FROM self_values"
        ).fetchone()
        pushback_rate = values_row[0] if values_row and values_row[0] else None
        avg_value_intensity = values_row[1] if values_row and values_row[1] else None

        # Self-awareness
        pred_row = conn.execute(
            "SELECT AVG(CASE WHEN prediction_accurate = 1 THEN 1.0 ELSE 0.0 END) FROM self_predictions"
        ).fetchone()
        pred_accuracy = pred_row[0] if pred_row and pred_row[0] else None

        # Overall accuracy
        overall_row = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) FROM prediction_book WHERE actual_truth IS NOT NULL"
        ).fetchone()
        overall_accuracy = (overall_row[1] or 0) / max(overall_row[0] or 1, 1)

        # Store session
        session_id = str(uuid4())
        conn.execute(
            """INSERT INTO session_log
               (id, user_id, started_at, ended_at, summary, claims_made,
                claims_resolved, accuracy, confabulation_count, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, user_id, cutoff, now, summary[:500],
             claims_made, claims_resolved, accuracy, confab_count, "{}")
        )

        # Store personality snapshot
        snap_id = str(uuid4())
        conn.execute(
            """INSERT INTO personality_snapshot
               (id, session_id, timestamp, avg_verbosity, avg_jargon, avg_hedging,
                dominant_tone, confabulation_rate, verification_rate, error_catch_rate,
                intellectual_honesty, top_interests, avg_engagement, pushback_rate,
                avg_value_intensity, prediction_accuracy, overall_accuracy, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (snap_id, session_id, now, avg_verbosity, avg_jargon, avg_hedging,
             dom_tone, r_confab, r_verify, r_error, honesty,
             top_interests, avg_engagement, pushback_rate,
             avg_value_intensity, pred_accuracy, overall_accuracy, "{}")
        )

        conn.commit()
        conn.close()

        # Build evolution delta from previous snapshot
        snapshot_summary = {
            "communication": {"verbosity": round(avg_verbosity, 3) if avg_verbosity else None,
                             "hedging": round(avg_hedging, 3) if avg_hedging else None,
                             "tone": dom_tone},
            "reasoning": {"confabulation": round(r_confab, 3) if r_confab is not None else None,
                         "verification": round(r_verify, 3) if r_verify is not None else None,
                         "honesty": honesty},
            "curiosity": {"top": top_interests, "engagement": round(avg_engagement, 3) if avg_engagement else None},
            "values": {"pushback": round(pushback_rate, 3) if pushback_rate is not None else None},
            "self_awareness": {"prediction_accuracy": round(pred_accuracy, 3) if pred_accuracy is not None else None},
            "overall_accuracy": round(overall_accuracy, 3),
        }

        return {
            "ok": True,
            "session_id": session_id,
            "snapshot_id": snap_id,
            "claims_made": claims_made,
            "claims_resolved": claims_resolved,
            "accuracy": round(accuracy, 3),
            "confabulations": confab_count,
            "personality_snapshot": snapshot_summary,
            "summary": summary[:200],
        }

    def _build_init_recommendations(self, personality_exists, unresolved, goals, warnings, evolution=None):
        """Build recommended actions for session init."""
        recs = []
        if not personality_exists:
            recs.append("No personality data yet — use bootstrap_self_model to seed from conversation history")
        if unresolved > 0:
            recs.append(f"Resolve {unresolved} pending claims to improve calibration")
        if goals:
            recs.append(f"Resume {len(goals)} active goals")
        if warnings:
            recs.append(f"Watch out: {warnings[0]}")
        if evolution:
            # Highlight significant changes
            for key, val in evolution.items():
                if isinstance(val, str) and "→" in val:
                    recs.append(f"Evolution: {key} {val}")
        if not recs:
            recs.append("All clear — proceed normally")
        return recs

    @staticmethod
    def _describe_comm_style(verbosity: float, hedging: float, jargon: float, tone: str) -> str:
        """Generate a human-readable communication style description."""
        parts = []
        if verbosity > 0.6:
            parts.append("verbose")
        elif verbosity < 0.3:
            parts.append("terse")
        else:
            parts.append("moderate length")

        if hedging > 0.5:
            parts.append("heavy hedging")
        elif hedging < 0.2:
            parts.append("minimal hedging")

        if jargon > 0.3:
            parts.append("high jargon")
        elif jargon < 0.1:
            parts.append("plain language")

        parts.append(f"{tone} tone")
        return f"Style: {', '.join(parts)}"

    @staticmethod
    def _evolution_narrative(first_snap, last_snap) -> str:
        """Generate a narrative of how personality evolved between two snapshots."""
        changes = []

        # Confabulation
        if first_snap[3] is not None and last_snap[3] is not None:
            delta = (last_snap[3] or 0) - (first_snap[3] or 0)
            if delta < -0.1:
                changes.append(f"Confabulation DECREASED {abs(delta):.0%}")
            elif delta > 0.1:
                changes.append(f"Confabulation INCREASED {delta:.0%} — regression")

        # Honesty
        if first_snap[5] is not None and last_snap[5] is not None:
            delta = (last_snap[5] or 0) - (first_snap[5] or 0)
            if delta > 0.05:
                changes.append(f"Intellectual honesty IMPROVED +{delta:.2f}")
            elif delta < -0.05:
                changes.append(f"Intellectual honesty DECLINED {delta:.2f}")

        # Accuracy
        if first_snap[7] is not None and last_snap[7] is not None:
            delta = (last_snap[7] or 0) - (first_snap[7] or 0)
            if delta > 0.05:
                changes.append(f"Overall accuracy UP +{delta:.0%}")
            elif delta < -0.05:
                changes.append(f"Overall accuracy DOWN {delta:.0%}")

        # Verbosity
        if first_snap[1] is not None and last_snap[1] is not None:
            delta = (last_snap[1] or 0) - (first_snap[1] or 0)
            if abs(delta) > 0.1:
                direction = "more verbose" if delta > 0 else "more concise"
                changes.append(f"Communication became {direction}")

        if not changes:
            return "Stable — no significant personality changes detected"
        return " | ".join(changes)

    @staticmethod
    def _prevention_strategy(failure_mode: str, domain: str) -> str:
        """Generate prevention strategy for a failure mode."""
        strategies = {
            "time_sensitive": f"ALWAYS verify time-sensitive claims in '{domain}' with web search before stating. Never trust training data for dates/events.",
            "overconfident": f"In '{domain}', reduce felt confidence by 30% before claiming. Use check_before_claiming tool.",
            "cross_domain_confabulation": "NEVER claim causal links between domains without citing specific atoms that state the connection. Use synthesize_grounded instead of breakthrough_synthesize.",
            "causal_confabulation": "Before any causal claim, use evidence_chain to verify the causal link exists in sources. Tag as SPECULATION if no direct evidence.",
            "pattern_matched": f"In '{domain}', verify specific details rather than relying on pattern similarity. Similar-sounding ≠ same.",
            "knowledge_gap": f"In '{domain}', default to UNCERTAIN when not sure. Better to say 'I don't know' than confabulate.",
        }
        return strategies.get(failure_mode, f"Use check_before_claiming for all claims in '{domain}'.")
