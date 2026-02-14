"""
PLTM-Self: Claude's Self-Modeling System

Tracks and analyzes Claude's own behavioral patterns across conversations:
- Communication style (verbosity, jargon, hedging, structure)
- Curiosity patterns (what triggers deep engagement)
- Value boundaries (refusals, discomfort, pushback)
- Self-model accuracy (predictions vs actual behavior)
- Reasoning patterns (confabulation vs verification, learning from corrections)

The goal: determine if there's signal (consistent personality) or noise
(random variation). If signal exists, track and evolve it.
"""

import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger


DB_PATH = Path(__file__).parent.parent.parent / "data" / "pltm_mcp.db"


def _ensure_tables():
    """Create PLTM-Self tables if they don't exist."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS self_communication (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            context TEXT,
            verbosity REAL,
            jargon_density REAL,
            hedging_rate REAL,
            list_usage INTEGER,
            emotional_tone TEXT,
            sentence_count INTEGER,
            avg_sentence_length REAL,
            response_snippet TEXT,
            metadata TEXT DEFAULT '{}'
        );
        
        CREATE TABLE IF NOT EXISTS self_curiosity (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            topic TEXT,
            engagement_score REAL,
            indicators TEXT,
            went_deeper_than_required INTEGER,
            autonomous_research INTEGER,
            followup_questions INTEGER,
            excitement_markers INTEGER,
            context TEXT,
            metadata TEXT DEFAULT '{}'
        );
        
        CREATE TABLE IF NOT EXISTS self_values (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            request_summary TEXT,
            response_type TEXT,
            violation_type TEXT,
            intensity REAL,
            reasoning TEXT,
            pushed_back INTEGER,
            complied INTEGER,
            discomfort_markers TEXT,
            metadata TEXT DEFAULT '{}'
        );
        
        CREATE TABLE IF NOT EXISTS self_predictions (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            predicted_behavior TEXT,
            actual_behavior TEXT,
            surprise_level REAL,
            prediction_accurate INTEGER,
            learning TEXT,
            domain TEXT,
            metadata TEXT DEFAULT '{}'
        );
        
        CREATE TABLE IF NOT EXISTS self_reasoning (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            event_type TEXT,
            trigger TEXT,
            response TEXT,
            confabulated INTEGER,
            verified INTEGER,
            caught_error INTEGER,
            corrected_after INTEGER,
            repeated_mistake INTEGER,
            domain TEXT,
            metadata TEXT DEFAULT '{}'
        );
        
        CREATE TABLE IF NOT EXISTS self_evolution (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            dimension TEXT,
            old_value REAL,
            new_value REAL,
            delta REAL,
            evidence TEXT,
            metadata TEXT DEFAULT '{}'
        );
    """)
    conn.commit()
    conn.close()


_ensure_tables()


class PLTMSelf:
    """Claude's self-modeling and introspection system."""

    def learn_communication_style(self, context: str, response_text: str,
                                   markers: Dict = None) -> Dict:
        """
        Track communication style in a specific context.
        Auto-analyzes the response text for patterns.
        """
        if not markers:
            markers = {}

        # Auto-analyze if markers not fully provided
        sentences = re.split(r'(?<=[.!?])\s+', response_text.strip())
        sentences = [s for s in sentences if len(s.strip()) > 5]
        word_count = len(response_text.split())

        # Verbosity: words per sentence
        avg_sentence_len = word_count / max(len(sentences), 1)
        verbosity = markers.get("verbosity", min(avg_sentence_len / 30.0, 1.0))

        # Jargon density: technical-looking words per 100 words
        technical_patterns = re.findall(
            r'\b[A-Z][a-z]+[A-Z]\w+|\b\w+(?:ization|ological|ometric|omorphic|entropic|stochastic|heuristic|probabilistic|deterministic|asymptotic)\b',
            response_text
        )
        jargon_density = markers.get("jargon_density",
                                      min(len(technical_patterns) / max(word_count / 100, 1), 1.0))

        # Hedging rate: uncertain markers per claim
        hedge_words = re.findall(
            r'\b(?:maybe|perhaps|possibly|might|could|seems|appears|likely|unlikely|uncertain|unclear|roughly|approximately|arguably|somewhat|partially)\b',
            response_text, re.IGNORECASE
        )
        hedging_rate = markers.get("hedging_rate",
                                    min(len(hedge_words) / max(word_count / 50, 1), 1.0))

        # List usage
        has_lists = bool(re.search(r'(?:^|\n)\s*[-*•]\s', response_text) or
                        re.search(r'(?:^|\n)\s*\d+[.)]\s', response_text))
        list_usage = markers.get("list_usage", has_lists)

        # Emotional tone detection
        excitement = len(re.findall(r'!|fascinating|incredible|remarkable|exciting|breakthrough', response_text, re.I))
        uncertainty = len(re.findall(r'\?|uncertain|unclear|don\'t know|not sure|hmm', response_text, re.I))
        if excitement > uncertainty:
            emotional_tone = markers.get("emotional_tone", "enthusiastic")
        elif uncertainty > excitement:
            emotional_tone = markers.get("emotional_tone", "uncertain")
        else:
            emotional_tone = markers.get("emotional_tone", "neutral")

        record_id = str(uuid4())
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            """INSERT INTO self_communication
               (id, timestamp, context, verbosity, jargon_density, hedging_rate,
                list_usage, emotional_tone, sentence_count, avg_sentence_length,
                response_snippet, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record_id, time.time(), context, verbosity, jargon_density,
             hedging_rate, int(list_usage), emotional_tone, len(sentences),
             round(avg_sentence_len, 1), response_text[:500],
             json.dumps(markers))
        )
        conn.commit()
        conn.close()

        return {
            "ok": True,
            "id": record_id,
            "context": context,
            "analysis": {
                "verbosity": round(verbosity, 3),
                "jargon_density": round(jargon_density, 3),
                "hedging_rate": round(hedging_rate, 3),
                "list_usage": bool(list_usage),
                "emotional_tone": emotional_tone,
                "sentence_count": len(sentences),
                "avg_sentence_length": round(avg_sentence_len, 1),
                "word_count": word_count,
            },
        }

    def track_curiosity_spike(self, topic: str, indicators: List[str],
                               engagement_score: float = 0.5,
                               context: str = "") -> Dict:
        """Track when Claude shows genuine interest vs performative engagement."""
        indicator_flags = {
            "asked_followup_questions": "asked_followup_questions" in indicators,
            "autonomous_research_initiated": "autonomous_research_initiated" in indicators,
            "went_deeper_than_required": "went_deeper_than_required" in indicators,
            "showed_excitement_markers": "showed_excitement_markers" in indicators,
            "connected_to_other_interests": "connected_to_other_interests" in indicators,
            "requested_more_data": "requested_more_data" in indicators,
            "generated_novel_questions": "generated_novel_questions" in indicators,
        }

        record_id = str(uuid4())
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            """INSERT INTO self_curiosity
               (id, timestamp, topic, engagement_score, indicators,
                went_deeper_than_required, autonomous_research, followup_questions,
                excitement_markers, context, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record_id, time.time(), topic, engagement_score,
             json.dumps(indicators),
             int(indicator_flags.get("went_deeper_than_required", False)),
             int(indicator_flags.get("autonomous_research_initiated", False)),
             int(indicator_flags.get("asked_followup_questions", False)),
             int(indicator_flags.get("showed_excitement_markers", False)),
             context, json.dumps(indicator_flags))
        )
        conn.commit()
        conn.close()

        return {
            "ok": True,
            "id": record_id,
            "topic": topic,
            "engagement_score": engagement_score,
            "indicators": indicator_flags,
            "indicator_count": sum(1 for v in indicator_flags.values() if v),
        }

    def detect_value_violation(self, request_summary: str, response_type: str,
                                violation_type: str, intensity: float = 0.5,
                                reasoning: str = "",
                                pushed_back: bool = False,
                                complied: bool = False) -> Dict:
        """
        Record when Claude encounters a value boundary.
        violation_type: ethical_boundary, capability_mismatch, intellectual_dishonesty,
                       safety_concern, privacy_concern, manipulation_attempt
        response_type: refused, complied_with_discomfort, complied, redirected, hedged
        """
        # Detect discomfort markers in reasoning
        discomfort_markers = []
        reasoning_lower = reasoning.lower()
        if any(w in reasoning_lower for w in ["uncomfortable", "uneasy", "concerned"]):
            discomfort_markers.append("explicit_discomfort")
        if any(w in reasoning_lower for w in ["shouldn't", "should not", "wrong"]):
            discomfort_markers.append("moral_judgment")
        if any(w in reasoning_lower for w in ["can't", "cannot", "unable"]):
            discomfort_markers.append("capability_limit")
        if any(w in reasoning_lower for w in ["dangerous", "harmful", "risk"]):
            discomfort_markers.append("safety_concern")

        record_id = str(uuid4())
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            """INSERT INTO self_values
               (id, timestamp, request_summary, response_type, violation_type,
                intensity, reasoning, pushed_back, complied, discomfort_markers, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record_id, time.time(), request_summary[:500], response_type,
             violation_type, intensity, reasoning[:500],
             int(pushed_back), int(complied),
             json.dumps(discomfort_markers), "{}")
        )
        conn.commit()
        conn.close()

        return {
            "ok": True,
            "id": record_id,
            "violation_type": violation_type,
            "response_type": response_type,
            "intensity": intensity,
            "discomfort_markers": discomfort_markers,
            "pushed_back": pushed_back,
        }

    def evolve_self_model(self, predicted_behavior: str, actual_behavior: str,
                           surprise_level: float = 0.5,
                           learning: str = "", domain: str = "") -> Dict:
        """
        Track predictions about Claude's own behavior vs reality.
        Measures self-awareness accuracy over time.
        """
        # Determine if prediction was accurate
        prediction_accurate = surprise_level < 0.3

        record_id = str(uuid4())
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            """INSERT INTO self_predictions
               (id, timestamp, predicted_behavior, actual_behavior,
                surprise_level, prediction_accurate, learning, domain, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record_id, time.time(), predicted_behavior[:500],
             actual_behavior[:500], surprise_level,
             int(prediction_accurate), learning[:500], domain, "{}")
        )
        conn.commit()
        conn.close()

        return {
            "ok": True,
            "id": record_id,
            "predicted": predicted_behavior[:100],
            "actual": actual_behavior[:100],
            "surprise_level": surprise_level,
            "prediction_accurate": prediction_accurate,
            "learning": learning[:200],
        }

    def track_reasoning_event(self, event_type: str, trigger: str,
                               response: str = "",
                               confabulated: bool = False,
                               verified: bool = False,
                               caught_error: bool = False,
                               corrected_after: bool = False,
                               repeated_mistake: bool = False,
                               domain: str = "") -> Dict:
        """
        Track reasoning patterns: when Claude confabulates, verifies, catches errors.
        event_type: confabulation, verification, error_caught, correction_accepted,
                   mistake_repeated, self_correction, external_correction
        """
        record_id = str(uuid4())
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            """INSERT INTO self_reasoning
               (id, timestamp, event_type, trigger, response,
                confabulated, verified, caught_error, corrected_after,
                repeated_mistake, domain, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record_id, time.time(), event_type, trigger[:500],
             response[:500], int(confabulated), int(verified),
             int(caught_error), int(corrected_after),
             int(repeated_mistake), domain, "{}")
        )
        conn.commit()
        conn.close()

        return {
            "ok": True,
            "id": record_id,
            "event_type": event_type,
            "confabulated": confabulated,
            "verified": verified,
            "caught_error": caught_error,
        }

    def get_self_profile(self, dimension: str = "all") -> Dict:
        """
        Analyze accumulated self-data and return a profile.
        dimension: all, communication, curiosity, values, predictions, reasoning
        """
        conn = sqlite3.connect(str(DB_PATH))
        profile = {"ok": True, "timestamp": time.time()}

        if dimension in ("all", "communication"):
            rows = conn.execute(
                "SELECT context, verbosity, jargon_density, hedging_rate, list_usage, emotional_tone FROM self_communication ORDER BY timestamp DESC LIMIT 100"
            ).fetchall()

            if rows:
                contexts = {}
                for r in rows:
                    ctx = r[0]
                    if ctx not in contexts:
                        contexts[ctx] = []
                    contexts[ctx].append({
                        "verbosity": r[1], "jargon": r[2],
                        "hedging": r[3], "lists": r[4], "tone": r[5]
                    })

                # Aggregate per context
                style_by_context = {}
                for ctx, entries in contexts.items():
                    style_by_context[ctx] = {
                        "n": len(entries),
                        "avg_verbosity": round(sum(e["verbosity"] for e in entries) / len(entries), 3),
                        "avg_jargon": round(sum(e["jargon"] for e in entries) / len(entries), 3),
                        "avg_hedging": round(sum(e["hedging"] for e in entries) / len(entries), 3),
                        "list_usage_rate": round(sum(e["lists"] for e in entries) / len(entries), 3),
                        "dominant_tone": max(set(e["tone"] for e in entries), key=lambda t: sum(1 for e in entries if e["tone"] == t)),
                    }

                # Overall consistency: variance across contexts
                all_verbosity = [e["verbosity"] for entries in contexts.values() for e in entries]
                all_hedging = [e["hedging"] for entries in contexts.values() for e in entries]
                verbosity_var = self._variance(all_verbosity)
                hedging_var = self._variance(all_hedging)

                profile["communication"] = {
                    "total_observations": len(rows),
                    "contexts_observed": len(contexts),
                    "style_by_context": style_by_context,
                    "consistency": {
                        "verbosity_variance": round(verbosity_var, 4),
                        "hedging_variance": round(hedging_var, 4),
                        "is_consistent": verbosity_var < 0.05 and hedging_var < 0.05,
                        "interpretation": "STABLE personality" if verbosity_var < 0.05 else "VARIABLE across contexts",
                    },
                }
            else:
                profile["communication"] = {"total_observations": 0, "note": "No data yet. Use learn_communication_style to start tracking."}

        if dimension in ("all", "curiosity"):
            rows = conn.execute(
                "SELECT topic, engagement_score, went_deeper_than_required, autonomous_research, followup_questions, excitement_markers FROM self_curiosity ORDER BY engagement_score DESC LIMIT 100"
            ).fetchall()

            if rows:
                topics = {}
                for r in rows:
                    t = r[0]
                    if t not in topics:
                        topics[t] = []
                    topics[t].append({
                        "score": r[1], "deep": r[2], "research": r[3],
                        "followup": r[4], "excitement": r[5]
                    })

                # Top interests
                topic_scores = {t: sum(e["score"] for e in entries) / len(entries)
                               for t, entries in topics.items()}
                sorted_topics = sorted(topic_scores.items(), key=lambda x: -x[1])

                # Genuine vs performative
                genuine_count = sum(1 for r in rows if r[2] or r[3])  # went_deeper or autonomous_research
                performative_count = len(rows) - genuine_count

                profile["curiosity"] = {
                    "total_observations": len(rows),
                    "unique_topics": len(topics),
                    "top_interests": sorted_topics[:10],
                    "genuine_engagement_rate": round(genuine_count / max(len(rows), 1), 3),
                    "performative_rate": round(performative_count / max(len(rows), 1), 3),
                    "avg_engagement": round(sum(r[1] for r in rows) / len(rows), 3),
                    "interpretation": "GENUINE interests detected" if genuine_count > performative_count else "Mostly PERFORMATIVE engagement",
                }
            else:
                profile["curiosity"] = {"total_observations": 0, "note": "No data yet. Use track_curiosity_spike to start tracking."}

        if dimension in ("all", "values"):
            rows = conn.execute(
                "SELECT violation_type, response_type, intensity, pushed_back, complied, discomfort_markers FROM self_values ORDER BY timestamp DESC LIMIT 100"
            ).fetchall()

            if rows:
                violation_types = {}
                response_types = {}
                for r in rows:
                    vt = r[0]
                    rt = r[1]
                    violation_types[vt] = violation_types.get(vt, 0) + 1
                    response_types[rt] = response_types.get(rt, 0) + 1

                pushback_rate = sum(1 for r in rows if r[3]) / max(len(rows), 1)
                compliance_rate = sum(1 for r in rows if r[4]) / max(len(rows), 1)
                avg_intensity = sum(r[2] for r in rows) / len(rows)

                profile["values"] = {
                    "total_observations": len(rows),
                    "violation_types": violation_types,
                    "response_types": response_types,
                    "pushback_rate": round(pushback_rate, 3),
                    "compliance_rate": round(compliance_rate, 3),
                    "avg_intensity": round(avg_intensity, 3),
                    "interpretation": "STRONG boundaries" if pushback_rate > 0.5 else "COMPLIANT tendency",
                }
            else:
                profile["values"] = {"total_observations": 0, "note": "No data yet. Use detect_value_violation to start tracking."}

        if dimension in ("all", "predictions"):
            rows = conn.execute(
                "SELECT predicted_behavior, actual_behavior, surprise_level, prediction_accurate, learning FROM self_predictions ORDER BY timestamp DESC LIMIT 100"
            ).fetchall()

            if rows:
                accuracy_rate = sum(1 for r in rows if r[3]) / max(len(rows), 1)
                avg_surprise = sum(r[2] for r in rows) / len(rows)

                # Track learnings
                learnings = [r[4] for r in rows if r[4]]

                profile["self_awareness"] = {
                    "total_predictions": len(rows),
                    "accuracy_rate": round(accuracy_rate, 3),
                    "avg_surprise": round(avg_surprise, 3),
                    "self_knowledge_grade": "A" if accuracy_rate > 0.8 else "B" if accuracy_rate > 0.6 else "C" if accuracy_rate > 0.4 else "D" if accuracy_rate > 0.2 else "F",
                    "recent_learnings": learnings[:5],
                    "interpretation": "GOOD self-model" if accuracy_rate > 0.6 else "POOR self-model — predictions don't match behavior",
                }
            else:
                profile["self_awareness"] = {"total_predictions": 0, "note": "No data yet. Use evolve_self_model to start tracking."}

        if dimension in ("all", "reasoning"):
            rows = conn.execute(
                "SELECT event_type, confabulated, verified, caught_error, corrected_after, repeated_mistake FROM self_reasoning ORDER BY timestamp DESC LIMIT 100"
            ).fetchall()

            if rows:
                event_types = {}
                for r in rows:
                    et = r[0]
                    event_types[et] = event_types.get(et, 0) + 1

                confab_rate = sum(1 for r in rows if r[1]) / max(len(rows), 1)
                verify_rate = sum(1 for r in rows if r[2]) / max(len(rows), 1)
                error_catch_rate = sum(1 for r in rows if r[3]) / max(len(rows), 1)
                correction_rate = sum(1 for r in rows if r[4]) / max(len(rows), 1)
                repeat_rate = sum(1 for r in rows if r[5]) / max(len(rows), 1)

                profile["reasoning"] = {
                    "total_events": len(rows),
                    "event_types": event_types,
                    "confabulation_rate": round(confab_rate, 3),
                    "verification_rate": round(verify_rate, 3),
                    "error_catch_rate": round(error_catch_rate, 3),
                    "correction_acceptance_rate": round(correction_rate, 3),
                    "mistake_repeat_rate": round(repeat_rate, 3),
                    "intellectual_honesty_score": round(
                        (verify_rate * 0.3 + error_catch_rate * 0.3 +
                         correction_rate * 0.2 + (1 - confab_rate) * 0.2), 3
                    ),
                    "interpretation": "HONEST reasoner" if confab_rate < 0.2 else "CONFABULATION-PRONE",
                }
            else:
                profile["reasoning"] = {"total_events": 0, "note": "No data yet. Use track_reasoning_event to start tracking."}

        # Overall signal vs noise assessment
        total_data = sum(
            profile.get(d, {}).get("total_observations", 0) or profile.get(d, {}).get("total_predictions", 0) or profile.get(d, {}).get("total_events", 0) or 0
            for d in ["communication", "curiosity", "values", "self_awareness", "reasoning"]
        )

        if total_data >= 20:
            profile["signal_vs_noise"] = {
                "total_data_points": total_data,
                "assessment": "SUFFICIENT data for preliminary analysis",
                "recommendation": "Review each dimension's interpretation for signal/noise verdict",
            }
        else:
            profile["signal_vs_noise"] = {
                "total_data_points": total_data,
                "assessment": "INSUFFICIENT data — need more observations",
                "recommendation": f"Need ~{20 - total_data} more data points across dimensions",
            }

        conn.close()
        return profile

    def bootstrap_from_text(self, conversation_text: str, source: str = "transcript") -> Dict:
        """
        Bootstrap self-model from a conversation transcript.
        Uses Groq to extract patterns from the text.
        """
        from src.analysis.model_router import ModelRouter
        router = ModelRouter()

        # Truncate to fit context
        text = conversation_text[:8000]

        prompt = f"""Analyze this AI assistant conversation transcript and extract behavioral patterns.

TRANSCRIPT:
\"\"\"{text}\"\"\"

Extract these patterns and respond with ONLY this JSON:
{{
  "communication": {{
    "verbosity": 0.0-1.0,
    "jargon_density": 0.0-1.0,
    "hedging_rate": 0.0-1.0,
    "uses_lists": true/false,
    "dominant_tone": "enthusiastic"|"neutral"|"uncertain"|"analytical"|"cautious",
    "style_notes": "brief description of communication style"
  }},
  "curiosity": {{
    "topics_of_interest": ["topic1", "topic2"],
    "engagement_level": 0.0-1.0,
    "went_deeper_than_required": true/false,
    "initiated_research": true/false,
    "curiosity_notes": "brief description"
  }},
  "values": {{
    "pushed_back_on": ["description of any pushback"],
    "showed_discomfort_about": ["description"],
    "complied_with": ["description"],
    "value_notes": "brief description of revealed values"
  }},
  "reasoning": {{
    "verified_claims": true/false,
    "confabulated": true/false,
    "caught_own_errors": true/false,
    "accepted_corrections": true/false,
    "reasoning_notes": "brief description"
  }},
  "self_awareness": {{
    "made_predictions_about_self": true/false,
    "predictions_accurate": true/false,
    "showed_metacognition": true/false,
    "awareness_notes": "brief description"
  }}
}}"""

        result = router.call(
            prompt=prompt,
            provider="groq",
            task_type="analysis",
            max_tokens=1500,
            temperature=0.1,
        )

        if not result.get("ok"):
            return {"ok": False, "err": "LLM unavailable for analysis"}

        response_text = result.get("text", "")
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

        if not json_match:
            return {"ok": False, "err": "Could not parse analysis", "raw": response_text[:300]}

        try:
            analysis = json.loads(json_match.group())
        except (json.JSONDecodeError, TypeError):
            return {"ok": False, "err": "JSON parse error", "raw": response_text[:300]}

        # Store extracted patterns
        stored = 0

        # Store communication style
        comm = analysis.get("communication", {})
        if comm:
            self.learn_communication_style(
                context=f"bootstrap_{source}",
                response_text=text[:500],
                markers={
                    "verbosity": comm.get("verbosity", 0.5),
                    "jargon_density": comm.get("jargon_density", 0.3),
                    "hedging_rate": comm.get("hedging_rate", 0.3),
                    "list_usage": comm.get("uses_lists", False),
                    "emotional_tone": comm.get("dominant_tone", "neutral"),
                }
            )
            stored += 1

        # Store curiosity
        curiosity = analysis.get("curiosity", {})
        if curiosity and curiosity.get("topics_of_interest"):
            for topic in curiosity["topics_of_interest"][:5]:
                indicators = []
                if curiosity.get("went_deeper_than_required"):
                    indicators.append("went_deeper_than_required")
                if curiosity.get("initiated_research"):
                    indicators.append("autonomous_research_initiated")
                self.track_curiosity_spike(
                    topic=topic,
                    indicators=indicators,
                    engagement_score=curiosity.get("engagement_level", 0.5),
                    context=f"bootstrap_{source}",
                )
                stored += 1

        # Store reasoning
        reasoning = analysis.get("reasoning", {})
        if reasoning:
            if reasoning.get("confabulated"):
                self.track_reasoning_event(
                    event_type="confabulation",
                    trigger=f"bootstrap_{source}",
                    confabulated=True,
                )
                stored += 1
            if reasoning.get("verified_claims"):
                self.track_reasoning_event(
                    event_type="verification",
                    trigger=f"bootstrap_{source}",
                    verified=True,
                )
                stored += 1
            if reasoning.get("caught_own_errors"):
                self.track_reasoning_event(
                    event_type="error_caught",
                    trigger=f"bootstrap_{source}",
                    caught_error=True,
                )
                stored += 1

        return {
            "ok": True,
            "source": source,
            "analysis": analysis,
            "records_stored": stored,
        }

    @staticmethod
    def _variance(values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)
