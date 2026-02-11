"""Lightweight API server for Deep Claude Dashboard. Reads PLTM database read-only."""
import json
import sqlite3
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pltm_mcp.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def safe_round(val, digits=3):
    return round(val, digits) if val is not None else None


class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        try:
            if path == "/api/overview":
                self._json(self.get_overview())
            elif path == "/api/domains":
                self._json(self.get_domains())
            elif path == "/api/calibration":
                self._json(self.get_calibration(params))
            elif path == "/api/claims":
                self._json(self.get_claims(params))
            elif path == "/api/interventions":
                self._json(self.get_interventions(params))
            elif path == "/api/personality":
                self._json(self.get_personality())
            elif path == "/api/curiosity":
                self._json(self.get_curiosity())
            elif path == "/api/reasoning":
                self._json(self.get_reasoning())
            elif path == "/api/sessions":
                self._json(self.get_sessions())
            elif path == "/api/evolution":
                self._json(self.get_evolution())
            elif path == "/api/atoms":
                self._json(self.get_atoms(params))
            elif path == "/api/confabulations":
                self._json(self.get_confabulations())
            elif path == "/api/typed_memories":
                self._json(self.get_typed_memories(params))
            elif path == "/api/memory_stats":
                self._json(self.get_memory_stats())
            elif path == "/api/decay_forecast":
                self._json(self.get_decay_forecast(params))
            elif path == "/api/memory_clusters":
                self._json(self.get_memory_clusters(params))
            elif path == "/api/importance":
                self._json(self.get_importance(params))
            elif path == "/api/memory_audit":
                self._json(self.get_memory_audit(params))
            elif path == "/api/jury_stats":
                self._json(self.get_jury_stats())
            elif path == "/api/conflicts":
                self._json(self.get_conflicts(params))
            elif path == "/api/shared_memories":
                self._json(self.get_shared_memories(params))
            elif path == "/api/provenance":
                self._json(self.get_provenance_data(params))
            else:
                self._json({"error": "Not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def get_overview(self):
        conn = get_conn()
        overview = {}

        overview["atoms"] = conn.execute("SELECT COUNT(*) as n FROM atoms").fetchone()["n"]
        overview["claims"] = conn.execute("SELECT COUNT(*) as n FROM prediction_book").fetchone()["n"]
        overview["resolved"] = conn.execute("SELECT COUNT(*) as n FROM prediction_book WHERE actual_truth IS NOT NULL").fetchone()["n"]
        overview["correct"] = conn.execute("SELECT COUNT(*) as n FROM prediction_book WHERE was_correct = 1").fetchone()["n"]
        overview["accuracy"] = safe_round(overview["correct"] / max(overview["resolved"], 1))
        overview["interventions"] = conn.execute("SELECT COUNT(*) as n FROM epistemic_interventions").fetchone()["n"]
        overview["confabulations"] = conn.execute("SELECT COUNT(*) as n FROM confabulation_log").fetchone()["n"]
        overview["sessions"] = conn.execute("SELECT COUNT(*) as n FROM session_log").fetchone()["n"]
        overview["communication_records"] = conn.execute("SELECT COUNT(*) as n FROM self_communication").fetchone()["n"]
        overview["curiosity_records"] = conn.execute("SELECT COUNT(*) as n FROM self_curiosity").fetchone()["n"]
        overview["reasoning_records"] = conn.execute("SELECT COUNT(*) as n FROM self_reasoning").fetchone()["n"]
        overview["snapshots"] = conn.execute("SELECT COUNT(*) as n FROM personality_snapshot").fetchone()["n"]

        # Avg calibration error
        cal = conn.execute("SELECT AVG(calibration_error) as e FROM prediction_book WHERE calibration_error IS NOT NULL").fetchone()
        overview["avg_calibration_error"] = safe_round(cal["e"])

        conn.close()
        return overview

    def get_domains(self):
        conn = get_conn()
        domains = []

        rows = conn.execute(
            "SELECT domain, COUNT(*) as total, "
            "SUM(CASE WHEN actual_truth IS NOT NULL THEN 1 ELSE 0 END) as resolved, "
            "SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct, "
            "SUM(CASE WHEN was_correct = 0 AND actual_truth IS NOT NULL THEN 1 ELSE 0 END) as failures, "
            "AVG(felt_confidence) as avg_felt, AVG(calibration_error) as avg_error "
            "FROM prediction_book WHERE domain IS NOT NULL GROUP BY domain ORDER BY total DESC"
        ).fetchall()

        for r in rows:
            d = dict(r)
            d["accuracy"] = safe_round(d["correct"] / max(d["resolved"], 1))
            d["avg_felt"] = safe_round(d["avg_felt"])
            d["avg_error"] = safe_round(d["avg_error"])

            # Calibration cache
            cal = conn.execute(
                "SELECT accuracy_ratio, overconfidence_ratio FROM calibration_cache WHERE domain = ?",
                (d["domain"],)
            ).fetchone()
            if cal:
                d["accuracy_ratio"] = safe_round(cal["accuracy_ratio"])
                d["overconfidence_ratio"] = safe_round(cal["overconfidence_ratio"])

            # Interventions
            intv = conn.execute(
                "SELECT COUNT(*) as n FROM epistemic_interventions WHERE domain = ?", (d["domain"],)
            ).fetchone()
            d["interventions"] = intv["n"]

            domains.append(d)

        conn.close()
        return domains

    def get_calibration(self, params):
        conn = get_conn()
        rows = conn.execute(
            "SELECT domain, total_claims, verified_claims, correct_claims, "
            "accuracy_ratio, avg_felt_confidence, avg_calibration_error, overconfidence_ratio, last_updated "
            "FROM calibration_cache ORDER BY total_claims DESC"
        ).fetchall()
        result = [dict(r) for r in rows]
        for r in result:
            for k in ["accuracy_ratio", "avg_felt_confidence", "avg_calibration_error", "overconfidence_ratio"]:
                r[k] = safe_round(r[k])
        conn.close()
        return result

    def get_claims(self, params):
        conn = get_conn()
        domain = params.get("domain", [None])[0]
        limit = int(params.get("limit", [50])[0])

        if domain:
            rows = conn.execute(
                "SELECT id, timestamp, claim, domain, felt_confidence, epistemic_status, "
                "has_verified, was_correct, calibration_error "
                "FROM prediction_book WHERE domain = ? ORDER BY timestamp DESC LIMIT ?",
                (domain, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, timestamp, claim, domain, felt_confidence, epistemic_status, "
                "has_verified, was_correct, calibration_error "
                "FROM prediction_book ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            d["felt_confidence"] = safe_round(d["felt_confidence"])
            d["calibration_error"] = safe_round(d["calibration_error"])
            if d["timestamp"]:
                d["time_str"] = datetime.fromtimestamp(d["timestamp"]).strftime("%Y-%m-%d %H:%M")
            result.append(d)

        conn.close()
        return result

    def get_interventions(self, params):
        conn = get_conn()
        domain = params.get("domain", [None])[0]

        if domain:
            rows = conn.execute(
                "SELECT timestamp, claim, domain, felt_confidence, adjusted_confidence, action_taken, outcome "
                "FROM epistemic_interventions WHERE domain = ? ORDER BY timestamp DESC LIMIT 50",
                (domain,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT timestamp, claim, domain, felt_confidence, adjusted_confidence, action_taken, outcome "
                "FROM epistemic_interventions ORDER BY timestamp DESC LIMIT 50"
            ).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            d["felt_confidence"] = safe_round(d["felt_confidence"])
            d["adjusted_confidence"] = safe_round(d["adjusted_confidence"])
            if d["timestamp"]:
                d["time_str"] = datetime.fromtimestamp(d["timestamp"]).strftime("%Y-%m-%d %H:%M")
            result.append(d)

        conn.close()
        return result

    def get_personality(self):
        conn = get_conn()
        result = {}

        # Communication by context
        comm = conn.execute(
            "SELECT context, AVG(verbosity) as v, AVG(jargon_density) as j, "
            "AVG(hedging_rate) as h, COUNT(*) as n "
            "FROM self_communication GROUP BY context ORDER BY n DESC"
        ).fetchall()
        result["communication"] = [
            {"context": r["context"], "verbosity": safe_round(r["v"]),
             "jargon": safe_round(r["j"]), "hedging": safe_round(r["h"]), "samples": r["n"]}
            for r in comm
        ]

        # Tones
        tones = conn.execute(
            "SELECT emotional_tone, COUNT(*) as n FROM self_communication "
            "GROUP BY emotional_tone ORDER BY n DESC"
        ).fetchall()
        result["tones"] = [dict(r) for r in tones]

        # Values
        values = conn.execute(
            "SELECT response_type, violation_type, intensity, reasoning, pushed_back, complied "
            "FROM self_values ORDER BY intensity DESC LIMIT 20"
        ).fetchall()
        result["values"] = [dict(r) for r in values]

        conn.close()
        return result

    def get_curiosity(self):
        conn = get_conn()
        rows = conn.execute(
            "SELECT topic, AVG(engagement_score) as avg_eng, COUNT(*) as n, "
            "SUM(went_deeper_than_required) as deeper, SUM(autonomous_research) as auto_research "
            "FROM self_curiosity GROUP BY topic ORDER BY avg_eng DESC"
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["avg_eng"] = safe_round(d["avg_eng"])
            d["genuine_ratio"] = safe_round((d["deeper"] or 0 + d["auto_research"] or 0) / max(d["n"], 1))
            result.append(d)
        conn.close()
        return result

    def get_reasoning(self):
        conn = get_conn()
        rows = conn.execute(
            "SELECT id, timestamp, event_type, trigger, confabulated, verified, caught_error, domain "
            "FROM self_reasoning ORDER BY timestamp DESC LIMIT 50"
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d["timestamp"]:
                d["time_str"] = datetime.fromtimestamp(d["timestamp"]).strftime("%Y-%m-%d %H:%M")
            result.append(d)

        # Summary
        summary = conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(confabulated) as confab, SUM(verified) as verified, SUM(caught_error) as errors "
            "FROM self_reasoning"
        ).fetchone()
        s = dict(summary)
        total = max(s["total"], 1)
        s["confab_rate"] = safe_round((s["confab"] or 0) / total)
        s["verify_rate"] = safe_round((s["verified"] or 0) / total)
        s["error_rate"] = safe_round((s["errors"] or 0) / total)

        conn.close()
        return {"events": result, "summary": s}

    def get_sessions(self):
        conn = get_conn()
        rows = conn.execute(
            "SELECT id, user_id, started_at, ended_at, summary, claims_made, "
            "claims_resolved, accuracy, confabulation_count "
            "FROM session_log ORDER BY started_at DESC LIMIT 20"
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("started_at"):
                d["time_str"] = datetime.fromtimestamp(d["started_at"]).strftime("%Y-%m-%d %H:%M")
            d["accuracy"] = safe_round(d.get("accuracy"))
            result.append(d)
        conn.close()
        return result

    def get_evolution(self):
        conn = get_conn()
        rows = conn.execute(
            "SELECT id, session_id, timestamp, avg_verbosity, avg_jargon, avg_hedging, "
            "dominant_tone, confabulation_rate, verification_rate, error_catch_rate, "
            "intellectual_honesty, top_interests, avg_engagement, pushback_rate, "
            "avg_value_intensity, prediction_accuracy, overall_accuracy, metadata "
            "FROM personality_snapshot ORDER BY timestamp DESC"
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("timestamp"):
                d["time_str"] = datetime.fromtimestamp(d["timestamp"]).strftime("%Y-%m-%d %H:%M")
            for k in ["avg_verbosity", "avg_jargon", "avg_hedging", "confabulation_rate",
                       "verification_rate", "error_catch_rate", "intellectual_honesty",
                       "avg_engagement", "pushback_rate", "avg_value_intensity",
                       "prediction_accuracy", "overall_accuracy"]:
                d[k] = safe_round(d.get(k))
            result.append(d)
        conn.close()
        return result

    def get_atoms(self, params):
        conn = get_conn()
        conn.execute("PRAGMA busy_timeout = 5000")
        limit = int(params.get("limit", [100])[0])
        search = params.get("q", [None])[0]

        if search:
            rows = conn.execute(
                "SELECT subject, predicate, object, confidence, atom_type, graph "
                "FROM atoms WHERE subject LIKE ? OR predicate LIKE ? OR object LIKE ? "
                "ORDER BY confidence DESC LIMIT ?",
                (f"%{search}%", f"%{search}%", f"%{search}%", limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT subject, predicate, object, confidence, atom_type, graph "
                "FROM atoms ORDER BY last_accessed DESC LIMIT ?",
                (limit,)
            ).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            d["source"] = d.pop("graph", "")
            result.append(d)
        conn.close()
        return result

    def get_confabulations(self):
        conn = get_conn()
        rows = conn.execute("SELECT * FROM confabulation_log ORDER BY timestamp DESC LIMIT 20").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("timestamp"):
                d["time_str"] = datetime.fromtimestamp(d["timestamp"]).strftime("%Y-%m-%d %H:%M")
            result.append(d)
        conn.close()
        return result


    def get_typed_memories(self, params):
        conn = get_conn()
        conn.execute("PRAGMA busy_timeout = 5000")
        user_id = params.get("user_id", [None])[0]
        mtype = params.get("type", [None])[0]
        limit = int(params.get("limit", [100])[0])
        search = params.get("q", [None])[0]

        query = "SELECT id, memory_type, user_id, content, strength, created_at, last_accessed, access_count, confidence, tags FROM typed_memories WHERE 1=1"
        args = []
        if user_id:
            query += " AND user_id = ?"
            args.append(user_id)
        if mtype:
            query += " AND memory_type = ?"
            args.append(mtype)
        if search:
            query += " AND content LIKE ?"
            args.append(f"%{search}%")
        query += " ORDER BY last_accessed DESC LIMIT ?"
        args.append(limit)

        try:
            rows = conn.execute(query, args).fetchall()
        except Exception:
            conn.close()
            return []
        result = []
        for r in rows:
            d = dict(r)
            d["strength"] = safe_round(d.get("strength"))
            d["confidence"] = safe_round(d.get("confidence"))
            if d.get("created_at"):
                d["created_str"] = datetime.fromtimestamp(d["created_at"]).strftime("%Y-%m-%d %H:%M")
            if d.get("last_accessed"):
                d["accessed_str"] = datetime.fromtimestamp(d["last_accessed"]).strftime("%Y-%m-%d %H:%M")
            result.append(d)
        conn.close()
        return result

    def get_memory_stats(self):
        conn = get_conn()
        conn.execute("PRAGMA busy_timeout = 5000")
        stats = {}
        try:
            stats["total"] = conn.execute("SELECT COUNT(*) as n FROM typed_memories").fetchone()["n"]
            types = conn.execute("SELECT memory_type, COUNT(*) as n, AVG(strength) as avg_str, AVG(confidence) as avg_conf FROM typed_memories GROUP BY memory_type").fetchall()
            stats["by_type"] = [{"type": r["memory_type"], "count": r["n"], "avg_strength": safe_round(r["avg_str"]), "avg_confidence": safe_round(r["avg_conf"])} for r in types]
            users = conn.execute("SELECT user_id, COUNT(*) as n FROM typed_memories GROUP BY user_id ORDER BY n DESC").fetchall()
            stats["by_user"] = [{"user_id": r["user_id"], "count": r["n"]} for r in users]
            low = conn.execute("SELECT COUNT(*) as n FROM typed_memories WHERE strength < 0.3").fetchone()["n"]
            stats["low_strength"] = low
            high_access = conn.execute("SELECT COUNT(*) as n FROM typed_memories WHERE access_count > 5").fetchone()["n"]
            stats["high_access"] = high_access
        except Exception:
            stats["total"] = 0
            stats["by_type"] = []
            stats["by_user"] = []
        conn.close()
        return stats

    def get_decay_forecast(self, params):
        conn = get_conn()
        conn.execute("PRAGMA busy_timeout = 5000")
        user_id = params.get("user_id", ["default"])[0]
        import math
        HALF_LIVES = {"episodic": 48, "semantic": 720, "belief": 168, "procedural": 2160}
        now = time.time()
        try:
            rows = conn.execute(
                "SELECT id, memory_type, content, strength, last_accessed, access_count FROM typed_memories WHERE user_id = ? AND strength > 0.1",
                (user_id,)
            ).fetchall()
        except Exception:
            conn.close()
            return {"forecasts": [], "summary": {}}
        forecasts = []
        for r in rows:
            d = dict(r)
            hl = HALF_LIVES.get(d["memory_type"], 168)
            hours_since = (now - (d["last_accessed"] or d.get("created_at", now))) / 3600
            rehearsal_bonus = min((d["access_count"] or 0) * 0.1, 0.5)
            effective_hl = hl * (1 + rehearsal_bonus)
            current = d["strength"] * math.exp(-0.693 * hours_since / effective_hl)
            forecast_7d = d["strength"] * math.exp(-0.693 * (hours_since + 168) / effective_hl)
            forecasts.append({
                "id": d["id"][:8],
                "type": d["memory_type"],
                "content": (d["content"] or "")[:60],
                "current_strength": safe_round(current),
                "strength_7d": safe_round(forecast_7d),
                "half_life_hours": safe_round(effective_hl),
                "hours_since_access": safe_round(hours_since),
            })
        forecasts.sort(key=lambda x: x["strength_7d"])
        at_risk = [f for f in forecasts if f["strength_7d"] < 0.2]
        conn.close()
        return {"forecasts": forecasts[:50], "at_risk_count": len(at_risk), "total": len(forecasts)}

    def get_memory_clusters(self, params):
        conn = get_conn()
        conn.execute("PRAGMA busy_timeout = 5000")
        user_id = params.get("user_id", ["default"])[0]
        try:
            rows = conn.execute(
                "SELECT id, memory_type, content, tags, confidence, strength FROM typed_memories WHERE user_id = ?",
                (user_id,)
            ).fetchall()
        except Exception:
            conn.close()
            return {"clusters": []}
        tag_groups = {}
        for r in rows:
            d = dict(r)
            tags = d.get("tags") or ""
            if isinstance(tags, str):
                try:
                    tag_list = json.loads(tags) if tags.startswith("[") else [t.strip() for t in tags.split(",") if t.strip()]
                except Exception:
                    tag_list = [tags] if tags else ["untagged"]
            else:
                tag_list = tags or ["untagged"]
            key = ", ".join(sorted(tag_list[:3])) if tag_list else "untagged"
            if key not in tag_groups:
                tag_groups[key] = {"topic": key, "memories": [], "types": {}}
            tag_groups[key]["memories"].append({"content": (d["content"] or "")[:80], "type": d["memory_type"], "confidence": safe_round(d.get("confidence"))})
            t = d["memory_type"] or "unknown"
            tag_groups[key]["types"][t] = tag_groups[key]["types"].get(t, 0) + 1
        clusters = []
        for k, v in sorted(tag_groups.items(), key=lambda x: -len(x[1]["memories"])):
            clusters.append({"topic": v["topic"], "size": len(v["memories"]), "type_breakdown": v["types"], "memories": v["memories"][:5]})
        conn.close()
        return {"clusters": clusters, "total_clusters": len(clusters)}

    def get_importance(self, params):
        conn = get_conn()
        conn.execute("PRAGMA busy_timeout = 5000")
        user_id = params.get("user_id", ["default"])[0]
        TYPE_WEIGHTS = {"procedural": 1.3, "belief": 1.2, "semantic": 1.0, "episodic": 0.8}
        now = time.time()
        try:
            rows = conn.execute(
                "SELECT id, memory_type, content, strength, confidence, access_count, created_at, last_accessed FROM typed_memories WHERE user_id = ?",
                (user_id,)
            ).fetchall()
        except Exception:
            conn.close()
            return {"ranked": []}
        ranked = []
        for r in rows:
            d = dict(r)
            tw = TYPE_WEIGHTS.get(d["memory_type"], 1.0)
            freq = min((d["access_count"] or 0) / 10.0, 1.0)
            age_hours = max((now - (d["created_at"] or now)) / 3600, 1)
            recency = 1.0 / (1.0 + age_hours / 168)
            score = tw * 0.3 + (d["strength"] or 0.5) * 0.25 + (d["confidence"] or 0.5) * 0.2 + freq * 0.15 + recency * 0.1
            ranked.append({
                "id": d["id"][:8],
                "type": d["memory_type"],
                "content": (d["content"] or "")[:80],
                "importance": safe_round(score),
                "strength": safe_round(d.get("strength")),
                "confidence": safe_round(d.get("confidence")),
                "access_count": d["access_count"] or 0,
            })
        ranked.sort(key=lambda x: -x["importance"])
        conn.close()
        return {"ranked": ranked[:50], "total": len(ranked)}

    def get_memory_audit(self, params):
        conn = get_conn()
        conn.execute("PRAGMA busy_timeout = 5000")
        user_id = params.get("user_id", ["default"])[0]
        audit = {"health_score": 100, "issues": []}
        try:
            total = conn.execute("SELECT COUNT(*) as n FROM typed_memories WHERE user_id = ?", (user_id,)).fetchone()["n"]
            audit["total_memories"] = total
            if total == 0:
                audit["health_score"] = 0
                audit["health_label"] = "empty"
                conn.close()
                return audit
            low_str = conn.execute("SELECT COUNT(*) as n FROM typed_memories WHERE user_id = ? AND strength < 0.2", (user_id,)).fetchone()["n"]
            low_conf = conn.execute("SELECT COUNT(*) as n FROM typed_memories WHERE user_id = ? AND confidence < 0.3", (user_id,)).fetchone()["n"]
            stale = conn.execute("SELECT COUNT(*) as n FROM typed_memories WHERE user_id = ? AND last_accessed < ?", (user_id, time.time() - 604800)).fetchone()["n"]
            types = conn.execute("SELECT memory_type, COUNT(*) as n FROM typed_memories WHERE user_id = ? GROUP BY memory_type", (user_id,)).fetchall()
            audit["type_distribution"] = {r["memory_type"]: r["n"] for r in types}
            penalty = 0
            if low_str > total * 0.3:
                penalty += 15
                audit["issues"].append(f"{low_str} memories below 0.2 strength")
            if low_conf > total * 0.2:
                penalty += 10
                audit["issues"].append(f"{low_conf} memories below 0.3 confidence")
            if stale > total * 0.5:
                penalty += 15
                audit["issues"].append(f"{stale} memories not accessed in 7+ days")
            if len(audit["type_distribution"]) < 2:
                penalty += 10
                audit["issues"].append("Low type diversity")
            audit["low_strength"] = low_str
            audit["low_confidence"] = low_conf
            audit["stale_count"] = stale
            audit["health_score"] = max(0, 100 - penalty)
            audit["health_label"] = "excellent" if audit["health_score"] >= 80 else "good" if audit["health_score"] >= 60 else "fair" if audit["health_score"] >= 40 else "poor"
        except Exception as e:
            audit["error"] = str(e)
        conn.close()
        return audit

    def get_jury_stats(self):
        conn = get_conn()
        conn.execute("PRAGMA busy_timeout = 5000")
        stats = {}
        try:
            rows = conn.execute("SELECT judge_name, evaluations, approvals, rejections, quarantines, avg_confidence, avg_latency_ms FROM meta_judge_stats").fetchall()
            stats["judges"] = [dict(r) for r in rows]
            for j in stats["judges"]:
                j["avg_confidence"] = safe_round(j.get("avg_confidence"))
                j["avg_latency_ms"] = safe_round(j.get("avg_latency_ms"))
        except Exception:
            stats["judges"] = []
        try:
            fb = conn.execute("SELECT feedback_type, COUNT(*) as n FROM meta_judge_feedback GROUP BY feedback_type").fetchall()
            stats["feedback"] = {r["feedback_type"]: r["n"] for r in fb}
        except Exception:
            stats["feedback"] = {}
        conn.close()
        return stats

    def get_conflicts(self, params):
        conn = get_conn()
        conn.execute("PRAGMA busy_timeout = 5000")
        user_id = params.get("user_id", ["default"])[0]
        conflicts = []
        try:
            beliefs = conn.execute(
                "SELECT id, content, confidence, evidence_for, evidence_against FROM typed_memories WHERE user_id = ? AND memory_type = 'belief'",
                (user_id,)
            ).fetchall()
            for b in beliefs:
                d = dict(b)
                against = d.get("evidence_against") or ""
                if against and against != "[]":
                    conflicts.append({
                        "memory_id": d["id"][:8],
                        "content": (d["content"] or "")[:80],
                        "confidence": safe_round(d.get("confidence")),
                        "evidence_against": against[:100],
                    })
        except Exception:
            pass
        conn.close()
        return {"conflicts": conflicts, "count": len(conflicts)}

    def get_shared_memories(self, params):
        conn = get_conn()
        conn.execute("PRAGMA busy_timeout = 5000")
        try:
            rows = conn.execute(
                "SELECT memory_id, owner_id, shared_with, permission, shared_at FROM shared_memories ORDER BY shared_at DESC LIMIT 50"
            ).fetchall()
            result = [dict(r) for r in rows]
        except Exception:
            result = []
        conn.close()
        return {"shared": result, "count": len(result)}

    def get_provenance_data(self, params):
        conn = get_conn()
        conn.execute("PRAGMA busy_timeout = 5000")
        memory_id = params.get("memory_id", [None])[0]
        try:
            if memory_id:
                rows = conn.execute(
                    "SELECT * FROM memory_provenance WHERE memory_id = ?", (memory_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memory_provenance ORDER BY recorded_at DESC LIMIT 50"
                ).fetchall()
            result = [dict(r) for r in rows]
        except Exception:
            result = []
        conn.close()
        return {"provenance": result, "count": len(result)}


if __name__ == "__main__":
    port = 8787
    server = HTTPServer(("0.0.0.0", port), APIHandler)
    print(f"Deep Claude API running on http://localhost:{port}")
    print(f"Database: {DB_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
