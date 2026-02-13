"""
Meta-Criticality Observer & Emergence Detector

Observes systems operating at criticality and PREDICTS when emergence will occur.

Architecture:
    MetaCriticalityObserver orchestrates 4 real-time metrics:
    1. r  — criticality ratio (entropy/integration) from SelfOrganizedCriticality
    2. Φ  — integrated information from PhiIntegrationCalculator
    3. dH/dt — entropy change rate (computed from criticality history)
    4. bridges — cross-domain bridge count from KnowledgeNetworkGraph

    Emergence hypothesis: emergence is imminent when:
    - r ∈ [0.8, 1.0]  (critical zone, edge of chaos)
    - Φ increasing      (integration growing)
    - dH/dt ≈ 0         (entropy stabilized — order crystallizing)
    - bridges > threshold (cross-domain connections enabling novel combinations)

    The observer LEARNS from outcomes: after predicting emergence, it records
    whether emergence actually occurred, then updates its signature weights
    via Bayesian posterior updates. Over time, it calibrates which metric
    combinations are most predictive for THIS specific knowledge system.

Persistence:
    - emergence_observations: raw metric snapshots
    - emergence_predictions: predictions with outcome tracking
    - emergence_signatures: learned Bayesian weights
"""

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import aiosqlite
from loguru import logger


# ── Emergence Thresholds (initial priors, learned over time) ─────────────────

DEFAULT_THRESHOLDS = {
    "r_min": 0.8,       # Critical zone lower bound
    "r_max": 1.0,       # Critical zone upper bound (not 1.2 — we want the sweet spot)
    "phi_slope_min": 0.01,  # Φ must be increasing by at least this per observation
    "dh_dt_max": 0.05,  # Entropy change rate must be near zero (|dH/dt| < this)
    "bridge_min": 3,     # Minimum cross-domain bridges
    "emergence_threshold": 0.7,  # Probability above which we predict emergence
}

# Metric weights (Bayesian priors — updated from outcomes)
DEFAULT_WEIGHTS = {
    "r_in_zone": 0.30,      # Criticality ratio in sweet spot
    "phi_increasing": 0.25,  # Φ trending up
    "entropy_stable": 0.25,  # dH/dt near zero
    "bridges_above": 0.20,   # Sufficient cross-domain bridges
}


@dataclass
class EmergenceObservation:
    """A single observation of system state."""
    obs_id: str
    timestamp: float
    r: float                  # criticality ratio
    entropy: float            # raw entropy
    integration: float        # raw integration
    phi: float                # integrated information
    phi_delta: float          # change in Φ since last observation
    dh_dt: float              # entropy change rate
    bridge_count: int         # cross-domain bridges
    zone: str                 # subcritical / critical / supercritical
    emergence_probability: float
    prediction: str           # "imminent", "possible", "unlikely", "dormant"


@dataclass
class EmergenceSignature:
    """A learned pattern of metrics that preceded emergence."""
    sig_id: str
    r_range: Tuple[float, float]
    phi_slope: float
    dh_dt_range: Tuple[float, float]
    bridge_range: Tuple[int, int]
    n_observed: int           # times this pattern was observed
    n_emerged: int            # times emergence actually followed
    posterior: float          # P(emergence | signature) — Bayesian posterior
    created_at: float
    updated_at: float


class MetaCriticalityObserver:
    """
    Watches a knowledge system operating at criticality and predicts emergence.

    Integrates with existing PLTM subsystems:
    - SelfOrganizedCriticality → r, entropy, integration, zone
    - PhiIntegrationCalculator → Φ per domain
    - KnowledgeNetworkGraph → bridge count, network stats

    Learns from outcomes via Bayesian signature updates.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent.parent.parent / "data" / "pltm_mcp.db"
        self._conn: Optional[aiosqlite.Connection] = None

        # External system references (set via connect())
        self._criticality = None   # SelfOrganizedCriticality
        self._phi_calc = None      # PhiIntegrationCalculator
        self._knowledge_graph = None  # KnowledgeNetworkGraph
        self._store = None         # SQLiteGraphStore

        # In-memory state
        self._observations: List[EmergenceObservation] = []
        self._weights = dict(DEFAULT_WEIGHTS)
        self._thresholds = dict(DEFAULT_THRESHOLDS)
        self._last_phi: Optional[float] = None
        self._last_entropy: Optional[float] = None
        self._last_obs_time: Optional[float] = None

    async def connect(self, store=None, criticality=None,
                      phi_calc=None, knowledge_graph=None):
        """Connect to DB and register external systems."""
        self._conn = await aiosqlite.connect(str(self.db_path))
        await self._conn.execute("PRAGMA journal_mode=WAL")

        if store:
            self._store = store
        if criticality:
            self._criticality = criticality
        if phi_calc:
            self._phi_calc = phi_calc
        if knowledge_graph:
            self._knowledge_graph = knowledge_graph

        await self._ensure_tables()
        await self._load_weights()
        await self._load_recent_observations()
        logger.info("MetaCriticalityObserver connected — emergence detection active")

    async def _ensure_tables(self):
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS emergence_observations (
                obs_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                r REAL, entropy REAL, integration REAL,
                phi REAL, phi_delta REAL, dh_dt REAL,
                bridge_count INTEGER, zone TEXT,
                emergence_probability REAL,
                prediction TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_eo_ts ON emergence_observations(timestamp);

            CREATE TABLE IF NOT EXISTS emergence_predictions (
                pred_id TEXT PRIMARY KEY,
                obs_id TEXT NOT NULL,
                predicted_at REAL NOT NULL,
                probability REAL NOT NULL,
                prediction TEXT NOT NULL,
                outcome TEXT DEFAULT 'pending',
                outcome_recorded_at REAL,
                outcome_details TEXT DEFAULT '{}',
                FOREIGN KEY (obs_id) REFERENCES emergence_observations(obs_id)
            );
            CREATE INDEX IF NOT EXISTS idx_ep_outcome ON emergence_predictions(outcome);

            CREATE TABLE IF NOT EXISTS emergence_signatures (
                sig_id TEXT PRIMARY KEY,
                r_min REAL, r_max REAL,
                phi_slope REAL,
                dh_dt_min REAL, dh_dt_max REAL,
                bridge_min INTEGER, bridge_max INTEGER,
                n_observed INTEGER DEFAULT 0,
                n_emerged INTEGER DEFAULT 0,
                posterior REAL DEFAULT 0.5,
                created_at REAL, updated_at REAL
            );

            CREATE TABLE IF NOT EXISTS emergence_weights (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        await self._conn.commit()

    async def _load_weights(self):
        """Load learned weights from DB, or use defaults."""
        cursor = await self._conn.execute("SELECT key, value FROM emergence_weights")
        rows = await cursor.fetchall()
        if rows:
            for key, value in rows:
                if key in self._weights:
                    self._weights[key] = value
                elif key in self._thresholds:
                    self._thresholds[key] = value

    async def _save_weights(self):
        """Persist current weights to DB."""
        now = time.time()
        for key, value in {**self._weights, **self._thresholds}.items():
            await self._conn.execute(
                "INSERT OR REPLACE INTO emergence_weights (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, now),
            )
        await self._conn.commit()

    async def _load_recent_observations(self):
        """Load last N observations for dH/dt and Φ delta computation."""
        cursor = await self._conn.execute(
            "SELECT obs_id, timestamp, r, entropy, integration, phi, phi_delta, dh_dt, bridge_count, zone, emergence_probability, prediction "
            "FROM emergence_observations ORDER BY timestamp DESC LIMIT 20"
        )
        rows = await cursor.fetchall()
        for row in reversed(rows):  # oldest first
            obs = EmergenceObservation(
                obs_id=row[0], timestamp=row[1], r=row[2], entropy=row[3],
                integration=row[4], phi=row[5], phi_delta=row[6], dh_dt=row[7],
                bridge_count=row[8], zone=row[9], emergence_probability=row[10],
                prediction=row[11],
            )
            self._observations.append(obs)

        if self._observations:
            last = self._observations[-1]
            self._last_phi = last.phi
            self._last_entropy = last.entropy
            self._last_obs_time = last.timestamp

    # ═══════════════════════════════════════════════════════════════════════════
    # OBSERVE — Take a snapshot of all 4 metrics
    # ═══════════════════════════════════════════════════════════════════════════

    async def observe(self) -> Dict[str, Any]:
        """
        Take a full observation of the system's criticality state.

        Measures all 4 metrics, computes emergence probability,
        persists the observation, and returns a prediction.
        """
        now = time.time()

        # ── Metric 1: Criticality ratio (r) ──────────────────────────────────
        r, entropy, integration, zone = await self._measure_criticality()

        # ── Metric 2: Φ (integrated information) ─────────────────────────────
        phi = await self._measure_phi()

        # ── Metric 3: dH/dt (entropy change rate) ────────────────────────────
        dh_dt = self._compute_dh_dt(entropy, now)

        # ── Metric 4: Cross-domain bridges ────────────────────────────────────
        bridge_count = await self._count_bridges()

        # ── Φ delta ───────────────────────────────────────────────────────────
        phi_delta = (phi - self._last_phi) if self._last_phi is not None else 0.0

        # ── Compute emergence probability ─────────────────────────────────────
        probability, component_scores = self._compute_emergence_probability(
            r, phi_delta, dh_dt, bridge_count
        )

        # ── Classify prediction ───────────────────────────────────────────────
        prediction = self._classify_prediction(probability)

        # ── Build observation ─────────────────────────────────────────────────
        obs = EmergenceObservation(
            obs_id=f"eo_{uuid4().hex[:8]}",
            timestamp=now,
            r=round(r, 4),
            entropy=round(entropy, 4),
            integration=round(integration, 4),
            phi=round(phi, 4),
            phi_delta=round(phi_delta, 4),
            dh_dt=round(dh_dt, 4),
            bridge_count=bridge_count,
            zone=zone,
            emergence_probability=round(probability, 4),
            prediction=prediction,
        )

        # ── Persist ───────────────────────────────────────────────────────────
        await self._persist_observation(obs)
        self._observations.append(obs)
        if len(self._observations) > 100:
            self._observations = self._observations[-100:]

        # Update state for next delta computation
        self._last_phi = phi
        self._last_entropy = entropy
        self._last_obs_time = now

        # ── If emergence predicted, create a trackable prediction ─────────────
        pred_id = None
        if prediction in ("imminent", "possible"):
            pred_id = await self._create_prediction(obs)

        return {
            "obs_id": obs.obs_id,
            "metrics": {
                "r": obs.r,
                "phi": obs.phi,
                "phi_delta": obs.phi_delta,
                "dh_dt": obs.dh_dt,
                "bridges": obs.bridge_count,
                "zone": obs.zone,
            },
            "component_scores": component_scores,
            "emergence_probability": obs.emergence_probability,
            "prediction": prediction,
            "pred_id": pred_id,
            "weights": dict(self._weights),
            "observation_count": len(self._observations),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # LEARN — Record outcomes and update Bayesian weights
    # ═══════════════════════════════════════════════════════════════════════════

    async def record_outcome(self, pred_id: str, did_emerge: bool,
                             details: str = "") -> Dict[str, Any]:
        """
        After observing whether emergence actually happened, learn the pattern.

        Updates:
        1. The prediction record with outcome
        2. Matching signatures with Bayesian posterior
        3. Global weights based on which components were predictive
        """
        # Get the prediction
        cursor = await self._conn.execute(
            "SELECT obs_id, probability, prediction FROM emergence_predictions WHERE pred_id = ?",
            (pred_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return {"ok": False, "err": "prediction not found"}

        obs_id, predicted_prob, predicted_label = row

        # Get the observation metrics
        cursor = await self._conn.execute(
            "SELECT r, entropy, phi, phi_delta, dh_dt, bridge_count FROM emergence_observations WHERE obs_id = ?",
            (obs_id,),
        )
        obs_row = await cursor.fetchone()
        if not obs_row:
            return {"ok": False, "err": "observation not found"}

        r, entropy, phi, phi_delta, dh_dt, bridge_count = obs_row

        # Record outcome
        outcome = "emerged" if did_emerge else "did_not_emerge"
        await self._conn.execute(
            "UPDATE emergence_predictions SET outcome = ?, outcome_recorded_at = ?, outcome_details = ? WHERE pred_id = ?",
            (outcome, time.time(), json.dumps({"details": details}), pred_id),
        )

        # Update or create signature
        await self._update_signature(r, phi_delta, dh_dt, bridge_count, did_emerge)

        # Update global weights based on which components were correct
        await self._update_weights(r, phi_delta, dh_dt, bridge_count, did_emerge)

        await self._conn.commit()

        # Compute calibration
        calibration = await self._compute_calibration()

        return {
            "ok": True,
            "pred_id": pred_id,
            "predicted": predicted_label,
            "predicted_prob": predicted_prob,
            "actual": outcome,
            "correct": (did_emerge and predicted_label in ("imminent", "possible")) or
                       (not did_emerge and predicted_label in ("unlikely", "dormant")),
            "weights_after": dict(self._weights),
            "calibration": calibration,
        }

    async def _update_signature(self, r, phi_delta, dh_dt, bridge_count, did_emerge):
        """Find or create a matching signature and update its posterior."""
        # Quantize metrics into signature bins
        r_bin = (round(r - 0.1, 1), round(r + 0.1, 1))
        dh_bin = (round(dh_dt - 0.05, 2), round(dh_dt + 0.05, 2))
        b_bin = (max(0, bridge_count - 2), bridge_count + 2)

        # Look for existing signature in this bin
        cursor = await self._conn.execute(
            """SELECT sig_id, n_observed, n_emerged, posterior FROM emergence_signatures
               WHERE r_min <= ? AND r_max >= ? AND bridge_min <= ? AND bridge_max >= ?
               LIMIT 1""",
            (r, r, bridge_count, bridge_count),
        )
        row = await cursor.fetchone()

        if row:
            sig_id, n_obs, n_em, prior = row
            n_obs += 1
            if did_emerge:
                n_em += 1
            # Bayesian update: posterior = (prior * likelihood) / evidence
            # Simplified: use beta distribution update
            posterior = n_em / n_obs  # Maximum likelihood estimate
            # Smooth with prior (beta(1,1) = uniform)
            alpha = n_em + 1
            beta_param = (n_obs - n_em) + 1
            posterior = alpha / (alpha + beta_param)

            await self._conn.execute(
                "UPDATE emergence_signatures SET n_observed = ?, n_emerged = ?, posterior = ?, updated_at = ? WHERE sig_id = ?",
                (n_obs, n_em, round(posterior, 4), time.time(), sig_id),
            )
        else:
            # Create new signature
            sig_id = f"es_{uuid4().hex[:8]}"
            posterior = 0.75 if did_emerge else 0.25  # Informative first observation
            await self._conn.execute(
                """INSERT INTO emergence_signatures
                   (sig_id, r_min, r_max, phi_slope, dh_dt_min, dh_dt_max,
                    bridge_min, bridge_max, n_observed, n_emerged, posterior, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)""",
                (sig_id, r_bin[0], r_bin[1], round(phi_delta, 3),
                 dh_bin[0], dh_bin[1], b_bin[0], b_bin[1],
                 1 if did_emerge else 0, round(posterior, 4),
                 time.time(), time.time()),
            )

    async def _update_weights(self, r, phi_delta, dh_dt, bridge_count, did_emerge):
        """
        Update global metric weights based on which components were predictive.

        If emergence happened and a component was in the "good" range, boost it.
        If emergence didn't happen and a component was in the "good" range, dampen it.
        """
        t = self._thresholds
        learning_rate = 0.05

        # Score each component (was it in the "emergence" range?)
        components = {
            "r_in_zone": 1.0 if t["r_min"] <= r <= t["r_max"] else 0.0,
            "phi_increasing": 1.0 if phi_delta >= t["phi_slope_min"] else 0.0,
            "entropy_stable": 1.0 if abs(dh_dt) <= t["dh_dt_max"] else 0.0,
            "bridges_above": 1.0 if bridge_count >= t["bridge_min"] else 0.0,
        }

        for key, in_range in components.items():
            if did_emerge and in_range:
                # Component was predictive — boost weight
                self._weights[key] = min(0.5, self._weights[key] + learning_rate)
            elif not did_emerge and in_range:
                # Component was in range but emergence didn't happen — dampen
                self._weights[key] = max(0.05, self._weights[key] - learning_rate * 0.5)

        # Renormalize weights to sum to 1.0
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: round(v / total, 4) for k, v in self._weights.items()}

        await self._save_weights()

    # ═══════════════════════════════════════════════════════════════════════════
    # METRIC MEASUREMENT (delegates to existing subsystems)
    # ═══════════════════════════════════════════════════════════════════════════

    async def _measure_criticality(self) -> Tuple[float, float, float, str]:
        """Get r, entropy, integration, zone from SelfOrganizedCriticality."""
        if self._criticality:
            try:
                state = await self._criticality.get_criticality_state()
                return state["r"], state["e"], state["i"], state["z"]
            except Exception as e:
                logger.warning(f"Criticality measurement failed: {e}")

        # Fallback: measure directly from store
        if self._store:
            try:
                atoms = await self._store.get_all_atoms()
                if not atoms:
                    return 1.0, 0.5, 0.5, "critical"

                confidences = [a.confidence for a in atoms if hasattr(a, 'confidence')]
                if confidences:
                    mean_c = sum(confidences) / len(confidences)
                    variance = sum((c - mean_c) ** 2 for c in confidences) / len(confidences)
                    entropy = min(1.0, variance * 4)
                else:
                    entropy = 0.5

                integration = sum(confidences) / len(confidences) if confidences else 0.5
                r = entropy / max(integration, 0.01)

                if r < 0.8:
                    zone = "subcritical"
                elif r > 1.2:
                    zone = "supercritical"
                else:
                    zone = "critical"

                return r, entropy, integration, zone
            except Exception as e:
                logger.warning(f"Fallback criticality measurement failed: {e}")

        return 1.0, 0.5, 0.5, "critical"

    async def _measure_phi(self) -> float:
        """Get global Φ from PhiIntegrationCalculator."""
        if self._phi_calc and self._store:
            try:
                atoms = await self._store.get_all_atoms()
                if len(atoms) < 3:
                    return 0.0
                result = self._phi_calc.calculate("global", atoms)
                return result.phi
            except Exception as e:
                logger.warning(f"Phi measurement failed: {e}")

        return 0.0

    def _compute_dh_dt(self, current_entropy: float, current_time: float) -> float:
        """Compute entropy change rate from history."""
        if self._last_entropy is not None and self._last_obs_time is not None:
            dt = current_time - self._last_obs_time
            if dt > 0:
                return (current_entropy - self._last_entropy) / dt
        return 0.0

    async def _count_bridges(self) -> int:
        """Count cross-domain bridges from KnowledgeNetworkGraph."""
        if self._knowledge_graph:
            try:
                result = await self._knowledge_graph.find_bridges(top_k=100)
                return len(result.get("bridges", []))
            except Exception as e:
                logger.warning(f"Bridge count failed: {e}")
        return 0

    # ═══════════════════════════════════════════════════════════════════════════
    # EMERGENCE PROBABILITY COMPUTATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _compute_emergence_probability(
        self, r: float, phi_delta: float, dh_dt: float, bridge_count: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute P(emergence) from the 4 metrics using learned weights.

        Each component produces a score in [0, 1], then weighted sum.
        """
        t = self._thresholds
        w = self._weights

        # Component 1: r in critical zone [0.8, 1.0]
        if t["r_min"] <= r <= t["r_max"]:
            # Peak at center of zone
            center = (t["r_min"] + t["r_max"]) / 2
            distance = abs(r - center) / ((t["r_max"] - t["r_min"]) / 2)
            r_score = 1.0 - distance * 0.3  # Slight penalty for being off-center
        elif r < t["r_min"]:
            r_score = max(0, 1.0 - (t["r_min"] - r) * 3)  # Decay below zone
        else:
            r_score = max(0, 1.0 - (r - t["r_max"]) * 3)  # Decay above zone

        # Component 2: Φ increasing
        if phi_delta >= t["phi_slope_min"]:
            phi_score = min(1.0, phi_delta / (t["phi_slope_min"] * 5))  # Saturates at 5x threshold
        else:
            phi_score = max(0, 0.3 + phi_delta * 10)  # Some credit for near-zero

        # Component 3: Entropy stable (|dH/dt| near zero)
        abs_dh = abs(dh_dt)
        if abs_dh <= t["dh_dt_max"]:
            entropy_score = 1.0 - (abs_dh / t["dh_dt_max"]) * 0.3
        else:
            entropy_score = max(0, 1.0 - (abs_dh - t["dh_dt_max"]) * 5)

        # Component 4: Bridges above threshold
        if bridge_count >= t["bridge_min"]:
            bridge_score = min(1.0, bridge_count / (t["bridge_min"] * 3))
        else:
            bridge_score = bridge_count / max(t["bridge_min"], 1)

        # Weighted sum
        probability = (
            w["r_in_zone"] * r_score +
            w["phi_increasing"] * phi_score +
            w["entropy_stable"] * entropy_score +
            w["bridges_above"] * bridge_score
        )
        probability = round(max(0.0, min(1.0, probability)), 4)

        component_scores = {
            "r_score": round(r_score, 3),
            "phi_score": round(phi_score, 3),
            "entropy_score": round(entropy_score, 3),
            "bridge_score": round(bridge_score, 3),
        }

        # Check for matching learned signatures — boost if we've seen this pattern emerge before
        # (done synchronously from in-memory cache for speed)
        signature_boost = self._check_signatures(r, phi_delta, dh_dt, bridge_count)
        if signature_boost > 0:
            probability = min(1.0, probability + signature_boost * 0.15)
            component_scores["signature_boost"] = round(signature_boost, 3)

        return probability, component_scores

    def _check_signatures(self, r, phi_delta, dh_dt, bridge_count) -> float:
        """Check if current metrics match any learned high-emergence signatures."""
        # This will be populated from DB on connect; for now return 0
        # In production, we'd cache signatures in memory
        return 0.0

    def _classify_prediction(self, probability: float) -> str:
        """Classify emergence probability into a human-readable prediction."""
        if probability >= 0.7:
            return "imminent"
        elif probability >= 0.5:
            return "possible"
        elif probability >= 0.3:
            return "unlikely"
        else:
            return "dormant"

    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════════

    async def _persist_observation(self, obs: EmergenceObservation):
        await self._conn.execute(
            """INSERT INTO emergence_observations
               (obs_id, timestamp, r, entropy, integration, phi, phi_delta, dh_dt,
                bridge_count, zone, emergence_probability, prediction)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (obs.obs_id, obs.timestamp, obs.r, obs.entropy, obs.integration,
             obs.phi, obs.phi_delta, obs.dh_dt, obs.bridge_count, obs.zone,
             obs.emergence_probability, obs.prediction),
        )
        await self._conn.commit()

    async def _create_prediction(self, obs: EmergenceObservation) -> str:
        pred_id = f"ep_{uuid4().hex[:8]}"
        await self._conn.execute(
            """INSERT INTO emergence_predictions
               (pred_id, obs_id, predicted_at, probability, prediction)
               VALUES (?, ?, ?, ?, ?)""",
            (pred_id, obs.obs_id, obs.timestamp, obs.emergence_probability, obs.prediction),
        )
        await self._conn.commit()
        return pred_id

    # ═══════════════════════════════════════════════════════════════════════════
    # QUERY / ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════

    async def get_history(self, limit: int = 20) -> Dict[str, Any]:
        """Get recent observation history with trend analysis."""
        cursor = await self._conn.execute(
            """SELECT obs_id, timestamp, r, phi, phi_delta, dh_dt, bridge_count,
                      zone, emergence_probability, prediction
               FROM emergence_observations ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()

        observations = []
        for row in rows:
            observations.append({
                "id": row[0][:12], "ts": row[1],
                "r": row[2], "phi": row[3], "phi_d": row[4],
                "dh_dt": row[5], "bridges": row[6], "zone": row[7],
                "p": row[8], "pred": row[9],
            })

        # Trend analysis
        if len(observations) >= 3:
            recent_probs = [o["p"] for o in observations[:5]]
            avg_prob = sum(recent_probs) / len(recent_probs)
            if recent_probs[0] > recent_probs[-1] * 1.2:
                trend = "rising"
            elif recent_probs[0] < recent_probs[-1] * 0.8:
                trend = "falling"
            else:
                trend = "stable"
        else:
            avg_prob = 0.0
            trend = "insufficient_data"

        return {
            "n": len(observations),
            "avg_probability": round(avg_prob, 3),
            "trend": trend,
            "observations": observations[:10],
            "weights": dict(self._weights),
        }

    async def get_predictions(self, include_pending: bool = True,
                              limit: int = 20) -> Dict[str, Any]:
        """Get predictions with their outcomes."""
        where = ""
        if not include_pending:
            where = "WHERE outcome != 'pending'"

        cursor = await self._conn.execute(
            f"""SELECT pred_id, obs_id, predicted_at, probability, prediction,
                       outcome, outcome_recorded_at
                FROM emergence_predictions {where}
                ORDER BY predicted_at DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()

        predictions = []
        correct = 0
        total_with_outcome = 0
        for row in rows:
            pred = {
                "id": row[0], "obs": row[1][:12], "ts": row[2],
                "prob": row[3], "pred": row[4], "outcome": row[5],
            }
            predictions.append(pred)
            if row[5] != "pending":
                total_with_outcome += 1
                if (row[5] == "emerged" and row[4] in ("imminent", "possible")) or \
                   (row[5] == "did_not_emerge" and row[4] in ("unlikely", "dormant")):
                    correct += 1

        accuracy = correct / total_with_outcome if total_with_outcome > 0 else None

        return {
            "n": len(predictions),
            "accuracy": round(accuracy, 3) if accuracy is not None else None,
            "correct": correct,
            "total_evaluated": total_with_outcome,
            "predictions": predictions[:10],
        }

    async def get_signatures(self, limit: int = 10) -> Dict[str, Any]:
        """Get learned emergence signatures sorted by posterior probability."""
        cursor = await self._conn.execute(
            """SELECT sig_id, r_min, r_max, phi_slope, dh_dt_min, dh_dt_max,
                      bridge_min, bridge_max, n_observed, n_emerged, posterior
               FROM emergence_signatures ORDER BY posterior DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()

        signatures = []
        for row in rows:
            signatures.append({
                "id": row[0], "r": [row[1], row[2]],
                "phi_slope": row[3], "dh_dt": [row[4], row[5]],
                "bridges": [row[6], row[7]],
                "n_obs": row[8], "n_em": row[9],
                "posterior": row[10],
            })

        return {"n": len(signatures), "signatures": signatures}

    async def _compute_calibration(self) -> Dict[str, Any]:
        """Compute calibration: how well do predicted probabilities match actual rates?"""
        cursor = await self._conn.execute(
            """SELECT ep.probability, ep.outcome
               FROM emergence_predictions ep
               WHERE ep.outcome != 'pending'"""
        )
        rows = await cursor.fetchall()
        if not rows:
            return {"n": 0}

        # Bin predictions into buckets
        bins = defaultdict(lambda: {"total": 0, "emerged": 0})
        for prob, outcome in rows:
            bucket = round(prob, 1)  # 0.0, 0.1, ..., 1.0
            bins[bucket]["total"] += 1
            if outcome == "emerged":
                bins[bucket]["emerged"] += 1

        calibration = {}
        for bucket in sorted(bins.keys()):
            data = bins[bucket]
            actual_rate = data["emerged"] / data["total"]
            calibration[str(bucket)] = {
                "predicted": bucket,
                "actual": round(actual_rate, 3),
                "n": data["total"],
            }

        return {"n": len(rows), "bins": calibration}

    async def get_dashboard(self) -> Dict[str, Any]:
        """
        Full emergence dashboard: current state + history + predictions + signatures.
        One-call summary for session handoff.
        """
        # Current observation
        current = await self.observe()

        # Recent history summary
        history = await self.get_history(limit=5)

        # Prediction accuracy
        predictions = await self.get_predictions(include_pending=False, limit=50)

        # Top signatures
        signatures = await self.get_signatures(limit=3)

        return {
            "current": {
                "prediction": current["prediction"],
                "probability": current["emergence_probability"],
                "metrics": current["metrics"],
            },
            "trend": history["trend"],
            "accuracy": predictions.get("accuracy"),
            "n_predictions": predictions["total_evaluated"],
            "top_signatures": signatures["signatures"][:3],
            "weights": current["weights"],
        }

    async def close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None
