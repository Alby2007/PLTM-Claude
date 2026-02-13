"""
Tests for MetaCriticalityObserver / Emergence Detector.

Tests cover:
- Observation with all 4 metrics
- Emergence probability computation
- Prediction classification
- Outcome recording + Bayesian weight updates
- Signature learning
- History and dashboard queries
- Fallback behavior when subsystems unavailable
"""

import time
from pathlib import Path

import pytest
import aiosqlite


class TestMetaCriticalityObserver:

    @pytest.fixture
    async def observer(self, tmp_path):
        from src.meta.emergence_detector import MetaCriticalityObserver
        db_path = tmp_path / "test.db"
        obs = MetaCriticalityObserver(db_path)
        await obs.connect()
        yield obs
        await obs.close()

    # ── Table creation ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_tables_created(self, observer):
        cursor = await observer._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [r[0] for r in await cursor.fetchall()]
        assert "emergence_observations" in tables
        assert "emergence_predictions" in tables
        assert "emergence_signatures" in tables
        assert "emergence_weights" in tables

    # ── Observation ───────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_observe_returns_all_fields(self, observer):
        result = await observer.observe()
        assert "obs_id" in result
        assert "metrics" in result
        assert "emergence_probability" in result
        assert "prediction" in result
        assert "weights" in result

        metrics = result["metrics"]
        assert "r" in metrics
        assert "phi" in metrics
        assert "dh_dt" in metrics
        assert "bridges" in metrics
        assert "zone" in metrics

    @pytest.mark.asyncio
    async def test_observe_persists(self, observer):
        result = await observer.observe()
        cursor = await observer._conn.execute(
            "SELECT obs_id FROM emergence_observations WHERE obs_id = ?",
            (result["obs_id"],),
        )
        row = await cursor.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_multiple_observations(self, observer):
        r1 = await observer.observe()
        r2 = await observer.observe()
        assert r1["obs_id"] != r2["obs_id"]
        assert len(observer._observations) == 2

    # ── Probability computation ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_probability_in_range(self, observer):
        result = await observer.observe()
        p = result["emergence_probability"]
        assert 0.0 <= p <= 1.0

    def test_compute_emergence_critical_zone(self, observer):
        """r in [0.8, 1.0] should give high r_score."""
        prob, scores = observer._compute_emergence_probability(
            r=0.9, phi_delta=0.05, dh_dt=0.01, bridge_count=5
        )
        assert scores["r_score"] > 0.7
        assert prob > 0.5

    def test_compute_emergence_subcritical(self, observer):
        """r far below 0.8 should give low r_score."""
        prob, scores = observer._compute_emergence_probability(
            r=0.3, phi_delta=0.0, dh_dt=0.2, bridge_count=0
        )
        assert scores["r_score"] < 0.5
        assert prob < 0.5

    def test_compute_emergence_all_good(self, observer):
        """All metrics in ideal range → high probability."""
        prob, scores = observer._compute_emergence_probability(
            r=0.9, phi_delta=0.1, dh_dt=0.001, bridge_count=10
        )
        assert prob >= 0.7

    def test_compute_emergence_all_bad(self, observer):
        """All metrics out of range → low probability."""
        prob, scores = observer._compute_emergence_probability(
            r=2.0, phi_delta=-0.1, dh_dt=0.5, bridge_count=0
        )
        assert prob < 0.3

    # ── Prediction classification ─────────────────────────────────────────────

    def test_classify_imminent(self, observer):
        assert observer._classify_prediction(0.8) == "imminent"

    def test_classify_possible(self, observer):
        assert observer._classify_prediction(0.55) == "possible"

    def test_classify_unlikely(self, observer):
        assert observer._classify_prediction(0.35) == "unlikely"

    def test_classify_dormant(self, observer):
        assert observer._classify_prediction(0.1) == "dormant"

    # ── Outcome recording + weight updates ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_record_outcome_not_found(self, observer):
        result = await observer.record_outcome("nonexistent", True)
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_record_outcome_success(self, observer):
        # Force a prediction by manipulating thresholds to make probability high
        observer._thresholds["emergence_threshold"] = 0.0  # everything is "imminent"
        obs_result = await observer.observe()

        # If a prediction was created
        pred_id = obs_result.get("pred_id")
        if pred_id:
            result = await observer.record_outcome(pred_id, True, "novel insight emerged")
            assert result["ok"] is True
            assert result["actual"] == "emerged"
            assert "weights_after" in result

    @pytest.mark.asyncio
    async def test_weight_update_on_emergence(self, observer):
        """Weights should shift when we record outcomes."""
        original_weights = dict(observer._weights)

        # Create observation + prediction manually
        obs_id = "test_obs_1"
        await observer._conn.execute(
            """INSERT INTO emergence_observations
               (obs_id, timestamp, r, entropy, integration, phi, phi_delta, dh_dt,
                bridge_count, zone, emergence_probability, prediction)
               VALUES (?, ?, 0.9, 0.4, 0.5, 0.6, 0.05, 0.01, 5, 'critical', 0.8, 'imminent')""",
            (obs_id, time.time()),
        )
        pred_id = "test_pred_1"
        await observer._conn.execute(
            """INSERT INTO emergence_predictions
               (pred_id, obs_id, predicted_at, probability, prediction)
               VALUES (?, ?, ?, 0.8, 'imminent')""",
            (pred_id, obs_id, time.time()),
        )
        await observer._conn.commit()

        result = await observer.record_outcome(pred_id, True, "cross-domain insight")
        assert result["ok"] is True

        # Weights should have changed
        new_weights = result["weights_after"]
        assert new_weights != original_weights

    @pytest.mark.asyncio
    async def test_weight_normalization(self, observer):
        """Weights should always sum to ~1.0 after updates."""
        obs_id = "test_obs_2"
        await observer._conn.execute(
            """INSERT INTO emergence_observations
               (obs_id, timestamp, r, entropy, integration, phi, phi_delta, dh_dt,
                bridge_count, zone, emergence_probability, prediction)
               VALUES (?, ?, 0.9, 0.4, 0.5, 0.6, 0.05, 0.01, 5, 'critical', 0.8, 'imminent')""",
            (obs_id, time.time()),
        )
        pred_id = "test_pred_2"
        await observer._conn.execute(
            """INSERT INTO emergence_predictions
               (pred_id, obs_id, predicted_at, probability, prediction)
               VALUES (?, ?, ?, 0.8, 'imminent')""",
            (pred_id, obs_id, time.time()),
        )
        await observer._conn.commit()

        await observer.record_outcome(pred_id, True)
        total = sum(observer._weights.values())
        assert abs(total - 1.0) < 0.01

    # ── Signature learning ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_signature_created_on_outcome(self, observer):
        obs_id = "test_obs_3"
        await observer._conn.execute(
            """INSERT INTO emergence_observations
               (obs_id, timestamp, r, entropy, integration, phi, phi_delta, dh_dt,
                bridge_count, zone, emergence_probability, prediction)
               VALUES (?, ?, 0.9, 0.4, 0.5, 0.6, 0.05, 0.01, 5, 'critical', 0.8, 'imminent')""",
            (obs_id, time.time()),
        )
        pred_id = "test_pred_3"
        await observer._conn.execute(
            """INSERT INTO emergence_predictions
               (pred_id, obs_id, predicted_at, probability, prediction)
               VALUES (?, ?, ?, 0.8, 'imminent')""",
            (pred_id, obs_id, time.time()),
        )
        await observer._conn.commit()

        await observer.record_outcome(pred_id, True)

        sigs = await observer.get_signatures()
        assert sigs["n"] >= 1
        assert sigs["signatures"][0]["n_obs"] >= 1

    # ── History and dashboard ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_get_history_empty(self, observer):
        result = await observer.get_history()
        assert result["n"] == 0
        assert result["trend"] == "insufficient_data"

    @pytest.mark.asyncio
    async def test_get_history_with_data(self, observer):
        await observer.observe()
        await observer.observe()
        await observer.observe()

        result = await observer.get_history()
        assert result["n"] == 3
        assert "avg_probability" in result

    @pytest.mark.asyncio
    async def test_get_predictions_empty(self, observer):
        result = await observer.get_predictions()
        assert result["n"] == 0

    @pytest.mark.asyncio
    async def test_get_dashboard(self, observer):
        result = await observer.get_dashboard()
        assert "current" in result
        assert "trend" in result
        assert "weights" in result
        assert "prediction" in result["current"]
        assert "probability" in result["current"]
        assert "metrics" in result["current"]

    # ── dH/dt computation ─────────────────────────────────────────────────────

    def test_dh_dt_no_history(self, observer):
        """First observation should have dH/dt = 0."""
        dh = observer._compute_dh_dt(0.5, time.time())
        assert dh == 0.0

    def test_dh_dt_with_history(self, observer):
        """dH/dt should reflect entropy change rate."""
        observer._last_entropy = 0.4
        observer._last_obs_time = time.time() - 10  # 10 seconds ago
        dh = observer._compute_dh_dt(0.5, time.time())
        # (0.5 - 0.4) / 10 ≈ 0.01
        assert abs(dh - 0.01) < 0.005

    def test_dh_dt_decreasing(self, observer):
        """Negative dH/dt when entropy decreasing."""
        observer._last_entropy = 0.6
        observer._last_obs_time = time.time() - 10
        dh = observer._compute_dh_dt(0.4, time.time())
        assert dh < 0

    # ── Component score edge cases ────────────────────────────────────────────

    def test_phi_delta_negative(self, observer):
        """Negative phi_delta should give low phi_score."""
        _, scores = observer._compute_emergence_probability(
            r=0.9, phi_delta=-0.1, dh_dt=0.01, bridge_count=5
        )
        assert scores["phi_score"] < 0.5

    def test_high_entropy_change(self, observer):
        """High |dH/dt| should give low entropy_score."""
        _, scores = observer._compute_emergence_probability(
            r=0.9, phi_delta=0.05, dh_dt=0.5, bridge_count=5
        )
        assert scores["entropy_score"] < 0.3

    def test_zero_bridges(self, observer):
        """Zero bridges should give zero bridge_score."""
        _, scores = observer._compute_emergence_probability(
            r=0.9, phi_delta=0.05, dh_dt=0.01, bridge_count=0
        )
        assert scores["bridge_score"] == 0.0

    # ── Fallback behavior ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_observe_without_subsystems(self, observer):
        """Should work even without criticality/phi/graph subsystems."""
        assert observer._criticality is None
        assert observer._phi_calc is None
        assert observer._knowledge_graph is None

        result = await observer.observe()
        assert result["prediction"] in ("imminent", "possible", "unlikely", "dormant")

    @pytest.mark.asyncio
    async def test_observation_count_capped(self, observer):
        """In-memory observations should be capped at 100."""
        for _ in range(110):
            await observer.observe()
        assert len(observer._observations) == 100
