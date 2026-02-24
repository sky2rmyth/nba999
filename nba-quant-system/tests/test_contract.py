"""Tests for NBA quant system contract requirements."""
from __future__ import annotations

import os
from unittest import mock

import numpy as np
import pytest

os.environ.pop("SUPABASE_URL", None)
os.environ.pop("SUPABASE_KEY", None)

from app import supabase_client


@pytest.fixture(autouse=True)
def _reset_supabase():
    supabase_client._client = None
    supabase_client._available = None
    yield
    supabase_client._client = None
    supabase_client._available = None


# ---------- Star rating thresholds (Requirement 6) ----------

class TestStarRating:
    def test_edge_to_stars_5_at_15(self):
        from app.rating_engine import _edge_to_stars
        assert _edge_to_stars(15.0) == 5

    def test_edge_to_stars_5_above_15(self):
        from app.rating_engine import _edge_to_stars
        assert _edge_to_stars(20.0) == 5

    def test_edge_to_stars_4_at_12(self):
        from app.rating_engine import _edge_to_stars
        assert _edge_to_stars(12.0) == 4

    def test_edge_to_stars_4_at_14(self):
        from app.rating_engine import _edge_to_stars
        assert _edge_to_stars(14.9) == 4

    def test_edge_to_stars_3_at_9(self):
        from app.rating_engine import _edge_to_stars
        assert _edge_to_stars(9.0) == 3

    def test_edge_to_stars_2_at_7(self):
        from app.rating_engine import _edge_to_stars
        assert _edge_to_stars(7.0) == 2

    def test_edge_to_stars_1_at_5(self):
        from app.rating_engine import _edge_to_stars
        assert _edge_to_stars(5.0) == 1

    def test_edge_to_stars_0_below_5(self):
        from app.rating_engine import _edge_to_stars
        assert _edge_to_stars(4.9) == 0

    def test_stars_display(self):
        from app.rating_engine import stars_display
        assert stars_display(5) == "★★★★★"
        assert stars_display(3) == "★★★"
        assert stars_display(1) == "★"
        assert stars_display(0) == "☆"


# ---------- Edge scoring (Requirement 4) ----------

class TestEdgeScoring:
    def test_compute_edge_score_basic(self):
        from app.rating_engine import compute_edge_score
        # 60% probability vs 50% implied = 10% edge
        score = compute_edge_score(0.60, 0.50)
        assert score == 10.0

    def test_compute_edge_score_zero(self):
        from app.rating_engine import compute_edge_score
        score = compute_edge_score(0.50, 0.50)
        assert score == 0.0

    def test_compute_edge_score_capped_at_100(self):
        from app.rating_engine import compute_edge_score
        score = compute_edge_score(1.0, 0.0)
        assert score == 100.0

    def test_compute_ev_positive(self):
        from app.rating_engine import compute_ev
        # 60% prob, 1.91 odds → EV = 0.60 * 1.91 - 1 = 0.146
        ev = compute_ev(0.60, 1.91)
        assert ev > 0

    def test_compute_ev_negative(self):
        from app.rating_engine import compute_ev
        # 40% prob, 1.91 odds → EV = 0.40 * 1.91 - 1 = -0.236
        ev = compute_ev(0.40, 1.91)
        assert ev < 0


# ---------- Kelly fraction (Requirement 8) ----------

class TestKellyStake:
    def test_kelly_positive_edge(self):
        from app.rating_engine import compute_kelly_stake
        result = compute_kelly_stake(0.60, 1.91, bankroll=10000.0)
        assert result["recommended_stake"] > 0
        assert result["bankroll_after_bet"] < 10000.0

    def test_kelly_negative_edge(self):
        from app.rating_engine import compute_kelly_stake
        result = compute_kelly_stake(0.40, 1.91, bankroll=10000.0)
        assert result["recommended_stake"] == 0.0
        assert result["bankroll_after_bet"] == 10000.0

    def test_kelly_half_fraction(self):
        from app.rating_engine import compute_kelly_stake
        full = compute_kelly_stake(0.60, 1.91, bankroll=10000.0, fraction=1.0)
        half = compute_kelly_stake(0.60, 1.91, bankroll=10000.0, fraction=0.5)
        assert abs(half["recommended_stake"] - full["recommended_stake"] / 2) < 0.01


# ---------- Monte Carlo simulation (Requirement 3) ----------

class TestMonteCarlo:
    def test_simulation_returns_home_win_probability(self):
        from app.game_simulator import run_possession_simulation
        result = run_possession_simulation(
            game_id=12345,
            predicted_home_score=110.0,
            predicted_away_score=105.0,
            home_variance=64.0,
            away_variance=64.0,
            spread_line=-3.5,
            total_line=215.0,
        )
        assert "home_win_probability" in result
        assert 0.0 <= result["home_win_probability"] <= 1.0

    def test_simulation_minimum_10000_runs(self):
        from app.game_simulator import run_possession_simulation
        result = run_possession_simulation(
            game_id=99999,
            predicted_home_score=108.0,
            predicted_away_score=106.0,
            home_variance=64.0,
            away_variance=64.0,
            spread_line=-2.0,
            total_line=214.0,
        )
        assert result["simulation_count"] >= 10000

    def test_simulation_has_required_fields(self):
        from app.game_simulator import run_possession_simulation
        result = run_possession_simulation(
            game_id=11111,
            predicted_home_score=112.0,
            predicted_away_score=108.0,
            home_variance=64.0,
            away_variance=64.0,
            spread_line=-4.0,
            total_line=220.0,
        )
        required_fields = [
            "spread_cover_probability", "over_probability",
            "home_win_probability", "expected_home_score",
            "expected_visitor_score", "predicted_margin",
            "predicted_total", "simulation_count",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"


# ---------- Chinese language (Requirement 2) ----------

class TestChineseOutput:
    def test_model_status_report_no_english_mae(self):
        from app.i18n_cn import cn
        report = cn("model_status_report",
                     version="v5", available="✅",
                     training_samples=500,
                     mae_display="4.5 / 3.2",
                     sc_acc="55.0%", to_acc="52.0%",
                     last_trained="2025-01-01")
        assert "MAE" not in report
        assert "平均误差" in report

    def test_model_status_report_matches_contract(self):
        from app.i18n_cn import cn
        report = cn("model_status_report",
                     version="v5", available="✅",
                     training_samples=500,
                     mae_display="4.5 / 3.2",
                     sc_acc="55.0%", to_acc="52.0%",
                     last_trained="2025-01-01")
        assert "模型状态报告" in report
        assert "版本" in report
        assert "模型状态" in report
        assert "训练样本" in report
        assert "让分准确率" in report
        assert "大小分准确率" in report
        assert "最后训练时间" in report

    def test_model_loaded_string_matches_contract(self):
        from app.i18n_cn import cn
        assert "Supabase已加载" in cn("model_loaded")

    def test_review_no_games_message(self):
        from app.i18n_cn import cn
        msg = cn("review_no_games")
        assert "复盘系统" in msg
        assert "当前没有可复盘比赛" in msg
        assert "比赛尚未结束" in msg


# ---------- Review safety (Requirement 9) ----------

class TestReviewSafety:
    def test_review_generates_report_with_predictions(self, tmp_path):
        """Review generates review_latest.json when predictions exist."""
        predictions = [
            {
                "game_id": 1,
                "spread_pick": "home",
                "total_pick": "over",
                "predicted_margin": 5.0,
                "predicted_total": 215.0,
            },
        ]
        review_rows = [
            {"game_id": 1, "spread_hit": True, "ou_hit": True},
        ]
        with mock.patch("app.review_engine.load_latest_predictions", return_value=predictions):
            with mock.patch("app.review_engine.fetch_game_result") as mock_fetch:
                mock_fetch.return_value = {"home_team": "Team A", "visitor_team": "Team B", "home_score": 110, "visitor_score": 105, "spread": 0, "total": 0}
                with mock.patch("app.supabase_client.save_review_result"):
                    with mock.patch("app.supabase_client.fetch_recent_review_results", return_value=review_rows):
                        import os
                        old_cwd = os.getcwd()
                        os.chdir(tmp_path)
                        try:
                            from app.review_engine import run_review
                            run_review()
                        finally:
                            os.chdir(old_cwd)

        import json
        report = json.loads((tmp_path / "review_latest.json").read_text())
        assert report["games"] == 1
        assert "spread_hit_rate" in report
        assert "total_hit_rate" in report

    def test_review_uses_predictions(self):
        """Review loads predictions from predictions via load_latest_predictions."""
        with mock.patch("app.review_engine.load_latest_predictions", return_value=[]) as mock_load:
            with mock.patch("app.supabase_client.fetch_recent_review_results", return_value=[]):
                from app.review_engine import run_review
                run_review()  # empty predictions should not crash
                mock_load.assert_called_once()

    def test_review_writes_to_review_results(self):
        """Review engine persists results via save_review_result."""
        predictions = [
            {
                "game_id": 1,
                "spread_pick": "home",
                "total_pick": "over",
                "predicted_margin": 5.0,
                "predicted_total": 215.0,
            },
        ]
        with mock.patch("app.review_engine.load_latest_predictions", return_value=predictions):
            with mock.patch("app.review_engine.fetch_game_result") as mock_fetch:
                mock_fetch.return_value = {"home_team": "Team A", "visitor_team": "Team B", "home_score": 110, "visitor_score": 105, "spread": 0, "total": 0}
                with mock.patch("app.supabase_client.save_review_result") as mock_save:
                    with mock.patch("app.supabase_client.fetch_recent_review_results", return_value=[]):
                        from app.review_engine import run_review
                        run_review()
                        mock_save.assert_called_once()

    def test_review_computes_rates_from_review_results(self):
        """Hit rates are computed from review_results, not predictions."""
        review_rows = [
            {"game_id": 1, "spread_hit": True, "ou_hit": False},
            {"game_id": 2, "spread_hit": False, "ou_hit": True},
        ]
        with mock.patch("app.review_engine.load_latest_predictions", return_value=[]):
            with mock.patch("app.supabase_client.fetch_recent_review_results", return_value=review_rows):
                from app.review_engine import run_review
                import os, json, tempfile
                with tempfile.TemporaryDirectory() as td:
                    old_cwd = os.getcwd()
                    os.chdir(td)
                    try:
                        run_review()
                        report = json.loads(open("review_latest.json").read())
                    finally:
                        os.chdir(old_cwd)
                assert report["games"] == 2
                assert report["spread_hit_rate"] == 0.5
                assert report["total_hit_rate"] == 0.5


# ---------- Deduplication helper ----------

class TestDeduplicatePredictions:
    def test_keeps_latest_per_game_id(self):
        from app.review_engine import _deduplicate_predictions
        preds = [
            {"game_id": 1, "created_at": "2025-01-15T01:00:00", "payload": "old"},
            {"game_id": 1, "created_at": "2025-01-15T02:00:00", "payload": "new"},
            {"game_id": 2, "created_at": "2025-01-15T01:00:00", "payload": "only"},
        ]
        result = _deduplicate_predictions(preds)
        assert len(result) == 2
        by_gid = {r["game_id"]: r for r in result}
        assert by_gid[1]["payload"] == "new"
        assert by_gid[2]["payload"] == "only"

    def test_single_prediction_unchanged(self):
        from app.review_engine import _deduplicate_predictions
        preds = [{"game_id": 1, "created_at": "2025-01-15T01:00:00"}]
        assert _deduplicate_predictions(preds) == preds

    def test_empty_list(self):
        from app.review_engine import _deduplicate_predictions
        assert _deduplicate_predictions([]) == []

    def test_missing_created_at_treated_as_empty_string(self):
        from app.review_engine import _deduplicate_predictions
        preds = [
            {"game_id": 1, "payload": "no_ts"},
            {"game_id": 1, "created_at": "2025-01-15T01:00:00", "payload": "with_ts"},
        ]
        result = _deduplicate_predictions(preds)
        assert len(result) == 1
        assert result[0]["payload"] == "with_ts"


# ---------- Supabase fetch predictions (Requirement 10/13) ----------

class TestSupabaseFetchPredictions:
    def test_fetch_predictions_for_date_returns_matching(self):
        fake_client = mock.MagicMock()
        fake_client.table.return_value.select.return_value.execute.return_value.data = [
            {"payload": {"game_date": "2025-01-15", "game_id": 42, "is_final_prediction": True}},
            {"payload": {"game_date": "2025-01-14", "game_id": 41, "is_final_prediction": True}},
        ]
        supabase_client._client = fake_client
        supabase_client._available = True

        results = supabase_client.fetch_predictions_for_date("2025-01-15")
        assert len(results) == 1
        assert results[0]["game_id"] == 42

    def test_fetch_predictions_for_date_empty_when_not_configured(self):
        supabase_client._available = False
        results = supabase_client.fetch_predictions_for_date("2025-01-15")
        assert results == []


# ---------- Workflow schedule (Requirement 11) ----------

class TestWorkflowSchedule:
    def test_review_workflow_scheduled_at_7_utc(self):
        """Review workflow should be scheduled at 7:00 UTC."""
        import yaml
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent.parent
        with open(repo_root / ".github" / "workflows" / "review.yml") as f:
            wf = yaml.safe_load(f)
        # PyYAML parses 'on' as True; try both
        on_key = wf.get("on") or wf.get(True)
        schedules = on_key["schedule"]
        crons = [s["cron"] for s in schedules]
        assert "0 7 * * *" in crons
