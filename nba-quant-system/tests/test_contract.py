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
    def test_review_sends_no_games_message_when_no_finished_games(self):
        """Review gracefully exits with message when no finished games."""
        with mock.patch("app.review_engine.BallDontLieClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_game.return_value = {"id": 1, "status": "In Progress"}
            with mock.patch("app.supabase_client.fetch_all_predictions") as mock_fetch:
                mock_fetch.return_value = [
                    {"game_id": 1, "payload": {"spread_pick": "home", "total_pick": "大分"}},
                ]
                with mock.patch("app.review_engine.send_message") as mock_send:
                    from app.review_engine import run_review
                    run_review()

            assert mock_send.call_count == 1
            sent_text = mock_send.call_args[0][0]
            assert "当前没有可复盘比赛" in sent_text

    def test_review_sends_no_games_when_empty_response(self):
        """Review gracefully exits when there are no predictions."""
        with mock.patch("app.supabase_client.fetch_all_predictions") as mock_fetch:
            mock_fetch.return_value = []
            with mock.patch("app.review_engine.send_message") as mock_send:
                from app.review_engine import run_review
                run_review()

            sent_text = mock_send.call_args[0][0]
            assert "当前没有可复盘比赛" in sent_text

    def test_review_does_not_crash_on_api_error(self):
        """Review doesn't crash when API raises an exception."""
        with mock.patch("app.review_engine.BallDontLieClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.get_game.side_effect = RuntimeError("API error")
            with mock.patch("app.supabase_client.fetch_all_predictions") as mock_fetch:
                mock_fetch.return_value = [
                    {"game_id": 1, "payload": {"spread_pick": "home", "total_pick": "大分"}},
                ]
                with mock.patch("app.review_engine.send_message") as mock_send:
                    from app.review_engine import run_review
                    run_review()

            sent_text = mock_send.call_args[0][0]
            assert "当前没有可复盘比赛" in sent_text


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
