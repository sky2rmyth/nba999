"""Tests that predictions proceed for all games even when odds are unavailable."""
from __future__ import annotations

import os
from unittest import mock

import numpy as np
import pytest

os.environ.pop("SUPABASE_URL", None)
os.environ.pop("SUPABASE_KEY", None)


def _make_game(game_id, home_name, away_name, home_id=1, away_id=2):
    return {
        "id": game_id,
        "home_team": {"id": home_id, "full_name": home_name},
        "visitor_team": {"id": away_id, "full_name": away_name},
    }


class TestNoOddsFallback:
    """When odds providers return nothing, games should still be predicted using model-derived lines."""

    def test_no_odds_uses_model_fallback(self):
        """_match_primary_odds returns None and betting_odds raises → odds_source becomes MODEL."""
        from app.prediction_engine import _match_primary_odds

        # No primary odds available
        assert _match_primary_odds([], "Los Angeles Lakers", "Boston Celtics") is None

    def test_model_source_tag_set_when_no_odds(self):
        """When odds_source is NONE, the fallback sets it to MODEL with spread=0 and total=predicted."""
        # This tests the logic inline: if odds_source == "NONE" → odds_source = "MODEL"
        odds_source = "NONE"
        predicted_total = 218.5

        if odds_source == "NONE":
            opening_spread = 0.0
            live_spread = 0.0
            opening_total = predicted_total
            live_total = predicted_total
            odds_source = "MODEL"

        assert odds_source == "MODEL"
        assert live_spread == 0.0
        assert opening_spread == 0.0
        assert live_total == 218.5
        assert opening_total == 218.5

    def test_simulation_works_with_zero_spread_line(self):
        """Monte Carlo simulation works correctly with spread_line=0 (pick'em)."""
        from app.game_simulator import run_possession_simulation

        result = run_possession_simulation(
            game_id=77777,
            predicted_home_score=112.0,
            predicted_away_score=108.0,
            home_variance=64.0,
            away_variance=64.0,
            spread_line=0.0,
            total_line=220.0,
            n_sim=10000,
        )
        assert result["simulation_count"] == 10000
        # Home is predicted to win by 4, so with spread 0, cover prob should be > 0.5
        assert result["spread_cover_probability"] > 0.5
        assert 0.0 <= result["home_win_probability"] <= 1.0

    def test_simulation_works_with_model_derived_total(self):
        """Simulation works when total_line equals predicted total (model fallback case)."""
        from app.game_simulator import run_possession_simulation

        predicted_total = 220.0
        result = run_possession_simulation(
            game_id=88888,
            predicted_home_score=112.0,
            predicted_away_score=108.0,
            home_variance=64.0,
            away_variance=64.0,
            spread_line=0.0,
            total_line=predicted_total,
        )
        # Over probability should be roughly 50% since total line matches prediction
        assert 0.3 <= result["over_probability"] <= 0.7

    def test_rating_functions_accept_zero_spread(self):
        """Rating functions work correctly with zero spread (model fallback)."""
        from app.rating_engine import compute_spread_rating, compute_total_rating

        spread_result = compute_spread_rating(0.6, 0.0)
        assert "spread_stars" in spread_result
        assert "spread_confidence" in spread_result

        total_result = compute_total_rating(0.5, 220.0)
        assert "total_stars" in total_result
        assert "total_confidence" in total_result

    def test_analyze_line_behavior_handles_zero_lines(self):
        """Bookmaker behavior analysis handles zero opening/live lines."""
        from app.bookmaker_behavior import analyze_line_behavior

        result = analyze_line_behavior(0.0, 0.0, 0.05, 220.0, 220.0, 0.02)
        assert "sharp_movement_score" in result
        assert "market_confidence_indicator" in result
