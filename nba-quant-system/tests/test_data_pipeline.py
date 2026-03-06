"""Tests for data_pipeline advanced metrics functions."""
from __future__ import annotations

import pytest

from app.data_pipeline import (
    calculate_possessions,
    calculate_pace,
    offensive_rating,
    defensive_rating,
)


class TestCalculatePossessions:
    """Test calculate_possessions with the formula: FGA + 0.44 * FTA - ORB + TOV."""

    def test_typical_nba_stats(self):
        stats = {"fga": 88, "fta": 22, "offensive_rebounds": 10, "turnovers": 14}
        poss = calculate_possessions(stats)
        # 88 + 0.44*22 - 10 + 14 = 88 + 9.68 - 10 + 14 = 101.68
        assert abs(poss - 101.68) < 0.01

    def test_empty_stats_returns_minimum(self):
        poss = calculate_possessions({})
        assert poss == 1

    def test_zero_stats(self):
        stats = {"fga": 0, "fta": 0, "offensive_rebounds": 0, "turnovers": 0}
        poss = calculate_possessions(stats)
        assert poss == 1  # max(0, 1) = 1

    def test_high_scoring_game(self):
        stats = {"fga": 95, "fta": 30, "offensive_rebounds": 12, "turnovers": 16}
        poss = calculate_possessions(stats)
        # 95 + 0.44*30 - 12 + 16 = 95 + 13.2 - 12 + 16 = 112.2
        assert abs(poss - 112.2) < 0.01

    def test_never_returns_below_one(self):
        stats = {"fga": 0, "fta": 0, "offensive_rebounds": 100, "turnovers": 0}
        poss = calculate_possessions(stats)
        assert poss == 1


class TestCalculatePace:
    """Test calculate_pace (possessions per 48 minutes)."""

    def test_typical_pace(self):
        stats = {"fga": 88, "fta": 22, "offensive_rebounds": 10, "turnovers": 14}
        pace = calculate_pace(stats)
        # Possessions = 101.68, pace = 101.68 / 48 * 48 = 101.68
        assert abs(pace - 101.68) < 0.01

    def test_empty_stats(self):
        pace = calculate_pace({})
        assert pace == 1  # minimum possessions

    def test_pace_equals_possessions(self):
        """Pace per 48 minutes should equal possessions for a 48-minute game."""
        stats = {"fga": 90, "fta": 20, "offensive_rebounds": 10, "turnovers": 13}
        poss = calculate_possessions(stats)
        pace = calculate_pace(stats)
        assert abs(pace - poss) < 0.01


class TestOffensiveRating:
    """Test offensive_rating (points per 100 possessions)."""

    def test_typical_rating(self):
        # 110 points on 100 possessions = 110.0
        rating = offensive_rating(110, 100)
        assert abs(rating - 110.0) < 0.01

    def test_high_efficiency(self):
        # 120 points on 100 possessions = 120.0
        rating = offensive_rating(120, 100)
        assert abs(rating - 120.0) < 0.01

    def test_zero_possessions(self):
        rating = offensive_rating(110, 0)
        assert rating == 0

    def test_realistic_nba_range(self):
        # Typical NBA: ~110 points on ~100 possessions
        rating = offensive_rating(112, 100)
        assert 105 <= rating <= 120


class TestDefensiveRating:
    """Test defensive_rating (opponent points per 100 possessions)."""

    def test_typical_rating(self):
        rating = defensive_rating(105, 100)
        assert abs(rating - 105.0) < 0.01

    def test_zero_possessions(self):
        rating = defensive_rating(105, 0)
        assert rating == 0

    def test_realistic_nba_range(self):
        rating = defensive_rating(108, 100)
        assert 105 <= rating <= 120


class TestPossessionBasedMetrics:
    """Integration tests verifying metrics stay in expected NBA ranges."""

    def test_pace_in_expected_range(self):
        """With typical NBA stats, pace should be 95-105."""
        stats = {"fga": 88, "fta": 22, "offensive_rebounds": 10, "turnovers": 14}
        pace = calculate_pace(stats)
        assert 95 <= pace <= 105

    def test_off_rating_in_expected_range(self):
        """With typical NBA scoring, off rating should be 108-120."""
        poss = calculate_possessions(
            {"fga": 88, "fta": 22, "offensive_rebounds": 10, "turnovers": 14}
        )
        rating = offensive_rating(112, poss)
        assert 108 <= rating <= 120

    def test_predicted_total_in_range(self):
        """Simulated total from possession model should be 210-240."""
        home_stats = {"fga": 88, "fta": 22, "offensive_rebounds": 10, "turnovers": 14}
        away_stats = {"fga": 86, "fta": 20, "offensive_rebounds": 9, "turnovers": 15}
        home_poss = calculate_possessions(home_stats)
        away_poss = calculate_possessions(away_stats)
        home_off = offensive_rating(112, home_poss)
        away_off = offensive_rating(108, away_poss)
        pace = (calculate_pace(home_stats) + calculate_pace(away_stats)) / 2.0
        predicted_total = pace * (home_off / 100 + away_off / 100)
        assert 210 <= predicted_total <= 240
