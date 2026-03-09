"""Tests for data_pipeline advanced metrics functions."""
from __future__ import annotations

from unittest import mock

import pytest

from app.data_pipeline import (
    calculate_possessions,
    calculate_pace,
    offensive_rating,
    defensive_rating,
)
from app.feature_engineering import (
    calculate_possessions_from_boxscore,
    calculate_game_pace,
    calculate_ppp,
    calculate_three_point_rate,
    calculate_free_throw_rate,
    calculate_orb_rate,
    calculate_tov_rate,
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


class TestCalculatePossessionsFromBoxscore:
    """Test calculate_possessions_from_boxscore: FGA - OREB + TOV + 0.44 * FTA."""

    def test_typical_boxscore(self):
        bs = {"fga": 88, "oreb": 10, "turnovers": 14, "fta": 22}
        poss = calculate_possessions_from_boxscore(bs)
        # 88 - 10 + 14 + 0.44*22 = 88 - 10 + 14 + 9.68 = 101.68
        assert abs(poss - 101.68) < 0.01

    def test_empty_boxscore_returns_minimum(self):
        poss = calculate_possessions_from_boxscore({})
        assert poss == 1

    def test_high_scoring_boxscore(self):
        bs = {"fga": 95, "oreb": 12, "turnovers": 16, "fta": 30}
        poss = calculate_possessions_from_boxscore(bs)
        # 95 - 12 + 16 + 0.44*30 = 95 - 12 + 16 + 13.2 = 112.2
        assert abs(poss - 112.2) < 0.01

    def test_never_below_one(self):
        bs = {"fga": 0, "oreb": 100, "turnovers": 0, "fta": 0}
        poss = calculate_possessions_from_boxscore(bs)
        assert poss == 1


class TestCalculateGamePace:
    """Test calculate_game_pace: simple average clamped to [96, 103]."""

    def test_average(self):
        assert calculate_game_pace(100.0, 98.0) == 99.0

    def test_equal(self):
        assert calculate_game_pace(100.0, 100.0) == 100.0

    def test_high_paces_clamped(self):
        # (110 + 108) / 2 = 109 → clamped to 103
        assert calculate_game_pace(110.0, 108.0) == 103.0


class TestCalculatePPP:
    """Test calculate_ppp: off_rating / 100."""

    def test_typical(self):
        assert calculate_ppp(112.0) == 1.12

    def test_low(self):
        assert calculate_ppp(100.0) == 1.0

    def test_high(self):
        assert calculate_ppp(120.0) == 1.2


class TestCalculateThreePointRate:
    """Test calculate_three_point_rate: three_pa / fga."""

    def test_typical(self):
        assert abs(calculate_three_point_rate(33, 88) - 0.375) < 0.001

    def test_zero_fga(self):
        assert calculate_three_point_rate(10, 0) == 0.0

    def test_zero_three_pa(self):
        assert calculate_three_point_rate(0, 88) == 0.0


class TestCalculateFreeThrowRate:
    """Test calculate_free_throw_rate: fta / fga."""

    def test_typical(self):
        assert abs(calculate_free_throw_rate(22, 88) - 0.25) < 0.001

    def test_zero_fga(self):
        assert calculate_free_throw_rate(10, 0) == 0.0

    def test_zero_fta(self):
        assert calculate_free_throw_rate(0, 88) == 0.0


class TestCalculateOrbRate:
    """Test calculate_orb_rate: oreb / (oreb + opp_dreb)."""

    def test_typical(self):
        assert abs(calculate_orb_rate(10, 30) - 0.25) < 0.001

    def test_zero_total(self):
        assert calculate_orb_rate(0, 0) == 0.0

    def test_all_offensive(self):
        assert abs(calculate_orb_rate(10, 0) - 1.0) < 0.001


class TestCalculateTovRate:
    """Test calculate_tov_rate: turnovers / possessions."""

    def test_typical(self):
        assert abs(calculate_tov_rate(14, 100) - 0.14) < 0.001

    def test_zero_possessions(self):
        assert calculate_tov_rate(10, 0) == 0.0

    def test_zero_turnovers(self):
        assert calculate_tov_rate(0, 100) == 0.0


class TestPaceClamping:
    """Test calculate_game_pace clamping to [96, 103]."""

    def test_normal_pace_no_clamp(self):
        """Pace within range is unchanged."""
        result = calculate_game_pace(99.0, 99.0)
        assert result == 99.0

    def test_high_pace_clamped_to_103(self):
        """Fast teams get clamped to 103."""
        result = calculate_game_pace(110.0, 108.0)
        assert result == 103.0

    def test_low_pace_clamped_to_96(self):
        """Slow teams get clamped to 96."""
        result = calculate_game_pace(88.0, 90.0)
        assert result == 96.0

    def test_boundary_96(self):
        """Pace exactly at 96 stays at 96."""
        result = calculate_game_pace(96.0, 96.0)
        assert result == 96.0

    def test_boundary_103(self):
        """Pace exactly at 103 stays at 103."""
        result = calculate_game_pace(103.0, 103.0)
        assert result == 103.0


# ---------- API client: new endpoints ----------

class TestGameAdvancedStatsEndpoint:
    """Test game_advanced_stats endpoint registration and convenience methods."""

    def test_endpoint_registered(self):
        from app.api_client import BallDontLieClient
        assert "game_advanced_stats" in BallDontLieClient.ENDPOINTS
        spec = BallDontLieClient.ENDPOINTS["game_advanced_stats"]
        assert spec.path == "/game_advanced_stats"
        assert "game_ids[]" in spec.allowed_params
        assert "per_page" in spec.allowed_params

    def test_game_advanced_stats_calls_fetch(self):
        from app.api_client import BallDontLieClient
        with mock.patch("app.api_client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {
                "data": [{"team_id": 1, "off_rating": 112.5, "def_rating": 108.0, "pace": 100.1}],
                "meta": {},
            }
            mock_get.return_value.raise_for_status = mock.MagicMock()
            client = BallDontLieClient(api_key="test-key")
            result = client.game_advanced_stats(**{"game_ids[]": [42], "per_page": 100})
        assert len(result) == 1
        assert result[0]["off_rating"] == 112.5

    def test_module_level_get_game_advanced_stats(self):
        from app import api_client
        with mock.patch.object(api_client, "_default_client") as mock_client:
            mock_client.return_value.game_advanced_stats.return_value = [
                {"team_id": 1, "pace": 99.5},
            ]
            result = api_client.get_game_advanced_stats(42)
        assert len(result) == 1
        assert result[0]["pace"] == 99.5


class TestPlayerInjuriesEndpoint:
    """Test player_injuries endpoint registration and convenience methods."""

    def test_endpoint_registered(self):
        from app.api_client import BallDontLieClient
        assert "player_injuries" in BallDontLieClient.ENDPOINTS
        spec = BallDontLieClient.ENDPOINTS["player_injuries"]
        assert spec.path == "/player_injuries"
        assert "per_page" in spec.allowed_params

    def test_player_injuries_calls_fetch(self):
        from app.api_client import BallDontLieClient
        with mock.patch("app.api_client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {
                "data": [{"player_id": 10, "team_id": 1, "status": "Out"}],
                "meta": {},
            }
            mock_get.return_value.raise_for_status = mock.MagicMock()
            client = BallDontLieClient(api_key="test-key")
            result = client.player_injuries(per_page=100)
        assert len(result) == 1
        assert result[0]["status"] == "Out"

    def test_module_level_get_player_injuries(self):
        from app import api_client
        with mock.patch.object(api_client, "_default_client") as mock_client:
            mock_client.return_value.player_injuries.return_value = [
                {"player_id": 10, "team_id": 1, "status": "Out"},
            ]
            result = api_client.get_player_injuries()
        assert len(result) == 1
        assert result[0]["player_id"] == 10


# ---------- Page-based pagination ----------

class TestPageBasedPagination:
    """Test fetch_all_pages_paged auto-pagination with next_page."""

    def test_single_page(self):
        from app.api_client import BallDontLieClient
        with mock.patch("app.api_client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {
                "data": [{"id": 1}],
                "meta": {"next_page": None, "total_pages": 1},
            }
            mock_get.return_value.raise_for_status = mock.MagicMock()
            client = BallDontLieClient(api_key="test-key")
            result = client.fetch_all_pages_paged("game_advanced_stats", {"per_page": 25})
        assert len(result) == 1

    def test_multi_page(self):
        from app.api_client import BallDontLieClient
        responses = [
            {"data": [{"id": 1}], "meta": {"next_page": 2, "total_pages": 3}},
            {"data": [{"id": 2}], "meta": {"next_page": 3, "total_pages": 3}},
            {"data": [{"id": 3}], "meta": {"next_page": None, "total_pages": 3}},
        ]
        with mock.patch("app.api_client.requests.get") as mock_get:
            mock_get.return_value.raise_for_status = mock.MagicMock()
            mock_get.return_value.json.side_effect = responses
            client = BallDontLieClient(api_key="test-key")
            result = client.fetch_all_pages_paged("game_advanced_stats", {"per_page": 1})
        assert len(result) == 3
        assert [r["id"] for r in result] == [1, 2, 3]

    def test_page_limit_respected(self):
        from app.api_client import BallDontLieClient
        with mock.patch("app.api_client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {
                "data": [{"id": 1}],
                "meta": {"next_page": 2, "total_pages": 100},
            }
            mock_get.return_value.raise_for_status = mock.MagicMock()
            client = BallDontLieClient(api_key="test-key")
            result = client.fetch_all_pages_paged("game_advanced_stats", {"per_page": 1}, page_limit=2)
        # Only 2 pages fetched even though next_page keeps returning 2
        assert len(result) == 2


# ---------- Data pipeline: fetch_game_advanced_stats ----------

class TestFetchGameAdvancedStats:
    """Test data_pipeline.fetch_game_advanced_stats."""

    def test_returns_keyed_by_team_id(self):
        from app.data_pipeline import fetch_game_advanced_stats
        fake_data = [
            {"team_id": 1, "off_rating": 112.5, "def_rating": 108.0,
             "pace": 100.1, "ts_pct": 0.58, "efg_pct": 0.54,
             "ast_pct": 0.60, "reb_pct": 0.50, "tov_pct": 0.12},
            {"team_id": 2, "off_rating": 110.0, "def_rating": 109.5,
             "pace": 99.3, "ts_pct": 0.57, "efg_pct": 0.53,
             "ast_pct": 0.58, "reb_pct": 0.49, "tov_pct": 0.13},
        ]
        with mock.patch("app.data_pipeline.BallDontLieClient") as MockClient:
            MockClient.return_value.game_advanced_stats.return_value = fake_data
            result = fetch_game_advanced_stats(42)

        assert 1 in result
        assert 2 in result
        assert result[1]["off_rating"] == 112.5
        assert result[2]["pace"] == 99.3
        assert result[1]["efg_pct"] == 0.54

    def test_empty_response(self):
        from app.data_pipeline import fetch_game_advanced_stats
        with mock.patch("app.data_pipeline.BallDontLieClient") as MockClient:
            MockClient.return_value.game_advanced_stats.return_value = []
            result = fetch_game_advanced_stats(999)
        assert result == {}
