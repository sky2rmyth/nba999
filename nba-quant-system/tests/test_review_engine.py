"""Tests for review_engine hit-checking and rate calculation functions."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from app.review_engine import (
    calculate_rates,
    extract_prediction_fields,
    fetch_game_result,
    parse_prediction,
    spread_hit,
    total_hit,
)


# --- parse_prediction ---

class TestParsePrediction:
    def test_home_spread_pick(self):
        """Positive predicted_margin yields spread_pick 'home'."""
        row = {
            "game_id": 1,
            "payload": {
                "details": {
                    "simulation": {"predicted_margin": 5.0, "predicted_total": 215.0},
                    "total_rating": {"total_confidence": 60},
                }
            },
        }
        result = parse_prediction(row)
        assert result["spread_pick"] == "home"

    def test_away_spread_pick(self):
        """Negative predicted_margin yields spread_pick 'away'."""
        row = {
            "game_id": 2,
            "payload": {
                "details": {
                    "simulation": {"predicted_margin": -3.0, "predicted_total": 210.0},
                    "total_rating": {"total_confidence": 60},
                }
            },
        }
        result = parse_prediction(row)
        assert result["spread_pick"] == "away"

    def test_zero_margin_yields_away(self):
        """Zero predicted_margin yields spread_pick 'away'."""
        row = {
            "game_id": 3,
            "payload": {
                "details": {
                    "simulation": {"predicted_margin": 0, "predicted_total": 210.0},
                    "total_rating": {"total_confidence": 60},
                }
            },
        }
        result = parse_prediction(row)
        assert result["spread_pick"] == "away"

    def test_over_total_pick(self):
        """High total_confidence yields total_pick 'over'."""
        row = {
            "game_id": 4,
            "payload": {
                "details": {
                    "simulation": {"predicted_margin": 2.0, "predicted_total": 220.0},
                    "total_rating": {"total_confidence": 70},
                }
            },
        }
        result = parse_prediction(row)
        assert result["total_pick"] == "over"

    def test_under_total_pick(self):
        """Low total_confidence yields total_pick 'under'."""
        row = {
            "game_id": 5,
            "payload": {
                "details": {
                    "simulation": {"predicted_margin": 2.0, "predicted_total": 220.0},
                    "total_rating": {"total_confidence": 30},
                }
            },
        }
        result = parse_prediction(row)
        assert result["total_pick"] == "under"

    def test_missing_payload_defaults(self):
        """Missing payload falls back to 'away' and 'under'."""
        row = {"game_id": 6, "payload": {}}
        result = parse_prediction(row)
        assert result["spread_pick"] == "away"
        assert result["total_pick"] == "under"
        assert result["predicted_margin"] is None
        assert result["predicted_total"] is None

    def test_returns_game_id(self):
        """Returned dict contains game_id from input row."""
        row = {
            "game_id": 99,
            "payload": {
                "details": {
                    "simulation": {"predicted_margin": 1.0, "predicted_total": 200.0},
                    "total_rating": {"total_confidence": 55},
                }
            },
        }
        result = parse_prediction(row)
        assert result["game_id"] == 99


# --- spread_hit ---

class TestSpreadHit:
    def test_home_covers(self):
        """Home team wins by more than the spread."""
        row = {
            "final_home_score": 110,
            "final_visitor_score": 100,
            "spread": 5,
            "spread_pick": "home",
        }
        assert spread_hit(row) is True

    def test_home_does_not_cover(self):
        """Home team wins but does not cover the spread."""
        row = {
            "final_home_score": 103,
            "final_visitor_score": 100,
            "spread": 5,
            "spread_pick": "home",
        }
        assert spread_hit(row) is False

    def test_away_covers(self):
        """Away side covers when margin is less than spread."""
        row = {
            "final_home_score": 100,
            "final_visitor_score": 105,
            "spread": -3,
            "spread_pick": "away",
        }
        assert spread_hit(row) is True

    def test_away_does_not_cover(self):
        """Away side does not cover when margin exceeds spread."""
        row = {
            "final_home_score": 110,
            "final_visitor_score": 100,
            "spread": -3,
            "spread_pick": "away",
        }
        assert spread_hit(row) is False

    def test_unknown_side_returns_false(self):
        """Unknown spread_pick returns False."""
        row = {
            "final_home_score": 110,
            "final_visitor_score": 100,
            "spread": 0,
            "spread_pick": "unknown",
        }
        assert spread_hit(row) is False

    def test_exact_margin_equals_spread_home(self):
        """Margin exactly equals spread — not a hit for home."""
        row = {
            "final_home_score": 105,
            "final_visitor_score": 100,
            "spread": 5,
            "spread_pick": "home",
        }
        assert spread_hit(row) is False


# --- total_hit ---

class TestTotalHit:
    def test_over_hits(self):
        """Actual total exceeds the line."""
        row = {
            "final_home_score": 115,
            "final_visitor_score": 110,
            "total_pick": "over",
            "total_line": 220,
        }
        assert total_hit(row) is True

    def test_over_misses(self):
        """Actual total is under the line."""
        row = {
            "final_home_score": 100,
            "final_visitor_score": 105,
            "total_pick": "over",
            "total_line": 220,
        }
        assert total_hit(row) is False

    def test_under_hits(self):
        """Actual total is below the line."""
        row = {
            "final_home_score": 100,
            "final_visitor_score": 105,
            "total_pick": "under",
            "total_line": 220,
        }
        assert total_hit(row) is True

    def test_under_misses(self):
        """Actual total exceeds the line."""
        row = {
            "final_home_score": 115,
            "final_visitor_score": 110,
            "total_pick": "under",
            "total_line": 220,
        }
        assert total_hit(row) is False

    def test_unknown_total_returns_false(self):
        """Unknown total_pick returns False."""
        row = {
            "final_home_score": 110,
            "final_visitor_score": 100,
            "total_pick": "",
            "total_line": 220,
        }
        assert total_hit(row) is False

    def test_exact_total_equals_line_over(self):
        """Total exactly equals line — not a hit for over."""
        row = {
            "final_home_score": 110,
            "final_visitor_score": 110,
            "total_pick": "over",
            "total_line": 220,
        }
        assert total_hit(row) is False


# --- calculate_rates ---

class TestCalculateRates:
    def test_empty_rows(self):
        """Empty input returns all zeros."""
        assert calculate_rates([]) == (0, 0, 0)

    def test_all_hits(self):
        """100% hit rate returns 1.0 for all rates."""
        rows = [
            {"spread_hit": True, "ou_hit": True},
            {"spread_hit": True, "ou_hit": True},
        ]
        s, t, o = calculate_rates(rows)
        assert s == 1.0
        assert t == 1.0
        assert o == 1.0

    def test_no_hits(self):
        """0% hit rate returns 0 for all rates."""
        rows = [
            {"spread_hit": False, "ou_hit": False},
            {"spread_hit": False, "ou_hit": False},
        ]
        s, t, o = calculate_rates(rows)
        assert s == 0.0
        assert t == 0.0
        assert o == 0.0

    def test_mixed_hits(self):
        """Mixed results calculate correctly."""
        rows = [
            {"spread_hit": True, "ou_hit": False},
            {"spread_hit": False, "ou_hit": True},
            {"spread_hit": True, "ou_hit": True},
            {"spread_hit": False, "ou_hit": False},
        ]
        s, t, o = calculate_rates(rows)
        assert s == 0.5  # 2/4
        assert t == 0.5  # 2/4
        assert o == 0.5  # 4/8

    def test_single_row(self):
        """Single row with one hit."""
        rows = [{"spread_hit": True, "ou_hit": False}]
        s, t, o = calculate_rates(rows)
        assert s == 1.0
        assert t == 0.0
        assert o == 0.5


# --- extract_prediction_fields ---

class TestExtractPredictionFields:
    def test_home_spread_positive_margin(self):
        """Positive predicted_margin yields spread_pick 'home'."""
        row = {
            "game_id": 1,
            "payload": {
                "details": {
                    "simulation": {"predicted_margin": 5.0, "predicted_total": 215.0},
                }
            },
        }
        spread_pick, total_pick = extract_prediction_fields(row)
        assert spread_pick == "home"

    def test_away_spread_negative_margin(self):
        """Negative predicted_margin yields spread_pick 'away'."""
        row = {
            "game_id": 2,
            "payload": {
                "details": {
                    "simulation": {"predicted_margin": -3.0, "predicted_total": 210.0},
                }
            },
        }
        spread_pick, total_pick = extract_prediction_fields(row)
        assert spread_pick == "away"

    def test_zero_margin_yields_away(self):
        """Zero predicted_margin yields spread_pick 'away' (falsy)."""
        row = {
            "game_id": 3,
            "payload": {
                "details": {
                    "simulation": {"predicted_margin": 0, "predicted_total": 210.0},
                }
            },
        }
        spread_pick, _ = extract_prediction_fields(row)
        assert spread_pick == "away"

    def test_over_total_positive(self):
        """Positive predicted_total yields total_pick 'over'."""
        row = {
            "game_id": 4,
            "payload": {
                "details": {
                    "simulation": {"predicted_margin": 2.0, "predicted_total": 220.0},
                }
            },
        }
        _, total_pick = extract_prediction_fields(row)
        assert total_pick == "over"

    def test_under_total_negative(self):
        """Negative predicted_total yields total_pick 'under'."""
        row = {
            "game_id": 5,
            "payload": {
                "details": {
                    "simulation": {"predicted_margin": 2.0, "predicted_total": -5.0},
                }
            },
        }
        _, total_pick = extract_prediction_fields(row)
        assert total_pick == "under"

    def test_missing_payload_defaults(self):
        """Missing payload falls back to 'away' and 'under'."""
        row = {"game_id": 6, "payload": {}}
        spread_pick, total_pick = extract_prediction_fields(row)
        assert spread_pick == "away"
        assert total_pick == "under"

    def test_returns_tuple(self):
        """Returns a tuple of two strings."""
        row = {
            "game_id": 7,
            "payload": {
                "details": {
                    "simulation": {"predicted_margin": 1.0, "predicted_total": 200.0},
                }
            },
        }
        result = extract_prediction_fields(row)
        assert isinstance(result, tuple)
        assert len(result) == 2


# --- fetch_game_result ---

class TestFetchGameResult:
    @patch("app.review_engine.requests.get")
    def test_successful_fetch(self, mock_get):
        """Successful API call returns scores dict with spread and total."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {
                "status": "Final",
                "home_team_score": 105,
                "visitor_team_score": 98,
            }
        }
        mock_get.return_value = mock_resp
        result = fetch_game_result(12345)
        assert result["home_score"] == 105
        assert result["visitor_score"] == 98
        assert result["spread"] == 0
        assert result["total"] == 0

    @patch("app.review_engine.requests.get")
    def test_game_not_final_returns_none(self, mock_get):
        """Non-final game status returns None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {
                "status": "In Progress",
                "home_team_score": 50,
                "visitor_team_score": 48,
            }
        }
        mock_get.return_value = mock_resp
        result = fetch_game_result(12345)
        assert result is None

    @patch("app.review_engine.requests.get")
    def test_api_non_200_returns_none(self, mock_get):
        """Non-200 status code returns None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        mock_get.return_value = mock_resp
        result = fetch_game_result(12345)
        assert result is None

    @patch("app.review_engine.requests.get")
    def test_api_exception_returns_none(self, mock_get):
        """Network error returns None."""
        mock_get.side_effect = Exception("timeout")
        result = fetch_game_result(12345)
        assert result is None
