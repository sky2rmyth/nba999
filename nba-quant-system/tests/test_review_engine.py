"""Tests for review_engine hit-checking and rate calculation functions."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from app.review_engine import (
    build_review_message,
    build_review_summary,
    calc_spread_hit,
    calc_total_hit,
    calculate_rates,
    extract_prediction_fields,
    fetch_game_result,
    format_spread_text,
    format_total_text,
    parse_prediction,
    run_review,
    spread_hit,
    total_hit,
    zh_hit,
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
        """Successful API call returns scores dict with spread, total, and team names."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {
                "status": "Final",
                "home_team_score": 105,
                "visitor_team_score": 98,
                "home_team": {"full_name": "Los Angeles Lakers"},
                "visitor_team": {"full_name": "Golden State Warriors"},
            }
        }
        mock_get.return_value = mock_resp
        result = fetch_game_result(12345)
        assert result["home_score"] == 105
        assert result["visitor_score"] == 98
        assert result["spread"] == 0
        assert result["total"] == 0
        assert result["home_team"] == "Los Angeles Lakers"
        assert result["visitor_team"] == "Golden State Warriors"

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


# --- run_review Telegram notification ---

class TestRunReviewTelegram:
    @patch("app.review_engine.send_message")
    @patch("app.review_engine.fetch_game_result")
    @patch("app.review_engine.load_latest_predictions")
    def test_send_message_called_after_review(
        self, mock_load, mock_fetch, mock_send
    ):
        """run_review sends a Chinese Telegram message for each reviewed game."""
        mock_load.return_value = [
            {
                "game_id": 42,
                "payload": {"details": {"simulation": {"predicted_margin": 5.0, "predicted_total": 215.0}}},
            }
        ]
        mock_fetch.return_value = {
            "home_team": "Los Angeles Lakers",
            "visitor_team": "Golden State Warriors",
            "home_score": 110,
            "visitor_score": 100,
            "spread": 0,
            "total": 0,
        }

        with patch("app.supabase_client.save_review_result"), \
             patch("app.supabase_client.fetch_recent_review_results", return_value=[]):
            run_review()

        mock_send.assert_called_once()
        msg = mock_send.call_args[0][0]
        assert "NBA复盘结果" in msg
        assert "Los Angeles Lakers" in msg
        assert "Golden State Warriors" in msg
        assert "110" in msg
        assert "100" in msg
        assert "命中" in msg or "未中" in msg
        assert "True" not in msg
        assert "False" not in msg


# --- calc_spread_hit ---

class TestCalcSpreadHit:
    def test_home_covers_with_negative_spread(self):
        """Home favored by 4.5 and wins by 10 → hit."""
        assert calc_spread_hit("home", "LAL", "GSW", 113, 103, -4.5) is True

    def test_home_does_not_cover(self):
        """Home favored by 4.5 but wins by only 3 → miss."""
        assert calc_spread_hit("home", "LAL", "GSW", 106, 103, -4.5) is False

    def test_away_covers(self):
        """Away gets +4.5, home wins by only 3 → away covers."""
        assert calc_spread_hit("away", "LAL", "GSW", 106, 103, -4.5) is True

    def test_away_does_not_cover(self):
        """Away gets +4.5 but home wins by 10 → away miss."""
        assert calc_spread_hit("away", "LAL", "GSW", 113, 103, -4.5) is False

    def test_unknown_pick_returns_false(self):
        assert calc_spread_hit("unknown", "LAL", "GSW", 110, 100, 0) is False

    def test_exact_zero_adjusted_margin_home(self):
        """Adjusted margin exactly 0 is not a hit for home (push)."""
        assert calc_spread_hit("home", "LAL", "GSW", 105, 100, -5) is False


# --- calc_total_hit ---

class TestCalcTotalHit:
    def test_over_hits(self):
        assert calc_total_hit("over", 115, 110, 220) is True

    def test_over_misses(self):
        assert calc_total_hit("over", 100, 105, 220) is False

    def test_under_hits(self):
        assert calc_total_hit("under", 100, 105, 220) is True

    def test_under_misses(self):
        assert calc_total_hit("under", 115, 110, 220) is False

    def test_unknown_pick_returns_false(self):
        assert calc_total_hit("", 110, 100, 220) is False

    def test_exact_total_equals_line(self):
        """Total exactly equals line — not a hit for over (push)."""
        assert calc_total_hit("over", 110, 110, 220) is False


# --- zh_hit ---

class TestZhHit:
    def test_true_returns_hit(self):
        assert zh_hit(True) == "✅命中"

    def test_false_returns_miss(self):
        assert zh_hit(False) == "❌未中"


# --- format_spread_text ---

class TestFormatSpreadText:
    def test_home_pick(self):
        assert format_spread_text("home", "湖人", "勇士", -4.5) == "湖人 -4.5"

    def test_away_pick_negative_spread(self):
        """Away pick with negative spread shows + sign."""
        assert format_spread_text("away", "湖人", "勇士", -4.5) == "勇士 +4.5"

    def test_away_pick_positive_spread(self):
        """Away pick with positive spread shows - sign."""
        assert format_spread_text("away", "湖人", "勇士", 3.0) == "勇士 -3.0"


# --- format_total_text ---

class TestFormatTotalText:
    def test_over(self):
        assert format_total_text("over", 228.5) == "大分 228.5"

    def test_under(self):
        assert format_total_text("under", 228.5) == "小分 228.5"


# --- build_review_message ---

class TestBuildReviewMessage:
    def test_message_contains_chinese_elements(self):
        result = {
            "home_team": "湖人",
            "visitor_team": "勇士",
            "home_score": 113,
            "visitor_score": 103,
            "spread": -4.5,
            "total": 228.5,
        }
        pred = {"spread_pick": "home", "total_pick": "over"}
        msg = build_review_message(result, pred, True, False)
        assert "NBA复盘结果" in msg
        assert "湖人 vs 勇士" in msg
        assert "湖人 -4.5" in msg
        assert "大分 228.5" in msg
        assert "✅命中" in msg
        assert "❌未中" in msg
        assert "113 - 103" in msg
        assert "True" not in msg
        assert "False" not in msg


# --- build_review_summary ---

class TestBuildReviewSummary:
    def test_empty_data_returns_no_data_message(self):
        """No review rows returns the placeholder string."""
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.execute.return_value = MagicMock(data=[])
        result = build_review_summary(mock_client)
        assert result == "暂无复盘数据"

    def test_summary_with_data(self):
        """Summary includes rates and game count."""
        rows = [
            {"spread_hit": True, "ou_hit": False, "reviewed_at": "2026-02-20T10:00:00Z"},
            {"spread_hit": False, "ou_hit": True, "reviewed_at": "2026-02-21T10:00:00Z"},
        ]
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.execute.return_value = MagicMock(data=rows)
        result = build_review_summary(mock_client)
        assert "复盘报告" in result
        assert "50.0%" in result
        assert "复盘场次：2" in result

    def test_summary_last30_section(self):
        """Recent rows appear in 30-day rolling section."""
        rows = [
            {"spread_hit": True, "ou_hit": True, "reviewed_at": "2026-02-20T10:00:00"},
        ]
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.execute.return_value = MagicMock(data=rows)
        result = build_review_summary(mock_client)
        assert "近30天滚动表现" in result
        assert "样本数：1" in result
