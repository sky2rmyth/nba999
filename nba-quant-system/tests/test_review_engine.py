"""Tests for review_engine hit-checking and rate calculation functions."""
from __future__ import annotations

from app.review_engine import calculate_rates, spread_hit, total_hit


# --- spread_hit ---

class TestSpreadHit:
    def test_home_covers(self):
        """Home team wins by more than the spread."""
        row = {
            "final_home_score": 110,
            "final_visitor_score": 100,
            "spread": 5,
            "recommended_side": "home",
        }
        assert spread_hit(row) is True

    def test_home_does_not_cover(self):
        """Home team wins but does not cover the spread."""
        row = {
            "final_home_score": 103,
            "final_visitor_score": 100,
            "spread": 5,
            "recommended_side": "home",
        }
        assert spread_hit(row) is False

    def test_away_covers(self):
        """Away side covers when margin is less than spread."""
        row = {
            "final_home_score": 100,
            "final_visitor_score": 105,
            "spread": -3,
            "recommended_side": "away",
        }
        assert spread_hit(row) is True

    def test_away_does_not_cover(self):
        """Away side does not cover when margin exceeds spread."""
        row = {
            "final_home_score": 110,
            "final_visitor_score": 100,
            "spread": -3,
            "recommended_side": "away",
        }
        assert spread_hit(row) is False

    def test_unknown_side_returns_false(self):
        """Unknown recommended_side returns False."""
        row = {
            "final_home_score": 110,
            "final_visitor_score": 100,
            "spread": 0,
            "recommended_side": "unknown",
        }
        assert spread_hit(row) is False

    def test_exact_margin_equals_spread_home(self):
        """Margin exactly equals spread — not a hit for home."""
        row = {
            "final_home_score": 105,
            "final_visitor_score": 100,
            "spread": 5,
            "recommended_side": "home",
        }
        assert spread_hit(row) is False


# --- total_hit ---

class TestTotalHit:
    def test_over_hits(self):
        """Actual total exceeds the line."""
        row = {
            "final_home_score": 115,
            "final_visitor_score": 110,
            "recommended_total": "over",
            "total_line": 220,
        }
        assert total_hit(row) is True

    def test_over_misses(self):
        """Actual total is under the line."""
        row = {
            "final_home_score": 100,
            "final_visitor_score": 105,
            "recommended_total": "over",
            "total_line": 220,
        }
        assert total_hit(row) is False

    def test_under_hits(self):
        """Actual total is below the line."""
        row = {
            "final_home_score": 100,
            "final_visitor_score": 105,
            "recommended_total": "under",
            "total_line": 220,
        }
        assert total_hit(row) is True

    def test_under_misses(self):
        """Actual total exceeds the line."""
        row = {
            "final_home_score": 115,
            "final_visitor_score": 110,
            "recommended_total": "under",
            "total_line": 220,
        }
        assert total_hit(row) is False

    def test_unknown_total_returns_false(self):
        """Unknown recommended_total returns False."""
        row = {
            "final_home_score": 110,
            "final_visitor_score": 100,
            "recommended_total": "",
            "total_line": 220,
        }
        assert total_hit(row) is False

    def test_exact_total_equals_line_over(self):
        """Total exactly equals line — not a hit for over."""
        row = {
            "final_home_score": 110,
            "final_visitor_score": 110,
            "recommended_total": "over",
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
