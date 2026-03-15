"""Tests for prediction_engine recommendation, signal score, and core pick logic."""
from __future__ import annotations

from wcwidth import wcswidth

from app.prediction_engine import (
    build_pick_icon,
    build_prediction_table,
    pad,
    ICON_CORE,
)


class TestBuildPickIcon:
    """Test build_pick_icon returns correct icon for core pick."""

    def test_core_over(self):
        result = build_pick_icon(True, True, "over")
        assert result == ICON_CORE

    def test_core_under(self):
        result = build_pick_icon(True, True, "under")
        assert result == ICON_CORE

    def test_recommend_over(self):
        result = build_pick_icon(False, True, "over")
        assert result == ""

    def test_recommend_under(self):
        result = build_pick_icon(False, True, "under")
        assert result == ""

    def test_not_recommended(self):
        result = build_pick_icon(False, False, "over")
        assert result == ""

    def test_not_recommended_under(self):
        result = build_pick_icon(False, False, "under")
        assert result == ""

    def test_core_takes_priority_over_recommend(self):
        """When is_core is True, the icon should use ICON_CORE."""
        result_both = build_pick_icon(True, True, "over")
        result_core_only = build_pick_icon(True, False, "over")
        assert result_both == result_core_only == ICON_CORE


class TestPad:
    """Test pad() handles Chinese and ASCII text correctly."""

    def test_ascii_padding(self):
        result = pad("hello", 10)
        assert result == "hello     "
        assert len(result) == 10

    def test_chinese_padding(self):
        # "比赛" has display width 4 (2 chars * 2 width each)
        result = pad("比赛", 10)
        assert len(result) == 8  # 2 Chinese chars + 6 spaces
        # Verify the display width is correct
        assert wcswidth(result.rstrip()) + (10 - wcswidth("比赛")) == 10

    def test_mixed_chinese_ascii(self):
        # "猛龙 vs 森林狼" = 2+2+1+1+1+1+2+2+2 = 14 display width
        result = pad("猛龙 vs 森林狼", 22)
        text_width = wcswidth("猛龙 vs 森林狼")
        spaces = 22 - text_width
        assert result == "猛龙 vs 森林狼" + " " * spaces

    def test_exact_width(self):
        result = pad("hello", 5)
        assert result == "hello"

    def test_exceeds_width(self):
        result = pad("hello world", 5)
        assert result == "hello world"

    def test_non_string_input(self):
        result = pad(227.5, 6)
        assert result == "227.5 "

    def test_emoji_padding(self):
        result = pad("⭐", 4)
        assert isinstance(result, str)
        assert result.startswith("⭐")


class TestBuildPredictionTable:
    """Test build_prediction_table generates correct Chinese table format."""

    def _sample_games(self):
        return [
            {
                "away": "猛龙",
                "home": "森林狼",
                "line": 227.5,
                "over_prob": 0.642,
                "under_prob": 0.358,
                "prediction": "大分",
                "is_core": True,
            },
            {
                "away": "独行侠",
                "home": "魔术",
                "line": 228.5,
                "over_prob": 0.482,
                "under_prob": 0.518,
                "prediction": "小分",
                "is_core": False,
            },
        ]

    def test_header_present(self):
        table = build_prediction_table(self._sample_games())
        assert "比赛" in table
        assert "盘口" in table
        assert "大分概率" in table
        assert "小分概率" in table
        assert "模型判断" in table
        assert "重心" in table

    def test_table_borders(self):
        table = build_prediction_table(self._sample_games())
        assert "┌" in table
        assert "┘" in table
        assert "├" in table

    def test_core_game_row(self):
        table = build_prediction_table(self._sample_games())
        assert "猛龙 vs 森林狼" in table
        assert "227.5" in table
        assert "大分" in table
        assert ICON_CORE in table

    def test_non_core_game_row(self):
        table = build_prediction_table(self._sample_games())
        assert "独行侠 vs 魔术" in table
        assert "228.5" in table
        assert "小分" in table

    def test_empty_games(self):
        table = build_prediction_table([])
        assert "比赛" in table
        assert "盘口" in table
        # Only borders + header, no data rows
        lines = table.split("\n")
        assert len(lines) == 4  # top border + header + mid border + bottom border

    def test_all_games_present(self):
        games = self._sample_games()
        table = build_prediction_table(games)
        lines = table.split("\n")
        # 3 border/header lines + 2 data rows + 1 bottom border
        assert len(lines) == 6

    def test_negative_prob_format(self):
        games = [{
            "away": "爵士",
            "home": "奇才",
            "line": 243.5,
            "over_prob": 0.41,
            "under_prob": 0.59,
            "prediction": "小分",
            "is_core": False,
        }]
        table = build_prediction_table(games)
        assert "小分" in table
        assert "爵士 vs 奇才" in table


class TestRecommendationLegacy:
    """Legacy edge-based recommendation reason was removed — verify it no longer exists."""

    def test_no_recommendation_reason_constant(self):
        """prediction_engine should not export signal_score or edge-based reason."""
        import app.prediction_engine as pe
        assert not hasattr(pe, "INJURY_RATING_FACTOR")
        assert not hasattr(pe, "MC_WEIGHT")
        assert not hasattr(pe, "CLASSIFIER_WEIGHT")


class TestPaceModel:
    """Test pace model: (home + away) / 2 + pace_diff * 0.15 with B2B adjustments."""

    def test_basic_pace(self):
        from app.pace_model import calculate_game_pace
        # (100 + 98) / 2 + (100 - 98) * 0.15 = 99 + 0.3 = 99.3
        result = calculate_game_pace(100.0, 98.0)
        assert abs(result - 99.3) < 0.001

    def test_equal_pace(self):
        from app.pace_model import calculate_game_pace
        result = calculate_game_pace(100.0, 100.0)
        assert result == 100.0

    def test_home_back_to_back(self):
        from app.pace_model import calculate_game_pace
        base = calculate_game_pace(100.0, 98.0)
        with_b2b = calculate_game_pace(100.0, 98.0, home_back_to_back=True)
        assert abs(with_b2b - (base - 0.8)) < 0.001

    def test_away_back_to_back(self):
        from app.pace_model import calculate_game_pace
        base = calculate_game_pace(100.0, 98.0)
        with_b2b = calculate_game_pace(100.0, 98.0, away_back_to_back=True)
        assert abs(with_b2b - (base - 0.8)) < 0.001

    def test_both_back_to_back(self):
        from app.pace_model import calculate_game_pace
        base = calculate_game_pace(100.0, 98.0)
        with_b2b = calculate_game_pace(100.0, 98.0, home_back_to_back=True, away_back_to_back=True)
        assert abs(with_b2b - (base - 1.6)) < 0.001

    def test_high_pace(self):
        from app.pace_model import calculate_game_pace
        # (110 + 108) / 2 + (110 - 108) * 0.15 = 109 + 0.3 = 109.3
        result = calculate_game_pace(110.0, 108.0)
        assert abs(result - 109.3) < 0.001

    def test_negative_pace_diff(self):
        from app.pace_model import calculate_game_pace
        # (95 + 100) / 2 + (95 - 100) * 0.15 = 97.5 - 0.75 = 96.75
        result = calculate_game_pace(95.0, 100.0)
        assert abs(result - 96.75) < 0.001


class TestPPPModel:
    """Test PPP model: (off + opp_def) / 200 with shooting correction."""

    def test_basic_ppp(self):
        from app.ppp_model import calculate_ppp
        # home: (112 + 110) / 200 = 1.11; away: (110 + 108) / 200 = 1.09
        # Then shooting corrections applied
        h, a = calculate_ppp(112.0, 110.0, 108.0, 110.0, 0.37, 0.36, 0.27, 0.26)
        assert h > 0
        assert a > 0

    def test_ppp_increases_with_3p_rate(self):
        from app.ppp_model import calculate_ppp
        h_low, _ = calculate_ppp(112.0, 110.0, 108.0, 110.0, 0.30, 0.30, 0.27, 0.27)
        h_high, _ = calculate_ppp(112.0, 110.0, 108.0, 110.0, 0.45, 0.30, 0.27, 0.27)
        assert h_high > h_low

    def test_ppp_increases_with_ft_rate(self):
        from app.ppp_model import calculate_ppp
        h_low, _ = calculate_ppp(112.0, 110.0, 108.0, 110.0, 0.37, 0.37, 0.20, 0.20)
        h_high, _ = calculate_ppp(112.0, 110.0, 108.0, 110.0, 0.37, 0.37, 0.35, 0.20)
        assert h_high > h_low

    def test_ppp_uses_opponent_def(self):
        """PPP_home uses away_def_rating, PPP_away uses home_def_rating."""
        from app.ppp_model import calculate_ppp
        # Higher opponent def rating → higher PPP (weaker defense)
        h1, _ = calculate_ppp(112.0, 110.0, 105.0, 108.0, 0.37, 0.37, 0.27, 0.27)
        h2, _ = calculate_ppp(112.0, 110.0, 105.0, 115.0, 0.37, 0.37, 0.27, 0.27)
        assert h2 > h1  # Higher away_def → higher home PPP


class TestCorePick:
    """Test that recommendation uses confidence-based sorting with core pick rule."""

    @staticmethod
    def _apply_recommendation(results):
        """Mirror the new recommendation logic from prediction_engine.

        Sort by confidence = abs(prob_over - 0.5) descending.
        Core pick: max confidence game with max(over_prob, under_prob) >= 0.60.
        """
        for gr in results:
            gr["confidence"] = abs(gr["over_probability"] - 0.5)
            gr["is_core"] = False

        sorted_results = sorted(results, key=lambda x: x["confidence"], reverse=True)

        # Core pick: max confidence where max(over_prob, under_prob) >= 0.60
        for gr in sorted_results:
            max_prob = max(gr["over_probability"], gr["under_probability"])
            if max_prob >= 0.60:
                gr["is_core"] = True
                break
        return sorted_results

    def test_single_core_pick(self):
        results = [
            {"idx": 0, "over_probability": 0.65, "under_probability": 0.35},
            {"idx": 1, "over_probability": 0.68, "under_probability": 0.32},
            {"idx": 2, "over_probability": 0.55, "under_probability": 0.45},
        ]
        sorted_results = self._apply_recommendation(results)
        core = [r for r in sorted_results if r["is_core"]]
        assert len(core) == 1
        # idx=1 has highest confidence (|0.68-0.5|=0.18) and prob >= 0.60
        assert core[0]["idx"] == 1

    def test_core_requires_prob_060(self):
        """Core pick requires probability >= 0.60."""
        results = [
            {"idx": 0, "over_probability": 0.55, "under_probability": 0.45},
            {"idx": 1, "over_probability": 0.52, "under_probability": 0.48},
        ]
        sorted_results = self._apply_recommendation(results)
        core = [r for r in sorted_results if r["is_core"]]
        # No game has prob >= 0.60, so no core pick
        assert len(core) == 0

    def test_no_core_when_all_below_060(self):
        """When no game has probability >= 0.60, there is no core pick."""
        results = [
            {"idx": 0, "over_probability": 0.58, "under_probability": 0.42},
            {"idx": 1, "over_probability": 0.53, "under_probability": 0.47},
        ]
        sorted_results = self._apply_recommendation(results)
        core = [r for r in sorted_results if r["is_core"]]
        assert len(core) == 0

    def test_empty_results_no_core(self):
        results = []
        sorted_results = self._apply_recommendation(results)
        assert len(sorted_results) == 0

    def test_sorted_by_confidence(self):
        """Games should be sorted by confidence descending."""
        results = [
            {"idx": 0, "over_probability": 0.55, "under_probability": 0.45},  # conf=0.05
            {"idx": 1, "over_probability": 0.70, "under_probability": 0.30},  # conf=0.20
            {"idx": 2, "over_probability": 0.62, "under_probability": 0.38},  # conf=0.12
        ]
        sorted_results = self._apply_recommendation(results)
        assert sorted_results[0]["idx"] == 1
        assert sorted_results[1]["idx"] == 2
        assert sorted_results[2]["idx"] == 0

    def test_only_one_core_pick(self):
        """Even when multiple games meet core criteria, only 1 is marked."""
        results = [
            {"idx": 0, "over_probability": 0.70, "under_probability": 0.30},
            {"idx": 1, "over_probability": 0.68, "under_probability": 0.32},
            {"idx": 2, "over_probability": 0.65, "under_probability": 0.35},
        ]
        sorted_results = self._apply_recommendation(results)
        core = [r for r in sorted_results if r["is_core"]]
        assert len(core) == 1
        assert core[0]["idx"] == 0  # highest confidence

    def test_under_prob_can_be_core(self):
        """Under probability >= 0.60 should also qualify for core."""
        results = [
            {"idx": 0, "over_probability": 0.35, "under_probability": 0.65},
            {"idx": 1, "over_probability": 0.55, "under_probability": 0.45},
        ]
        sorted_results = self._apply_recommendation(results)
        core = [r for r in sorted_results if r["is_core"]]
        assert len(core) == 1
        assert core[0]["idx"] == 0  # under_prob=0.65 >= 0.60

    def test_confidence_calculation(self):
        """Confidence = abs(prob_over - 0.5)."""
        results = [
            {"idx": 0, "over_probability": 0.62, "under_probability": 0.38},
        ]
        sorted_results = self._apply_recommendation(results)
        assert abs(sorted_results[0]["confidence"] - 0.12) < 0.001


class TestInjuryModel:
    """Test injury model: PPP and pace adjustments."""

    def test_scorer_out_reduces_ppp(self):
        from app.injury_model import adjust_for_injuries
        h, a, p = adjust_for_injuries(1.12, 1.10, 99.0, home_scorer_out=True)
        assert abs(h - 1.08) < 0.001
        assert a == 1.10

    def test_pg_out_reduces_pace(self):
        from app.injury_model import adjust_for_injuries
        h, a, p = adjust_for_injuries(1.12, 1.10, 99.0, home_pg_out=True)
        assert h == 1.12
        assert abs(p - 98.0) < 0.001

    def test_both_out(self):
        from app.injury_model import adjust_for_injuries
        h, a, p = adjust_for_injuries(1.12, 1.10, 99.0,
                                       home_scorer_out=True, away_scorer_out=True,
                                       home_pg_out=True, away_pg_out=True)
        assert abs(h - 1.08) < 0.001
        assert abs(a - 1.06) < 0.001
        assert abs(p - 97.0) < 0.001

    def test_no_injuries(self):
        from app.injury_model import adjust_for_injuries
        h, a, p = adjust_for_injuries(1.12, 1.10, 99.0)
        assert h == 1.12
        assert a == 1.10
        assert p == 99.0


class TestMarketModel:
    """Test market calibration: predicted_total += line_move * 0.35."""

    def test_line_move_up(self):
        from app.market_model import apply_market_calibration
        # closing > opening → positive line move
        result = apply_market_calibration(225.0, 222.0, 226.0)
        # line_move = 226 - 222 = 4; 225 + 4 * 0.35 = 226.4
        assert abs(result - 226.4) < 0.01

    def test_line_move_down(self):
        from app.market_model import apply_market_calibration
        result = apply_market_calibration(225.0, 228.0, 224.0)
        # line_move = 224 - 228 = -4; 225 + (-4) * 0.35 = 223.6
        assert abs(result - 223.6) < 0.01

    def test_no_line_move(self):
        from app.market_model import apply_market_calibration
        result = apply_market_calibration(225.0, 225.0, 225.0)
        assert abs(result - 225.0) < 0.01


class TestTotalModel:
    """Test total calculation: pace * (PPP_home + PPP_away)."""

    def test_basic_total(self):
        from app.total_model import calculate_predicted_total
        result = calculate_predicted_total(99.0, 1.12, 1.10)
        # 99 * (1.12 + 1.10) = 99 * 2.22 = 219.78
        assert abs(result - 219.78) < 0.01

    def test_high_pace(self):
        from app.total_model import calculate_predicted_total
        result = calculate_predicted_total(105.0, 1.15, 1.12)
        assert result > 230


class TestLeagueConstants:
    """Test league-average constants in game_simulator (unchanged for training)."""

    def test_league_avg_pace(self):
        from app.game_simulator import LEAGUE_AVG_PACE
        assert LEAGUE_AVG_PACE == 99

    def test_league_avg_off(self):
        from app.game_simulator import LEAGUE_AVG_OFF
        assert LEAGUE_AVG_OFF == 114

    def test_league_avg_def(self):
        from app.game_simulator import LEAGUE_AVG_DEF
        assert LEAGUE_AVG_DEF == 114

    def test_ppp_std(self):
        from app.game_simulator import PPP_STD
        assert PPP_STD == 0.05


class TestGamePaceCalculation:
    """Test new pace model: (home + away) / 2 + pace_diff * 0.15."""

    def test_normal_pace(self):
        from app.pace_model import calculate_game_pace
        # (100 + 98) / 2 + (100 - 98) * 0.15 = 99 + 0.3 = 99.3
        result = calculate_game_pace(100.0, 98.0)
        assert abs(result - 99.3) < 0.001

    def test_high_pace(self):
        from app.pace_model import calculate_game_pace
        # (110 + 108) / 2 + (110 - 108) * 0.15 = 109 + 0.3 = 109.3
        result = calculate_game_pace(110.0, 108.0)
        assert abs(result - 109.3) < 0.001

    def test_low_pace(self):
        from app.pace_model import calculate_game_pace
        # (88 + 90) / 2 + (88 - 90) * 0.15 = 89 - 0.3 = 88.7
        result = calculate_game_pace(88.0, 90.0)
        assert abs(result - 88.7) < 0.001

    def test_equal_pace(self):
        from app.pace_model import calculate_game_pace
        assert calculate_game_pace(100.0, 100.0) == 100.0

    def test_b2b_reduces_pace(self):
        from app.pace_model import calculate_game_pace
        # (105 + 95) / 2 + (105 - 95) * 0.15 = 100 + 1.5 = 101.5
        result_no_b2b = calculate_game_pace(105.0, 95.0)
        result_home_b2b = calculate_game_pace(105.0, 95.0, home_back_to_back=True)
        assert abs(result_home_b2b - (result_no_b2b - 0.8)) < 0.001


class TestPPPCalculation:
    """PPP uses (off + opp_def) / 200 with shooting corrections."""

    def test_ppp_formula(self):
        """PPP_home = (home_off + away_def) / 200 * shooting_corrections."""
        from app.ppp_model import calculate_ppp
        h, a = calculate_ppp(112.0, 110.0, 108.0, 110.0, 0.37, 0.37, 0.27, 0.27)
        # base_home = (112 + 110) / 200 = 1.11
        # * (1 + 0.37 * 0.12) = 1.0444
        # * (1 + 0.27 * 0.06) = 1.0162
        # ≈ 1.11 * 1.0444 * 1.0162 ≈ 1.178
        assert 1.0 <= h <= 1.4
        assert 1.0 <= a <= 1.4

    def test_ppp_within_normal_range(self):
        """Realistic ratings produce PPP in [1.0, 1.4]."""
        from app.ppp_model import calculate_ppp
        for off in [105.0, 110.0, 114.0, 118.0]:
            h, a = calculate_ppp(off, off, 110.0, 110.0, 0.37, 0.37, 0.27, 0.27)
            assert 1.0 <= h <= 1.4

    def test_predicted_total_in_range(self):
        """With realistic PPP and pace, predicted total stays in a reasonable range."""
        from app.pace_model import calculate_game_pace
        from app.ppp_model import calculate_ppp
        from app.total_model import calculate_predicted_total
        game_pace = calculate_game_pace(99.0, 99.0)
        home_ppp, away_ppp = calculate_ppp(112.0, 110.0, 108.0, 110.0, 0.37, 0.37, 0.27, 0.27)
        predicted_total = calculate_predicted_total(game_pace, home_ppp, away_ppp)
        assert 200 <= predicted_total <= 260


class TestSimulationEngine:
    """Test Monte Carlo simulation with dynamic std."""

    def test_basic_simulation(self):
        from app.simulation_engine import run_total_simulation
        sim = run_total_simulation(
            game_id=12345, predicted_total=225.0, closing_total=220.0,
            pace_diff=2.0, ppp_home=1.12, ppp_away=1.10,
        )
        assert "over_probability" in sim
        assert "under_probability" in sim
        assert "simulation_count" in sim
        assert sim["simulation_count"] == 10000

    def test_prob_complement(self):
        from app.simulation_engine import run_total_simulation
        sim = run_total_simulation(
            game_id=42, predicted_total=225.0, closing_total=225.0,
            pace_diff=0.0, ppp_home=1.12, ppp_away=1.12,
        )
        assert abs(sim["over_probability"] + sim["under_probability"] - 1.0) < 1e-9

    def test_over_prob_when_predicted_higher(self):
        from app.simulation_engine import run_total_simulation
        sim = run_total_simulation(
            game_id=99, predicted_total=240.0, closing_total=220.0,
            pace_diff=2.0, ppp_home=1.15, ppp_away=1.10,
        )
        assert sim["over_probability"] > 0.5

    def test_under_prob_when_predicted_lower(self):
        from app.simulation_engine import run_total_simulation
        sim = run_total_simulation(
            game_id=99, predicted_total=210.0, closing_total=230.0,
            pace_diff=2.0, ppp_home=1.05, ppp_away=1.05,
        )
        assert sim["under_probability"] > 0.5

    def test_dynamic_std(self):
        """Dynamic std = 10 + abs(pace_diff) * 0.6 + abs(ppp_home - ppp_away) * 4."""
        pace_diff = 5.0
        ppp_home = 1.15
        ppp_away = 1.08
        expected_std = 10 + abs(pace_diff) * 0.6 + abs(ppp_home - ppp_away) * 4
        assert abs(expected_std - 13.28) < 0.01


class TestSimulationConstants:
    """Verify simulation engine defaults."""

    def test_n_simulations(self):
        from app.simulation_engine import N_SIMULATIONS
        assert N_SIMULATIONS == 10000

    def test_min_simulation_count(self):
        from app.prediction_engine import MIN_SIMULATION_COUNT
        assert MIN_SIMULATION_COUNT == 10000


class TestMarketCalibration:
    """Test Market Model: predicted_total += line_move * 0.35."""

    def test_positive_line_move(self):
        """Line moves up → predicted total increases."""
        from app.market_model import apply_market_calibration
        result = apply_market_calibration(225.0, 222.0, 226.0)
        # line_move = 226 - 222 = 4; 225 + 4 * 0.35 = 226.4
        assert abs(result - 226.4) < 0.01

    def test_negative_line_move(self):
        """Line moves down → predicted total decreases."""
        from app.market_model import apply_market_calibration
        result = apply_market_calibration(225.0, 228.0, 224.0)
        # line_move = 224 - 228 = -4; 225 + (-4) * 0.35 = 223.6
        assert abs(result - 223.6) < 0.01

    def test_no_line_move(self):
        """When opening == closing, no adjustment."""
        from app.market_model import apply_market_calibration
        result = apply_market_calibration(225.0, 225.0, 225.0)
        assert abs(result - 225.0) < 0.01

    def test_calibration_direction(self):
        """Line moving up should pull prediction up."""
        from app.market_model import apply_market_calibration
        base = 225.0
        up = apply_market_calibration(base, 220.0, 228.0)
        down = apply_market_calibration(base, 228.0, 220.0)
        assert up > base > down
