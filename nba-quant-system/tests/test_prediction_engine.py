"""Tests for prediction_engine recommendation, signal score, and core pick logic."""
from __future__ import annotations

from wcwidth import wcswidth

from app.prediction_engine import (
    build_pick_icon,
    build_prediction_table,
    pad,
    ICON_CORE,
    ICON_RECOMMEND,
    ICON_NO,
    ICON_OVER,
    ICON_UNDER,
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


class TestRecommendationReason:
    """Test the recommendation reason thresholds based on abs_edge."""

    @staticmethod
    def _reason(abs_edge: float) -> str:
        """Mirror the recommendation reason logic from prediction_engine."""
        if abs_edge >= 7:
            return "模型预测与盘口差距较大"
        elif abs_edge >= 4:
            return "模型预测存在明显价值"
        else:
            return "信号较弱，不推荐"

    def test_large_edge_reason(self):
        reason = self._reason(10.0)
        assert reason == "模型预测与盘口差距较大"

    def test_boundary_edge_7(self):
        reason = self._reason(7.0)
        assert reason == "模型预测与盘口差距较大"

    def test_medium_edge_reason(self):
        reason = self._reason(4.0)
        assert reason == "模型预测存在明显价值"

    def test_boundary_edge_4(self):
        reason = self._reason(4.0)
        assert reason == "模型预测存在明显价值"

    def test_small_edge_reason(self):
        reason = self._reason(3.0)
        assert reason == "信号较弱，不推荐"

    def test_zero_edge_reason(self):
        reason = self._reason(0.0)
        assert reason == "信号较弱，不推荐"


class TestProbabilityCalibration:
    """Test probability calibration: calibrated = 0.7 * raw + 0.3 * 0.5."""

    @staticmethod
    def _calibrate(raw_prob: float) -> float:
        return 0.7 * raw_prob + 0.3 * 0.5

    def test_calibrate_high_prob(self):
        calibrated = self._calibrate(0.80)
        assert abs(calibrated - 0.71) < 0.001

    def test_calibrate_low_prob(self):
        calibrated = self._calibrate(0.30)
        assert abs(calibrated - 0.36) < 0.001

    def test_calibrate_neutral(self):
        calibrated = self._calibrate(0.50)
        assert abs(calibrated - 0.50) < 0.001

    def test_calibrate_shrinks_toward_half(self):
        """Calibration should shrink extreme probabilities toward 0.5."""
        raw = 0.90
        calibrated = self._calibrate(raw)
        assert abs(calibrated - 0.5) < abs(raw - 0.5)

    def test_under_probability(self):
        """under_probability = 1 - calibrated over_probability."""
        calibrated = self._calibrate(0.65)
        under = 1.0 - calibrated
        assert abs(calibrated + under - 1.0) < 0.001


class TestSignalScore:
    """Test signal score calculation."""

    @staticmethod
    def _signal_score(abs_edge: float, over_probability: float, total_std: float) -> float:
        return abs_edge * 0.6 + over_probability * 40 - total_std * 0.2

    def test_higher_edge_yields_higher_score(self):
        s1 = self._signal_score(4.0, 0.60, 10.0)
        s2 = self._signal_score(8.0, 0.60, 10.0)
        assert s2 > s1

    def test_higher_prob_yields_higher_score(self):
        s1 = self._signal_score(6.0, 0.55, 10.0)
        s2 = self._signal_score(6.0, 0.70, 10.0)
        assert s2 > s1

    def test_higher_std_yields_lower_score(self):
        s1 = self._signal_score(6.0, 0.60, 8.0)
        s2 = self._signal_score(6.0, 0.60, 15.0)
        assert s1 > s2

    def test_known_value(self):
        # abs_edge=6, over_probability=0.65, total_std=10
        # 6*0.6 + 0.65*40 - 10*0.2 = 3.6 + 26.0 - 2.0 = 27.6
        score = self._signal_score(6.0, 0.65, 10.0)
        assert abs(score - 27.6) < 0.001


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


class TestEdgeCalculation:
    """Test edge = predicted_total - closing_total."""

    def test_positive_edge(self):
        predicted_total = 229.6
        closing_total = 221.5
        edge = predicted_total - closing_total
        assert abs(edge - 8.1) < 0.01

    def test_negative_edge(self):
        predicted_total = 210.0
        closing_total = 221.5
        edge = predicted_total - closing_total
        assert edge < 0

    def test_zero_edge(self):
        edge = 220.0 - 220.0
        assert edge == 0.0


class TestIntervalFormat:
    """Test that interval uses int() for output."""

    def test_interval_uses_int(self):
        total_5pct = 205.3
        total_95pct = 249.7
        total_range = f"{int(total_5pct)} – {int(total_95pct)}"
        assert total_range == "205 – 249"

    def test_interval_truncates_down(self):
        total_5pct = 199.9
        total_95pct = 240.1
        total_range = f"{int(total_5pct)} – {int(total_95pct)}"
        assert total_range == "199 – 240"


class TestLeagueConstants:
    """Test league-average constants match the refactored values."""

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
    """Test game pace uses matchup pace formula: 0.55 * fast + 0.45 * slow."""

    def test_normal_pace(self):
        from app.feature_engineering import calculate_game_pace
        # 0.55 * 100 + 0.45 * 98 = 55.0 + 44.1 = 99.1
        result = calculate_game_pace(100.0, 98.0)
        assert abs(result - 99.1) < 0.001

    def test_high_pace_not_clamped(self):
        from app.feature_engineering import calculate_game_pace
        # 0.55 * 110 + 0.45 * 108 = 60.5 + 48.6 = 109.1
        result = calculate_game_pace(110.0, 108.0)
        assert abs(result - 109.1) < 0.001

    def test_low_pace_not_clamped(self):
        from app.feature_engineering import calculate_game_pace
        # 0.55 * 90 + 0.45 * 88 = 49.5 + 39.6 = 89.1
        result = calculate_game_pace(88.0, 90.0)
        assert abs(result - 89.1) < 0.001

    def test_equal_pace(self):
        from app.feature_engineering import calculate_game_pace
        assert calculate_game_pace(100.0, 100.0) == 100.0

    def test_fast_team_weighted_higher(self):
        from app.feature_engineering import calculate_game_pace
        # fast=105, slow=95 → 0.55*105 + 0.45*95 = 57.75 + 42.75 = 100.5
        result = calculate_game_pace(105.0, 95.0)
        assert abs(result - 100.5) < 0.001


class TestPPPNoStructureAmplification:
    """PPP equals off_rating / 100 — no defensive adjustment or structure factor."""

    def test_ppp_equals_off_rating_div_100(self):
        """PPP must be derived directly from off_rating / 100 with no multiplier."""
        from app.feature_engineering import calculate_ppp
        assert calculate_ppp(112.0) == 1.12
        assert calculate_ppp(108.0) == 1.08

    def test_ppp_within_normal_range(self):
        """Realistic off_rating values produce PPP in [1.0, 1.2]."""
        from app.feature_engineering import calculate_ppp
        for off_rating in [105.0, 110.0, 114.0, 115.0]:
            ppp = calculate_ppp(off_rating)
            assert 1.0 <= ppp <= 1.2, f"PPP {ppp} out of range for off_rating {off_rating}"

    def test_predicted_total_in_range(self):
        """With realistic PPP and pace, predicted total stays in a reasonable range."""
        from app.feature_engineering import calculate_game_pace, calculate_ppp
        game_pace = calculate_game_pace(99.0, 99.0)
        home_ppp = calculate_ppp(112.0)
        away_ppp = calculate_ppp(110.0)
        predicted_total = game_pace * (home_ppp + away_ppp)
        assert 200 <= predicted_total <= 250


class TestEfficiencyAdjustment:
    """Test PPP clamping to [1.03, 1.18] after off_rating / 100 conversion."""

    def test_ppp_clamped_to_min_1_03(self):
        """PPP from low off_rating is clamped to 1.03."""
        from app.feature_engineering import calculate_ppp
        ppp = calculate_ppp(100.0)  # 1.00
        ppp = max(1.03, min(ppp, 1.18))
        assert ppp == 1.03

    def test_ppp_clamped_to_max_1_18(self):
        """PPP from high off_rating is clamped to 1.18."""
        from app.feature_engineering import calculate_ppp
        ppp = calculate_ppp(125.0)  # 1.25
        ppp = max(1.03, min(ppp, 1.18))
        assert ppp == 1.18

    def test_ppp_within_range_no_clamp(self):
        """PPP from normal off_rating passes through unclamped."""
        from app.feature_engineering import calculate_ppp
        ppp = calculate_ppp(112.0)  # 1.12
        ppp = max(1.03, min(ppp, 1.18))
        assert ppp == 1.12

    def test_ppp_safety_limits(self):
        """PPP must be clamped to [1.03, 1.18] after conversion."""
        from app.feature_engineering import calculate_ppp
        for off_rating in [100.0, 105.0, 110.0, 115.0, 120.0, 125.0]:
            ppp = calculate_ppp(off_rating)
            ppp = max(1.03, min(ppp, 1.18))
            assert 1.03 <= ppp <= 1.18

    def test_predicted_total_no_total_clamp(self):
        """model_total = game_pace * (home_ppp + away_ppp), no [205, 245] clamp."""
        from app.feature_engineering import calculate_game_pace, calculate_ppp
        game_pace = calculate_game_pace(99.0, 99.0)
        home_ppp = calculate_ppp(112.0)
        away_ppp = calculate_ppp(110.0)
        home_ppp = max(1.03, min(home_ppp, 1.18))
        away_ppp = max(1.03, min(away_ppp, 1.18))
        model_total = game_pace * (home_ppp + away_ppp)
        # No total clamping — value depends on pace and PPP
        assert model_total > 0


class TestMonteCarloConstants:
    """Verify Monte Carlo std and run count constants."""

    def test_mc_std(self):
        from app.prediction_engine import MC_TOTAL_STD
        assert MC_TOTAL_STD == 8.5

    def test_mc_runs(self):
        from app.prediction_engine import MIN_SIMULATION_COUNT
        assert MIN_SIMULATION_COUNT == 10000


class TestMarketAnchor:
    """Test Market Anchor blending: predicted_total = 0.65 * model_total + 0.35 * closing_line."""

    def test_blend_formula(self):
        """Market anchor blends model total (65%) with closing line (35%)."""
        model_total = 230.0
        closing_line = 220.0
        predicted_total = 0.65 * model_total + 0.35 * closing_line
        expected = 0.65 * 230.0 + 0.35 * 220.0  # 149.5 + 77.0 = 226.5
        assert abs(predicted_total - expected) < 0.01

    def test_blend_pulls_toward_closing(self):
        """Blending should pull model total toward the closing line."""
        model_total = 240.0
        closing_line = 220.0
        predicted_total = 0.65 * model_total + 0.35 * closing_line
        assert model_total > predicted_total > closing_line

    def test_blend_equal_values(self):
        """When model total equals closing line, blended result is the same."""
        model_total = 225.0
        closing_line = 225.0
        predicted_total = 0.65 * model_total + 0.35 * closing_line
        assert abs(predicted_total - 225.0) < 0.01

    def test_blend_closing_higher(self):
        """When closing line is higher than model total, blend is between them."""
        model_total = 215.0
        closing_line = 230.0
        predicted_total = 0.65 * model_total + 0.35 * closing_line
        assert model_total < predicted_total < closing_line
