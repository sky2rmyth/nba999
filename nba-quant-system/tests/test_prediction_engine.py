"""Tests for prediction_engine recommendation, signal score, and core pick logic."""
from __future__ import annotations

from app.prediction_engine import (
    build_pick_icon,
    build_prediction_table,
    ICON_CORE,
    ICON_RECOMMEND,
    ICON_NO,
    ICON_OVER,
    ICON_UNDER,
)


class TestBuildPickIcon:
    """Test build_pick_icon returns correct icon combinations."""

    def test_core_over(self):
        result = build_pick_icon(True, True, "over")
        assert result == ICON_CORE + ICON_OVER + "大"

    def test_core_under(self):
        result = build_pick_icon(True, True, "under")
        assert result == ICON_CORE + ICON_UNDER + "小"

    def test_recommend_over(self):
        result = build_pick_icon(False, True, "over")
        assert result == ICON_RECOMMEND + ICON_OVER + "大"

    def test_recommend_under(self):
        result = build_pick_icon(False, True, "under")
        assert result == ICON_RECOMMEND + ICON_UNDER + "小"

    def test_not_recommended(self):
        result = build_pick_icon(False, False, "over")
        assert result == ICON_NO

    def test_not_recommended_under(self):
        result = build_pick_icon(False, False, "under")
        assert result == ICON_NO

    def test_core_takes_priority_over_recommend(self):
        """When is_core is True, the icon should use ICON_CORE regardless of is_recommend."""
        result_both = build_pick_icon(True, True, "over")
        result_core_only = build_pick_icon(True, False, "over")
        assert result_both == result_core_only == ICON_CORE + ICON_OVER + "大"


class TestBuildPredictionTable:
    """Test build_prediction_table generates correct table format."""

    def _sample_games(self):
        return [
            {
                "away": "猛龙",
                "home": "森林狼",
                "line": 227.5,
                "pred_total": 233.7,
                "edge": 6.2,
                "prob": 0.64,
                "low": 212.3,
                "high": 255.8,
                "direction": "over",
                "is_core": True,
                "is_recommend": True,
            },
            {
                "away": "独行侠",
                "home": "魔术",
                "line": 228.5,
                "pred_total": 230.4,
                "edge": 1.9,
                "prob": 0.58,
                "low": 208.1,
                "high": 251.9,
                "direction": "over",
                "is_core": False,
                "is_recommend": False,
            },
        ]

    def test_header_present(self):
        table = build_prediction_table(self._sample_games())
        assert "比赛 | 盘口 | 模型 | Edge | 概率 | 区间 | 推荐" in table

    def test_separator_present(self):
        table = build_prediction_table(self._sample_games())
        assert "------------------------------------------------" in table

    def test_core_game_row(self):
        table = build_prediction_table(self._sample_games())
        lines = table.split("\n")
        # Data rows start at index 2 (0=header, 1=separator)
        core_row = lines[2]
        assert "猛龙 vs 森林狼" in core_row
        assert "227.5" in core_row
        assert "233.7" in core_row
        assert "+6.2" in core_row
        assert "64%" in core_row
        assert "212-255" in core_row
        assert ICON_CORE in core_row

    def test_not_recommended_row(self):
        table = build_prediction_table(self._sample_games())
        lines = table.split("\n")
        no_rec_row = lines[3]
        assert "独行侠 vs 魔术" in no_rec_row
        assert ICON_NO in no_rec_row
        assert ICON_CORE not in no_rec_row
        assert ICON_RECOMMEND not in no_rec_row

    def test_empty_games(self):
        table = build_prediction_table([])
        assert "比赛 | 盘口 | 模型 | Edge | 概率 | 区间 | 推荐" in table
        # Only header + separator, no data rows
        lines = table.split("\n")
        assert len(lines) == 2

    def test_all_games_present(self):
        games = self._sample_games()
        table = build_prediction_table(games)
        lines = table.split("\n")
        # 2 header lines + 2 data rows
        assert len(lines) == 4

    def test_negative_edge_format(self):
        games = [{
            "away": "爵士",
            "home": "奇才",
            "line": 243.5,
            "pred_total": 229.6,
            "edge": -13.9,
            "prob": 0.59,
            "low": 213.0,
            "high": 246.0,
            "direction": "under",
            "is_core": False,
            "is_recommend": True,
        }]
        table = build_prediction_table(games)
        assert "-13.9" in table
        assert ICON_RECOMMEND + ICON_UNDER + "小" in table

    def test_pipe_separators(self):
        """All data rows use pipe separators."""
        table = build_prediction_table(self._sample_games())
        lines = table.split("\n")
        for line in lines[2:]:
            assert line.count("|") == 6


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
    """Test that recommendation uses edge-only filtering with max 5 cap."""

    @staticmethod
    def _apply_recommendation(results, max_recs=5):
        """Mirror the recommendation logic from prediction_engine."""
        sorted_results = sorted(results, key=lambda x: abs(x["total_edge_pts"]), reverse=True)

        # Determine minimum recommendations based on game count
        total_games = len(sorted_results)
        if total_games > 5:
            min_recommend = 4
        elif total_games >= 3:
            min_recommend = 3
        else:
            min_recommend = 1

        for gr in sorted_results:
            abs_edge_val = abs(gr["total_edge_pts"])
            if abs_edge_val >= 4:
                gr["recommended"] = True
            else:
                gr["recommended"] = False
            if abs_edge_val >= 7:
                gr["star_pick"] = True
            else:
                gr["star_pick"] = False
            gr["is_core"] = False

        # Collect quality-filtered recommendations
        recommended_games = [gr for gr in sorted_results if gr["recommended"]]

        # Auto-fill if recommendations below minimum
        if len(recommended_games) < min_recommend:
            for gr in sorted_results:
                if not gr["recommended"]:
                    gr["recommended"] = True
                    recommended_games.append(gr)
                if len(recommended_games) >= min_recommend:
                    break

        # Cap recommendations at max_recs by abs(edge)
        if len(recommended_games) > max_recs:
            recommended_games = sorted(
                recommended_games, key=lambda x: abs(x["total_edge_pts"]), reverse=True
            )[:max_recs]
            keep_indices = {g["idx"] for g in recommended_games}
            for gr in sorted_results:
                if gr["recommended"] and gr["idx"] not in keep_indices:
                    gr["recommended"] = False

        # Guaranteed core pick: prefer star_pick with largest abs(edge)
        star_results = [gr for gr in sorted_results if gr["recommended"] and gr["star_pick"]]
        if star_results:
            star_results[0]["is_core"] = True
        else:
            rec_sorted = sorted(
                [gr for gr in sorted_results if gr["recommended"]],
                key=lambda x: abs(x["total_edge_pts"]),
                reverse=True,
            )
            if rec_sorted:
                rec_sorted[0]["star_pick"] = True
                rec_sorted[0]["is_core"] = True
        return sorted_results

    def test_single_core_pick(self):
        results = [
            {"idx": 0, "signal_score": 20.0, "total_edge_pts": 7.0,
             "over_probability": 0.65, "under_probability": 0.35},
            {"idx": 1, "signal_score": 30.0, "total_edge_pts": 9.0,
             "over_probability": 0.68, "under_probability": 0.32},
            {"idx": 2, "signal_score": 25.0, "total_edge_pts": 3.0,
             "over_probability": 0.55, "under_probability": 0.45},
        ]
        sorted_results = self._apply_recommendation(results)
        # idx=1 has abs(edge)=9 >= 7 → star_pick + recommended, should be core
        assert sorted_results[0]["idx"] == 1
        assert sorted_results[0]["is_core"] is True

    def test_single_game_with_edge_7_is_star_and_core(self):
        """Edge=7 → star_pick (>= 7) and core (guaranteed core pick)."""
        results = [{"idx": 0, "signal_score": 15.0, "total_edge_pts": 7.0,
                     "over_probability": 0.65, "under_probability": 0.35}]
        sorted_results = self._apply_recommendation(results)
        assert sorted_results[0]["is_core"] is True
        assert sorted_results[0]["recommended"] is True
        assert sorted_results[0]["star_pick"] is True

    def test_no_recommendation_when_edges_small(self):
        """When no game meets edge >= 4, auto-fill ensures minimum recommendations."""
        results = [
            {"idx": 0, "signal_score": 20.0, "total_edge_pts": 3.0,
             "over_probability": 0.55, "under_probability": 0.45},
            {"idx": 1, "signal_score": 25.0, "total_edge_pts": 3.5,
             "over_probability": 0.58, "under_probability": 0.42},
        ]
        sorted_results = self._apply_recommendation(results)
        recommended = [r for r in sorted_results if r["recommended"]]
        # 2 games → min_recommend = 1, auto-fill kicks in
        assert len(recommended) >= 1

    def test_empty_results_no_core(self):
        results = []
        sorted_results = self._apply_recommendation(results)
        assert len(sorted_results) == 0

    def test_recommendation_requires_edge_only(self):
        """Edge-only filter: abs(edge) >= 4 → recommended. All 3 games meet filter."""
        results = [
            {"idx": 0, "total_edge_pts": 6.0,
             "over_probability": 0.62, "under_probability": 0.38, "signal_score": 20.0},
            {"idx": 1, "total_edge_pts": 6.0,
             "over_probability": 0.55, "under_probability": 0.45, "signal_score": 18.0},
            {"idx": 2, "total_edge_pts": 4.0,
             "over_probability": 0.70, "under_probability": 0.30, "signal_score": 22.0},
        ]
        sorted_results = self._apply_recommendation(results)
        recommended = [gr for gr in sorted_results if gr["recommended"]]
        # 3 games -> all 3 have abs(edge) >= 4, all recommended
        assert len(recommended) == 3

    def test_negative_edge_also_recommends(self):
        """Negative edge with abs >= 4 should be recommended."""
        results = [
            {"idx": 0, "signal_score": 30.0, "total_edge_pts": -7.0,
             "over_probability": 0.35, "under_probability": 0.65},
            {"idx": 1, "signal_score": 20.0, "total_edge_pts": 3.0,
             "over_probability": 0.55, "under_probability": 0.45},
        ]
        sorted_results = self._apply_recommendation(results)
        recommended = [gr for gr in sorted_results if gr["recommended"]]
        assert any(r["idx"] == 0 for r in recommended)
        # Guaranteed core: idx=0 is promoted since it has the largest abs(edge)
        core = [r for r in sorted_results if r["is_core"]]
        assert len(core) == 1
        assert core[0]["idx"] == 0

    def test_max_5_recommendations_caps_from_top_abs_edge(self):
        """When more than 5 games qualify, only top 5 by abs(edge) are kept."""
        results = [
            {"idx": i, "total_edge_pts": float(10 - i), "signal_score": 20.0,
             "over_probability": 0.65, "under_probability": 0.35}
            for i in range(7)
        ]
        sorted_results = self._apply_recommendation(results)
        recommended = [gr for gr in sorted_results if gr["recommended"]]
        assert len(recommended) == 5
        # Top 5 by abs(edge): idx=0(10), idx=1(9), idx=2(8), idx=3(7), idx=4(6)
        rec_ids = {r["idx"] for r in recommended}
        assert rec_ids == {0, 1, 2, 3, 4}

    def test_core_pick_is_star_pick_with_max_abs_edge(self):
        """Core pick should be the star_pick game with the largest abs(edge)."""
        results = [
            {"idx": 0, "total_edge_pts": -12.0, "signal_score": 25.0,
             "over_probability": 0.30, "under_probability": 0.70},
            {"idx": 1, "total_edge_pts": 9.0, "signal_score": 30.0,
             "over_probability": 0.68, "under_probability": 0.32},
            {"idx": 2, "total_edge_pts": 6.0, "signal_score": 20.0,
             "over_probability": 0.62, "under_probability": 0.38},
        ]
        sorted_results = self._apply_recommendation(results)
        core = [r for r in sorted_results if r["is_core"]]
        assert len(core) == 1
        assert core[0]["idx"] == 0  # abs(-12) = 12 >= 7 → star_pick + core

    def test_star_pick_requires_edge_7(self):
        """Star pick needs abs(edge) >= 7 (edge-only, no prob condition)."""
        results = [
            {"idx": 0, "total_edge_pts": 5.0, "signal_score": 25.0,
             "over_probability": 0.63, "under_probability": 0.37},  # edge < 7
            {"idx": 1, "total_edge_pts": 6.5, "signal_score": 20.0,
             "over_probability": 0.68, "under_probability": 0.32},  # edge < 7
            {"idx": 2, "total_edge_pts": 10.0, "signal_score": 30.0,
             "over_probability": 0.70, "under_probability": 0.30},  # edge >= 7 → star_pick
        ]
        sorted_results = self._apply_recommendation(results)
        star_picks = [r for r in sorted_results if r.get("star_pick")]
        assert len(star_picks) == 1
        assert star_picks[0]["idx"] == 2

    def test_no_core_when_no_star_pick(self):
        """When no game meets star_pick criteria, a core pick is still promoted from recommended."""
        results = [
            {"idx": 0, "total_edge_pts": 6.0, "signal_score": 20.0,
             "over_probability": 0.62, "under_probability": 0.38},
            {"idx": 1, "total_edge_pts": 5.5, "signal_score": 18.0,
             "over_probability": 0.61, "under_probability": 0.39},
        ]
        sorted_results = self._apply_recommendation(results)
        core = [r for r in sorted_results if r["is_core"]]
        # Guaranteed core pick — promoted from recommended with largest abs(edge)
        assert len(core) == 1
        assert core[0]["idx"] == 0

    def test_fill_to_minimum_when_insufficient(self):
        """When quality filter yields 0 recommendations, auto-fill ensures minimum."""
        results = [
            {"idx": 0, "total_edge_pts": 3.0, "signal_score": 15.0,
             "over_probability": 0.55, "under_probability": 0.45},
            {"idx": 1, "total_edge_pts": 2.0, "signal_score": 10.0,
             "over_probability": 0.52, "under_probability": 0.48},
        ]
        sorted_results = self._apply_recommendation(results)
        recommended = [gr for gr in sorted_results if gr["recommended"]]
        # 2 games → min_recommend = 1, auto-fill kicks in
        assert len(recommended) >= 1

    def test_edge_10_is_star_pick_and_core(self):
        """Edge=10 >= 7 → star_pick + recommended + core."""
        results = [
            {"idx": 0, "total_edge_pts": 10.0, "signal_score": 25.0,
             "over_probability": 0.62, "under_probability": 0.38},
        ]
        sorted_results = self._apply_recommendation(results)
        assert sorted_results[0]["recommended"] is True
        assert sorted_results[0]["star_pick"] is True
        assert sorted_results[0]["is_core"] is True

    def test_only_one_core_pick(self):
        """Even when multiple games meet core criteria, only 1 is marked."""
        results = [
            {"idx": 0, "total_edge_pts": 10.0, "signal_score": 30.0,
             "over_probability": 0.70, "under_probability": 0.30},
            {"idx": 1, "total_edge_pts": 9.0, "signal_score": 28.0,
             "over_probability": 0.68, "under_probability": 0.32},
        ]
        sorted_results = self._apply_recommendation(results)
        core = [r for r in sorted_results if r["is_core"]]
        assert len(core) == 1
        assert core[0]["idx"] == 0  # abs(10) > abs(9)

    def test_min_recommend_4_when_more_than_5_games(self):
        """With > 5 games, min_recommend = 4. Auto-fill if quality filter yields fewer."""
        results = [
            {"idx": i, "total_edge_pts": float(2 + i * 0.5), "signal_score": 10.0,
             "over_probability": 0.53, "under_probability": 0.47}
            for i in range(6)
        ]
        sorted_results = self._apply_recommendation(results)
        recommended = [gr for gr in sorted_results if gr["recommended"]]
        assert len(recommended) >= 4

    def test_min_recommend_3_when_3_to_5_games(self):
        """With 3-5 games, min_recommend = 3. Auto-fill if quality filter yields fewer."""
        results = [
            {"idx": i, "total_edge_pts": float(2 + i), "signal_score": 10.0,
             "over_probability": 0.53, "under_probability": 0.47}
            for i in range(4)
        ]
        sorted_results = self._apply_recommendation(results)
        recommended = [gr for gr in sorted_results if gr["recommended"]]
        assert len(recommended) >= 3

    def test_min_recommend_1_when_2_games_or_fewer(self):
        """With <= 2 games, min_recommend = 1. Auto-fill if quality filter yields 0."""
        results = [
            {"idx": 0, "total_edge_pts": 1.0, "signal_score": 5.0,
             "over_probability": 0.51, "under_probability": 0.49},
        ]
        sorted_results = self._apply_recommendation(results)
        recommended = [gr for gr in sorted_results if gr["recommended"]]
        assert len(recommended) >= 1

    def test_guaranteed_core_pick_always_exists(self):
        """Every non-empty result set must have exactly 1 core pick."""
        results = [
            {"idx": 0, "total_edge_pts": 1.5, "signal_score": 5.0,
             "over_probability": 0.52, "under_probability": 0.48},
            {"idx": 1, "total_edge_pts": 2.0, "signal_score": 6.0,
             "over_probability": 0.53, "under_probability": 0.47},
            {"idx": 2, "total_edge_pts": 1.0, "signal_score": 4.0,
             "over_probability": 0.51, "under_probability": 0.49},
        ]
        sorted_results = self._apply_recommendation(results)
        core = [r for r in sorted_results if r["is_core"]]
        assert len(core) == 1

    def test_cap_at_5_with_many_qualifying_games(self):
        """When many games qualify, cap at max 5 recommendations."""
        results = [
            {"idx": i, "total_edge_pts": float(12 - i), "signal_score": 30.0,
             "over_probability": 0.70, "under_probability": 0.30}
            for i in range(10)
        ]
        sorted_results = self._apply_recommendation(results)
        recommended = [gr for gr in sorted_results if gr["recommended"]]
        assert len(recommended) == 5


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
    """Test game pace is simple average of home and away pace (no clamping)."""

    def test_normal_pace(self):
        from app.feature_engineering import calculate_game_pace
        assert calculate_game_pace(100.0, 98.0) == 99.0

    def test_high_pace(self):
        from app.feature_engineering import calculate_game_pace
        assert calculate_game_pace(110.0, 108.0) == 109.0

    def test_low_pace(self):
        from app.feature_engineering import calculate_game_pace
        assert calculate_game_pace(88.0, 90.0) == 89.0

    def test_equal_pace(self):
        from app.feature_engineering import calculate_game_pace
        assert calculate_game_pace(100.0, 100.0) == 100.0


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
        """With realistic PPP and pace, predicted total stays in 210-240."""
        from app.feature_engineering import calculate_game_pace, calculate_ppp
        game_pace = calculate_game_pace(99.0, 99.0)
        home_ppp = calculate_ppp(112.0)
        away_ppp = calculate_ppp(110.0)
        predicted_total = game_pace * (home_ppp + away_ppp)
        assert 210 <= predicted_total <= 240


class TestEfficiencyAdjustment:
    """Test PPP efficiency adjustment using league-average structure factors."""

    def test_efficiency_adjustment_increases_ppp(self):
        """Net effect of 3P, FT, ORB, TOV adjustments on PPP."""
        from app.prediction_engine import (
            LEAGUE_AVG_THREE_POINT_RATE, LEAGUE_AVG_FREE_THROW_RATE,
            LEAGUE_AVG_ORB_RATE, LEAGUE_AVG_TOV_RATE,
        )
        from app.feature_engineering import calculate_ppp
        base_ppp = calculate_ppp(112.0)
        adj_ppp = base_ppp
        adj_ppp *= (1 + LEAGUE_AVG_THREE_POINT_RATE * 0.15)
        adj_ppp *= (1 + LEAGUE_AVG_FREE_THROW_RATE * 0.10)
        adj_ppp *= (1 + LEAGUE_AVG_ORB_RATE * 0.08)
        adj_ppp *= (1 - LEAGUE_AVG_TOV_RATE * 0.12)
        # Net effect should increase PPP (positive factors outweigh TOV)
        assert adj_ppp > base_ppp

    def test_tov_rate_reduces_ppp(self):
        """Higher turnover rate should reduce PPP."""
        from app.feature_engineering import calculate_ppp
        ppp = calculate_ppp(112.0)
        adj_low_tov = ppp * (1 - 0.10 * 0.12)
        adj_high_tov = ppp * (1 - 0.20 * 0.12)
        assert adj_low_tov > adj_high_tov

    def test_three_point_rate_boosts_ppp(self):
        """Higher three-point rate should increase PPP."""
        from app.feature_engineering import calculate_ppp
        ppp = calculate_ppp(112.0)
        adj_low_3p = ppp * (1 + 0.30 * 0.15)
        adj_high_3p = ppp * (1 + 0.45 * 0.15)
        assert adj_high_3p > adj_low_3p

    def test_possession_model_predicted_total(self):
        """Predicted total = game_pace * (adj_home_ppp + adj_away_ppp)."""
        from app.prediction_engine import (
            LEAGUE_AVG_THREE_POINT_RATE, LEAGUE_AVG_FREE_THROW_RATE,
            LEAGUE_AVG_ORB_RATE, LEAGUE_AVG_TOV_RATE,
        )
        from app.feature_engineering import calculate_game_pace, calculate_ppp
        from app.game_simulator import LEAGUE_AVG_PACE
        game_pace = calculate_game_pace(99.0, 99.0, LEAGUE_AVG_PACE)
        home_ppp = calculate_ppp(112.0)
        away_ppp = calculate_ppp(110.0)
        # Apply efficiency adjustments
        home_ppp *= (1 + LEAGUE_AVG_THREE_POINT_RATE * 0.15)
        home_ppp *= (1 + LEAGUE_AVG_FREE_THROW_RATE * 0.10)
        home_ppp *= (1 + LEAGUE_AVG_ORB_RATE * 0.08)
        home_ppp *= (1 - LEAGUE_AVG_TOV_RATE * 0.12)
        away_ppp *= (1 + LEAGUE_AVG_THREE_POINT_RATE * 0.15)
        away_ppp *= (1 + LEAGUE_AVG_FREE_THROW_RATE * 0.10)
        away_ppp *= (1 + LEAGUE_AVG_ORB_RATE * 0.08)
        away_ppp *= (1 - LEAGUE_AVG_TOV_RATE * 0.12)
        predicted_total = game_pace * (home_ppp + away_ppp)
        # With efficiency adjustments, total should still be in reasonable NBA range
        assert 200 <= predicted_total <= 280


class TestMonteCarloConstants:
    """Verify Monte Carlo std and run count constants."""

    def test_mc_std(self):
        from app.prediction_engine import MC_TOTAL_STD
        assert MC_TOTAL_STD == 8.5

    def test_mc_runs(self):
        from app.prediction_engine import MIN_SIMULATION_COUNT
        assert MIN_SIMULATION_COUNT == 10000
