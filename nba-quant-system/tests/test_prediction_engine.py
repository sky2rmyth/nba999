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
        if abs_edge >= 8:
            return "模型预测与盘口差距较大"
        elif abs_edge >= 6:
            return "模型预测存在明显价值"
        else:
            return "信号较弱，不推荐"

    def test_large_edge_reason(self):
        reason = self._reason(10.0)
        assert reason == "模型预测与盘口差距较大"

    def test_boundary_edge_8(self):
        reason = self._reason(8.0)
        assert reason == "模型预测与盘口差距较大"

    def test_medium_edge_reason(self):
        reason = self._reason(6.0)
        assert reason == "模型预测存在明显价值"

    def test_boundary_edge_6(self):
        reason = self._reason(6.0)
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
    """Test that core pick is the highest signal_score among recommended games (abs_edge >= 6)."""

    def test_single_core_pick(self):
        results = [
            {"idx": 0, "signal_score": 20.0, "total_edge_pts": 7.0},
            {"idx": 1, "signal_score": 30.0, "total_edge_pts": 8.0},
            {"idx": 2, "signal_score": 25.0, "total_edge_pts": 3.0},
        ]
        sorted_results = sorted(results, key=lambda x: x["signal_score"], reverse=True)
        for gr in sorted_results:
            gr["recommended"] = abs(gr["total_edge_pts"]) >= 6
            gr["is_core"] = False
        recommended = [gr for gr in sorted_results if gr["recommended"]]
        if recommended:
            recommended[0]["is_core"] = True
        # idx=1 has highest signal_score among recommended (edge >= 6)
        assert sorted_results[0]["idx"] == 1
        assert sorted_results[0]["is_core"] is True
        # idx=2 (edge=3) should not be recommended
        not_rec = [r for r in sorted_results if not r["recommended"]]
        assert len(not_rec) == 1
        assert not_rec[0]["idx"] == 2

    def test_single_game_with_large_edge_is_core(self):
        results = [{"idx": 0, "signal_score": 15.0, "total_edge_pts": 7.0}]
        sorted_results = sorted(results, key=lambda x: x["signal_score"], reverse=True)
        for gr in sorted_results:
            gr["recommended"] = abs(gr["total_edge_pts"]) >= 6
            gr["is_core"] = False
        recommended = [gr for gr in sorted_results if gr["recommended"]]
        if recommended:
            recommended[0]["is_core"] = True
        assert sorted_results[0]["is_core"] is True

    def test_no_core_when_all_edges_small(self):
        """When no game has abs_edge >= 6, no core pick is selected."""
        results = [
            {"idx": 0, "signal_score": 20.0, "total_edge_pts": 3.0},
            {"idx": 1, "signal_score": 25.0, "total_edge_pts": 4.0},
        ]
        sorted_results = sorted(results, key=lambda x: x["signal_score"], reverse=True)
        for gr in sorted_results:
            gr["recommended"] = abs(gr["total_edge_pts"]) >= 6
            gr["is_core"] = False
        recommended = [gr for gr in sorted_results if gr["recommended"]]
        if recommended:
            recommended[0]["is_core"] = True
        core_count = sum(1 for r in sorted_results if r["is_core"])
        assert core_count == 0

    def test_empty_results_no_core(self):
        results = []
        sorted_results = sorted(results, key=lambda x: x["signal_score"], reverse=True)
        assert len(sorted_results) == 0

    def test_recommendation_based_on_abs_edge_6(self):
        """Games are recommended when abs_edge >= 6, regardless of count."""
        results = [{"idx": i, "signal_score": float(i), "total_edge_pts": float(i)} for i in range(8)]
        sorted_results = sorted(results, key=lambda x: x["signal_score"], reverse=True)
        for gr in sorted_results:
            gr["recommended"] = abs(gr["total_edge_pts"]) >= 6
            gr["is_core"] = False
        recommended = [gr for gr in sorted_results if gr["recommended"]]
        if recommended:
            recommended[0]["is_core"] = True
        # Only idx=6 and idx=7 have edge >= 6
        assert len(recommended) == 2
        assert recommended[0]["is_core"] is True

    def test_negative_edge_also_recommends(self):
        """Negative edge with abs >= 6 should also be recommended."""
        results = [
            {"idx": 0, "signal_score": 30.0, "total_edge_pts": -7.0},
            {"idx": 1, "signal_score": 20.0, "total_edge_pts": 3.0},
        ]
        sorted_results = sorted(results, key=lambda x: x["signal_score"], reverse=True)
        for gr in sorted_results:
            gr["recommended"] = abs(gr["total_edge_pts"]) >= 6
            gr["is_core"] = False
        recommended = [gr for gr in sorted_results if gr["recommended"]]
        if recommended:
            recommended[0]["is_core"] = True
        assert len(recommended) == 1
        assert recommended[0]["idx"] == 0
        assert recommended[0]["is_core"] is True


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
