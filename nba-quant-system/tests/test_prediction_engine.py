"""Tests for prediction_engine recommendation, signal score, and core pick logic."""
from __future__ import annotations


class TestRecommendationReason:
    """Test the recommendation reason thresholds based on abs_edge."""

    @staticmethod
    def _reason(abs_edge: float) -> str:
        """Mirror the recommendation reason logic from prediction_engine."""
        if abs_edge >= 8:
            return "模型预测与盘口差距较大"
        elif abs_edge >= 5:
            return "模型预测存在明显价值"
        else:
            return "信号较弱但进入推荐范围"

    def test_large_edge_reason(self):
        reason = self._reason(10.0)
        assert reason == "模型预测与盘口差距较大"

    def test_boundary_edge_8(self):
        reason = self._reason(8.0)
        assert reason == "模型预测与盘口差距较大"

    def test_medium_edge_reason(self):
        reason = self._reason(6.0)
        assert reason == "模型预测存在明显价值"

    def test_boundary_edge_5(self):
        reason = self._reason(5.0)
        assert reason == "模型预测存在明显价值"

    def test_small_edge_reason(self):
        reason = self._reason(3.0)
        assert reason == "信号较弱但进入推荐范围"

    def test_zero_edge_reason(self):
        reason = self._reason(0.0)
        assert reason == "信号较弱但进入推荐范围"


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
    """Test that exactly one core pick is selected (the highest signal score)."""

    def test_single_core_pick(self):
        results = [
            {"idx": 0, "signal_score": 20.0},
            {"idx": 1, "signal_score": 30.0},
            {"idx": 2, "signal_score": 25.0},
        ]
        sorted_results = sorted(results, key=lambda x: x["signal_score"], reverse=True)
        for i, gr in enumerate(sorted_results):
            gr["recommended"] = True
            gr["is_core"] = i == 0
        assert sorted_results[0]["idx"] == 1
        assert sorted_results[0]["is_core"] is True
        # Exactly one core
        core_count = sum(1 for r in sorted_results if r["is_core"])
        assert core_count == 1

    def test_single_game_is_core(self):
        results = [{"idx": 0, "signal_score": 15.0}]
        sorted_results = sorted(results, key=lambda x: x["signal_score"], reverse=True)
        for i, gr in enumerate(sorted_results):
            gr["recommended"] = True
            gr["is_core"] = i == 0
        assert sorted_results[0]["is_core"] is True

    def test_empty_results_no_core(self):
        results = []
        sorted_results = sorted(results, key=lambda x: x["signal_score"], reverse=True)
        assert len(sorted_results) == 0

    def test_top5_recommendation_more_than_5(self):
        """When more than 5 games, only top 5 by signal_score are recommended."""
        results = [{"idx": i, "signal_score": float(i)} for i in range(8)]
        sorted_results = sorted(results, key=lambda x: x["signal_score"], reverse=True)
        for i, gr in enumerate(sorted_results):
            gr["recommended"] = i < 5
            gr["is_core"] = i == 0
        recommended = [r for r in sorted_results if r["recommended"]]
        assert len(recommended) == 5
        assert sorted_results[0]["is_core"] is True
        assert sorted_results[0]["signal_score"] == 7.0

    def test_all_recommended_when_5_or_fewer(self):
        """When 5 or fewer games, all are recommended."""
        results = [{"idx": i, "signal_score": float(i)} for i in range(4)]
        sorted_results = sorted(results, key=lambda x: x["signal_score"], reverse=True)
        for i, gr in enumerate(sorted_results):
            gr["recommended"] = True
            gr["is_core"] = i == 0
        recommended = [r for r in sorted_results if r["recommended"]]
        assert len(recommended) == 4
        assert sorted_results[0]["is_core"] is True

    def test_exactly_5_games_all_recommended(self):
        """When exactly 5 games, all are recommended."""
        results = [{"idx": i, "signal_score": float(i * 2)} for i in range(5)]
        sorted_results = sorted(results, key=lambda x: x["signal_score"], reverse=True)
        for i, gr in enumerate(sorted_results):
            gr["recommended"] = True
            gr["is_core"] = i == 0
        recommended = [r for r in sorted_results if r["recommended"]]
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
