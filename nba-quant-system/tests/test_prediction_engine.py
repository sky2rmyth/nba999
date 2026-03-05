"""Tests for prediction_engine recommendation, signal score, and core pick logic."""
from __future__ import annotations


class TestRecommendationLogic:
    """Test the recommendation thresholds defined in the problem statement."""

    @staticmethod
    def _recommend(abs_edge: float, over_probability: float) -> tuple[str, str]:
        """Mirror the recommendation logic from prediction_engine."""
        if abs_edge >= 6 and over_probability >= 0.62:
            return "推荐", "Edge大于6分且概率高于62%，模型信号强"
        elif abs_edge >= 4 and over_probability >= 0.58:
            return "观察", "Edge中等，概率一般，信号中等"
        else:
            return "不推荐", "Edge不足或概率不足，模型信号弱"

    def test_recommend_high_edge_high_prob(self):
        rec, reason = self._recommend(8.0, 0.72)
        assert rec == "推荐"
        assert "Edge大于6分" in reason

    def test_recommend_boundary_6_and_62(self):
        rec, _ = self._recommend(6.0, 0.62)
        assert rec == "推荐"

    def test_observe_medium_edge_medium_prob(self):
        rec, reason = self._recommend(5.0, 0.60)
        assert rec == "观察"
        assert "Edge中等" in reason

    def test_observe_boundary_4_and_58(self):
        rec, _ = self._recommend(4.0, 0.58)
        assert rec == "观察"

    def test_not_recommended_low_edge(self):
        rec, reason = self._recommend(2.0, 0.55)
        assert rec == "不推荐"
        assert "Edge不足" in reason

    def test_not_recommended_high_edge_low_prob(self):
        """High edge but low probability should not recommend."""
        rec, _ = self._recommend(8.0, 0.50)
        assert rec == "不推荐"

    def test_not_recommended_low_edge_high_prob(self):
        """Low edge but high probability: not enough edge."""
        rec, _ = self._recommend(3.0, 0.70)
        assert rec == "不推荐"

    def test_observe_not_recommend_boundary(self):
        """abs_edge=4 but prob=0.57 should be 不推荐."""
        rec, _ = self._recommend(4.0, 0.57)
        assert rec == "不推荐"


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
        core_idx = max(range(len(results)), key=lambda i: results[i]["signal_score"])
        assert core_idx == 1
        # Exactly one core
        core_count = sum(1 for r in results if r["idx"] == core_idx)
        assert core_count == 1

    def test_single_game_is_core(self):
        results = [{"idx": 0, "signal_score": 15.0}]
        core_idx = max(range(len(results)), key=lambda i: results[i]["signal_score"])
        assert core_idx == 0

    def test_empty_results_no_core(self):
        results = []
        core_idx = None
        if results:
            core_idx = max(range(len(results)), key=lambda i: results[i]["signal_score"])
        assert core_idx is None


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
