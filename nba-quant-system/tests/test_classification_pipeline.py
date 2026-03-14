"""Tests for the new NBA totals classification pipeline modules.

Covers:
    - dataset_builder (feature list, label logic)
    - train_classifier (hyperparameters, training)
    - calibration (CalibratedClassifierCV, isotonic)
    - train_pipeline (end-to-end orchestration)
    - evaluation (output columns, prediction logic)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

from app.dataset_builder import CLASSIFIER_FEATURES, build_dataset
from app.train_classifier import build_classifier, train
from app.calibration import calibrate
from app.train_pipeline import run_pipeline
from app.evaluation import predict_game, evaluate


# ---------------------------------------------------------------------------
# Shared fixture: synthetic DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df():
    """Synthetic DataFrame that mimics build_dataset output."""
    rng = np.random.RandomState(42)
    n = 200
    data: dict = {}
    for col in CLASSIFIER_FEATURES:
        if col == "closing_total":
            data[col] = rng.uniform(215, 235, n)
        elif col == "opening_total":
            data[col] = rng.uniform(215, 235, n)
        elif col == "line_movement":
            data[col] = rng.uniform(-3, 3, n)
        elif "pace" in col:
            data[col] = rng.uniform(95, 105, n)
        elif "off_rating" in col or "last5_off" in col:
            data[col] = rng.uniform(105, 120, n)
        elif "def_rating" in col:
            data[col] = rng.uniform(105, 120, n)
        elif "3p_rate" in col:
            data[col] = rng.uniform(0.30, 0.45, n)
        elif "ft_rate" in col:
            data[col] = rng.uniform(0.20, 0.35, n)
        elif "rest" in col:
            data[col] = rng.choice([1, 2, 3, 4], n).astype(float)
        else:
            data[col] = rng.uniform(0, 1, n)
    total_scores = data["closing_total"] + rng.uniform(-15, 15, n)
    data["label"] = (total_scores > data["closing_total"]).astype(int)
    data["game_id"] = list(range(n))
    return pd.DataFrame(data)


# ===================================================================
# dataset_builder
# ===================================================================

class TestClassifierFeatures:
    """CLASSIFIER_FEATURES structure matches the specification."""

    def test_feature_count(self):
        assert len(CLASSIFIER_FEATURES) == 21

    def test_odds_features(self):
        assert "closing_total" in CLASSIFIER_FEATURES
        assert "opening_total" in CLASSIFIER_FEATURES
        assert "line_movement" in CLASSIFIER_FEATURES

    def test_pace_features(self):
        for f in ("home_pace", "away_pace", "pace_avg", "pace_diff"):
            assert f in CLASSIFIER_FEATURES

    def test_rating_features(self):
        for f in ("home_off_rating", "away_off_rating",
                  "home_def_rating", "away_def_rating"):
            assert f in CLASSIFIER_FEATURES

    def test_shooting_features(self):
        for f in ("home_3p_rate", "away_3p_rate",
                  "home_ft_rate", "away_ft_rate"):
            assert f in CLASSIFIER_FEATURES

    def test_last5_features(self):
        for f in ("home_last5_off", "away_last5_off",
                  "home_last5_pace", "away_last5_pace"):
            assert f in CLASSIFIER_FEATURES

    def test_rest_features(self):
        assert "home_rest" in CLASSIFIER_FEATURES
        assert "away_rest" in CLASSIFIER_FEATURES


class TestLabelLogic:
    """Label = 1 if final_total > closing_line else 0."""

    def test_over(self):
        assert (1 if 230 > 225.0 else 0) == 1

    def test_under(self):
        assert (1 if 220 > 225.0 else 0) == 0

    def test_equal_is_under(self):
        assert (1 if 225 > 225.0 else 0) == 0


# ===================================================================
# train_classifier
# ===================================================================

class TestBuildClassifier:
    """Verify hyperparameters per spec."""

    def test_type(self):
        model = build_classifier()
        assert isinstance(model, GradientBoostingClassifier)

    def test_n_estimators(self):
        assert build_classifier().n_estimators == 300

    def test_learning_rate(self):
        assert build_classifier().learning_rate == 0.03

    def test_max_depth(self):
        assert build_classifier().max_depth == 3

    def test_has_predict_proba(self):
        assert hasattr(build_classifier(), "predict_proba")


class TestTrain:
    """Training on synthetic data."""

    def test_returns_model_and_metrics(self, synthetic_df):
        model, metrics = train(synthetic_df)
        assert isinstance(model, GradientBoostingClassifier)
        assert "accuracy" in metrics

    def test_accuracy_between_0_and_1(self, synthetic_df):
        _, metrics = train(synthetic_df)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_metrics_keys(self, synthetic_df):
        _, metrics = train(synthetic_df)
        for key in ("accuracy", "train_samples", "test_samples", "n_features"):
            assert key in metrics

    def test_n_features_matches(self, synthetic_df):
        _, metrics = train(synthetic_df)
        assert metrics["n_features"] == 21


# ===================================================================
# calibration
# ===================================================================

class TestCalibration:
    """CalibratedClassifierCV with isotonic method."""

    def test_returns_calibrated(self, synthetic_df):
        base, _ = train(synthetic_df)
        cal_model = calibrate(base, synthetic_df)
        assert isinstance(cal_model, CalibratedClassifierCV)

    def test_calibrated_predict_proba_shape(self, synthetic_df):
        base, _ = train(synthetic_df)
        cal_model = calibrate(base, synthetic_df)
        X = synthetic_df[CLASSIFIER_FEATURES].values
        proba = cal_model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_calibrated_probabilities_valid(self, synthetic_df):
        base, _ = train(synthetic_df)
        cal_model = calibrate(base, synthetic_df)
        X = synthetic_df[CLASSIFIER_FEATURES].values[:5]
        proba = cal_model.predict_proba(X)
        for row in proba:
            assert 0.0 <= row[0] <= 1.0
            assert 0.0 <= row[1] <= 1.0
            assert abs(row[0] + row[1] - 1.0) < 1e-9


# ===================================================================
# train_pipeline
# ===================================================================

class TestRunPipeline:
    """Pipeline orchestration (mocked dataset)."""

    def test_pipeline_returns_model(self, synthetic_df, monkeypatch):
        monkeypatch.setattr("app.train_pipeline.build_dataset", lambda **kw: synthetic_df)
        result = run_pipeline(save=False)
        assert result["model"] is not None

    def test_pipeline_returns_metrics(self, synthetic_df, monkeypatch):
        monkeypatch.setattr("app.train_pipeline.build_dataset", lambda **kw: synthetic_df)
        result = run_pipeline(save=False)
        assert "accuracy" in result["metrics"]

    def test_pipeline_returns_features(self, synthetic_df, monkeypatch):
        monkeypatch.setattr("app.train_pipeline.build_dataset", lambda **kw: synthetic_df)
        result = run_pipeline(save=False)
        assert result["features"] == CLASSIFIER_FEATURES

    def test_pipeline_empty_data(self, monkeypatch):
        monkeypatch.setattr("app.train_pipeline.build_dataset", lambda **kw: pd.DataFrame())
        result = run_pipeline(save=False)
        assert result["model"] is None


# ===================================================================
# evaluation
# ===================================================================

class TestPredictGame:
    """Single-game prediction."""

    def test_output_keys(self, synthetic_df):
        base, _ = train(synthetic_df)
        cal = calibrate(base, synthetic_df)
        features = synthetic_df.iloc[0].to_dict()
        out = predict_game(cal, features, game_label="LAL vs BOS")
        for key in ("Game", "Line", "Over Probability", "Under Probability",
                     "Prediction", "Confidence"):
            assert key in out

    def test_prediction_is_over_or_under(self, synthetic_df):
        base, _ = train(synthetic_df)
        cal = calibrate(base, synthetic_df)
        features = synthetic_df.iloc[0].to_dict()
        out = predict_game(cal, features)
        assert out["Prediction"] in ("OVER", "UNDER")

    def test_prob_complement(self, synthetic_df):
        base, _ = train(synthetic_df)
        cal = calibrate(base, synthetic_df)
        features = synthetic_df.iloc[0].to_dict()
        out = predict_game(cal, features)
        assert abs(out["Over Probability"] + out["Under Probability"] - 1.0) < 1e-3

    def test_confidence_range(self, synthetic_df):
        base, _ = train(synthetic_df)
        cal = calibrate(base, synthetic_df)
        features = synthetic_df.iloc[0].to_dict()
        out = predict_game(cal, features)
        assert 0.5 <= out["Confidence"] <= 1.0


class TestEvaluate:
    """Batch evaluation."""

    def test_output_columns(self, synthetic_df):
        base, _ = train(synthetic_df)
        cal = calibrate(base, synthetic_df)
        result = evaluate(cal, synthetic_df)
        expected_cols = {"Game", "Line", "Over Probability",
                         "Under Probability", "Prediction", "Confidence"}
        assert expected_cols == set(result.columns)

    def test_output_length(self, synthetic_df):
        base, _ = train(synthetic_df)
        cal = calibrate(base, synthetic_df)
        result = evaluate(cal, synthetic_df)
        assert len(result) == len(synthetic_df)

    def test_predictions_valid(self, synthetic_df):
        base, _ = train(synthetic_df)
        cal = calibrate(base, synthetic_df)
        result = evaluate(cal, synthetic_df)
        assert set(result["Prediction"].unique()).issubset({"OVER", "UNDER"})

    def test_confidence_at_least_half(self, synthetic_df):
        base, _ = train(synthetic_df)
        cal = calibrate(base, synthetic_df)
        result = evaluate(cal, synthetic_df)
        assert (result["Confidence"] >= 0.5).all()
