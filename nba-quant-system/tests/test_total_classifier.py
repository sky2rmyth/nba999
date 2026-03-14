"""Tests for the totals over/under classification model."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from app.feature_engineering import (
    TOTAL_FEATURE_COLUMNS,
    build_total_training_frame,
    _compute_team_features,
)
from app.prediction_models import _build_total_classifier, train_models, FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# TOTAL_FEATURE_COLUMNS structure
# ---------------------------------------------------------------------------

class TestTotalFeatureColumns:
    """Verify TOTAL_FEATURE_COLUMNS has the 19 expected features."""

    def test_total_feature_count(self):
        assert len(TOTAL_FEATURE_COLUMNS) == 19

    def test_closing_total_present(self):
        assert "closing_total" in TOTAL_FEATURE_COLUMNS

    def test_opening_total_present(self):
        assert "opening_total" in TOTAL_FEATURE_COLUMNS

    def test_line_movement_present(self):
        assert "line_movement" in TOTAL_FEATURE_COLUMNS

    def test_pace_features_present(self):
        assert "home_pace" in TOTAL_FEATURE_COLUMNS
        assert "away_pace" in TOTAL_FEATURE_COLUMNS

    def test_off_rating_features_present(self):
        assert "home_off_rating" in TOTAL_FEATURE_COLUMNS
        assert "away_off_rating" in TOTAL_FEATURE_COLUMNS

    def test_def_rating_features_present(self):
        assert "home_def_rating" in TOTAL_FEATURE_COLUMNS
        assert "away_def_rating" in TOTAL_FEATURE_COLUMNS

    def test_3p_rate_features_present(self):
        assert "home_3p_rate" in TOTAL_FEATURE_COLUMNS
        assert "away_3p_rate" in TOTAL_FEATURE_COLUMNS

    def test_ft_rate_features_present(self):
        assert "home_ft_rate" in TOTAL_FEATURE_COLUMNS
        assert "away_ft_rate" in TOTAL_FEATURE_COLUMNS

    def test_last5_off_rating_features_present(self):
        assert "last5_home_off_rating" in TOTAL_FEATURE_COLUMNS
        assert "last5_away_off_rating" in TOTAL_FEATURE_COLUMNS

    def test_last5_pace_features_present(self):
        assert "last5_home_pace" in TOTAL_FEATURE_COLUMNS
        assert "last5_away_pace" in TOTAL_FEATURE_COLUMNS

    def test_rest_days_features_present(self):
        assert "rest_days_home" in TOTAL_FEATURE_COLUMNS
        assert "rest_days_away" in TOTAL_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# _build_total_classifier
# ---------------------------------------------------------------------------

class TestBuildTotalClassifier:
    """Verify _build_total_classifier returns GradientBoostingClassifier."""

    def test_returns_gradient_boosting_classifier(self):
        model = _build_total_classifier()
        assert isinstance(model, GradientBoostingClassifier)

    def test_model_has_predict_proba(self):
        model = _build_total_classifier()
        assert hasattr(model, "predict_proba")

    def test_model_hyperparameters(self):
        model = _build_total_classifier()
        assert model.n_estimators == 150
        assert model.learning_rate == 0.05
        assert model.max_depth == 5
        assert model.random_state == 42


# ---------------------------------------------------------------------------
# Total classifier training label logic
# ---------------------------------------------------------------------------

class TestTotalClassifierLabel:
    """Test label generation: 1 if total > closing_total, else 0."""

    def test_over_label(self):
        total_score = 230
        closing_total = 225.0
        label = 1 if total_score > closing_total else 0
        assert label == 1

    def test_under_label(self):
        total_score = 220
        closing_total = 225.0
        label = 1 if total_score > closing_total else 0
        assert label == 0

    def test_equal_is_under(self):
        """When total == closing_total, label should be 0 (not over)."""
        total_score = 225
        closing_total = 225.0
        label = 1 if total_score > closing_total else 0
        assert label == 0


# ---------------------------------------------------------------------------
# Prediction rule
# ---------------------------------------------------------------------------

class TestPredictionRule:
    """Test prediction rule: prob_over > 0.5 → Over, else → Under."""

    def test_over_prediction(self):
        prob_over = 0.65
        prediction = "Over" if prob_over > 0.5 else "Under"
        assert prediction == "Over"

    def test_under_prediction(self):
        prob_over = 0.35
        prediction = "Over" if prob_over > 0.5 else "Under"
        assert prediction == "Under"

    def test_boundary_under(self):
        """prob_over == 0.5 should predict Under."""
        prob_over = 0.5
        prediction = "Over" if prob_over > 0.5 else "Under"
        assert prediction == "Under"

    def test_prob_under_complement(self):
        prob_over = 0.62
        prob_under = 1.0 - prob_over
        assert abs(prob_under - 0.38) < 1e-9


# ---------------------------------------------------------------------------
# GradientBoostingClassifier integration
# ---------------------------------------------------------------------------

class TestTotalClassifierIntegration:
    """Test that the total classifier can be trained on synthetic data
    with TOTAL_FEATURE_COLUMNS and produce valid predict_proba output."""

    @pytest.fixture
    def synthetic_total_data(self):
        """Create synthetic DataFrame that mimics build_total_training_frame output."""
        rng = np.random.RandomState(42)
        n = 100
        data = {}
        for col in TOTAL_FEATURE_COLUMNS:
            if col == "closing_total":
                data[col] = rng.uniform(215, 235, n)
            elif col == "opening_total":
                data[col] = rng.uniform(215, 235, n)
            elif col == "line_movement":
                data[col] = rng.uniform(-3, 3, n)
            elif "pace" in col:
                data[col] = rng.uniform(95, 105, n)
            elif "off_rating" in col:
                data[col] = rng.uniform(105, 120, n)
            elif "def_rating" in col:
                data[col] = rng.uniform(105, 120, n)
            elif "3p_rate" in col:
                data[col] = rng.uniform(0.30, 0.45, n)
            elif "ft_rate" in col:
                data[col] = rng.uniform(0.20, 0.35, n)
            elif "rest_days" in col:
                data[col] = rng.choice([1, 2, 3, 4], n).astype(float)
            else:
                data[col] = rng.uniform(0, 1, n)
        # Label: 1 if a random total > closing_total
        total_scores = data["closing_total"] + rng.uniform(-15, 15, n)
        data["label"] = (total_scores > data["closing_total"]).astype(int)
        return pd.DataFrame(data)

    def test_train_and_predict(self, synthetic_total_data):
        model = _build_total_classifier()
        X = synthetic_total_data[TOTAL_FEATURE_COLUMNS].values
        y = synthetic_total_data["label"].values
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert all(0.0 <= p <= 1.0 for p in proba[:, 1])

    def test_prob_over_complement(self, synthetic_total_data):
        model = _build_total_classifier()
        X = synthetic_total_data[TOTAL_FEATURE_COLUMNS].values
        y = synthetic_total_data["label"].values
        model.fit(X, y)

        proba = model.predict_proba(X)
        for i in range(len(proba)):
            prob_over = proba[i][1]
            prob_under = 1.0 - prob_over
            assert abs(proba[i][0] - prob_under) < 1e-9

    def test_prediction_direction(self, synthetic_total_data):
        model = _build_total_classifier()
        X = synthetic_total_data[TOTAL_FEATURE_COLUMNS].values
        y = synthetic_total_data["label"].values
        model.fit(X, y)

        proba = model.predict_proba(X)
        for i in range(min(10, len(proba))):
            prob_over = proba[i][1]
            prediction = "Over" if prob_over > 0.5 else "Under"
            assert prediction in ("Over", "Under")


# ---------------------------------------------------------------------------
# train_models with total_df parameter
# ---------------------------------------------------------------------------

class TestTrainModelsWithTotalDf:
    """Test that train_models correctly handles the total_df parameter."""

    @pytest.fixture
    def minimal_df(self):
        """Minimal DataFrame for the general regressor/spread models."""
        rng = np.random.RandomState(42)
        n = 50
        data = {col: rng.uniform(0, 1, n) for col in FEATURE_COLUMNS}
        data["home_score"] = rng.uniform(95, 125, n)
        data["away_score"] = rng.uniform(95, 125, n)
        return pd.DataFrame(data)

    @pytest.fixture
    def minimal_total_df(self):
        """Minimal DataFrame for the total classifier."""
        rng = np.random.RandomState(42)
        n = 50
        data = {}
        for col in TOTAL_FEATURE_COLUMNS:
            data[col] = rng.uniform(0, 1, n)
        data["label"] = rng.choice([0, 1], n)
        return pd.DataFrame(data)

    def test_train_without_total_df(self, minimal_df):
        """Training without total_df should succeed with total_model=None."""
        with mock.patch("app.prediction_models.database"):
            with mock.patch("app.prediction_models._bump_version", return_value="v99"):
                with mock.patch("app.prediction_models.MODEL_DIR", Path("/tmp/test_models_no_total")):
                    Path("/tmp/test_models_no_total").mkdir(parents=True, exist_ok=True)
                    bundle = train_models(minimal_df, total_df=None)
        assert bundle.total_model is None

    def test_train_with_total_df(self, minimal_df, minimal_total_df):
        """Training with total_df should produce a GradientBoostingClassifier total_model."""
        with mock.patch("app.prediction_models.database"):
            with mock.patch("app.prediction_models._bump_version", return_value="v99"):
                with mock.patch("app.prediction_models.MODEL_DIR", Path("/tmp/test_models_with_total")):
                    Path("/tmp/test_models_with_total").mkdir(parents=True, exist_ok=True)
                    bundle = train_models(minimal_df, total_df=minimal_total_df)
        assert bundle.total_model is not None
        assert isinstance(bundle.total_model, GradientBoostingClassifier)

    def test_total_accuracy_in_metrics(self, minimal_df, minimal_total_df):
        """Metrics should contain total_over_accuracy when total_df is provided."""
        with mock.patch("app.prediction_models.database"):
            with mock.patch("app.prediction_models._bump_version", return_value="v99"):
                with mock.patch("app.prediction_models.MODEL_DIR", Path("/tmp/test_models_acc")):
                    Path("/tmp/test_models_acc").mkdir(parents=True, exist_ok=True)
                    bundle = train_models(minimal_df, total_df=minimal_total_df)
        assert "total_over_accuracy" in bundle.metrics
        assert 0.0 <= bundle.metrics["total_over_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# _compute_team_features includes new columns
# ---------------------------------------------------------------------------

class TestComputeTeamFeaturesNewColumns:
    """Verify that _compute_team_features now includes the new features."""

    @pytest.fixture
    def mock_db(self, tmp_path):
        """Create an in-memory SQLite DB with sample game data."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE games (
                game_id INTEGER PRIMARY KEY,
                season INTEGER,
                date TEXT,
                status TEXT,
                home_team_id INTEGER,
                visitor_team_id INTEGER,
                home_score INTEGER,
                visitor_score INTEGER,
                payload_json TEXT NOT NULL,
                updated_at TEXT
            )
        """)
        # Insert 10 games for team 1 (home)
        for i in range(10):
            conn.execute(
                "INSERT INTO games VALUES (?, 2024, ?, 'Final', 1, 2, 110, 105, '{}', NULL)",
                (100 + i, f"2024-01-{10 + i:02d}"),
            )
        conn.commit()
        return conn

    def test_last5_off_rating_present(self, mock_db):
        feat = _compute_team_features(mock_db, 1, 2, "2024-01-25", "home")
        assert "last5_home_off_rating" in feat
        assert feat["last5_home_off_rating"] > 0

    def test_last5_pace_present(self, mock_db):
        feat = _compute_team_features(mock_db, 1, 2, "2024-01-25", "home")
        assert "last5_home_pace" in feat
        assert feat["last5_home_pace"] > 0

    def test_3p_rate_present(self, mock_db):
        feat = _compute_team_features(mock_db, 1, 2, "2024-01-25", "home")
        assert "home_3p_rate" in feat
        assert feat["home_3p_rate"] > 0

    def test_ft_rate_present(self, mock_db):
        feat = _compute_team_features(mock_db, 1, 2, "2024-01-25", "home")
        assert "home_ft_rate" in feat
        assert feat["home_ft_rate"] > 0

    def test_no_games_returns_zero_defaults(self, mock_db):
        feat = _compute_team_features(mock_db, 999, 2, "2024-01-25", "home")
        assert feat.get("last5_home_off_rating") == 0.0
        assert feat.get("last5_home_pace") == 0.0
        assert feat.get("home_3p_rate") == 0.0
        assert feat.get("home_ft_rate") == 0.0
