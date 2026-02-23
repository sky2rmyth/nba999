"""Tests for ensure_models() model source logic in retrain_engine."""
from __future__ import annotations

import os
from unittest import mock

import pytest

os.environ.pop("SUPABASE_URL", None)
os.environ.pop("SUPABASE_KEY", None)

from app import supabase_client


@pytest.fixture(autouse=True)
def _reset_supabase():
    supabase_client._client = None
    supabase_client._available = None
    yield
    supabase_client._client = None
    supabase_client._available = None


# ---------- ensure_models: cached local model ----------

def test_ensure_models_returns_cached_when_local_exists():
    """When models exist locally, ensure_models returns them with 'Using cached model' source."""
    fake_bundle = mock.MagicMock()
    fake_bundle.version = "v5"

    with mock.patch("app.retrain_engine.load_models", return_value=fake_bundle):
        from app.retrain_engine import ensure_models
        result = ensure_models(force=False)

    assert result is fake_bundle
    assert result.source == "Using cached model"


# ---------- ensure_models: loaded from Supabase ----------

def test_ensure_models_downloads_from_supabase_when_local_missing():
    """When local models are missing but Supabase has them, source is 'Loaded from Supabase'."""
    fake_bundle = mock.MagicMock()
    fake_bundle.version = "v3"

    # First call: no local models; second call (after download): models available
    with mock.patch("app.retrain_engine.load_models", side_effect=[None, fake_bundle]):
        with mock.patch("app.supabase_client.download_models_from_storage", return_value=True):
            from app.retrain_engine import ensure_models
            result = ensure_models(force=False)

    assert result is fake_bundle
    assert result.source == "Loaded from Supabase"


# ---------- ensure_models: force retrain even when cached ----------

def test_ensure_models_force_triggers_training():
    """When force=True, ensure_models trains even when local models exist."""
    fake_cached = mock.MagicMock()
    fake_cached.version = "v5"

    fake_trained = mock.MagicMock()
    fake_trained.version = "v6"
    fake_trained.algorithm = "lightgbm"
    fake_trained.metrics = {}
    fake_trained.duration = 1.0

    fake_df = mock.MagicMock()
    fake_df.empty = False

    with mock.patch("app.retrain_engine.load_models", return_value=fake_cached):
        with mock.patch("app.retrain_engine._db_has_completed_games", return_value=True):
            with mock.patch("app.retrain_engine.build_training_frame", return_value=fake_df):
                with mock.patch("app.retrain_engine.train_models", return_value=fake_trained):
                    with mock.patch("app.retrain_engine.FEATURE_COLUMNS", ["f"] * 50):
                        from app.retrain_engine import ensure_models
                        result = ensure_models(force=True)

    assert result is fake_trained
    assert result.source == "Trained new model"


# ---------- ModelBundle has source attribute ----------

def test_model_bundle_has_source_attribute():
    """ModelBundle initializes with source='unknown' by default."""
    from app.prediction_models import ModelBundle
    bundle = ModelBundle(None, None, "test")
    assert bundle.source == "unknown"
