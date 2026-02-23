"""Tests for unified model status and review workflow changes."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from unittest import mock

import pytest

os.environ.pop("SUPABASE_URL", None)
os.environ.pop("SUPABASE_KEY", None)

from app import supabase_client
from app.database import get_conn, init_db, insert_prediction, DB_PATH
from app.prediction_models import MODEL_DIR, MODEL_FILES, VERSION_FILE, _current_version


@pytest.fixture(autouse=True)
def _reset_supabase():
    supabase_client._client = None
    supabase_client._available = None
    yield
    supabase_client._client = None
    supabase_client._available = None


@pytest.fixture()
def _fresh_db(tmp_path, monkeypatch):
    """Create a fresh in-memory–style SQLite DB for each test."""
    db_file = tmp_path / "test.sqlite"
    monkeypatch.setattr("app.database.DB_PATH", db_file)
    init_db()
    yield db_file


# ---------- prediction_models: version fallback removed ----------

def test_current_version_returns_unknown_when_no_file(tmp_path, monkeypatch):
    """Without model_version.json, _current_version returns 'unknown' (not v1)."""
    monkeypatch.setattr("app.prediction_models.VERSION_FILE", tmp_path / "model_version.json")
    assert _current_version() == "unknown"


def test_current_version_reads_from_json(tmp_path, monkeypatch):
    """With model_version.json, _current_version returns the stored version."""
    vf = tmp_path / "model_version.json"
    vf.write_text(json.dumps({"version": "v5"}))
    monkeypatch.setattr("app.prediction_models.VERSION_FILE", vf)
    assert _current_version() == "v5"


# ---------- prediction_models: MODEL_FILES constant ----------

def test_model_files_constant():
    """MODEL_FILES lists the four expected model file names."""
    assert MODEL_FILES == ("home_model.pkl", "away_model.pkl", "spread_model.pkl", "total_model.pkl")


# ---------- database: is_final_prediction column ----------

def test_predictions_table_has_is_final_prediction(_fresh_db):
    """The predictions_snapshot table includes is_final_prediction column."""
    with get_conn() as conn:
        info = conn.execute("PRAGMA table_info(predictions_snapshot)").fetchall()
    columns = [row[1] for row in info]
    assert "is_final_prediction" in columns


def test_insert_prediction_marks_previous_as_non_final(_fresh_db):
    """Re-predicting the same game marks the old prediction as non-final."""
    base = {
        "game_id": 100,
        "prediction_time": "2025-01-15T12:00:00",
        "spread_pick": "home_cover",
        "spread_prob": 0.6,
        "total_pick": "over",
        "total_prob": 0.55,
        "confidence_score": 0.1,
        "star_rating": 3,
        "recommendation_index": 0.5,
        "expected_home_score": 110.0,
        "expected_visitor_score": 105.0,
        "simulation_variance": 64.0,
    }

    # First prediction
    insert_prediction("2025-01-15", base)
    # Second prediction (same game_id)
    insert_prediction("2025-01-15", {**base, "prediction_time": "2025-01-15T13:00:00"})

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT is_final_prediction FROM predictions_snapshot WHERE game_id=100 ORDER BY id"
        ).fetchall()

    assert len(rows) == 2
    assert rows[0]["is_final_prediction"] == 0  # old = non-final
    assert rows[1]["is_final_prediction"] == 1  # new = final


def test_insert_prediction_new_game_is_final(_fresh_db):
    """A prediction for a brand-new game_id is final by default."""
    base = {
        "game_id": 200,
        "prediction_time": "2025-01-15T12:00:00",
        "spread_pick": "home_cover",
        "spread_prob": 0.6,
        "total_pick": "over",
        "total_prob": 0.55,
        "confidence_score": 0.1,
        "star_rating": 3,
        "recommendation_index": 0.5,
        "expected_home_score": 110.0,
        "expected_visitor_score": 105.0,
        "simulation_variance": 64.0,
    }
    insert_prediction("2025-01-15", base)

    with get_conn() as conn:
        row = conn.execute(
            "SELECT is_final_prediction FROM predictions_snapshot WHERE game_id=200"
        ).fetchone()

    assert row["is_final_prediction"] == 1


# ---------- supabase_client: is_final_prediction in save_prediction ----------

def test_save_prediction_payload_contains_is_final():
    """save_prediction includes is_final_prediction=True in payload."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True

    supabase_client.save_prediction({"game_id": 42})

    inserted = fake_client.table.return_value.insert.call_args[0][0]
    payload = inserted["payload"]
    assert payload["is_final_prediction"] is True


def test_save_prediction_marks_previous_non_final():
    """save_prediction calls update to mark old predictions as non-final."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True

    supabase_client.save_prediction({"game_id": 42})

    # Verify update was called on the predictions table
    fake_client.table.assert_any_call("predictions")
    update_chain = fake_client.table.return_value.update
    assert update_chain.called


# ---------- supabase_client: fetch_latest_training_metrics ----------

def test_fetch_latest_training_metrics_returns_payload():
    """fetch_latest_training_metrics returns the payload from latest training log."""
    fake_client = mock.MagicMock()
    fake_client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = [
        {"payload": {"home_mae": 4.5, "data_points": 500}}
    ]
    supabase_client._client = fake_client
    supabase_client._available = True

    result = supabase_client.fetch_latest_training_metrics()
    assert result == {"home_mae": 4.5, "data_points": 500}


def test_fetch_latest_training_metrics_returns_none_when_unavailable():
    """fetch_latest_training_metrics returns None when Supabase is not configured."""
    supabase_client._available = False
    assert supabase_client.fetch_latest_training_metrics() is None


# ---------- model_status: caching ----------

def test_model_status_caches_result():
    """get_model_status returns cached result on second call."""
    from app import model_status
    model_status._cached_status = None  # reset

    fake_status = {"version": "v3", "model_available": True, "training_samples": 100,
                   "last_trained": "2025-01-01", "metrics": {}}
    model_status._cached_status = fake_status

    assert model_status.get_model_status() is fake_status

    # Clean up
    model_status._cached_status = None
