"""Tests for supabase_client module."""
from __future__ import annotations

import os
from unittest import mock

import pytest

# Ensure Supabase credentials are NOT set during import
os.environ.pop("SUPABASE_URL", None)
os.environ.pop("SUPABASE_KEY", None)

from app import supabase_client


@pytest.fixture(autouse=True)
def _reset_client():
    """Reset module-level state between tests."""
    supabase_client._client = None
    supabase_client._available = None
    supabase_client._table_columns_cache.clear()
    yield
    supabase_client._client = None
    supabase_client._available = None
    supabase_client._table_columns_cache.clear()


# --- _get_client ---

def test_get_client_no_credentials():
    """Without env vars, _get_client returns None."""
    with mock.patch.dict(os.environ, {}, clear=True):
        assert supabase_client._get_client() is None
        assert supabase_client._available is False


def test_get_client_with_credentials():
    """With env vars, _get_client initializes the Supabase client."""
    fake_client = mock.MagicMock()
    with mock.patch.dict(os.environ, {"SUPABASE_URL": "https://x.supabase.co", "SUPABASE_KEY": "key123"}):
        with mock.patch.dict(
            "sys.modules",
            {"supabase": mock.MagicMock(create_client=mock.MagicMock(return_value=fake_client))},
        ):
            client = supabase_client._get_client()
    assert client is fake_client
    assert supabase_client._available is True


# --- _ensure_tables ---

def test_ensure_tables_checks_all_four():
    """_ensure_tables checks predictions, simulation_logs, training_logs, review_results."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._ensure_tables()
    tables_checked = [call[0][0] for call in fake_client.table.call_args_list]
    assert "predictions" in tables_checked
    assert "simulation_logs" in tables_checked
    assert "training_logs" in tables_checked
    assert "review_results" in tables_checked


# --- save_prediction ---

def test_save_prediction_skips_when_not_configured():
    """save_prediction is a no-op without credentials."""
    supabase_client._available = False
    supabase_client.save_prediction({"game_id": 1})  # should not raise


def test_save_prediction_does_not_raise_on_failure():
    """save_prediction swallows exceptions so the pipeline never crashes."""
    fake_client = mock.MagicMock()
    fake_client.table.return_value.insert.return_value.execute.side_effect = RuntimeError("write failed")
    supabase_client._client = fake_client
    supabase_client._available = True
    supabase_client.save_prediction({"game_id": 1})  # should not raise


def test_save_prediction_uses_payload_jsonb():
    """save_prediction stores all data inside game_id + payload JSONB."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True

    supabase_client.save_prediction({
        "game_id": 42,
        "game_date": "2025-01-15",
        "home_team": "Lakers",
        "away_team": "Celtics",
        "spread_line": -3.5,
        "total_line": 220.5,
        "spread_pick": "home_cover",
        "total_pick": "over",
        "spread_confidence": 65.0,
        "total_confidence": 58.0,
        "model_version": "v2",
        "simulation_runs": 10000,
    })

    inserted = fake_client.table.return_value.insert.call_args[0][0]
    assert inserted["game_id"] == 42
    assert "payload" in inserted
    assert len(inserted) == 2  # only 'game_id' and 'payload' at top level
    payload = inserted["payload"]
    assert payload["game_id"] == 42
    assert payload["game_date"] == "2025-01-15"
    assert payload["spread_line"] == -3.5
    assert payload["total_line"] == 220.5
    assert payload["spread_confidence"] == 65.0
    assert payload["total_confidence"] == 58.0
    assert payload["model_version"] == "v2"
    assert payload["simulation_runs"] == 10000
    assert "created_at" in payload


# --- save_simulation_log ---

def test_save_simulation_log_does_not_raise_on_failure():
    """save_simulation_log swallows exceptions so the pipeline never crashes."""
    fake_client = mock.MagicMock()
    fake_client.table.return_value.insert.return_value.execute.side_effect = RuntimeError("fail")
    supabase_client._client = fake_client
    supabase_client._available = True
    supabase_client.save_simulation_log({"game_id": 1})  # should not raise


def test_save_simulation_log_skips_when_not_configured():
    """save_simulation_log is a no-op without credentials."""
    supabase_client._available = False
    supabase_client.save_simulation_log({"game_id": 1})  # should not raise


def test_save_simulation_log_uses_payload_jsonb():
    """save_simulation_log stores all data inside game_id + payload JSONB."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True

    supabase_client.save_simulation_log({
        "game_id": 42,
        "model_version": "v2",
        "simulation_runs": 10000,
        "spread_cover_probability": 0.65,
        "over_probability": 0.58,
        "expected_home_score": 112.5,
        "expected_visitor_score": 108.3,
    })

    inserted = fake_client.table.return_value.insert.call_args[0][0]
    assert inserted["game_id"] == 42
    assert "payload" in inserted
    assert len(inserted) == 2  # only 'game_id' and 'payload' at top level
    payload = inserted["payload"]
    assert payload["game_id"] == 42
    assert payload["model_version"] == "v2"
    assert payload["simulation_runs"] == 10000
    assert payload["spread_cover_probability"] == 0.65
    assert payload["over_probability"] == 0.58
    assert payload["expected_home_score"] == 112.5
    assert payload["expected_visitor_score"] == 108.3
    assert "timestamp" in payload


# --- save_training_log ---

def test_save_training_log_does_not_raise_on_failure():
    """save_training_log swallows exceptions so the pipeline never crashes."""
    fake_client = mock.MagicMock()
    fake_client.table.return_value.insert.return_value.execute.side_effect = RuntimeError("fail")
    supabase_client._client = fake_client
    supabase_client._available = True
    supabase_client.save_training_log({"model_version": "v1"})  # should not raise


def test_save_training_log_skips_when_not_configured():
    """save_training_log is a no-op without credentials."""
    supabase_client._available = False
    supabase_client.save_training_log({"model_version": "v1"})  # should not raise


def test_save_training_log_uses_payload_jsonb():
    """save_training_log stores all data inside a 'payload' JSONB column."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True

    supabase_client.save_training_log({
        "model_version": "v2",
        "feature_count": 50,
        "algorithm": "xgboost",
        "home_mae": 4.5,
    })

    inserted = fake_client.table.return_value.insert.call_args[0][0]
    assert "payload" in inserted
    assert len(inserted) == 1  # only 'payload' key at top level
    payload = inserted["payload"]
    assert payload["model_version"] == "v2"
    assert payload["feature_count"] == 50
    assert payload["algorithm"] == "xgboost"
    assert payload["home_mae"] == 4.5
    assert "timestamp" in payload


# --- save_review_result ---

def test_save_review_result_does_not_raise_on_failure():
    """save_review_result swallows exceptions so the workflow never crashes."""
    fake_client = mock.MagicMock()
    fake_client.table.return_value.upsert.return_value.execute.side_effect = RuntimeError("fail")
    supabase_client._client = fake_client
    supabase_client._available = True
    # Pre-populate cache so RPC is not called
    supabase_client._table_columns_cache["review_results"] = {"game_id", "reviewed_at"}
    supabase_client.save_review_result({"game_id": 1})  # should not raise


def test_save_review_result_includes_fields():
    """save_review_result record contains expected fields when all columns exist."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True

    # Pre-populate column cache so _get_table_columns is not called
    supabase_client._table_columns_cache["review_results"] = {
        "game_id", "game_date", "home_team", "away_team",
        "spread_pick", "total_pick", "spread_correct", "total_correct",
        "final_home_score", "final_visitor_score", "reviewed_at",
    }

    supabase_client.save_review_result({
        "game_id": 42,
        "game_date": "2025-01-15",
        "home_team": "Lakers",
        "away_team": "Celtics",
        "spread_pick": "home_cover",
        "total_pick": "over",
        "spread_correct": True,
        "total_correct": False,
        "final_home_score": 112,
        "final_visitor_score": 108,
    })

    upserted = fake_client.table.return_value.upsert.call_args[0][0]
    assert upserted["game_id"] == 42
    assert upserted["game_date"] == "2025-01-15"
    assert upserted["spread_correct"] is True
    assert upserted["total_correct"] is False
    assert upserted["final_home_score"] == 112
    assert "reviewed_at" in upserted
    # Verify on_conflict is set to game_id
    assert fake_client.table.return_value.upsert.call_args[1]["on_conflict"] == "game_id"


def test_save_review_result_skips_when_not_configured():
    """save_review_result is a no-op without credentials."""
    supabase_client._available = False
    supabase_client.save_review_result({"game_id": 1})  # should not raise


def test_save_review_result_filters_unknown_columns():
    """save_review_result drops fields not present in the DB table."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True

    # Simulate a table that only has game_id and reviewed_at columns
    supabase_client._table_columns_cache["review_results"] = {"game_id", "reviewed_at"}

    supabase_client.save_review_result({
        "game_id": 42,
        "home_team": "Lakers",
        "away_team": "Celtics",
        "spread_correct": True,
        "unknown_field": "ignored",
    })

    upserted = fake_client.table.return_value.upsert.call_args[0][0]
    assert upserted["game_id"] == 42
    assert "reviewed_at" in upserted
    # Fields not in the DB columns should be excluded
    assert "home_team" not in upserted
    assert "away_team" not in upserted
    assert "spread_correct" not in upserted
    assert "unknown_field" not in upserted


def test_save_review_result_skips_when_columns_unknown():
    """When column discovery fails, no fields pass filtering and upsert is skipped."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True

    # Column cache set to None (simulating failed discovery)
    supabase_client._table_columns_cache["review_results"] = None

    supabase_client.save_review_result({
        "game_id": 42,
        "home_team": "Lakers",
        "extra_field": "kept",
    })

    fake_client.table.return_value.upsert.assert_not_called()


# --- upload_models_to_storage ---

def test_upload_models_skips_when_not_configured():
    """upload_models_to_storage returns False without credentials."""
    supabase_client._available = False
    assert supabase_client.upload_models_to_storage("/tmp/models") is False


def test_upload_models_uploads_all_files(tmp_path):
    """upload_models_to_storage uploads each model file to the bucket."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True

    for fname in supabase_client._MODEL_STORAGE_FILES:
        (tmp_path / fname).write_bytes(b"fake-model-data")

    result = supabase_client.upload_models_to_storage(tmp_path)
    assert result is True

    upload_calls = fake_client.storage.from_.return_value.upload.call_args_list
    uploaded_names = [call[0][0] for call in upload_calls]
    for fname in supabase_client._MODEL_STORAGE_FILES:
        assert fname in uploaded_names


def test_upload_models_handles_missing_files(tmp_path):
    """upload_models_to_storage skips files that don't exist locally."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True

    # Only create one file
    (tmp_path / "home_model.pkl").write_bytes(b"data")

    result = supabase_client.upload_models_to_storage(tmp_path)
    assert result is True
    assert fake_client.storage.from_.return_value.upload.call_count == 1


# --- download_models_from_storage ---

def test_download_models_skips_when_not_configured():
    """download_models_from_storage returns False without credentials."""
    supabase_client._available = False
    assert supabase_client.download_models_from_storage("/tmp/models") is False


def test_download_models_downloads_all_files(tmp_path):
    """download_models_from_storage writes files from the bucket to disk."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True

    fake_client.storage.from_.return_value.download.return_value = b"model-bytes"

    result = supabase_client.download_models_from_storage(tmp_path)
    assert result is True

    for fname in supabase_client._MODEL_STORAGE_FILES:
        assert (tmp_path / fname).exists()
        assert (tmp_path / fname).read_bytes() == b"model-bytes"


def test_download_models_returns_false_when_required_missing(tmp_path):
    """download_models_from_storage returns False if required models fail to download."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True

    # All downloads fail
    fake_client.storage.from_.return_value.download.side_effect = RuntimeError("not found")

    result = supabase_client.download_models_from_storage(tmp_path)
    assert result is False


# --- get_table_columns ---

def test_get_table_columns_returns_set():
    """get_table_columns returns a set of column names."""
    supabase_client._table_columns_cache["my_table"] = {"id", "name"}
    assert supabase_client.get_table_columns("my_table") == {"id", "name"}


def test_get_table_columns_returns_empty_set_when_unknown():
    """get_table_columns returns empty set when columns are unknown."""
    supabase_client._table_columns_cache["my_table"] = None
    assert supabase_client.get_table_columns("my_table") == set()


def test_get_table_columns_returns_empty_set_when_not_configured():
    """get_table_columns returns empty set without credentials."""
    supabase_client._available = False
    assert supabase_client.get_table_columns("any_table") == set()


def test_get_table_columns_uses_information_schema():
    """_get_table_columns queries information_schema.columns via RPC."""
    fake_client = mock.MagicMock()
    fake_client.rpc.return_value.execute.return_value = mock.MagicMock(
        data=[{"column_name": "game_id"}, {"column_name": "score"}]
    )
    supabase_client._client = fake_client
    supabase_client._available = True

    cols = supabase_client.get_table_columns("review_results")
    assert cols == {"game_id", "score"}
    fake_client.rpc.assert_called_once()
    call_args = fake_client.rpc.call_args[0]
    assert call_args[0] == "sql"
    assert "information_schema.columns" in call_args[1]["query"]
    assert "review_results" in call_args[1]["query"]


# --- adaptive_upsert ---

def test_adaptive_upsert_filters_columns():
    """adaptive_upsert drops fields not present in the table."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True
    supabase_client._table_columns_cache["test_table"] = {"game_id", "score"}

    supabase_client.adaptive_upsert("test_table", {"game_id": 1, "score": 100, "extra": "ignored"})

    upserted = fake_client.table.return_value.upsert.call_args[0][0]
    assert upserted == {"game_id": 1, "score": 100}
    assert "extra" not in upserted


def test_adaptive_upsert_skips_when_columns_unknown():
    """adaptive_upsert skips upsert when column discovery returns None."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True
    supabase_client._table_columns_cache["test_table"] = None

    supabase_client.adaptive_upsert("test_table", {"game_id": 1, "extra": "kept"})

    fake_client.table.return_value.upsert.assert_not_called()


def test_adaptive_upsert_skips_when_not_configured():
    """adaptive_upsert is a no-op without credentials."""
    supabase_client._available = False
    supabase_client.adaptive_upsert("test_table", {"game_id": 1})  # should not raise


def test_adaptive_upsert_skips_when_no_valid_columns():
    """adaptive_upsert does nothing when all columns are filtered out."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True
    supabase_client._table_columns_cache["test_table"] = {"id"}

    supabase_client.adaptive_upsert("test_table", {"extra": "no_match"})

    fake_client.table.return_value.upsert.assert_not_called()


def test_adaptive_upsert_custom_conflict():
    """adaptive_upsert passes through the conflict parameter."""
    fake_client = mock.MagicMock()
    supabase_client._client = fake_client
    supabase_client._available = True
    supabase_client._table_columns_cache["test_table"] = {"id", "value"}

    supabase_client.adaptive_upsert("test_table", {"id": 1, "value": "x"}, conflict="id")

    assert fake_client.table.return_value.upsert.call_args[1]["on_conflict"] == "id"


# --- ensure_review_schema ---

def test_ensure_review_schema_skips_when_not_configured():
    """ensure_review_schema is a no-op without credentials."""
    supabase_client._available = False
    supabase_client.ensure_review_schema()  # should not raise


def test_ensure_review_schema_skips_when_table_inaccessible():
    """ensure_review_schema handles inaccessible table gracefully."""
    fake_client = mock.MagicMock()
    fake_client.table.return_value.select.return_value.limit.return_value.execute.side_effect = RuntimeError("missing")
    supabase_client._client = fake_client
    supabase_client._available = True
    supabase_client.ensure_review_schema()  # should not raise


def test_ensure_review_schema_adds_missing_columns():
    """ensure_review_schema calls RPC for columns not in the existing schema."""
    fake_client = mock.MagicMock()
    # Simulate table with only game_id column
    fake_client.table.return_value.select.return_value.limit.return_value.execute.return_value = mock.MagicMock(
        data=[{"game_id": "1"}]
    )
    supabase_client._client = fake_client
    supabase_client._available = True

    supabase_client.ensure_review_schema()

    # Should have called rpc for each missing required column
    rpc_calls = fake_client.rpc.call_args_list
    assert len(rpc_calls) > 0
    # Verify the SQL contains ALTER TABLE
    for call in rpc_calls:
        sql = call[0][1]["sql"]
        assert "ALTER TABLE review_results" in sql
        assert "ADD COLUMN IF NOT EXISTS" in sql


def test_ensure_review_schema_no_rpc_when_all_present():
    """ensure_review_schema does not call RPC when all columns are present."""
    fake_client = mock.MagicMock()
    fake_client.table.return_value.select.return_value.limit.return_value.execute.return_value = mock.MagicMock(
        data=[{
            "game_id": "1", "spread_pick": "home", "total_pick": "over",
            "spread_hit": True, "ou_hit": False, "final_home_score": 100,
            "final_visitor_score": 95, "reviewed_at": "2025-01-01T00:00:00Z",
        }]
    )
    supabase_client._client = fake_client
    supabase_client._available = True

    supabase_client.ensure_review_schema()

    fake_client.rpc.assert_not_called()


def test_ensure_review_schema_handles_rpc_failure():
    """ensure_review_schema does not crash when RPC call fails."""
    fake_client = mock.MagicMock()
    fake_client.table.return_value.select.return_value.limit.return_value.execute.return_value = mock.MagicMock(
        data=[{"game_id": "1"}]
    )
    fake_client.rpc.return_value.execute.side_effect = RuntimeError("rpc failed")
    supabase_client._client = fake_client
    supabase_client._available = True

    supabase_client.ensure_review_schema()  # should not raise
