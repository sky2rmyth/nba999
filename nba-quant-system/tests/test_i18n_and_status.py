"""Tests for i18n_cn module and model_status Supabase restore."""
from __future__ import annotations

import os
from unittest import mock

import pytest

os.environ.pop("SUPABASE_URL", None)
os.environ.pop("SUPABASE_KEY", None)

from app import supabase_client
from app.i18n_cn import cn


@pytest.fixture(autouse=True)
def _reset_supabase():
    supabase_client._client = None
    supabase_client._available = None
    yield
    supabase_client._client = None
    supabase_client._available = None


# ---------- i18n_cn ----------

def test_cn_returns_plain_string():
    """cn() returns a plain string for a known key with no formatting."""
    assert cn("model_loaded") == "🧠 模型来源：Supabase已加载"


def test_cn_returns_formatted_string():
    """cn() formats kwargs into the template."""
    result = cn("feature_count", feature_count=42)
    assert result == "特征数量: 42"


def test_cn_returns_key_when_unknown():
    """cn() returns the key itself when the key is not in the dictionary."""
    assert cn("nonexistent_key") == "nonexistent_key"


def test_cn_training_messages_are_chinese():
    """All training-related messages are in Chinese."""
    assert "学习" in cn("training_start")
    assert "完成" in cn("training_done")
    assert "模型" in cn("model_trained")


def test_cn_model_status_report_template():
    """model_status_report template can be formatted with expected kwargs."""
    result = cn("model_status_report",
                version="v5",
                available="✅",
                training_samples=500,
                mae_display="4.5 / 3.2",
                sc_acc="55.0%",
                to_acc="52.0%",
                last_trained="2025-01-01")
    assert "v5" in result
    assert "500" in result
    assert "模型状态报告" in result


# ---------- model_status: Supabase restore ----------

def test_model_status_tries_supabase_when_local_missing(tmp_path, monkeypatch):
    """model_status downloads from Supabase when local models are missing."""
    from app import model_status
    model_status._cached_status = None

    monkeypatch.setattr("app.model_status.MODEL_DIR", tmp_path)
    monkeypatch.setattr("app.model_status.MODEL_FILES",
                        ("home_model.pkl", "away_model.pkl"))

    def fake_download(model_dir):
        (tmp_path / "home_model.pkl").write_bytes(b"data")
        (tmp_path / "away_model.pkl").write_bytes(b"data")
        return True

    with mock.patch("app.supabase_client.download_models_from_storage",
                    side_effect=fake_download):
        with mock.patch("app.model_status._current_version", return_value="v3"):
            status = model_status._load_model_status()

    assert status["model_available"] is True
    model_status._cached_status = None
