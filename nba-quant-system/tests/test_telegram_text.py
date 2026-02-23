"""Tests for telegram_text and ProgressTracker Chinese output."""
from __future__ import annotations

from app.telegram_text import TEXT
from app.telegram_bot import ProgressTracker


# ---------- telegram_text: TEXT dict ----------

def test_text_keys_present():
    """TEXT contains all required keys."""
    expected = {"start", "load_model", "fetch_data", "simulation", "saving", "done", "model_source"}
    assert expected == set(TEXT.keys())


def test_text_values_are_chinese():
    """All TEXT values contain Chinese characters (no English status words)."""
    english_words = {"System Starting", "Fetching", "Loading", "Running", "Saving", "Completed"}
    for value in TEXT.values():
        for word in english_words:
            assert word not in value, f"English text '{word}' found in TEXT value '{value}'"


# ---------- ProgressTracker: Chinese stages ----------

def test_progress_tracker_stages_use_chinese():
    """ProgressTracker.STAGES labels are Chinese, not English."""
    english_labels = {"System Starting", "Fetching Games Data", "Loading Models",
                      "Running Monte Carlo Simulation", "Saving Results", "Completed"}
    for _, label in ProgressTracker.STAGES:
        assert label not in english_labels, f"English label '{label}' still in STAGES"


def test_progress_tracker_build_text_no_english():
    """_build_text output contains no hardcoded English status text."""
    tracker = ProgressTracker()
    tracker.current_stage = 0
    text = tracker._build_text()
    assert "System Starting" not in text
    assert "Fetching Games Data" not in text
    assert "Loading Models" not in text
    assert "NBA Quant System" not in text
    assert "量化预测系统" in text


# ---------- ProgressTracker: Supabase stage filtering ----------

def test_progress_tracker_supabase_skips_boot_stages():
    """When model_source is 'supabase', only load_model, simulation, done are visible."""
    tracker = ProgressTracker()
    tracker.model_source = "supabase"
    visible = tracker._visible_stages()
    visible_indices = {idx for idx, _, _ in visible}
    assert visible_indices == {2, 3, 5}


def test_progress_tracker_default_shows_all_stages():
    """When model_source is None, all stages are visible."""
    tracker = ProgressTracker()
    visible = tracker._visible_stages()
    assert len(visible) == len(ProgressTracker.STAGES)


def test_progress_tracker_non_supabase_shows_all_stages():
    """When model_source is not 'supabase', all stages are visible."""
    tracker = ProgressTracker()
    tracker.model_source = "cached"
    visible = tracker._visible_stages()
    assert len(visible) == len(ProgressTracker.STAGES)


def test_progress_tracker_supabase_build_text_excludes_start():
    """When model_source is 'supabase', _build_text omits start/fetch_data/saving."""
    tracker = ProgressTracker()
    tracker.model_source = "supabase"
    tracker.current_stage = 2
    text = tracker._build_text()
    assert TEXT["start"] not in text
    assert TEXT["fetch_data"] not in text
    assert TEXT["saving"] not in text
    assert TEXT["load_model"] in text
    assert TEXT["simulation"] in text
    assert TEXT["done"] in text
