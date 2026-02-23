"""Supabase primary persistent storage.

If SUPABASE_URL and SUPABASE_KEY are set, data is persisted to Supabase.
When configured, write failures raise exceptions so the workflow fails loudly.
If credentials are not set, calls are skipped (SQLite acts as local cache).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_client: Any = None
_available: bool | None = None


def _get_client() -> Any:
    global _client, _available
    if _available is False:
        return None
    if _client is not None:
        return _client
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        logger.debug("Supabase credentials not configured — skipping")
        _available = False
        return None
    from supabase import create_client  # type: ignore
    _client = create_client(url, key)
    _available = True
    logger.info("Supabase client initialized")
    _ensure_tables()
    return _client


def _ensure_tables() -> None:
    """Verify that required tables exist by attempting a select.

    Tables must be created in Supabase dashboard or via migration.
    This only logs whether they are accessible.
    """
    client = _client
    if client is None:
        return
    for table in ("predictions", "simulation_logs", "training_logs", "review_results"):
        try:
            client.table(table).select("*").limit(1).execute()
            logger.info("Supabase table '%s' accessible", table)
        except Exception:
            logger.warning("Supabase table '%s' not accessible — create it in Supabase dashboard", table)


def save_prediction(row: dict[str, Any]) -> None:
    """Persist a prediction row to Supabase.  Raises on failure."""
    client = _get_client()
    if client is None:
        return
    record = {
        "game_id": row.get("game_id"),
        "game_date": row.get("game_date"),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "spread_line": row.get("spread_line"),
        "total_line": row.get("total_line"),
        "spread_pick": row.get("spread_pick"),
        "total_pick": row.get("total_pick"),
        "spread_confidence": row.get("spread_confidence"),
        "total_confidence": row.get("total_confidence"),
        "model_version": row.get("model_version", "v1"),
        "simulation_runs": row.get("simulation_runs", 10000),
        "spread_edge": row.get("spread_edge", 0.0),
        "total_edge": row.get("total_edge", 0.0),
        "expected_home_score": row.get("expected_home_score"),
        "expected_visitor_score": row.get("expected_visitor_score"),
        "confidence_score": row.get("confidence_score"),
        "star_rating": row.get("star_rating"),
        "odds_source": row.get("odds_source"),
        "feature_count": row.get("feature_count", 0),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "details_json": json.dumps(row.get("details", {}), ensure_ascii=False),
    }
    client.table("predictions").insert(record).execute()
    logger.info("Supabase: prediction saved for game %s", row.get("game_id"))


def save_simulation_log(row: dict[str, Any]) -> None:
    """Persist a simulation log to Supabase.  Raises on failure."""
    client = _get_client()
    if client is None:
        return
    record = {
        "game_id": row.get("game_id"),
        "model_version": row.get("model_version", "v1"),
        "simulation_runs": row.get("simulation_runs", 10000),
        "spread_cover_probability": row.get("spread_cover_probability"),
        "over_probability": row.get("over_probability"),
        "expected_home_score": row.get("expected_home_score"),
        "expected_visitor_score": row.get("expected_visitor_score"),
        "score_distribution_variance": row.get("score_distribution_variance"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    client.table("simulation_logs").insert(record).execute()


def save_training_log(row: dict[str, Any]) -> None:
    """Persist a training log to Supabase.  Raises on failure."""
    client = _get_client()
    if client is None:
        return
    record = {
        "model_version": row.get("model_version", "v1"),
        "feature_count": row.get("feature_count", 0),
        "algorithm": row.get("algorithm"),
        "data_points": row.get("data_points", 0),
        "home_mae": row.get("home_mae"),
        "home_rmse": row.get("home_rmse"),
        "away_mae": row.get("away_mae"),
        "away_rmse": row.get("away_rmse"),
        "training_seconds": row.get("training_seconds"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    client.table("training_logs").insert(record).execute()
    logger.info("Supabase: training log saved for version %s", row.get("model_version"))


def save_review_result(row: dict[str, Any]) -> None:
    """Persist a review result to Supabase.  Raises on failure."""
    client = _get_client()
    if client is None:
        return
    record = {
        "game_id": row.get("game_id"),
        "game_date": row.get("game_date"),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "spread_pick": row.get("spread_pick"),
        "total_pick": row.get("total_pick"),
        "spread_correct": row.get("spread_correct"),
        "total_correct": row.get("total_correct"),
        "final_home_score": row.get("final_home_score"),
        "final_visitor_score": row.get("final_visitor_score"),
        "spread_rate": row.get("spread_rate"),
        "total_rate": row.get("total_rate"),
        "roi": row.get("roi"),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    client.table("review_results").insert(record).execute()
    logger.info("Supabase: review result saved for game %s", row.get("game_id"))
