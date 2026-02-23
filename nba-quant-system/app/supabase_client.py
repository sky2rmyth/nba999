"""Optional Supabase persistent storage.

If SUPABASE_URL and SUPABASE_KEY are set, data is persisted to Supabase.
Otherwise all calls silently no-op so the system keeps working with SQLite.
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
    try:
        from supabase import create_client  # type: ignore
        _client = create_client(url, key)
        _available = True
        logger.info("Supabase client initialized")
        _ensure_tables()
        return _client
    except Exception:
        logger.warning("Supabase client init failed", exc_info=True)
        _available = False
        return None


def _ensure_tables() -> None:
    """Verify that required tables exist by attempting a select.

    Tables must be created in Supabase dashboard or via migration.
    This only logs whether they are accessible.
    """
    client = _client
    if client is None:
        return
    for table in ("predictions", "simulation_logs", "training_logs"):
        try:
            client.table(table).select("*").limit(1).execute()
            logger.info("Supabase table '%s' accessible", table)
        except Exception:
            logger.warning("Supabase table '%s' not accessible — create it in Supabase dashboard", table)


def save_prediction(row: dict[str, Any]) -> None:
    """Persist a prediction row to Supabase."""
    client = _get_client()
    if client is None:
        return
    try:
        record = {
            "game_id": row.get("game_id"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "model_version": row.get("model_version", "v1"),
            "feature_count": row.get("feature_count", 0),
            "simulation_runs": row.get("simulation_runs", 10000),
            "spread_edge": row.get("spread_edge", 0.0),
            "total_edge": row.get("total_edge", 0.0),
            "spread_pick": row.get("spread_pick"),
            "total_pick": row.get("total_pick"),
            "expected_home_score": row.get("expected_home_score"),
            "expected_visitor_score": row.get("expected_visitor_score"),
            "confidence_score": row.get("confidence_score"),
            "star_rating": row.get("star_rating"),
            "odds_source": row.get("odds_source"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details_json": json.dumps(row.get("details", {}), ensure_ascii=False),
        }
        client.table("predictions").insert(record).execute()
        logger.info("Supabase: prediction saved for game %s", row.get("game_id"))
    except Exception:
        logger.warning("Supabase: failed to save prediction", exc_info=True)


def save_simulation_log(row: dict[str, Any]) -> None:
    """Persist a simulation log to Supabase."""
    client = _get_client()
    if client is None:
        return
    try:
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
    except Exception:
        logger.warning("Supabase: failed to save simulation log", exc_info=True)


def save_training_log(row: dict[str, Any]) -> None:
    """Persist a training log to Supabase."""
    client = _get_client()
    if client is None:
        return
    try:
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
    except Exception:
        logger.warning("Supabase: failed to save training log", exc_info=True)
