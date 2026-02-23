"""Supabase primary persistent storage.

If SUPABASE_URL and SUPABASE_KEY are set, data is persisted to Supabase.
When configured, write failures raise exceptions so the workflow fails loudly.
If credentials are not set, calls are skipped (SQLite acts as local cache).
"""
from __future__ import annotations

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
    """Persist a prediction row to Supabase.

    All prediction data is stored inside a single ``payload`` JSONB column.
    Previous predictions for the same game_id are marked as non-final.
    Errors are caught so the prediction pipeline never crashes if logging fails.
    """
    client = _get_client()
    if client is None:
        return
    record = dict(row)
    record.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    record["is_final_prediction"] = True
    try:
        # Mark previous predictions for this game as non-final
        client.table("predictions").update(
            {"payload": {"is_final_prediction": False}}
        ).eq("game_id", record["game_id"]).execute()
    except Exception:
        logger.debug("Supabase: could not update previous predictions — continuing")
    try:
        client.table("predictions").insert({
            "game_id": record["game_id"],
            "payload": record,
        }).execute()
        logger.info("Supabase: prediction saved for game %s", row.get("game_id"))
    except Exception:
        logger.exception("Supabase: failed to save prediction — continuing")


def save_simulation_log(row: dict[str, Any]) -> None:
    """Persist a simulation log to Supabase.

    All data is stored inside ``game_id`` + ``payload`` JSONB columns.
    Errors are caught so the prediction pipeline never crashes if logging fails.
    """
    client = _get_client()
    if client is None:
        return
    record = dict(row)
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    try:
        client.table("simulation_logs").insert({
            "game_id": record["game_id"],
            "payload": record,
        }).execute()
        logger.info("Supabase: simulation log saved for game %s", row.get("game_id"))
    except Exception:
        logger.exception("Supabase: failed to save simulation log — continuing")


def save_training_log(row: dict[str, Any]) -> None:
    """Persist a training log to Supabase.

    All data is stored inside a single ``payload`` JSONB column.
    Errors are caught so the prediction pipeline never crashes if logging fails.
    """
    client = _get_client()
    if client is None:
        return
    record = dict(row)
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    try:
        client.table("training_logs").insert({"payload": record}).execute()
        logger.info("Supabase: training log saved for version %s", row.get("model_version"))
    except Exception:
        logger.exception("Supabase: failed to save training log — continuing")


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


def fetch_latest_training_metrics() -> dict[str, Any] | None:
    """Fetch the latest training log payload from Supabase.

    Returns the payload dict or None if unavailable.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        resp = client.table("training_logs").select("payload").order(
            "id", desc=True
        ).limit(1).execute()
        if resp.data:
            return resp.data[0].get("payload")
    except Exception:
        logger.debug("Supabase: could not fetch latest training metrics")
    return None
