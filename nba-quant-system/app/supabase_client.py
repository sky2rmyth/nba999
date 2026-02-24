"""Supabase primary persistent storage.

If SUPABASE_URL and SUPABASE_KEY are set, data is persisted to Supabase.
When configured, write failures raise exceptions so the workflow fails loudly.
If credentials are not set, calls are silently skipped.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

_client: Any = None
_available: bool | None = None
_table_columns_cache: dict[str, set[str] | None] = {}


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


def _get_table_columns(table: str) -> set[str] | None:
    """Discover column names for *table* from ``information_schema.columns``.

    Uses a Supabase RPC call to query the real PostgreSQL schema so that
    columns are returned even when the table is empty.  The result is
    cached so the query runs at most once per process.
    """
    if table in _table_columns_cache:
        return _table_columns_cache[table]

    client = _get_client()
    if client is None:
        return None

    try:
        sql = (
            "SELECT column_name FROM information_schema.columns"
            f" WHERE table_name = '{table}'"
        )
        resp = client.rpc("sql", {"query": sql}).execute()
        if resp.data:
            columns = {row["column_name"] for row in resp.data}
            _table_columns_cache[table] = columns
            logger.debug("Discovered columns for '%s': %s", table, columns)
            return columns
    except Exception:
        logger.debug("Could not discover columns for table '%s'", table, exc_info=True)

    _table_columns_cache[table] = None
    return None


def get_table_columns(table: str) -> set[str]:
    """Return known column names for *table*.

    Public wrapper around ``_get_table_columns``.  Returns an empty set when
    the columns cannot be determined (table empty or Supabase unavailable).
    """
    return _get_table_columns(table) or set()


def adaptive_upsert(
    table: str,
    record: dict[str, Any],
    conflict: str = "game_id",
) -> None:
    """Upsert *record* into *table*, automatically filtering unknown columns.

    Only fields whose names match existing columns (discovered from
    ``information_schema``) are written.  When column discovery fails,
    no data is written to prevent crashes from unknown column names.
    """
    client = _get_client()
    if client is None:
        return

    cols = get_table_columns(table)
    filtered = {k: v for k, v in record.items() if k in cols}

    if not filtered:
        logger.warning("No valid columns to write for table '%s'. Attempted: %s", table, list(record.keys()))
        return

    client.table(table).upsert(filtered, on_conflict=conflict).execute()


def save_prediction(row: dict[str, Any]) -> None:
    """Persist a prediction row to Supabase.

    All prediction data is stored inside a single ``payload`` JSONB column.
    The ``is_final_prediction`` flag is set to ``True`` in the payload so
    consumers can identify the latest prediction per game.
    Errors are caught so the prediction pipeline never crashes if logging fails.
    """
    client = _get_client()
    if client is None:
        return
    record = dict(row)
    record.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    record["is_final_prediction"] = True
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
    """Persist a review result to Supabase via UPSERT on game_id.

    Each game_id appears only once in the ``review_results`` table.  If a
    result already exists for the game, it is updated instead of duplicated.
    Uses ``adaptive_upsert`` so that unknown columns are silently dropped
    and schema changes never crash the workflow.
    """
    record = dict(row)
    record.setdefault("reviewed_at", datetime.now(timezone.utc).isoformat())

    try:
        adaptive_upsert("review_results", record)
        logger.info("Supabase: review result saved for game %s", row.get("game_id"))
    except Exception:
        logger.exception("Supabase: failed to save review result for game %s — continuing", row.get("game_id"))


def fetch_recent_review_results(days: int = 30) -> list[dict[str, Any]]:
    """Fetch review results from the last *days* days from Supabase.

    Returns a list of review result dicts or an empty list.
    """
    client = _get_client()
    if client is None:
        return []
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        resp = (
            client.table("review_results")
            .select("*")
            .gte("reviewed_at", cutoff)
            .execute()
        )
        return resp.data or []
    except Exception:
        logger.debug("Supabase: could not fetch recent review results", exc_info=True)
    return []


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


def fetch_predictions_for_date(game_date: str) -> list[dict[str, Any]]:
    """Fetch predictions for a given date from Supabase.

    Returns a list of prediction payload dicts or an empty list.
    """
    client = _get_client()
    if client is None:
        return []
    try:
        resp = client.table("predictions").select("payload").execute()
        results = []
        for row in resp.data or []:
            payload = row.get("payload") or {}
            if payload.get("game_date") == game_date and payload.get("is_final_prediction"):
                results.append(payload)
        return results
    except Exception:
        logger.debug("Supabase: could not fetch predictions for %s", game_date)
    return []


def fetch_all_predictions() -> list[dict[str, Any]]:
    """Fetch all prediction records from Supabase ordered by created_at desc.

    Results are ordered newest-first so that client-side deduplication
    (``_deduplicate_predictions``) retains only the latest prediction per
    ``game_id``.

    Returns a list of raw rows (each containing ``id``, ``game_id``,
    ``payload``, and optionally ``game_date``).
    """
    client = _get_client()
    if client is None:
        return []
    try:
        resp = client.table("predictions").select("*").order("created_at", desc=True).execute()
        return resp.data or []
    except Exception:
        logger.debug("Supabase: could not fetch all predictions")
    return []


def update_prediction_game_date(record_id: int, game_date: str) -> None:
    """Set the ``game_date`` column for a prediction row identified by *record_id*."""
    client = _get_client()
    if client is None:
        return
    client.table("predictions").update(
        {"game_date": game_date}
    ).eq("id", record_id).execute()
    logger.info("Supabase: updated game_date=%s for prediction id=%s", game_date, record_id)


# ---------------------------------------------------------------------------
# Supabase Storage helpers for persistent model files
# ---------------------------------------------------------------------------

_STORAGE_BUCKET = "models"
_MODEL_STORAGE_FILES = (
    "home_model.pkl",
    "away_model.pkl",
    "spread_model.pkl",
    "total_model.pkl",
    "model_version.json",
)


def upload_models_to_storage(model_dir: str | os.PathLike) -> bool:
    """Upload trained model files to Supabase Storage bucket.

    Returns True if all files were uploaded successfully, False otherwise.
    """
    from pathlib import Path

    client = _get_client()
    if client is None:
        logger.debug("Supabase not configured — model upload skipped")
        return False
    model_path = Path(model_dir)
    success = True
    for filename in _MODEL_STORAGE_FILES:
        filepath = model_path / filename
        if not filepath.exists():
            logger.warning("Model file %s not found — skipping upload", filename)
            continue
        try:
            data = filepath.read_bytes()
            # Remove existing file first (upsert)
            try:
                client.storage.from_(_STORAGE_BUCKET).remove([filename])
            except Exception:
                logger.debug("Supabase Storage: could not remove existing %s", filename)
            client.storage.from_(_STORAGE_BUCKET).upload(filename, data)
            logger.info("Supabase Storage: uploaded %s", filename)
        except Exception:
            logger.exception("Supabase Storage: failed to upload %s", filename)
            success = False
    return success


def download_models_from_storage(model_dir: str | os.PathLike) -> bool:
    """Download model files from Supabase Storage bucket.

    Returns True if all required model files were downloaded, False otherwise.
    """
    from pathlib import Path

    client = _get_client()
    if client is None:
        logger.debug("Supabase not configured — model download skipped")
        return False
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for filename in _MODEL_STORAGE_FILES:
        try:
            data = client.storage.from_(_STORAGE_BUCKET).download(filename)
            (model_path / filename).write_bytes(data)
            logger.info("Supabase Storage: downloaded %s", filename)
            downloaded += 1
        except Exception:
            logger.warning("Supabase Storage: %s not available", filename)
    # At minimum the two score models must be present
    required = ("home_model.pkl", "away_model.pkl")
    ok = all((model_path / f).exists() for f in required)
    if ok:
        logger.info("Supabase Storage: models restored (%d files)", downloaded)
    return ok
