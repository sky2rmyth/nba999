"""Model status reporter — sends current model state to Telegram."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .prediction_models import MODEL_DIR, MODEL_FILES, VERSION_FILE, _current_version
from .telegram_bot import send_message

logger = logging.getLogger(__name__)

# In-memory cache: loaded once per execution.
_cached_status: dict | None = None


def _load_model_status() -> dict:
    """Build model status from models/ directory and Supabase training_logs.

    Result is cached in ``_cached_status`` so it is only computed once per run.
    """
    global _cached_status
    if _cached_status is not None:
        return _cached_status

    # --- Version from model_version.json ---
    version = _current_version()

    # --- Verify model files ---
    missing = [f for f in MODEL_FILES if not (MODEL_DIR / f).exists()]
    model_available = len(missing) == 0

    # --- Latest metrics: prefer Supabase, fall back to local DB ---
    metrics: dict = {}
    training_samples = 0
    last_trained = "N/A"

    try:
        from .supabase_client import fetch_latest_training_metrics
        supa_metrics = fetch_latest_training_metrics()
        if supa_metrics:
            metrics = supa_metrics
            training_samples = supa_metrics.get("data_points", 0)
            last_trained = supa_metrics.get("timestamp", "N/A")
    except Exception:
        logger.debug("Supabase training metrics unavailable")

    if not metrics:
        try:
            from .database import get_conn, init_db
            init_db()
            with get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM model_history ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row:
                    last_trained = row["trained_at"]
                    training_samples = row["data_points"]
                    metrics = json.loads(row["metrics_json"])
        except Exception:
            logger.debug("Could not fetch model history from local DB")

    _cached_status = {
        "version": version,
        "model_available": model_available,
        "training_samples": training_samples,
        "last_trained": last_trained,
        "metrics": metrics,
    }
    return _cached_status


def get_model_status() -> dict:
    """Return cached model status dict (loads once per execution)."""
    return _load_model_status()


def run_model_status() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    status = get_model_status()
    version = status["version"]
    model_available = status["model_available"]
    training_samples = status["training_samples"]
    last_trained = status["last_trained"]
    metrics = status["metrics"]

    home_mae = metrics.get("home_mae", "N/A")
    away_mae = metrics.get("away_mae", "N/A")
    sc_acc = metrics.get("spread_cover_accuracy", "N/A")
    to_acc = metrics.get("total_over_accuracy", "N/A")

    if isinstance(home_mae, float):
        home_mae = f"{home_mae:.2f}"
    if isinstance(away_mae, float):
        away_mae = f"{away_mae:.2f}"

    # Combined MAE display
    if home_mae != "N/A" and away_mae != "N/A":
        mae_display = f"{home_mae} / {away_mae}"
    else:
        mae_display = "N/A"

    if isinstance(sc_acc, float):
        sc_acc = f"{sc_acc:.1%}"
    if isinstance(to_acc, float):
        to_acc = f"{to_acc:.1%}"

    msg = (
        f"📈 模型状态报告\n\n"
        f"版本: {version}\n"
        f"模型可用: {'✅' if model_available else '❌'}\n"
        f"训练样本: {training_samples}\n"
        f"MAE: {mae_display}\n"
        f"让分覆盖准确率: {sc_acc}\n"
        f"大小分准确率: {to_acc}\n"
        f"最后训练: {last_trained}"
    )
    send_message(msg)
    logger.info("Model status sent to Telegram")


if __name__ == "__main__":
    run_model_status()
