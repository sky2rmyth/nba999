from __future__ import annotations

import logging

from .data_pipeline import bootstrap_historical_data
from .database import get_conn, init_db
from .feature_engineering import FEATURE_COLUMNS, build_training_frame
from .i18n_cn import cn
from .prediction_models import load_models, train_models, _current_version, MODEL_DIR
from .rating_engine import is_spread_correct, is_total_correct

logger = logging.getLogger(__name__)

PERFORMANCE_THRESHOLD = 0.45  # retrain if accuracy drops below 45%
MIN_RETRAIN_GAMES = 50  # minimum new finished games before retraining


def _try_send_telegram(text: str) -> None:
    """Send Telegram message, silently ignore failures."""
    try:
        from .telegram_bot import send_message
        send_message(text)
    except Exception:
        logger.debug("Telegram send skipped: %s", text)


def _db_has_completed_games() -> bool:
    init_db()
    with get_conn() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM games WHERE status LIKE 'Final%'"
        ).fetchone()[0]
    return count > 0


def _check_performance_degradation() -> bool:
    """Check if recent prediction accuracy dropped below threshold."""
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT p.spread_pick, p.total_pick, p.live_spread, p.live_total,
                       r.final_home_score, r.final_visitor_score
                FROM predictions_snapshot p
                JOIN results r ON p.game_id = r.game_id
                WHERE p.is_final_prediction = 1
                ORDER BY p.created_at DESC LIMIT 30
                """
            ).fetchall()
        if len(rows) < 10:
            return False
        hits = 0
        total = 0
        for r in rows:
            margin = r["final_home_score"] - r["final_visitor_score"]
            total_pts = r["final_home_score"] + r["final_visitor_score"]
            if r["live_spread"] is not None:
                hits += int(is_spread_correct(r["spread_pick"], margin, r["live_spread"]))
                total += 1
            if r["live_total"] is not None:
                hits += int(is_total_correct(r["total_pick"], total_pts, r["live_total"]))
                total += 1
        if total == 0:
            return False
        accuracy = hits / total
        logger.info("Recent accuracy: %.1f%% (%d/%d)", accuracy * 100, hits, total)
        if accuracy < PERFORMANCE_THRESHOLD:
            logger.info("Performance degradation detected (%.1f%% < %.1f%%) — triggering retrain",
                        accuracy * 100, PERFORMANCE_THRESHOLD * 100)
            return True
    except Exception:
        logger.debug("Performance check failed", exc_info=True)
    return False


def _count_new_finished_games() -> int:
    """Return the number of finished games not yet reviewed."""
    try:
        with get_conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) c FROM games g LEFT JOIN results r "
                "ON g.game_id=r.game_id "
                "WHERE g.status LIKE 'Final%%' AND r.game_id IS NULL"
            ).fetchone()[0]
    except Exception:
        return 0


def should_retrain(force: bool = False) -> bool:
    if force:
        return True
    if _check_performance_degradation():
        return True
    new_games = _count_new_finished_games()
    return new_games >= MIN_RETRAIN_GAMES or load_models() is None


def ensure_models(force: bool = False):
    # Step 1: If models exist locally and not forced, use cached
    cached = load_models()
    if cached is not None and not force:
        logger.info("MODEL SOURCE: Using cached model | Version: %s", cached.version)
        _try_send_telegram(cn("model_cached"))
        cached.source = "cached"
        return cached

    # Step 2: Try restoring from Supabase Storage before training
    try:
        from .supabase_client import download_models_from_storage
        if download_models_from_storage(MODEL_DIR):
            restored = load_models()
            if restored is not None:
                logger.info("MODEL SOURCE: Loaded from Supabase | Version: %s", restored.version)
                _try_send_telegram(cn("model_loaded"))
                restored.source = "supabase"
                return restored
    except Exception:
        logger.debug("Supabase Storage restore failed", exc_info=True)

    # Step 3: Check if enough new games to justify retraining
    new_games = _count_new_finished_games()
    if cached is not None and new_games < MIN_RETRAIN_GAMES:
        logger.info("Only %d new finished games (< %d) — reusing cached model",
                     new_games, MIN_RETRAIN_GAMES)
        _try_send_telegram(cn("skip_retrain_insufficient", min_games=MIN_RETRAIN_GAMES))
        cached.source = "cached"
        return cached

    # Step 4: Train new models + upload to Supabase
    _try_send_telegram(cn("training_start"))

    if not _db_has_completed_games():
        logger.info("FIRST RUN TRAINING STARTED")
        bootstrap_historical_data()
    logger.info("Building training dataset...")
    _try_send_telegram(cn("building_dataset"))
    df = build_training_frame()
    if df.empty:
        raise RuntimeError("No historical data available for training")

    feature_count = len(FEATURE_COLUMNS)
    sample_count = len(df)
    _try_send_telegram(cn("feature_count", feature_count=feature_count))
    _try_send_telegram(cn("sample_count", sample_count=sample_count))

    logger.info("Training models...")
    bundle = train_models(df)

    # Send training report to Telegram
    duration = getattr(bundle, "duration", 0)
    sc_acc = bundle.metrics.get("spread_cover_accuracy", 0)
    to_acc = bundle.metrics.get("total_over_accuracy", 0)
    report = cn("training_report",
                version=bundle.version,
                sample_count=sample_count,
                feature_count=feature_count,
                duration=duration,
                sc_acc=sc_acc,
                to_acc=to_acc)
    _try_send_telegram(report)
    _try_send_telegram(cn("training_done"))

    # --- Log training to Supabase ---
    from .supabase_client import save_training_log
    save_training_log({
        "model_version": bundle.version,
        "feature_count": feature_count,
        "algorithm": bundle.algorithm,
        "data_points": sample_count,
        "home_mae": bundle.metrics.get("home_mae"),
        "home_rmse": bundle.metrics.get("home_rmse"),
        "away_mae": bundle.metrics.get("away_mae"),
        "away_rmse": bundle.metrics.get("away_rmse"),
        "training_seconds": duration,
    })

    # --- Upload models to Supabase Storage ---
    from .supabase_client import upload_models_to_storage
    logger.info("Uploading models to Supabase Storage...")
    _try_send_telegram(cn("upload_models"))
    if not upload_models_to_storage(MODEL_DIR):
        raise RuntimeError("Failed to upload models to Supabase Storage")
    logger.info("Models upload complete")

    logger.info("MODEL SOURCE: Trained new model | Algorithm: %s | Version: %s", bundle.algorithm, bundle.version)
    _try_send_telegram(cn("model_trained"))
    bundle.source = "trained"
    return bundle
