from __future__ import annotations

import logging

from .data_pipeline import bootstrap_historical_data
from .database import get_conn, init_db
from .feature_engineering import build_training_frame
from .prediction_models import load_models, train_models

logger = logging.getLogger(__name__)


def _db_has_completed_games() -> bool:
    init_db()
    with get_conn() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM games WHERE status LIKE 'Final%'"
        ).fetchone()[0]
    return count > 0


def should_retrain(force: bool = False) -> bool:
    if force:
        return True
    with get_conn() as conn:
        pending = conn.execute(
            "SELECT COUNT(*) c FROM games g LEFT JOIN results r ON g.game_id=r.game_id WHERE g.status LIKE 'Final%' AND r.game_id IS NULL"
        ).fetchone()[0]
    return pending >= 50 or load_models() is None


def ensure_models(force: bool = False):
    first_run = load_models() is None
    if should_retrain(force=force):
        if first_run:
            logger.info("FIRST RUN TRAINING STARTED")
            if not _db_has_completed_games():
                bootstrap_historical_data()
        logger.info("Building training dataset...")
        df = build_training_frame()
        if df.empty:
            raise RuntimeError("No historical data available for training")
        logger.info("Training models...")
        bundle = train_models(df)
        logger.info("Training executed: YES | Algorithm: %s", bundle.algorithm)
        return bundle
    bundle = load_models()
    if bundle is None:
        raise RuntimeError("Model files missing")
    logger.info("Training executed: NO | Models loaded from disk")
    return bundle
