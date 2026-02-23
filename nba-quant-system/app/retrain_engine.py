from __future__ import annotations

import logging

from .data_pipeline import bootstrap_historical_data
from .database import get_conn, init_db
from .feature_engineering import FEATURE_COLUMNS, build_training_frame
from .prediction_models import load_models, train_models

logger = logging.getLogger(__name__)


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
        _try_send_telegram("构建训练数据集...")
        df = build_training_frame()
        if df.empty:
            raise RuntimeError("No historical data available for training")

        feature_count = len(FEATURE_COLUMNS)
        sample_count = len(df)
        _try_send_telegram(f"特征数量: {feature_count}")
        _try_send_telegram(f"训练样本数: {sample_count}")

        logger.info("Training models...")
        bundle = train_models(df)

        # Send training report to Telegram
        duration = getattr(bundle, "duration", 0)
        report = (
            "📊 模型训练报告\n\n"
            "训练方式: Regression Score Model\n"
            f"模型: LightGBM\n"
            f"训练样本: {sample_count}\n"
            f"特征数量: {feature_count}\n"
            f"训练耗时: {duration:.1f} 秒\n"
            "主队得分模型: 完成\n"
            "客队得分模型: 完成"
        )
        _try_send_telegram(report)

        logger.info("Training executed: YES | Algorithm: %s", bundle.algorithm)
        return bundle
    bundle = load_models()
    if bundle is None:
        raise RuntimeError("Model files missing")
    logger.info("Training executed: NO | Models loaded from disk")
    return bundle
