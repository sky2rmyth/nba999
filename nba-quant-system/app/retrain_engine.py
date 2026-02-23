from __future__ import annotations

from .database import get_conn
from .feature_engineering import build_training_frame
from .prediction_models import load_models, train_models


def should_retrain(force: bool = False) -> bool:
    if force:
        return True
    with get_conn() as conn:
        pending = conn.execute(
            "SELECT COUNT(*) c FROM games g LEFT JOIN results r ON g.game_id=r.game_id WHERE g.status LIKE 'Final%' AND r.game_id IS NULL"
        ).fetchone()[0]
    return pending >= 50 or load_models() is None


def ensure_models(force: bool = False):
    if should_retrain(force=force):
        df = build_training_frame()
        if df.empty:
            raise RuntimeError("No historical data available for training")
        return train_models(df)
    bundle = load_models()
    if bundle is None:
        raise RuntimeError("Model files missing")
    return bundle
