from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from . import database
from .feature_engineering import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelBundle:
    def __init__(self, home_score_model, away_score_model, algorithm: str):
        self.home_score_model = home_score_model
        self.away_score_model = away_score_model
        self.algorithm = algorithm


def _build_regressor():
    try:
        from lightgbm import LGBMRegressor  # type: ignore

        return "lightgbm", LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        ), LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        return "gradient_boosting", GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42,
        ), GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42,
        )


def train_models(df: pd.DataFrame) -> ModelBundle:
    feature_count = len(FEATURE_COLUMNS)
    sample_count = len(df)
    logger.info("Training samples: %d", sample_count)
    logger.info("Feature count: %d", feature_count)

    if feature_count < 30:
        raise RuntimeError(
            f"Feature count {feature_count} < 30 — pipeline requires minimum 30 features"
        )

    X = df[FEATURE_COLUMNS].values
    y_home = df["home_score"].values
    y_away = df["away_score"].values

    alg, home_model, away_model = _build_regressor()

    X_train, X_test, yh_train, yh_test = train_test_split(
        X, y_home, test_size=0.2, random_state=42
    )
    _, _, ya_train, ya_test = train_test_split(
        X, y_away, test_size=0.2, random_state=42
    )

    start_time = time.time()

    home_model.fit(X_train, yh_train)
    away_model.fit(X_train, ya_train)

    duration = round(time.time() - start_time, 2)
    logger.info("Training duration: %.2f seconds", duration)

    # Evaluation metrics
    home_pred = home_model.predict(X_test)
    away_pred = away_model.predict(X_test)
    metrics = {
        "home_mae": float(mean_absolute_error(yh_test, home_pred)),
        "home_rmse": float(np.sqrt(mean_squared_error(yh_test, home_pred))),
        "away_mae": float(mean_absolute_error(ya_test, away_pred)),
        "away_rmse": float(np.sqrt(mean_squared_error(ya_test, away_pred))),
        "training_seconds": duration,
    }
    logger.info("Home Score Model MAE: %.2f  RMSE: %.2f", metrics["home_mae"], metrics["home_rmse"])
    logger.info("Away Score Model MAE: %.2f  RMSE: %.2f", metrics["away_mae"], metrics["away_rmse"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "home_score_model.pkl", "wb") as f:
        pickle.dump(home_model, f)
    with open(MODEL_DIR / "away_score_model.pkl", "wb") as f:
        pickle.dump(away_model, f)

    database.log_model("home_score", alg, sample_count, metrics, str(MODEL_DIR / "home_score_model.pkl"))
    database.log_model("away_score", alg, sample_count, metrics, str(MODEL_DIR / "away_score_model.pkl"))

    bundle = ModelBundle(home_model, away_model, alg)
    bundle.metrics = metrics
    bundle.sample_count = sample_count
    bundle.feature_count = feature_count
    bundle.duration = duration
    return bundle


def load_models() -> ModelBundle | None:
    hp = MODEL_DIR / "home_score_model.pkl"
    ap = MODEL_DIR / "away_score_model.pkl"
    if not hp.exists() or not ap.exists():
        return None
    with open(hp, "rb") as f:
        home = pickle.load(f)
    with open(ap, "rb") as f:
        away = pickle.load(f)
    return ModelBundle(home, away, "loaded")
