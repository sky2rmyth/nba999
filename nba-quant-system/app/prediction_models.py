from __future__ import annotations

import json
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
VERSION_FILE = MODEL_DIR / "model_version.json"


MODEL_FILES = ("home_model.pkl", "away_model.pkl", "spread_model.pkl", "total_model.pkl")


def _current_version() -> str:
    """Read persisted model version string."""
    if VERSION_FILE.exists():
        try:
            return json.loads(VERSION_FILE.read_text()).get("version", "unknown")
        except Exception:
            pass
    return "unknown"


def _bump_version() -> str:
    """Increment and persist the model version."""
    cur = _current_version()
    try:
        num = int(cur.lstrip("v")) + 1
    except ValueError:
        num = 2
    new_ver = f"v{num}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VERSION_FILE.write_text(json.dumps({"version": new_ver}))
    return new_ver


class ModelBundle:
    def __init__(self, home_score_model, away_score_model, algorithm: str,
                 spread_cover_model=None, total_model=None):
        self.home_score_model = home_score_model
        self.away_score_model = away_score_model
        self.spread_cover_model = spread_cover_model
        self.total_model = total_model
        self.algorithm = algorithm
        self.version = _current_version()
        self.metrics: dict = {}
        self.sample_count: int = 0
        self.feature_count: int = 0
        self.duration: float = 0.0
        self.source: str = "unknown"


def _build_regressor():
    lgbm_params = dict(
        n_estimators=300, learning_rate=0.05, max_depth=8,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    try:
        from lightgbm import LGBMRegressor  # type: ignore

        return "lightgbm", LGBMRegressor(**lgbm_params), LGBMRegressor(**lgbm_params)
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        gb_params = dict(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
        return "gradient_boosting", GradientBoostingRegressor(**gb_params), GradientBoostingRegressor(**gb_params)


def _build_classifier():
    """Build binary classifiers for spread cover and total over/under."""
    lgbm_params = dict(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    try:
        from lightgbm import LGBMClassifier  # type: ignore
        return LGBMClassifier(**lgbm_params), LGBMClassifier(**lgbm_params)
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        gb_params = dict(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
        return GradientBoostingClassifier(**gb_params), GradientBoostingClassifier(**gb_params)


def train_models(df: pd.DataFrame) -> ModelBundle:
    feature_count = len(FEATURE_COLUMNS)
    sample_count = len(df)
    logger.info("Training samples: %d", sample_count)
    logger.info("Feature count: %d", feature_count)

    if feature_count < 30:
        raise RuntimeError(
            f"Feature count {feature_count} < 30 — pipeline requires minimum 30 features"
        )

    new_version = _bump_version()
    logger.info("Model version: %s", new_version)

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

    # --- Hybrid: Spread Cover classifier ---
    # Uses home win as proxy for spread cover since historical spread lines
    # are not available in training data. The classifier learns team strength
    # patterns that correlate with covering; Monte Carlo handles the actual
    # spread line comparison at prediction time.
    spread_cover_model, total_model = _build_classifier()
    margin = df["home_score"].values - df["away_score"].values
    y_spread_cover = (margin > 0).astype(int)
    total_points = df["home_score"].values + df["away_score"].values
    median_total = float(np.median(total_points))
    y_total_over = (total_points > median_total).astype(int)

    Xsc_train, Xsc_test, ysc_train, ysc_test = train_test_split(
        X, y_spread_cover, test_size=0.2, random_state=42
    )
    Xto_train, Xto_test, yto_train, yto_test = train_test_split(
        X, y_total_over, test_size=0.2, random_state=42
    )
    spread_cover_model.fit(Xsc_train, ysc_train)
    total_model.fit(Xto_train, yto_train)

    duration = round(time.time() - start_time, 2)
    logger.info("Training duration: %.2f seconds", duration)

    # Evaluation metrics
    home_pred = home_model.predict(X_test)
    away_pred = away_model.predict(X_test)
    sc_acc = float(np.mean(spread_cover_model.predict(Xsc_test) == ysc_test))
    to_acc = float(np.mean(total_model.predict(Xto_test) == yto_test))

    metrics = {
        "home_mae": float(mean_absolute_error(yh_test, home_pred)),
        "home_rmse": float(np.sqrt(mean_squared_error(yh_test, home_pred))),
        "away_mae": float(mean_absolute_error(ya_test, away_pred)),
        "away_rmse": float(np.sqrt(mean_squared_error(ya_test, away_pred))),
        "spread_cover_accuracy": sc_acc,
        "total_over_accuracy": to_acc,
        "training_seconds": duration,
        "model_version": new_version,
    }
    logger.info("Home Score Model MAE: %.2f  RMSE: %.2f", metrics["home_mae"], metrics["home_rmse"])
    logger.info("Away Score Model MAE: %.2f  RMSE: %.2f", metrics["away_mae"], metrics["away_rmse"])
    logger.info("Spread Cover Accuracy: %.2f%%", sc_acc * 100)
    logger.info("Total Over Accuracy: %.2f%%", to_acc * 100)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "home_model.pkl", "wb") as f:
        pickle.dump(home_model, f)
    with open(MODEL_DIR / "away_model.pkl", "wb") as f:
        pickle.dump(away_model, f)
    with open(MODEL_DIR / "spread_model.pkl", "wb") as f:
        pickle.dump(spread_cover_model, f)
    with open(MODEL_DIR / "total_model.pkl", "wb") as f:
        pickle.dump(total_model, f)

    database.log_model("home_score", alg, sample_count, metrics, str(MODEL_DIR / "home_model.pkl"))
    database.log_model("away_score", alg, sample_count, metrics, str(MODEL_DIR / "away_model.pkl"))
    database.log_model("spread_cover", alg, sample_count, metrics, str(MODEL_DIR / "spread_model.pkl"))
    database.log_model("total_over", alg, sample_count, metrics, str(MODEL_DIR / "total_model.pkl"))

    # Supabase training log
    try:
        from .supabase_client import save_training_log
        save_training_log({
            "model_version": new_version,
            "feature_count": feature_count,
            "algorithm": alg,
            "data_points": sample_count,
            **metrics,
        })
    except Exception:
        logger.debug("Supabase training log skipped")

    # Upload models to Supabase Storage for persistence across runs
    try:
        from .supabase_client import upload_models_to_storage
        if upload_models_to_storage(MODEL_DIR):
            logger.info("Models uploaded to Supabase Storage successfully")
    except Exception:
        logger.debug("Supabase Storage upload failed", exc_info=True)

    bundle = ModelBundle(home_model, away_model, alg,
                         spread_cover_model=spread_cover_model,
                         total_model=total_model)
    bundle.version = new_version
    bundle.metrics = metrics
    bundle.sample_count = sample_count
    bundle.feature_count = feature_count
    bundle.duration = duration
    return bundle


def load_models() -> ModelBundle | None:
    hp = MODEL_DIR / "home_model.pkl"
    ap = MODEL_DIR / "away_model.pkl"
    if not hp.exists() or not ap.exists():
        return None
    with open(hp, "rb") as f:
        home = pickle.load(f)
    with open(ap, "rb") as f:
        away = pickle.load(f)

    spread_cover = None
    total = None
    sp = MODEL_DIR / "spread_model.pkl"
    tp = MODEL_DIR / "total_model.pkl"
    if sp.exists():
        with open(sp, "rb") as f:
            spread_cover = pickle.load(f)
    if tp.exists():
        with open(tp, "rb") as f:
            total = pickle.load(f)

    bundle = ModelBundle(home, away, "loaded",
                         spread_cover_model=spread_cover,
                         total_model=total)
    bundle.version = _current_version()
    return bundle
