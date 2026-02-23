from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from . import database
from .feature_engineering import FEATURE_COLUMNS

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelBundle:
    def __init__(self, spread_model, total_model, algorithm: str):
        self.spread_model = spread_model
        self.total_model = total_model
        self.algorithm = algorithm


def _pick_algorithm():
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        return "lightgbm", LGBMClassifier(n_estimators=200, learning_rate=0.05), LGBMClassifier(n_estimators=200, learning_rate=0.05)
    except Exception:
        return "gradient_boosting", GradientBoostingClassifier(random_state=42), GradientBoostingClassifier(random_state=42)


def train_models(df: pd.DataFrame) -> ModelBundle:
    X = df[FEATURE_COLUMNS]
    y_spread = df["spread_label"]
    y_total = df["total_label"]
    alg, spread_model, total_model = _pick_algorithm()
    if len(df) < 30:
        alg = "logistic_regression"
        spread_model = LogisticRegression(max_iter=500)
        total_model = LogisticRegression(max_iter=500)

    X_train, X_test, ys_train, ys_test = train_test_split(X, y_spread, test_size=0.2, random_state=42)
    _, _, yt_train, yt_test = train_test_split(X, y_total, test_size=0.2, random_state=42)

    spread_model.fit(X_train, ys_train)
    total_model.fit(X_train, yt_train)

    spread_pred = spread_model.predict(X_test)
    total_pred = total_model.predict(X_test)
    metrics = {
        "spread_acc": float(accuracy_score(ys_test, spread_pred)),
        "total_acc": float(accuracy_score(yt_test, total_pred)),
    }
    if len(set(ys_test)) > 1:
        metrics["spread_auc"] = float(roc_auc_score(ys_test, spread_model.predict_proba(X_test)[:, 1]))
    if len(set(yt_test)) > 1:
        metrics["total_auc"] = float(roc_auc_score(yt_test, total_model.predict_proba(X_test)[:, 1]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "spread_model.pkl", "wb") as f:
        pickle.dump(spread_model, f)
    with open(MODEL_DIR / "total_model.pkl", "wb") as f:
        pickle.dump(total_model, f)

    database.log_model("spread", alg, len(df), metrics, str(MODEL_DIR / "spread_model.pkl"))
    database.log_model("total", alg, len(df), metrics, str(MODEL_DIR / "total_model.pkl"))
    return ModelBundle(spread_model, total_model, alg)


def load_models() -> ModelBundle | None:
    sp = MODEL_DIR / "spread_model.pkl"
    tp = MODEL_DIR / "total_model.pkl"
    if not sp.exists() or not tp.exists():
        return None
    with open(sp, "rb") as f:
        spread = pickle.load(f)
    with open(tp, "rb") as f:
        total = pickle.load(f)
    return ModelBundle(spread, total, "loaded")
