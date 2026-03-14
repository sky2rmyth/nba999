"""Evaluation / prediction output for the NBA totals classifier.

Given a calibrated model and a feature row (or DataFrame), produce the
output columns required by the specification:

    Game | Line | Over Probability | Under Probability | Prediction | Confidence
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .dataset_builder import CLASSIFIER_FEATURES

logger = logging.getLogger(__name__)


def predict_game(model, features: dict, game_label: str = "") -> dict:
    """Generate a single-game prediction.

    Parameters
    ----------
    model:
        Fitted classifier exposing ``predict_proba``.
    features:
        Dictionary containing all :data:`~app.dataset_builder.CLASSIFIER_FEATURES`.
    game_label:
        Human-readable game identifier (e.g. ``"LAL vs BOS"``).

    Returns
    -------
    dict
        ``{Game, Line, Over Probability, Under Probability, Prediction, Confidence}``
    """
    X = np.array([[features[c] for c in CLASSIFIER_FEATURES]])
    proba = model.predict_proba(X)[0]

    prob_over = float(proba[1])
    prob_under = 1.0 - prob_over
    prediction = "OVER" if prob_over > 0.5 else "UNDER"
    confidence = round(max(prob_over, prob_under), 4)

    return {
        "Game": game_label,
        "Line": features.get("closing_total", 0.0),
        "Over Probability": round(prob_over, 4),
        "Under Probability": round(prob_under, 4),
        "Prediction": prediction,
        "Confidence": confidence,
    }


def evaluate(model, df: pd.DataFrame) -> pd.DataFrame:
    """Batch-evaluate games and return a DataFrame with prediction columns.

    Parameters
    ----------
    model:
        Fitted classifier exposing ``predict_proba``.
    df:
        DataFrame containing :data:`~app.dataset_builder.CLASSIFIER_FEATURES`.
        An optional ``"game_id"`` column is used as the ``Game`` label.

    Returns
    -------
    pd.DataFrame
        Columns: ``Game, Line, Over Probability, Under Probability,
        Prediction, Confidence``.
    """
    X = df[CLASSIFIER_FEATURES].values
    proba = model.predict_proba(X)

    prob_over = proba[:, 1]
    prob_under = 1.0 - prob_over
    predictions = np.where(prob_over > 0.5, "OVER", "UNDER")
    confidence = np.maximum(prob_over, prob_under)

    result = pd.DataFrame({
        "Game": df["game_id"].values if "game_id" in df.columns else range(len(df)),
        "Line": df["closing_total"].values,
        "Over Probability": np.round(prob_over, 4),
        "Under Probability": np.round(prob_under, 4),
        "Prediction": predictions,
        "Confidence": np.round(confidence, 4),
    })
    return result
