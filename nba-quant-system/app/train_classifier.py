"""Train a GradientBoostingClassifier for NBA totals over/under prediction.

Hyperparameters (per specification):
    n_estimators  = 300
    learning_rate = 0.03
    max_depth     = 3
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from .dataset_builder import CLASSIFIER_FEATURES

logger = logging.getLogger(__name__)


def build_classifier() -> GradientBoostingClassifier:
    """Return an *untrained* GradientBoostingClassifier with the prescribed
    hyperparameters."""
    return GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        random_state=42,
    )


def train(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[GradientBoostingClassifier, dict]:
    """Train the totals classifier and return ``(model, metrics)``.

    Parameters
    ----------
    df:
        DataFrame containing :data:`~app.dataset_builder.CLASSIFIER_FEATURES`
        and a ``"label"`` column.
    test_size:
        Fraction of data reserved for evaluation.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    tuple
        ``(trained_model, metrics_dict)``
    """
    X = df[CLASSIFIER_FEATURES].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    model = build_classifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test))
    logger.info("Totals classifier accuracy: %.2f%%", accuracy * 100)

    metrics = {
        "accuracy": accuracy,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": len(CLASSIFIER_FEATURES),
    }
    return model, metrics
