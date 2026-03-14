"""Probability calibration for the NBA totals classifier.

Wraps a trained classifier with :class:`~sklearn.calibration.CalibratedClassifierCV`
using the **isotonic** method so that ``predict_proba`` outputs are
well-calibrated.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from .dataset_builder import CLASSIFIER_FEATURES

logger = logging.getLogger(__name__)


def calibrate(
    base_model,
    df: pd.DataFrame,
    *,
    cv: int = 5,
) -> CalibratedClassifierCV:
    """Return a calibrated version of *base_model*.

    Parameters
    ----------
    base_model:
        A **fitted** sklearn classifier that exposes ``predict_proba``.
    df:
        DataFrame containing :data:`~app.dataset_builder.CLASSIFIER_FEATURES`
        and a ``"label"`` column.  Used as the calibration set.
    cv:
        Number of cross-validation folds used by the calibrator.

    Returns
    -------
    CalibratedClassifierCV
        Calibrated model whose ``predict_proba`` is isotonic-calibrated.
    """
    X = df[CLASSIFIER_FEATURES].values
    y = df["label"].values

    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=cv,
    )
    calibrated.fit(X, y)

    logger.info("Calibration complete (isotonic, cv=%d)", cv)
    return calibrated
