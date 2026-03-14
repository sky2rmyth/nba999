"""End-to-end training pipeline for the NBA totals classifier.

Steps:
    1. Build dataset  (``dataset_builder.build_dataset``)
    2. Train classifier  (``train_classifier.train``)
    3. Calibrate  (``calibration.calibrate``)
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

from .calibration import calibrate
from .dataset_builder import CLASSIFIER_FEATURES, build_dataset
from .train_classifier import train

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def run_pipeline(
    db_path: Path | None = None,
    *,
    test_size: float = 0.2,
    save: bool = True,
) -> dict:
    """Execute the full training pipeline.

    Parameters
    ----------
    db_path:
        Optional path to the SQLite database.  Falls back to the default
        :data:`~app.database.DB_PATH`.
    test_size:
        Fraction of data held out for evaluation during training.
    save:
        Whether to persist the calibrated model to disk.

    Returns
    -------
    dict
        ``{"model", "metrics", "features"}`` — the calibrated model, training
        metrics, and the list of feature column names.
    """
    # 1. Build dataset
    kwargs = {} if db_path is None else {"db_path": db_path}
    df = build_dataset(**kwargs)

    if df.empty:
        logger.warning("No training data — pipeline aborted")
        return {"model": None, "metrics": {}, "features": CLASSIFIER_FEATURES}

    # 2. Train
    model, metrics = train(df, test_size=test_size)

    # 3. Calibrate
    calibrated_model = calibrate(model, df)

    # 4. Persist
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MODEL_DIR / "total_classifier.pkl"
        with open(out_path, "wb") as fh:
            pickle.dump(calibrated_model, fh)
        logger.info("Calibrated model saved to %s", out_path)

    logger.info("Pipeline complete — accuracy %.2f%%", metrics.get("accuracy", 0) * 100)
    return {
        "model": calibrated_model,
        "metrics": metrics,
        "features": CLASSIFIER_FEATURES,
    }
