"""Market calibration model for NBA totals prediction.

Applies a deviation correction when the model total diverges too far
from the closing line.
"""
from __future__ import annotations


def apply_market_calibration(
    model_total: float,
    closing_total: float,
) -> float:
    """Apply market deviation correction.

    When the model total deviates more than 12 points from the closing
    line, compress the difference to prevent runaway predictions.

    Parameters
    ----------
    model_total : float
        Model-derived total after pace/PPP calculation and market anchor.
    closing_total : float
        Closing total line from oddsmakers.

    Returns
    -------
    float
        Adjusted model total.
    """
    if abs(model_total - closing_total) > 12:
        model_total = closing_total + (model_total - closing_total) * 0.6
    return model_total
