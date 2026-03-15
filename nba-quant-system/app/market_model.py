"""Market calibration model for NBA totals prediction.

Adjusts the predicted total based on line movement between
opening and closing totals.
"""
from __future__ import annotations


def apply_market_calibration(
    predicted_total: float,
    opening_total: float,
    closing_total: float,
) -> float:
    """Apply market line-movement calibration.

    Parameters
    ----------
    predicted_total : float
        Model-derived predicted total before market adjustment.
    opening_total : float
        Opening total line from oddsmakers.
    closing_total : float
        Closing total line from oddsmakers.

    Returns
    -------
    float
        Adjusted predicted total.
    """
    line_move = closing_total - opening_total
    predicted_total += line_move * 0.35
    return predicted_total
