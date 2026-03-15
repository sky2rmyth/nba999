"""Total calculation model for NBA totals prediction.

Combines game pace and PPP to produce the predicted total.
"""
from __future__ import annotations


def calculate_predicted_total(
    game_pace: float,
    ppp_home: float,
    ppp_away: float,
    closing_total: float = 0.0,
) -> float:
    """Calculate the predicted total from pace and PPP with market anchor.

    Parameters
    ----------
    game_pace : float
        Expected game pace (possessions per 48 minutes).
    ppp_home : float
        Home team's points per possession.
    ppp_away : float
        Away team's points per possession.
    closing_total : float
        Closing total line from oddsmakers for market anchoring.

    Returns
    -------
    float
        Predicted total score.
    """
    model_total = game_pace * (ppp_home + ppp_away)
    predicted_total = model_total * 0.65 + closing_total * 0.35
    return predicted_total
