"""Total calculation model for NBA totals prediction.

Combines game pace and PPP to produce the predicted total.
"""
from __future__ import annotations


def calculate_predicted_total(
    game_pace: float,
    ppp_home: float,
    ppp_away: float,
) -> float:
    """Calculate the predicted total from pace and PPP.

    Parameters
    ----------
    game_pace : float
        Expected game pace (possessions per 48 minutes).
    ppp_home : float
        Home team's points per possession.
    ppp_away : float
        Away team's points per possession.

    Returns
    -------
    float
        Predicted total score.
    """
    return game_pace * (ppp_home + ppp_away)
