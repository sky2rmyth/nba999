"""Total calculation model for NBA totals prediction.

Combines game pace and PPP to produce the predicted total.
Uses PPP–Pace linkage: when both teams have high PPP, the pace
determines whether the model allows a higher total (fast pace)
or suppresses it (slow pace).
"""
from __future__ import annotations

HIGH_PPP_THRESHOLD = 1.14
FAST_PACE_THRESHOLD = 104
FAST_PACE_BOOST = 1.03
SLOW_PACE_SUPPRESS = 0.95
EXTREME_DEV_LIMIT = 15
EXTREME_DEV_COMPRESS = 0.6


def calculate_predicted_total(
    game_pace: float,
    ppp_home: float,
    ppp_away: float,
    closing_total: float = 0.0,
) -> float:
    """Calculate the predicted total from pace and PPP with market anchor.

    When both teams have PPP above :data:`HIGH_PPP_THRESHOLD`, the game
    pace determines the adjustment: fast pace (≥ 104) boosts the total
    while slow pace suppresses it, linking offensive efficiency to tempo.

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
    # --- Dual high PPP detection ---
    high_offense = ppp_home > HIGH_PPP_THRESHOLD and ppp_away > HIGH_PPP_THRESHOLD

    model_total = game_pace * (ppp_home + ppp_away)

    # --- PPP–Pace linkage ---
    if high_offense:
        if game_pace >= FAST_PACE_THRESHOLD:
            model_total = model_total * FAST_PACE_BOOST
        else:
            model_total = model_total * SLOW_PACE_SUPPRESS

    predicted_total = model_total * 0.65 + closing_total * 0.35

    # --- Extreme deviation protection ---
    edge = predicted_total - closing_total
    if abs(edge) > EXTREME_DEV_LIMIT:
        predicted_total = closing_total + edge * EXTREME_DEV_COMPRESS

    return predicted_total
