"""Total calculation model for NBA totals prediction.

Combines game pace and PPP to produce the predicted total.
Includes dual-high-PPP suppression to prevent runaway "over" predictions
when both teams have strong offenses.
"""
from __future__ import annotations

HIGH_PPP_THRESHOLD = 1.14
PACE_SUPPRESSION = 0.97
TOTAL_SUPPRESSION = 0.90
EXTREME_DEV_LIMIT = 15
EXTREME_DEV_CAP = 12


def calculate_predicted_total(
    game_pace: float,
    ppp_home: float,
    ppp_away: float,
    closing_total: float = 0.0,
) -> float:
    """Calculate the predicted total from pace and PPP with market anchor.

    When both teams have PPP above :data:`HIGH_PPP_THRESHOLD`, a
    suppression mechanism reduces the pace and model total to prevent
    runaway "over" predictions.

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
    high_offense_flag = ppp_home > HIGH_PPP_THRESHOLD and ppp_away > HIGH_PPP_THRESHOLD

    # --- Pace suppression for dual high offense ---
    if high_offense_flag:
        game_pace = game_pace * PACE_SUPPRESSION

    model_total = game_pace * (ppp_home + ppp_away)

    # --- Model total suppression for dual high offense ---
    if high_offense_flag:
        model_total = model_total * TOTAL_SUPPRESSION

    predicted_total = model_total * 0.65 + closing_total * 0.35

    # --- Extreme deviation correction ---
    if predicted_total - closing_total > EXTREME_DEV_LIMIT:
        predicted_total = closing_total + EXTREME_DEV_CAP

    return predicted_total
