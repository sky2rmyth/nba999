"""Total calculation model for NBA totals prediction.

Combines game pace and PPP to produce the predicted total.
Uses market-line anchoring: the model total is allowed to deviate
from the closing total by only a limited fraction (0.35), with a
hard cap of ±12 points and high-PPP cooling.
"""
from __future__ import annotations

HIGH_PPP_THRESHOLD = 1.17
MAX_EDGE = 12
ANCHOR_WEIGHT = 0.35
HIGH_PPP_COOL = 0.97


def calculate_predicted_total(
    game_pace: float,
    ppp_home: float,
    ppp_away: float,
    closing_total: float = 0.0,
) -> float:
    """Calculate the predicted total from pace and PPP with market anchor.

    The raw total (pace × combined PPP) is anchored to the closing line:
    only 35 % of the deviation is kept.  A hard cap of ±12 prevents
    extreme outliers, and dual-high-PPP games receive a 3 % cooling
    factor to counteract systematic over-prediction.

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
    # --- Step 1: Raw total ---
    raw_total = game_pace * (ppp_home + ppp_away)

    # --- Step 2: Market-line anchoring (only 35% deviation allowed) ---
    model_total = closing_total + (raw_total - closing_total) * ANCHOR_WEIGHT

    # --- Step 3: Hard cap at ±12 ---
    edge = model_total - closing_total
    if edge > MAX_EDGE:
        model_total = closing_total + MAX_EDGE
    elif edge < -MAX_EDGE:
        model_total = closing_total - MAX_EDGE

    # --- Step 4: High-PPP cooling (prevent all-over bias) ---
    if ppp_home > HIGH_PPP_THRESHOLD and ppp_away > HIGH_PPP_THRESHOLD:
        model_total *= HIGH_PPP_COOL

    return model_total
