"""Pace model for NBA totals prediction.

Calculates expected game pace from team pace values with
back-to-back fatigue adjustments.
"""
from __future__ import annotations


def calculate_game_pace(
    home_pace: float,
    away_pace: float,
    home_back_to_back: bool = False,
    away_back_to_back: bool = False,
) -> float:
    """Calculate expected game pace.

    Parameters
    ----------
    home_pace : float
        Home team's pace (possessions per 48 minutes).
    away_pace : float
        Away team's pace (possessions per 48 minutes).
    home_back_to_back : bool
        Whether the home team is on a back-to-back.
    away_back_to_back : bool
        Whether the away team is on a back-to-back.

    Returns
    -------
    float
        Expected game pace.
    """
    pace_diff = home_pace - away_pace
    pace_base = (home_pace + away_pace) / 2
    pace = pace_base + pace_diff * 0.15

    if home_back_to_back:
        pace -= 0.8
    if away_back_to_back:
        pace -= 0.8

    return pace
