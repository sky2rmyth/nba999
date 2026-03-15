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
    league_pace = 100

    pace_raw = (home_pace + away_pace) / 2

    if home_back_to_back:
        pace_raw -= 0.8
    if away_back_to_back:
        pace_raw -= 0.8

    game_pace = pace_raw * 0.6 + league_pace * 0.4

    game_pace = max(96, min(game_pace, 106))
    return game_pace
