"""Injury impact model for NBA totals prediction.

Adjusts PPP and pace when key players are absent.
"""
from __future__ import annotations

# PPP reduction when a primary scorer is ruled out (midpoint of 0.03–0.05 range)
SCORER_OUT_PPP_PENALTY = 0.04

# Pace reduction when a point guard is ruled out
PG_OUT_PACE_PENALTY = 1.0


def adjust_for_injuries(
    ppp_home: float,
    ppp_away: float,
    game_pace: float,
    home_scorer_out: bool = False,
    away_scorer_out: bool = False,
    home_pg_out: bool = False,
    away_pg_out: bool = False,
) -> tuple[float, float, float]:
    """Adjust PPP and pace for player absences.

    Parameters
    ----------
    ppp_home, ppp_away : float
        Points-per-possession values before injury adjustment.
    game_pace : float
        Game pace before injury adjustment.
    home_scorer_out, away_scorer_out : bool
        Whether each team's primary scorer is ruled out.
    home_pg_out, away_pg_out : bool
        Whether each team's point guard is ruled out.

    Returns
    -------
    tuple[float, float, float]
        ``(ppp_home, ppp_away, game_pace)`` after adjustments.
    """
    if home_scorer_out:
        ppp_home -= SCORER_OUT_PPP_PENALTY
    if away_scorer_out:
        ppp_away -= SCORER_OUT_PPP_PENALTY

    if home_pg_out:
        game_pace -= PG_OUT_PACE_PENALTY
    if away_pg_out:
        game_pace -= PG_OUT_PACE_PENALTY

    return ppp_home, ppp_away, game_pace
