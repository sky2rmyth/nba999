"""PPP (Points Per Possession) model for NBA totals prediction.

Calculates PPP for home and away teams using offensive and defensive
ratings with shooting structure corrections.
"""
from __future__ import annotations


def calculate_ppp(
    home_off_rating: float,
    away_off_rating: float,
    home_def_rating: float,
    away_def_rating: float,
    home_3p_rate: float,
    away_3p_rate: float,
    home_ft_rate: float,
    away_ft_rate: float,
) -> tuple[float, float]:
    """Calculate Points Per Possession for both teams.

    PPP combines a team's offensive rating with the opponent's defensive
    rating and applies shooting structure corrections for three-point and
    free-throw rates.

    Parameters
    ----------
    home_off_rating, away_off_rating : float
        Offensive ratings (points per 100 possessions).
    home_def_rating, away_def_rating : float
        Defensive ratings (opponent points per 100 possessions).
    home_3p_rate, away_3p_rate : float
        Three-point attempt rate (3PA / FGA).
    home_ft_rate, away_ft_rate : float
        Free-throw attempt rate (FTA / FGA).

    Returns
    -------
    tuple[float, float]
        ``(ppp_home, ppp_away)``
    """
    league_ppp = 1.12

    ppp_home = (home_off_rating / away_def_rating) * league_ppp
    ppp_away = (away_off_rating / home_def_rating) * league_ppp

    ppp_home = max(1.02, min(ppp_home, 1.18))
    ppp_away = max(1.02, min(ppp_away, 1.18))

    return ppp_home, ppp_away
