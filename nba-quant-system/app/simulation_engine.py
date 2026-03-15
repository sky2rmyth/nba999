"""Monte Carlo simulation engine for NBA totals prediction.

Uses dynamic standard deviation based on pace and PPP differences
to generate a distribution of total scores and compute over/under
probabilities.
"""
from __future__ import annotations

import numpy as np

N_SIMULATIONS = 10000


def run_total_simulation(
    game_id: int,
    predicted_total: float,
    closing_total: float,
    pace_diff: float,
    ppp_home: float,
    ppp_away: float,
    n_sim: int = N_SIMULATIONS,
) -> dict[str, float]:
    """Run Monte Carlo simulation for the total.

    Parameters
    ----------
    game_id : int
        Unique game identifier (used as random seed for reproducibility).
    predicted_total : float
        Model-derived predicted total.
    closing_total : float
        Closing total line from oddsmakers.
    pace_diff : float
        Pace difference (home_pace - away_pace).
    ppp_home : float
        Home team's points per possession.
    ppp_away : float
        Away team's points per possession.
    n_sim : int
        Number of simulations to run (default 10,000).

    Returns
    -------
    dict[str, float]
        Keys: ``over_probability``, ``under_probability``, ``simulation_count``.
    """
    seed = int(game_id) % 1_000_000
    rng = np.random.default_rng(seed)

    # Dynamic standard deviation
    std = 10 + abs(pace_diff) * 0.6 + abs(ppp_home - ppp_away) * 4

    simulated_totals = rng.normal(predicted_total, std, n_sim)

    over_prob = float(np.mean(simulated_totals > closing_total))
    under_prob = 1.0 - over_prob

    return {
        "over_probability": over_prob,
        "under_probability": under_prob,
        "simulation_count": n_sim,
    }
