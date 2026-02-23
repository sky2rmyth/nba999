from __future__ import annotations

import numpy as np


def run_possession_simulation(
    game_id: int,
    home_off_rating: float,
    visitor_off_rating: float,
    pace: float,
    spread_line: float,
    total_line: float,
    n_sim: int = 10000,
) -> dict[str, float]:
    seed = int(game_id) % 1_000_000
    rng = np.random.default_rng(seed)
    possessions = np.clip(rng.normal(pace, 4.0, n_sim), 85, 110)

    home_ppp = np.clip(rng.normal(home_off_rating / 100.0, 0.08, n_sim), 0.8, 1.4)
    visitor_ppp = np.clip(rng.normal(visitor_off_rating / 100.0, 0.08, n_sim), 0.8, 1.4)

    home_scores = possessions * home_ppp
    visitor_scores = possessions * visitor_ppp

    margins = home_scores - visitor_scores
    totals = home_scores + visitor_scores

    spread_cover_prob = float(np.mean(margins + spread_line > 0))
    over_prob = float(np.mean(totals > total_line))
    variance = float(np.var(totals) + np.var(margins))

    return {
        "spread_cover_probability": spread_cover_prob,
        "over_probability": over_prob,
        "expected_home_score": float(np.mean(home_scores)),
        "expected_visitor_score": float(np.mean(visitor_scores)),
        "score_distribution_variance": variance,
        "simulation_count": n_sim,
    }
