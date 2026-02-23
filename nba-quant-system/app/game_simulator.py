from __future__ import annotations

import numpy as np


def run_possession_simulation(
    game_id: int,
    predicted_home_score: float,
    predicted_away_score: float,
    home_variance: float,
    away_variance: float,
    spread_line: float,
    total_line: float,
    n_sim: int = 10000,
) -> dict[str, float]:
    seed = int(game_id) % 1_000_000
    rng = np.random.default_rng(seed)

    home_std = max(np.sqrt(home_variance), 8.0)
    away_std = max(np.sqrt(away_variance), 8.0)

    home_scores = rng.normal(predicted_home_score, home_std, n_sim)
    away_scores = rng.normal(predicted_away_score, away_std, n_sim)

    # Clip to reasonable NBA score ranges (historical min ~60, max ~160)
    home_scores = np.clip(home_scores, 70, 170)
    away_scores = np.clip(away_scores, 70, 170)

    margins = home_scores - away_scores
    totals = home_scores + away_scores

    spread_cover_prob = float(np.mean(margins + spread_line > 0))
    over_prob = float(np.mean(totals > total_line))

    margin_mean = float(np.mean(margins))
    margin_std = float(np.std(margins))
    total_mean = float(np.mean(totals))
    total_std = float(np.std(totals))
    variance = float(np.var(totals) + np.var(margins))

    return {
        "spread_cover_probability": spread_cover_prob,
        "over_probability": over_prob,
        "expected_home_score": float(np.mean(home_scores)),
        "expected_visitor_score": float(np.mean(away_scores)),
        "predicted_margin": margin_mean,
        "predicted_total": total_mean,
        "margin_std": margin_std,
        "total_std": total_std,
        "score_distribution_variance": variance,
        "simulation_count": n_sim,
    }
