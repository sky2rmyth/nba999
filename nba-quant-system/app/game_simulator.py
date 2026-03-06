from __future__ import annotations

import numpy as np

# League-wide average parameters
LEAGUE_AVG_PACE = 99
LEAGUE_AVG_DEF = 113

# Maximum allowed total standard deviation to limit simulation variance
MAX_TOTAL_STD = 11
# Z-score for 90% confidence interval (5th–95th percentile)
Z_SCORE_90PCT = 1.65
# Minimum absolute edge (points) required to recommend a game
MIN_RECOMMEND_EDGE = 6


# Simulation noise parameters
PACE_STD = 2        # Standard deviation for pace variation across simulations
PPP_STD = 0.06      # Standard deviation for points-per-possession variation


def run_possession_simulation(
    game_id: int,
    game_pace: float,
    home_adj_ppp: float,
    away_adj_ppp: float,
    predicted_total: float,
    closing_total: float,
    spread_line: float,
    n_sim: int = 10000,
) -> dict[str, float]:
    seed = int(game_id) % 1_000_000
    rng = np.random.default_rng(seed)

    pace_sims = rng.normal(game_pace, PACE_STD, n_sim)
    home_ppp_sims = rng.normal(home_adj_ppp, PPP_STD, n_sim)
    away_ppp_sims = rng.normal(away_adj_ppp, PPP_STD, n_sim)

    home_scores = pace_sims * home_ppp_sims
    away_scores = pace_sims * away_ppp_sims

    totals = home_scores + away_scores
    margins = home_scores - away_scores

    # Standard deviation capped at MAX_TOTAL_STD
    total_std = float(np.std(totals))
    if total_std > MAX_TOTAL_STD:
        total_std = MAX_TOTAL_STD

    # Probability calculation from simulation counts
    over_probability = float(np.mean(totals > closing_total))
    under_probability = 1.0 - over_probability

    # Edge calculation
    edge = predicted_total - closing_total
    abs_edge = abs(edge)

    # Recommendation based on absolute edge
    recommend = abs_edge >= MIN_RECOMMEND_EDGE

    # Parametric interval based on capped total_std
    total_5pct = predicted_total - Z_SCORE_90PCT * total_std
    total_95pct = predicted_total + Z_SCORE_90PCT * total_std

    # Spread and margin statistics
    spread_cover_prob = float(np.mean(margins + spread_line > 0))
    margin_mean = float(np.mean(margins))
    margin_std = float(np.std(margins))
    total_mean = float(np.mean(totals))

    variance = float(np.var(totals) + np.var(margins))
    home_win_prob = float(np.mean(margins > 0))

    spread_5pct = float(np.percentile(margins, 5))
    spread_95pct = float(np.percentile(margins, 95))

    return {
        "predicted_total": predicted_total,
        "edge": edge,
        "over_probability": over_probability,
        "under_probability": under_probability,
        "simulation_low": total_5pct,
        "simulation_high": total_95pct,
        "recommend": recommend,
        "spread_cover_probability": spread_cover_prob,
        "home_win_probability": home_win_prob,
        "expected_home_score": float(np.mean(home_scores)),
        "expected_visitor_score": float(np.mean(away_scores)),
        "predicted_margin": margin_mean,
        "margin_std": margin_std,
        "total_std": total_std,
        "score_distribution_variance": variance,
        "simulation_count": n_sim,
        "spread_5pct": spread_5pct,
        "spread_95pct": spread_95pct,
        "total_5pct": total_5pct,
        "total_95pct": total_95pct,
    }
