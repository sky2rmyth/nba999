from __future__ import annotations


def _edge_to_stars(edge_pct: float) -> int:
    """Map edge percentage to star rating."""
    edge = abs(edge_pct)
    if edge > 12.0:
        return 5
    if edge > 8.0:
        return 4
    if edge > 5.0:
        return 3
    if edge > 2.0:
        return 2
    return 1


def compute_spread_rating(spread_cover_prob: float, market_spread: float) -> dict[str, float | int]:
    """Compute spread recommendation from simulation probability vs implied market."""
    implied_prob = 0.5  # market line implies ~50%
    edge = (spread_cover_prob - implied_prob) * 100.0
    stars = _edge_to_stars(edge)
    confidence = round(spread_cover_prob * 100.0, 1)
    recommendation = round(abs(edge) * 10, 1)
    return {
        "spread_edge": round(edge, 2),
        "spread_confidence": confidence,
        "spread_stars": stars,
        "spread_recommendation_index": recommendation,
    }


def compute_total_rating(over_prob: float, market_total: float) -> dict[str, float | int]:
    """Compute total recommendation from simulation probability vs implied market."""
    implied_prob = 0.5  # market total implies ~50%
    edge = (over_prob - implied_prob) * 100.0
    stars = _edge_to_stars(edge)
    confidence = round(over_prob * 100.0, 1)
    recommendation = round(abs(edge) * 10, 1)
    return {
        "total_edge": round(edge, 2),
        "total_confidence": confidence,
        "total_stars": stars,
        "total_recommendation_index": recommendation,
    }


def compute_ratings(edge: float, variance: float, market_confidence: float) -> dict[str, float | int]:
    """Legacy wrapper for backward compatibility."""
    edge_pct = abs(edge) * 100.0
    stars = _edge_to_stars(edge_pct)
    tightness = max(0.0, min(1.0, 1.0 / (1.0 + variance / 150.0)))
    confidence = max(0.0, min(1.0, 0.45 * abs(edge) + 0.30 * tightness + 0.25 * market_confidence))
    recommendation = round(confidence * 100, 1)
    return {
        "confidence_score": round(confidence, 4),
        "star_rating": stars,
        "recommendation_index": recommendation,
    }
