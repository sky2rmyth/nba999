from __future__ import annotations


def compute_ratings(edge: float, variance: float, market_confidence: float) -> dict[str, float | int]:
    tightness = max(0.0, min(1.0, 1.0 / (1.0 + variance / 150.0)))
    confidence = max(0.0, min(1.0, 0.45 * abs(edge) + 0.30 * tightness + 0.25 * market_confidence))
    recommendation = round(confidence * 100, 1)
    stars = max(1, min(5, int(round(confidence * 5))))
    return {
        "confidence_score": round(confidence, 4),
        "star_rating": stars,
        "recommendation_index": recommendation,
    }
