from __future__ import annotations


def analyze_line_behavior(opening_spread: float | None, live_spread: float | None, model_spread_edge: float, opening_total: float | None, live_total: float | None, model_total_edge: float) -> dict[str, float]:
    spread_move = 0.0 if opening_spread is None or live_spread is None else live_spread - opening_spread
    total_move = 0.0 if opening_total is None or live_total is None else live_total - opening_total
    sharp_movement = min(1.0, (abs(spread_move) + abs(total_move) / 2.0) / 4.0)

    reverse_line_movement = 1.0 if (spread_move * model_spread_edge < 0 or total_move * model_total_edge < 0) else 0.0
    market_confidence = max(0.0, min(1.0, 0.55 * sharp_movement + 0.45 * (1.0 - reverse_line_movement)))
    return {
        "sharp_movement_score": sharp_movement,
        "reverse_line_movement": reverse_line_movement,
        "market_confidence_indicator": market_confidence,
    }
