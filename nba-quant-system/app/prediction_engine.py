from __future__ import annotations

from datetime import datetime

import pandas as pd

from .api_client import BallDontLieClient
from .bookmaker_behavior import analyze_line_behavior
from .database import get_conn, insert_prediction
from .data_pipeline import bootstrap_historical_data, sync_date_games
from .feature_engineering import FEATURE_COLUMNS
from .game_simulator import run_possession_simulation
from .odds_tracker import parse_main_market, store_opening_and_live
from .rating_engine import compute_ratings
from .retrain_engine import ensure_models
from .team_translation import zh_name
from .telegram_bot import send_message


def _recent_margin(team_id: int) -> float:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT home_team_id, visitor_team_id, home_score, visitor_score FROM games
            WHERE status LIKE 'Final%' AND (home_team_id=? OR visitor_team_id=?)
            ORDER BY date DESC LIMIT 10""",
            (team_id, team_id),
        ).fetchall()
    if not rows:
        return 0.0
    vals = []
    for r in rows:
        vals.append((r[2] - r[3]) if r[0] == team_id else (r[3] - r[2]))
    return sum(vals) / len(vals)


def run_prediction(target_date: str | None = None) -> None:
    target_date = target_date or datetime.utcnow().strftime("%Y-%m-%d")
    bootstrap_historical_data()
    model_bundle = ensure_models()
    client = BallDontLieClient()
    games = sync_date_games(target_date)

    lines = [f"🏀 NBA每日预测｜{target_date}", "", "━━━━━━━━━━━━━━━━"]

    for g in games:
        game_id = g["id"]
        home = g["home_team"]
        vis = g["visitor_team"]

        opening_payload = {"data": client.betting_odds(**{"game_ids[]": [game_id], "per_page": 100})}
        live_payload = {"data": client.betting_odds(**{"game_ids[]": [game_id], "per_page": 100})}
        store_opening_and_live(game_id, opening_payload, live_payload)
        opening_spread, opening_total, _ = parse_main_market(opening_payload)
        live_spread, live_total, _ = parse_main_market(live_payload)
        opening_spread = opening_spread if opening_spread is not None else 0.0
        live_spread = live_spread if live_spread is not None else opening_spread
        opening_total = opening_total if opening_total is not None else 220.0
        live_total = live_total if live_total is not None else opening_total

        hrm = _recent_margin(home["id"])
        vrm = _recent_margin(vis["id"])
        feat = pd.DataFrame([
            {
                "home_recent_margin": hrm,
                "visitor_recent_margin": vrm,
                "margin_diff": hrm - vrm,
                "opening_spread": opening_spread,
                "live_spread": live_spread,
                "opening_total": opening_total,
                "live_total": live_total,
            }
        ])[FEATURE_COLUMNS]

        spread_prob = float(model_bundle.spread_model.predict_proba(feat)[:, 1][0])
        total_prob = float(model_bundle.total_model.predict_proba(feat)[:, 1][0])

        sim = run_possession_simulation(
            game_id=game_id,
            home_off_rating=112 + hrm,
            visitor_off_rating=112 + vrm,
            pace=98,
            spread_line=live_spread,
            total_line=live_total,
            n_sim=10000,
        )
        spread_edge = sim["spread_cover_probability"] - 0.5
        total_edge = sim["over_probability"] - 0.5
        market = analyze_line_behavior(opening_spread, live_spread, spread_edge, opening_total, live_total, total_edge)
        ratings = compute_ratings(max(abs(spread_edge), abs(total_edge)), sim["score_distribution_variance"], market["market_confidence_indicator"])

        spread_pick = f"{zh_name(home['full_name'])}让分" if spread_prob >= 0.5 else f"{zh_name(vis['full_name'])}受让"
        total_pick = "大分" if total_prob >= 0.5 else "小分"

        lines.extend(
            [
                f"{zh_name(vis['full_name'])} vs {zh_name(home['full_name'])}",
                f"让：{opening_spread:+.1f} → {spread_pick} {'⭐' * int(ratings['star_rating'])} {spread_prob*100:.0f}% 指{int(ratings['recommendation_index'])}",
                f"大：{opening_total:.1f} → {total_pick} {'⭐' * max(1, int(ratings['star_rating'])-1)} {total_prob*100:.0f}% 指{int(ratings['recommendation_index'])}",
                "",
                "━━━━━━━━━━━━━━━━",
            ]
        )

        insert_prediction(
            snapshot_date=target_date,
            row={
                "game_id": game_id,
                "prediction_time": datetime.utcnow().isoformat(),
                "spread_pick": spread_pick,
                "spread_prob": spread_prob,
                "total_pick": total_pick,
                "total_prob": total_prob,
                "confidence_score": ratings["confidence_score"],
                "star_rating": ratings["star_rating"],
                "recommendation_index": ratings["recommendation_index"],
                "expected_home_score": sim["expected_home_score"],
                "expected_visitor_score": sim["expected_visitor_score"],
                "simulation_variance": sim["score_distribution_variance"],
                "opening_spread": opening_spread,
                "live_spread": live_spread,
                "opening_total": opening_total,
                "live_total": live_total,
                "details": {"simulation": sim, "market": market},
            },
        )

    send_message("\n".join(lines))


if __name__ == "__main__":
    run_prediction()
