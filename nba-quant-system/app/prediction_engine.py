from __future__ import annotations

import logging
import sys
from datetime import datetime

import pandas as pd

from .api_client import BallDontLieClient
from .bookmaker_behavior import analyze_line_behavior
from .database import get_conn, insert_prediction
from .data_pipeline import bootstrap_historical_data, sync_date_games
from .feature_engineering import FEATURE_COLUMNS
from .game_simulator import run_possession_simulation
from .odds_provider import fetch_today_odds, extract_opening_line, extract_live_line
from .odds_tracker import parse_main_market, store_opening_and_live
from .prediction_models import MODEL_DIR
from .rating_engine import compute_ratings
from .retrain_engine import ensure_models
from .team_translation import zh_name
from .telegram_bot import send_message

logger = logging.getLogger(__name__)

MIN_SIMULATION_COUNT = 10000


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


def _verify_models_present() -> bool:
    sp = MODEL_DIR / "spread_model.pkl"
    tp = MODEL_DIR / "total_model.pkl"
    return sp.exists() and tp.exists()


def _match_primary_odds(primary_odds: list, home_name: str, visitor_name: str) -> dict | None:
    """Match a game to primary odds data by team names."""
    for event in primary_odds:
        event_home = event.get("home_team", "")
        event_away = event.get("away_team", "")
        if (home_name and event_home and
            (home_name.lower() in event_home.lower() or event_home.lower() in home_name.lower())) and \
           (visitor_name and event_away and
            (visitor_name.lower() in event_away.lower() or event_away.lower() in visitor_name.lower())):
            return event
    return None


def run_prediction(target_date: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    target_date = target_date or datetime.utcnow().strftime("%Y-%m-%d")

    # --- Step 1: Bootstrap historical data ---
    bootstrap_historical_data()

    # --- Step 2: Ensure models (auto-train on first run) ---
    models_existed_before = _verify_models_present()
    model_bundle = ensure_models()
    if not _verify_models_present():
        logger.error("Models missing after ensure_models — aborting")
        sys.exit(1)
    logger.info("Models present: YES")
    training_executed = not models_existed_before

    # --- Step 3: Fetch today's games ---
    client = BallDontLieClient()
    games = sync_date_games(target_date)

    # --- Step 3b: Fetch primary odds (the-odds-api) ---
    primary_odds = fetch_today_odds()

    lines = [f"🏀 NBA每日预测｜{target_date}", "", "━━━━━━━━━━━━━━━━"]
    saved_count = 0
    odds_valid_count = 0

    for g in games:
        game_id = g["id"]
        home = g["home_team"]
        vis = g["visitor_team"]

        opening_spread = None
        live_spread = None
        opening_total = None
        live_total = None
        odds_source = "NONE"

        # --- PRIMARY: the-odds-api.com ---
        matched_event = _match_primary_odds(
            primary_odds, home.get("full_name", ""), vis.get("full_name", "")
        )
        if matched_event:
            opening = extract_opening_line(matched_event)
            live = extract_live_line(matched_event)
            if opening.get("home_spread") is not None and opening.get("total_points") is not None:
                opening_spread = float(opening["home_spread"])
                opening_total = float(opening["total_points"])
                live_spread = float(live["home_spread"]) if live.get("home_spread") is not None else opening_spread
                live_total = float(live["total_points"]) if live.get("total_points") is not None else opening_total
                odds_source = "PRIMARY"
                logger.info("Odds Source: PRIMARY (game %s)", game_id)

        # --- FALLBACK: balldontlie betting_odds ---
        if odds_source == "NONE":
            try:
                odds_data = client.betting_odds(game_ids=game_id, per_page=100)
                logger.info("Odds API debug | game_id=%s | provider_count=%d | response_length=%d",
                            game_id, len(odds_data), len(str(odds_data)))
                opening_payload = {"data": odds_data}
                live_payload = {"data": odds_data}
                store_opening_and_live(game_id, opening_payload, live_payload)
                o_spread, o_total, _ = parse_main_market(opening_payload)
                l_spread, l_total, _ = parse_main_market(live_payload)
                if o_spread is not None and o_total is not None:
                    opening_spread = o_spread
                    opening_total = o_total
                    live_spread = l_spread if l_spread is not None else o_spread
                    live_total = l_total if l_total is not None else o_total
                    odds_source = "BALLDONTLIE"
                    logger.info("Odds Source: BALLDONTLIE (game %s)", game_id)
            except Exception:
                logger.warning("betting_odds unavailable for game %s", game_id, exc_info=True)

        # --- BOTH FAILED: skip prediction ---
        if odds_source == "NONE":
            logger.warning("Odds Source: NONE (game %s) – skipping prediction", game_id)
            lines.extend(
                [
                    f"{zh_name(vis['full_name'])} vs {zh_name(home['full_name'])}",
                    "盘口：暂无数据",
                    "",
                    "━━━━━━━━━━━━━━━━",
                ]
            )
            continue

        odds_valid_count += 1

        logger.info("Loaded odds for game %s", game_id)
        logger.info("  Opening Spread: %s", opening_spread)
        logger.info("  Live Spread: %s", live_spread)
        logger.info("  Opening Total: %s", opening_total)
        logger.info("  Live Total: %s", live_total)

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

        # --- Step 4: Monte Carlo simulation ---
        sim = run_possession_simulation(
            game_id=game_id,
            home_off_rating=112 + hrm,
            visitor_off_rating=112 + vrm,
            pace=98,
            spread_line=live_spread,
            total_line=live_total,
            n_sim=MIN_SIMULATION_COUNT,
        )

        # --- Step 5: Verify simulation count ---
        sim_count = sim.get("simulation_count", 0)
        if sim_count < MIN_SIMULATION_COUNT:
            logger.error("Simulation count %d < %d for game %s — aborting", sim_count, MIN_SIMULATION_COUNT, game_id)
            sys.exit(1)
        logger.info("Simulation runs: %d (game_id=%s)", sim_count, game_id)

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

        # --- Step 6: Save prediction to database ---
        insert_prediction(
            snapshot_date=target_date,
            row={
                "game_id": game_id,
                "home_team": home.get("full_name", ""),
                "away_team": vis.get("full_name", ""),
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
                "simulation_runs": sim_count,
                "odds_source": odds_source,
                "details": {"simulation": sim, "market": market},
            },
        )
        saved_count += 1

    # --- Step 7: Fail if no games have valid odds ---
    if games and odds_valid_count == 0:
        raise RuntimeError("No betting odds returned from any provider")

    # --- Step 8: Verify predictions were saved ---
    if games and saved_count == 0:
        logger.error("No predictions saved — aborting workflow")
        sys.exit(1)

    # --- Step 9: Database validation summary ---
    logger.info("Saved predictions: %d", saved_count)
    logger.info("Models present: YES")
    logger.info("Training executed: %s", "YES" if training_executed else "NO")

    # --- Step 10: Send Telegram (only after training ✔, simulation ✔, database save ✔) ---
    send_message("\n".join(lines))
    logger.info("Telegram message sent successfully")


if __name__ == "__main__":
    run_prediction()
