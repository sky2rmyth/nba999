from __future__ import annotations

import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from .api_client import BallDontLieClient
from .bookmaker_behavior import analyze_line_behavior
from .database import get_conn, insert_prediction
from .data_pipeline import bootstrap_historical_data, sync_date_games
from .feature_engineering import FEATURE_COLUMNS, _compute_team_features
from .game_simulator import run_possession_simulation
from .odds_provider import fetch_today_odds, extract_opening_line, extract_live_line
from .odds_tracker import parse_main_market, store_opening_and_live
from .prediction_models import MODEL_DIR, MODEL_FILES
from .rating_engine import compute_spread_rating, compute_total_rating
from .retrain_engine import ensure_models
from .team_translation import zh_name
from .telegram_bot import send_message, ProgressTracker

logger = logging.getLogger(__name__)

MIN_SIMULATION_COUNT = 10000

# Hybrid model blending weights: Monte Carlo simulation vs classifier model.
# MC simulation captures game-specific dynamics; classifier captures historical patterns.
MC_WEIGHT = 0.6
CLASSIFIER_WEIGHT = 0.4


def _verify_models_present() -> bool:
    return all((MODEL_DIR / f).exists() for f in MODEL_FILES)


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


def _build_prediction_features(home_id: int, away_id: int) -> pd.DataFrame:
    """Build feature row for prediction using live game data."""
    with get_conn() as conn:
        # Use tomorrow's date to include all available data
        future_date = "9999-12-31"
        home_feat = _compute_team_features(conn, home_id, away_id, future_date, "home")
        away_feat = _compute_team_features(conn, away_id, home_id, future_date, "away")

    row = {}
    row.update(home_feat)
    row.update(away_feat)
    row["home_indicator"] = 1.0
    home_pace = row.get("home_pace", 98.0)
    away_pace = row.get("away_pace", 98.0)
    row["pace_interaction"] = home_pace * away_pace / 100.0

    feat = pd.DataFrame([row])
    for col in FEATURE_COLUMNS:
        if col not in feat.columns:
            feat[col] = 0.0
    return feat[FEATURE_COLUMNS]


def run_prediction(target_date: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    target_date = target_date or datetime.utcnow().strftime("%Y-%m-%d")

    # --- Progress tracker ---
    progress = ProgressTracker()
    progress.start()  # 🟡 System Starting

    # --- Step 1: Bootstrap historical data ---
    progress.advance(1)  # 🔵 Fetching Games Data
    bootstrap_historical_data()

    # --- Step 2: Ensure models (auto-train on first run) ---
    progress.advance(2)  # 🧠 Loading Models
    model_bundle = ensure_models()
    if not _verify_models_present():
        logger.error("Models missing after ensure_models — aborting")
        sys.exit(1)
    logger.info("Models present: YES | Version: %s | Source: %s",
                model_bundle.version, getattr(model_bundle, "source", "unknown"))

    # --- Verification: feature count ---
    feature_count = len(FEATURE_COLUMNS)
    logger.info("Feature count: %d", feature_count)
    if feature_count < 30:
        logger.error("Feature count %d < 30 — aborting", feature_count)
        sys.exit(1)

    # --- Step 3: Fetch today's games ---
    client = BallDontLieClient()
    games = sync_date_games(target_date)

    # --- Step 3b: Fetch primary odds (the-odds-api) ---
    primary_odds = fetch_today_odds()

    # --- Monte Carlo phase ---
    progress.advance(3)  # ⚙️ Running Monte Carlo Simulation

    lines = [f"🏀 NBA每日预测｜{target_date}", "", "━━━━━━━━━━━━━━━━"]
    saved_count = 0
    odds_valid_count = 0

    for idx, g in enumerate(games):
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

        # --- Build features and predict scores ---
        feat = _build_prediction_features(home["id"], vis["id"])

        predicted_home_score = float(model_bundle.home_score_model.predict(feat)[0])
        predicted_away_score = float(model_bundle.away_score_model.predict(feat)[0])

        predicted_margin = predicted_home_score - predicted_away_score
        predicted_total = predicted_home_score + predicted_away_score

        logger.info("Predicted Home Score: %.1f  Away Score: %.1f", predicted_home_score, predicted_away_score)
        logger.info("Predicted Margin: %.1f  Total: %.1f", predicted_margin, predicted_total)

        # --- Hybrid: Spread Cover & Total model predictions ---
        spread_cover_prob_model = None
        total_over_prob_model = None
        if model_bundle.spread_cover_model is not None:
            try:
                spread_cover_prob_model = float(model_bundle.spread_cover_model.predict_proba(feat)[0][1])
                logger.info("Spread Cover Model Prob: %.2f%%", spread_cover_prob_model * 100)
            except Exception:
                pass
        if model_bundle.total_model is not None:
            try:
                total_over_prob_model = float(model_bundle.total_model.predict_proba(feat)[0][1])
                logger.info("Total Over Model Prob: %.2f%%", total_over_prob_model * 100)
            except Exception:
                pass

        # --- Compute variance from recent games ---
        home_feat_row = feat.iloc[0]
        home_var = max(float(home_feat_row.get("home_scoring_variance", 64.0)), 64.0)
        away_var = max(float(home_feat_row.get("away_scoring_variance", 64.0)), 64.0)

        # --- Step 4: Monte Carlo simulation ---
        sim = run_possession_simulation(
            game_id=game_id,
            predicted_home_score=predicted_home_score,
            predicted_away_score=predicted_away_score,
            home_variance=home_var,
            away_variance=away_var,
            spread_line=live_spread,
            total_line=live_total,
            n_sim=MIN_SIMULATION_COUNT,
        )

        progress.set_game_progress(
            f"⚙️ Game {idx + 1}/{len(games)}: {zh_name(vis['full_name'])} vs {zh_name(home['full_name'])} ✅"
        )

        # --- Step 5: Verify simulation count ---
        sim_count = sim.get("simulation_count", 0)
        if sim_count < MIN_SIMULATION_COUNT:
            logger.error("Simulation count %d < %d for game %s — aborting", sim_count, MIN_SIMULATION_COUNT, game_id)
            sys.exit(1)
        logger.info("Simulation runs: %d (game_id=%s)", sim_count, game_id)

        # --- Hybrid spread decision: combine Monte Carlo + classifier ---
        mc_spread_prob = sim["spread_cover_probability"]
        if spread_cover_prob_model is not None:
            combined_spread_prob = MC_WEIGHT * mc_spread_prob + CLASSIFIER_WEIGHT * spread_cover_prob_model
        else:
            combined_spread_prob = mc_spread_prob

        if predicted_margin > -live_spread:
            spread_pick = f"主队 {zh_name(home['full_name'])} {live_spread:+.1f}"
            spread_pick_label = "home_cover"
        else:
            spread_pick = f"客队 {zh_name(vis['full_name'])} {-live_spread:+.1f}（受让）"
            spread_pick_label = "away_cover"

        # --- Hybrid total decision: combine Monte Carlo + classifier ---
        mc_total_prob = sim["over_probability"]
        if total_over_prob_model is not None:
            combined_total_prob = MC_WEIGHT * mc_total_prob + CLASSIFIER_WEIGHT * total_over_prob_model
        else:
            combined_total_prob = mc_total_prob

        if predicted_total > live_total:
            total_pick = "大分"
        else:
            total_pick = "小分"

        # --- Rating from simulation edge ---
        spread_rating = compute_spread_rating(combined_spread_prob, live_spread)
        total_rating = compute_total_rating(combined_total_prob, live_total)

        spread_edge = combined_spread_prob - 0.5
        total_edge = combined_total_prob - 0.5
        market = analyze_line_behavior(opening_spread, live_spread, spread_edge, opening_total, live_total, total_edge)

        spread_confidence = spread_rating["spread_confidence"]
        total_confidence = total_rating["total_confidence"]
        spread_stars = spread_rating["spread_stars"]
        total_stars = total_rating["total_stars"]

        # --- Telegram format (Part 8) ---
        lines.extend([
            f"{zh_name(vis['full_name'])} vs {zh_name(home['full_name'])}",
            "",
            "让分：",
            f"主队 {zh_name(home['full_name'])} {live_spread:+.1f}",
            f"推荐：{spread_pick}",
            f"信心：{spread_confidence:.0f}%",
            "",
            "大小：",
            f"{live_total:.1f}",
            f"推荐：{total_pick}",
            f"信心：{total_confidence:.0f}%",
            "",
            "━━━━━━━━━━━━━━━━",
        ])

        # --- Step 6: Save prediction to database ---
        spread_prob = combined_spread_prob
        total_prob = combined_total_prob
        overall_confidence = max(abs(spread_edge), abs(total_edge))
        overall_stars = max(spread_stars, total_stars)
        recommendation_idx = max(spread_rating["spread_recommendation_index"],
                                 total_rating["total_recommendation_index"])

        prediction_row = {
            "game_id": game_id,
            "home_team": home.get("full_name", ""),
            "away_team": vis.get("full_name", ""),
            "prediction_time": datetime.utcnow().isoformat(),
            "spread_pick": spread_pick,
            "spread_prob": spread_prob,
            "total_pick": total_pick,
            "total_prob": total_prob,
            "confidence_score": round(overall_confidence, 4),
            "star_rating": overall_stars,
            "recommendation_index": recommendation_idx,
            "expected_home_score": sim["expected_home_score"],
            "expected_visitor_score": sim["expected_visitor_score"],
            "simulation_variance": sim["score_distribution_variance"],
            "opening_spread": opening_spread,
            "live_spread": live_spread,
            "opening_total": opening_total,
            "live_total": live_total,
            "simulation_runs": sim_count,
            "odds_source": odds_source,
            "model_version": model_bundle.version,
            "feature_count": feature_count,
            "spread_edge": round(spread_edge, 4),
            "total_edge": round(total_edge, 4),
            "details": {"simulation": sim, "market": market,
                        "spread_rating": spread_rating, "total_rating": total_rating},
        }

        insert_prediction(snapshot_date=target_date, row=prediction_row)
        saved_count += 1

        # --- Supabase persistence (mandatory when configured) ---
        from .supabase_client import save_prediction, save_simulation_log
        save_prediction({
            **prediction_row,
            "game_date": target_date,
            "spread_line": live_spread,
            "total_line": live_total,
            "spread_confidence": spread_confidence,
            "total_confidence": total_confidence,
        })
        save_simulation_log({
            "game_id": game_id,
            "model_version": model_bundle.version,
            "simulation_runs": sim_count,
            **sim,
        })

    # --- Saving phase ---
    progress.advance(4)  # 💾 Saving Results

    # --- Step 7: Fail if no games have valid odds ---
    if games and odds_valid_count == 0:
        raise RuntimeError("No betting odds returned from any provider")

    # --- Step 8: Verify predictions were saved ---
    if games and saved_count == 0:
        logger.error("No predictions saved — aborting workflow")
        sys.exit(1)

    # --- Step 9: Database validation summary ---
    model_source = getattr(model_bundle, "source", "unknown")
    logger.info("Saved predictions: %d", saved_count)
    logger.info("Models present: YES")
    logger.info("MODEL SOURCE: %s", model_source)
    logger.info("Model version: %s", model_bundle.version)

    # --- Step 10: Send Telegram (only after training ✔, simulation ✔, database save ✔) ---
    send_message("\n".join(lines))
    logger.info("Telegram message sent successfully")

    # --- Completed ---
    progress.finish()  # ✅ Completed


if __name__ == "__main__":
    run_prediction()
