from __future__ import annotations

import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from .api_client import BallDontLieClient
from .bookmaker_behavior import analyze_line_behavior
from .database import get_conn, insert_prediction
from .data_pipeline import bootstrap_historical_data, sync_date_games, fetch_player_injuries
from .feature_engineering import (
    FEATURE_COLUMNS, TOTAL_FEATURE_COLUMNS,
    _compute_team_features, calculate_game_pace, calculate_ppp,
    calculate_three_point_rate, calculate_free_throw_rate,
    calculate_orb_rate, calculate_tov_rate,
)
from .game_simulator import run_possession_simulation
from .odds_provider import fetch_today_odds, extract_opening_line, extract_live_line
from .odds_tracker import parse_main_market, store_opening_and_live
from .prediction_models import MODEL_DIR, MODEL_FILES
from .rating_engine import (
    compute_spread_rating, compute_total_rating,
    compute_edge_score, stars_display,
)
from .retrain_engine import ensure_models
from .team_translation import zh_name
from .telegram_bot import send_message, ProgressTracker

logger = logging.getLogger(__name__)

MIN_SIMULATION_COUNT = 10000

# Hybrid model blending weights: Monte Carlo simulation vs classifier model.
# MC simulation captures game-specific dynamics; classifier captures historical patterns.
MC_WEIGHT = 0.6
CLASSIFIER_WEIGHT = 0.4

# Blending weights for last-10 games vs full-season ratings (stability improvement)
RECENT_WEIGHT = 0.6
SEASON_WEIGHT = 0.4

# Blending weights for ML prediction vs rating-based prediction
ML_WEIGHT = 0.7
RATING_WEIGHT = 0.3

# Minimum pace divisor to avoid division by zero in rating calculations
MIN_PACE_DIVISOR = 80

# Probability calibration: shrink raw probability toward neutral (0.5)
PROB_RAW_WEIGHT = 0.7
PROB_NEUTRAL_WEIGHT = 0.3
NEUTRAL_PROBABILITY = 0.5

# Injury adjustment: reduce offensive rating when a team has player(s) ruled out
INJURY_RATING_FACTOR = 0.96

# Monte Carlo simulation standard deviation for total
MC_TOTAL_STD = 8.5

# League average total for bias calibration
LEAGUE_AVG_TOTAL = 229

# League-average rate constants for possession-based efficiency adjustment
LEAGUE_AVG_THREE_POINT_RATE = 0.37
LEAGUE_AVG_FREE_THROW_RATE = 0.27
LEAGUE_AVG_ORB_RATE = 0.25
LEAGUE_AVG_TOV_RATE = 0.13

ICON_CORE = "⭐"
ICON_RECOMMEND = "✅"
ICON_NO = "❌"
ICON_OVER = "🟢"
ICON_UNDER = "🔵"


def build_pick_icon(is_core, is_recommend, direction):
    """Return an icon string for the pick recommendation status.

    Args:
        is_core: Whether this is the core pick of the day.
        is_recommend: Whether this game is recommended.
        direction: 'over' or 'under' indicating the predicted direction.
    """

    if direction == "over":
        d = ICON_OVER + "大"
    else:
        d = ICON_UNDER + "小"

    if is_core:
        return ICON_CORE + d

    if is_recommend:
        return ICON_RECOMMEND + d

    return ICON_NO


def build_prediction_table(games):
    """Build a pipe-separated table string for Telegram output.

    Args:
        games: List of dicts with keys: away, home, line, pred_total,
               edge, prob, low, high, direction, is_core, is_recommend.
    """

    lines = []

    header = (
        "比赛 | 盘口 | 模型 | Edge | 概率 | 区间 | 推荐\n"
        "------------------------------------------------"
    )

    lines.append(header)

    for g in games:

        match = f"{g['away']} vs {g['home']}"
        line = g["line"]
        model = round(g["pred_total"], 1)
        edge = round(g["edge"], 1)

        prob = int(g["prob"] * 100)

        interval = f"{int(g['low'])}-{int(g['high'])}"

        pick = build_pick_icon(
            g["is_core"],
            g["is_recommend"],
            g["direction"]
        )

        row = (
            f"{match} | "
            f"{line} | "
            f"{model} | "
            f"{edge:+} | "
            f"{prob}% | "
            f"{interval} | "
            f"{pick}"
        )

        lines.append(row)

    return "\n".join(lines)


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
    """Build feature row for prediction using live game data.

    Returns a DataFrame that contains *at least* ``FEATURE_COLUMNS``.  Extra
    columns produced by ``_compute_team_features`` (e.g. ``last5_*``,
    ``*_3p_rate``, ``*_ft_rate``) are preserved so that the total classifier
    can use them.
    """
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
    # Keep all columns; callers select FEATURE_COLUMNS or TOTAL_FEATURE_COLUMNS as needed.
    return feat


def run_prediction(target_date: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    target_date = target_date or datetime.utcnow().strftime("%Y-%m-%d")

    # --- Progress tracker ---
    progress = ProgressTracker()

    # --- Step 2: Ensure models first to determine source ---
    model_bundle = ensure_models()
    if not _verify_models_present():
        logger.error("Models missing after ensure_models — aborting")
        sys.exit(1)
    model_source = getattr(model_bundle, "source", "unknown")
    logger.info("Models present: YES | Version: %s | Source: %s",
                model_bundle.version, model_source)

    # Set model source on tracker so Supabase-loaded models skip boot stages
    progress.model_source = model_source
    progress.start()

    # --- Step 1: Bootstrap historical data ---
    progress.advance(1)
    bootstrap_historical_data()

    # --- Loading Models stage ---
    progress.advance(2)

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
    progress.advance(3)

    saved_count = 0
    odds_valid_count = 0
    telegram_count = 0
    game_results: list[dict] = []  # Collect per-game data for core pick selection

    # --- Fetch player injuries for adjustment ---
    try:
        injury_list = fetch_player_injuries()
    except Exception:
        logger.warning("Could not fetch player injuries — skipping injury adjustment")
        injury_list = []

    # Build a set of team_ids that have injured players marked as "out"
    teams_with_star_out: set[int] = set()
    for inj in injury_list:
        status = str(inj.get("status", "")).lower()
        if status == "out":
            team = inj.get("team", {})
            team_id = team.get("id") if isinstance(team, dict) else None
            if team_id is not None:
                teams_with_star_out.add(int(team_id))

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

        if odds_source != "NONE":
            odds_valid_count += 1

        logger.info("Loaded odds for game %s (source: %s)", game_id, odds_source)
        logger.info("  Opening Spread: %s", opening_spread)
        logger.info("  Live Spread: %s", live_spread)
        logger.info("  Opening Total: %s", opening_total)
        logger.info("  Live Total: %s", live_total)

        # --- Build features and predict scores ---
        feat = _build_prediction_features(home["id"], vis["id"])

        predicted_home_score = float(model_bundle.home_score_model.predict(feat[FEATURE_COLUMNS])[0])
        predicted_away_score = float(model_bundle.away_score_model.predict(feat[FEATURE_COLUMNS])[0])

        # --- Model input upgrade: blend last-10 and season ratings ---
        feat_row = feat.iloc[0]
        season_home_off = float(feat_row.get("home_off_rating", 110.0))
        season_away_off = float(feat_row.get("away_off_rating", 110.0))
        season_home_def = float(feat_row.get("home_def_rating", 110.0))
        season_away_def = float(feat_row.get("away_def_rating", 110.0))
        season_home_pace = float(feat_row.get("home_pace", 98.0))
        season_away_pace = float(feat_row.get("away_pace", 98.0))

        home_avg10 = float(feat_row.get("home_avg_score_last10", predicted_home_score))
        home_allowed10 = float(feat_row.get("home_avg_allowed_last10", predicted_away_score))
        away_avg10 = float(feat_row.get("away_avg_score_last10", predicted_away_score))
        away_allowed10 = float(feat_row.get("away_avg_allowed_last10", predicted_home_score))

        last10_home_pace = (home_avg10 + home_allowed10) / 2.14
        last10_away_pace = (away_avg10 + away_allowed10) / 2.14

        last10_home_off = (home_avg10 / max(last10_home_pace, MIN_PACE_DIVISOR)) * 100.0
        last10_away_off = (away_avg10 / max(last10_away_pace, MIN_PACE_DIVISOR)) * 100.0
        last10_home_def = (home_allowed10 / max(last10_home_pace, MIN_PACE_DIVISOR)) * 100.0
        last10_away_def = (away_allowed10 / max(last10_away_pace, MIN_PACE_DIVISOR)) * 100.0

        # Blended ratings: 0.6 last-10 + 0.4 season
        home_off = RECENT_WEIGHT * last10_home_off + SEASON_WEIGHT * season_home_off
        away_off = RECENT_WEIGHT * last10_away_off + SEASON_WEIGHT * season_away_off
        home_def = RECENT_WEIGHT * last10_home_def + SEASON_WEIGHT * season_home_def
        away_def = RECENT_WEIGHT * last10_away_def + SEASON_WEIGHT * season_away_def

        home_pace_blend = 0.7 * last10_home_pace + 0.3 * season_home_pace
        away_pace_blend = 0.7 * last10_away_pace + 0.3 * season_away_pace

        # Injury adjustment: reduce off_rating when team has player(s) ruled out
        if home["id"] in teams_with_star_out:
            home_off *= INJURY_RATING_FACTOR
            logger.info("Injury adjustment applied: %s off_rating * %.2f", home["full_name"], INJURY_RATING_FACTOR)
        if vis["id"] in teams_with_star_out:
            away_off *= INJURY_RATING_FACTOR
            logger.info("Injury adjustment applied: %s off_rating * %.2f", vis["full_name"], INJURY_RATING_FACTOR)

        # Game pace via matchup pace formula (0.55 * fast + 0.45 * slow)
        game_pace = calculate_game_pace(home_pace_blend, away_pace_blend)

        # PPP derived from (possibly injury-adjusted) off_rating
        home_ppp = calculate_ppp(home_off)
        away_ppp = calculate_ppp(away_off)

        # PPP safety limits
        home_ppp = max(1.03, min(home_ppp, 1.18))
        away_ppp = max(1.03, min(away_ppp, 1.18))

        # Final predicted total from possession model (before market anchor)
        model_total = game_pace * (home_ppp + away_ppp)

        # Keep ML-based margin for spread analysis
        predicted_margin = predicted_home_score - predicted_away_score

        logger.info("Predicted Home Score: %.1f  Away Score: %.1f", predicted_home_score, predicted_away_score)
        logger.info("Predicted Margin: %.1f  Model Total: %.1f", predicted_margin, model_total)

        # --- Skip games without valid odds ---
        if odds_source == "NONE":
            logger.warning("Odds Source: NONE (game %s) – skipping game (no line available)", game_id)
            continue

        # --- Market Anchor: blend model total with closing line ---
        predicted_total = 0.65 * model_total + 0.35 * live_total

        # --- Debug output ---
        total_edge_debug = predicted_total - live_total
        print("====== MODEL DEBUG ======")
        print("Game Pace:", game_pace)
        print("Home PPP:", home_ppp)
        print("Away PPP:", away_ppp)
        print("Predicted Total:", predicted_total)
        print("Closing Line:", live_total)
        print("Edge:", total_edge_debug)
        print("=========================")

        # --- Hybrid: Spread Cover & Total model predictions ---
        spread_cover_prob_model = None
        total_over_prob_model = None
        if model_bundle.spread_cover_model is not None:
            try:
                spread_cover_prob_model = float(model_bundle.spread_cover_model.predict_proba(feat[FEATURE_COLUMNS])[0][1])
                logger.info("Spread Cover Model Prob: %.2f%%", spread_cover_prob_model * 100)
            except Exception:
                pass

        # --- Total classifier: build feature vector with TOTAL_FEATURE_COLUMNS ---
        if model_bundle.total_model is not None and live_total is not None:
            try:
                feat_row = feat.iloc[0]
                total_feat_row = {
                    "closing_total": live_total,
                    "opening_total": opening_total if opening_total is not None else live_total,
                    "line_movement": (live_total - opening_total) if opening_total is not None else 0.0,
                    "home_pace": float(feat_row.get("home_pace", 98.0)),
                    "away_pace": float(feat_row.get("away_pace", 98.0)),
                    "home_off_rating": float(feat_row.get("home_off_rating", 110.0)),
                    "away_off_rating": float(feat_row.get("away_off_rating", 110.0)),
                    "home_def_rating": float(feat_row.get("home_def_rating", 110.0)),
                    "away_def_rating": float(feat_row.get("away_def_rating", 110.0)),
                    "home_3p_rate": float(feat_row.get("home_3p_rate", 0.37)),
                    "away_3p_rate": float(feat_row.get("away_3p_rate", 0.37)),
                    "home_ft_rate": float(feat_row.get("home_ft_rate", 0.27)),
                    "away_ft_rate": float(feat_row.get("away_ft_rate", 0.27)),
                    "last5_home_off_rating": float(feat_row.get("last5_home_off_rating", 110.0)),
                    "last5_away_off_rating": float(feat_row.get("last5_away_off_rating", 110.0)),
                    "last5_home_pace": float(feat_row.get("last5_home_pace", 98.0)),
                    "last5_away_pace": float(feat_row.get("last5_away_pace", 98.0)),
                    "rest_days_home": float(feat_row.get("home_rest_days", 2.0)),
                    "rest_days_away": float(feat_row.get("away_rest_days", 2.0)),
                }
                total_feat_df = pd.DataFrame([total_feat_row])[TOTAL_FEATURE_COLUMNS]
                total_over_prob_model = float(model_bundle.total_model.predict_proba(total_feat_df)[0][1])
            except Exception:
                logger.debug("Total classifier prediction failed", exc_info=True)

        # --- Total classifier output & prediction rule ---
        if total_over_prob_model is not None:
            prob_over = total_over_prob_model
            prob_under = 1.0 - prob_over
            if prob_over > 0.5:
                total_prediction = "Over"
                total_pick = "大分"
            else:
                total_prediction = "Under"
                total_pick = "小分"

            logger.info("Predicted Over Probability: %.2f%%", prob_over * 100)
            logger.info("Predicted Under Probability: %.2f%%", prob_under * 100)
            logger.info("Prediction: %s", total_prediction)
            logger.info("Closing Line: %.1f", live_total)
            logger.info("Actual Result: pending")

            print("====== TOTAL CLASSIFIER ======")
            print("Predicted Over Probability:", round(prob_over, 4))
            print("Predicted Under Probability:", round(prob_under, 4))
            print("Prediction:", total_prediction)
            print("Closing Line:", live_total)
            print("Actual Result: pending")
            print("==============================")
        else:
            prob_over = None
            prob_under = None
            total_prediction = None

        # --- Step 4: Monte Carlo simulation ---
        sim = run_possession_simulation(
            game_id=game_id,
            game_pace=game_pace,
            home_adj_ppp=home_ppp,
            away_adj_ppp=away_ppp,
            predicted_total=predicted_total,
            closing_total=live_total,
            spread_line=live_spread,
            n_sim=MIN_SIMULATION_COUNT,
            total_std=MC_TOTAL_STD,
        )

        print("Simulation Low:", sim["simulation_low"])
        print("Simulation High:", sim["simulation_high"])
        print("Simulation Std:", sim["total_std"])
        print("Over Probability:", sim["over_probability"])
        print("Under Probability:", sim["under_probability"])
        print("=========================")

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
        if prob_over is not None:
            combined_total_prob = MC_WEIGHT * mc_total_prob + CLASSIFIER_WEIGHT * prob_over
        else:
            combined_total_prob = mc_total_prob

        # --- Probability calibration: shrink toward neutral ---
        calibrated_total_prob = PROB_RAW_WEIGHT * combined_total_prob + PROB_NEUTRAL_WEIGHT * NEUTRAL_PROBABILITY
        combined_total_prob = calibrated_total_prob

        # Total pick: use classifier prediction rule when available, else use model total
        if total_prediction is not None:
            # total_pick already set by classifier above
            pass
        elif predicted_total > live_total:
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

        # --- Edge scoring ---
        overall_edge_raw = max(abs(spread_edge), abs(total_edge)) * 100.0
        edge_score = compute_edge_score(
            max(combined_spread_prob, combined_total_prob)
        )
        clv_projection = round((live_spread - opening_spread) if opening_spread and live_spread else 0.0, 2)

        # --- Total edge & signal score ---
        total_edge_pts = predicted_total - live_total
        abs_edge = abs(total_edge_pts)
        total_std = sim["total_std"]
        over_probability = combined_total_prob

        signal_score = (
            abs_edge * 0.6
            + over_probability * 40
            - total_std * 0.2
        )

        # --- Recommendation reason (based on abs_edge) ---
        if abs_edge >= 7:
            reason = "模型预测与盘口差距较大"
        elif abs_edge >= 4:
            reason = "模型预测存在明显价值"
        else:
            reason = "信号较弱，不推荐"

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
            "edge_score": edge_score,
            "clv_projection": clv_projection,
            "home_win_probability": sim.get("home_win_probability", 0.0),
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

        # --- Collect game result for core pick selection ---
        total_range = f"{int(sim['total_5pct'])} – {int(sim['total_95pct'])}"
        under_probability = 1.0 - over_probability

        game_results.append({
            "idx": len(game_results),
            "game_id": game_id,
            "home": home,
            "vis": vis,
            "live_total": live_total,
            "predicted_total": predicted_total,
            "total_edge_pts": total_edge_pts,
            "over_probability": over_probability,
            "under_probability": under_probability,
            "total_range": total_range,
            "low": sim['total_5pct'],
            "high": sim['total_95pct'],
            "reason": reason,
            "signal_score": signal_score,
            "odds_source": odds_source,
        })

    # --- Daily recommendation ---
    # Edge-only filter: abs(edge) < 4 → not recommended.
    # abs(edge) >= 4 and < 7 → recommended.
    # abs(edge) >= 7 → star_pick.
    # Min recommendations based on game count; auto-fill if needed.
    # Guaranteed 1 core (star) pick per day.
    # Cap at 5 recommendations.
    MAX_DAILY_RECOMMENDATIONS = 5
    if game_results:
        sorted_results = sorted(game_results, key=lambda x: abs(x["total_edge_pts"]), reverse=True)

        # Determine minimum recommendations based on game count
        total_games = len(sorted_results)
        if total_games >= 5:
            min_recommend = 4
        elif total_games == 4:
            min_recommend = 3
        else:
            min_recommend = 1

        for gr in sorted_results:
            abs_edge_val = abs(gr["total_edge_pts"])
            if abs_edge_val >= 4:
                gr["recommended"] = True
            else:
                gr["recommended"] = False
            if abs_edge_val >= 7:
                gr["star_pick"] = True
            else:
                gr["star_pick"] = False
            gr["is_core"] = False

        # Collect quality-filtered recommendations
        recommended_games = [gr for gr in sorted_results if gr["recommended"]]

        # Auto-fill if recommendations below minimum
        if len(recommended_games) < min_recommend:
            for gr in sorted_results:
                if not gr["recommended"]:
                    gr["recommended"] = True
                    recommended_games.append(gr)
                if len(recommended_games) >= min_recommend:
                    break

        # Cap recommendations at MAX_DAILY_RECOMMENDATIONS by abs(edge)
        if len(recommended_games) > MAX_DAILY_RECOMMENDATIONS:
            recommended_games = sorted(
                recommended_games, key=lambda x: abs(x["total_edge_pts"]), reverse=True
            )[:MAX_DAILY_RECOMMENDATIONS]
            # Mark excess as not recommended
            keep_indices = {g["idx"] for g in recommended_games}
            for gr in sorted_results:
                if gr["recommended"] and gr["idx"] not in keep_indices:
                    gr["recommended"] = False

        # Guaranteed core pick: prefer star_pick with largest abs(edge)
        star_results = [gr for gr in sorted_results if gr["recommended"] and gr["star_pick"]]
        if star_results:
            star_results[0]["is_core"] = True
        else:
            # No star_pick qualifies — promote the recommended game with largest abs(edge)
            rec_sorted = sorted(
                [gr for gr in sorted_results if gr["recommended"]],
                key=lambda x: abs(x["total_edge_pts"]),
                reverse=True,
            )
            if rec_sorted:
                rec_sorted[0]["star_pick"] = True
                rec_sorted[0]["is_core"] = True
    else:
        sorted_results = []

    # --- Build table output for all games ---
    predictions = []
    for gr in sorted_results:
        direction = "over" if gr["total_edge_pts"] > 0 else "under"
        prob = gr["over_probability"] if direction == "over" else gr["under_probability"]
        predictions.append({
            "away": zh_name(gr["vis"]["full_name"]),
            "home": zh_name(gr["home"]["full_name"]),
            "line": gr["live_total"],
            "pred_total": gr["predicted_total"],
            "edge": gr["total_edge_pts"],
            "prob": prob,
            "low": gr["low"],
            "high": gr["high"],
            "direction": direction,
            "is_core": gr["is_core"],
            "is_recommend": gr["recommended"],
        })

    table = build_prediction_table(predictions)

    # --- Saving phase ---
    progress.advance(4)

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
    msg = f"\n🏀 NBA量化预测｜{target_date}\n\n{table}\n"
    send_message(msg)
    logger.info("Telegram message sent successfully")

    # --- Completed ---
    progress.finish()


if __name__ == "__main__":
    run_prediction()
