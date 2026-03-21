from __future__ import annotations

import logging
import sys
from datetime import datetime

import pandas as pd
from wcwidth import wcswidth

from .api_client import BallDontLieClient
from .database import get_conn, insert_prediction
from .data_pipeline import bootstrap_historical_data, sync_date_games, fetch_player_injuries
from .feature_engineering import (
    FEATURE_COLUMNS,
    _compute_team_features,
)
from .injury_model import adjust_for_injuries
from .market_model import apply_market_calibration
from .odds_provider import fetch_today_odds, extract_opening_line, extract_live_line
from .odds_tracker import parse_main_market, store_opening_and_live
from .pace_model import calculate_game_pace as pace_model_calc
from .ppp_model import calculate_ppp as ppp_model_calc
from .prediction_models import MODEL_DIR, MODEL_FILES
from .retrain_engine import ensure_models
from .team_translation import zh_name
from .telegram_bot import send_message, ProgressTracker
from .total_model import calculate_predicted_total

logger = logging.getLogger(__name__)

MIN_SIMULATION_COUNT = 10000

ICON_CORE = "⭐"
ICON_RECOMMEND = "✅"
ICON_NO = "❌"
ICON_OVER = "🟢"
ICON_UNDER = "🔵"


def build_pick_icon(is_core, is_recommend, direction):
    """Return an icon string for the pick recommendation status.

    Args:
        is_core: Whether this is the core pick of the day.
        is_recommend: Whether this game is recommended (unused in new format).
        direction: 'over' or 'under' indicating the predicted direction (unused in new format).
    """
    if is_core:
        return ICON_CORE
    return ""


def pad(text, width):
    """Pad *text* to a fixed display *width* using :func:`wcswidth`.

    Chinese characters occupy 2 columns in a terminal, so plain
    ``str.ljust`` produces misaligned output.  This helper computes
    the real display width and appends the correct number of spaces.
    """
    text = str(text)
    w = wcswidth(text)
    if w < 0:
        # wcswidth returns -1 for non-printable chars; fall back to len
        w = len(text)
    if w < width:
        return text + " " * (width - w)
    return text


def build_prediction_table(games):
    """Build a fixed-width Chinese table for Telegram output.

    Args:
        games: List of dicts with keys: away, home, line, over_prob,
               under_prob, prediction, is_core.
    """

    headers = [
        ("比赛", 22),
        ("盘口", 6),
        ("模型总分", 8),
        ("大分概率", 10),
        ("小分概率", 10),
        ("模型判断", 8),
        ("重心", 4),
    ]

    header_line = "│ " + " │ ".join(pad(h, w) for h, w in headers) + " │"
    display_width = wcswidth(header_line)

    lines = []
    lines.append("┌" + "─" * (display_width - 2) + "┐")
    lines.append(header_line)
    lines.append("├" + "─" * (display_width - 2) + "┤")

    for g in games:
        match = f"{g['away']} vs {g['home']}"
        line = str(g["line"])
        model_total = str(g.get("model_total", ""))
        over_prob = f"{g['over_prob']:.1%}" if isinstance(g['over_prob'], float) else str(g['over_prob'])
        under_prob = f"{g['under_prob']:.1%}" if isinstance(g['under_prob'], float) else str(g['under_prob'])
        prediction = g["prediction"]
        star = ICON_CORE if g.get("is_core") else ""

        row_data = [
            (match, 22),
            (line, 6),
            (model_total, 8),
            (over_prob, 10),
            (under_prob, 10),
            (prediction, 8),
            (star, 4),
        ]

        row = "│ " + " │ ".join(pad(v, w) for v, w in row_data) + " │"
        lines.append(row)

    lines.append("└" + "─" * (display_width - 2) + "┘")

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
    # Keep all columns; callers select FEATURE_COLUMNS as needed.
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

        # --- Build features ---
        feat = _build_prediction_features(home["id"], vis["id"])
        feat_row = feat.iloc[0]

        # --- Extract ratings and pace from features ---
        home_off_rating = float(feat_row.get("home_off_rating", 110.0))
        away_off_rating = float(feat_row.get("away_off_rating", 110.0))
        home_def_rating = float(feat_row.get("home_def_rating", 110.0))
        away_def_rating = float(feat_row.get("away_def_rating", 110.0))
        home_pace_val = float(feat_row.get("home_pace", 98.0))
        away_pace_val = float(feat_row.get("away_pace", 98.0))
        home_3p = float(feat_row.get("home_3p_rate", 0.37))
        away_3p = float(feat_row.get("away_3p_rate", 0.37))
        home_ft = float(feat_row.get("home_ft_rate", 0.27))
        away_ft = float(feat_row.get("away_ft_rate", 0.27))
        home_b2b = bool(feat_row.get("home_b2b", 0.0))
        away_b2b = bool(feat_row.get("away_b2b", 0.0))

        # --- Skip games without valid odds ---
        if odds_source == "NONE":
            logger.warning("Odds Source: NONE (game %s) – skipping game (no line available)", game_id)
            continue

        # --- Module 1: Pace Model ---
        pace_diff = home_pace_val - away_pace_val
        game_pace = pace_model_calc(
            home_pace_val, away_pace_val,
            home_back_to_back=home_b2b,
            away_back_to_back=away_b2b,
        )

        # --- Module 2: PPP Model ---
        home_ppp, away_ppp = ppp_model_calc(
            home_off_rating, away_off_rating,
            home_def_rating, away_def_rating,
            home_3p, away_3p,
            home_ft, away_ft,
        )

        # --- Module 3: Injury Model ---
        home_scorer_out = home["id"] in teams_with_star_out
        away_scorer_out = vis["id"] in teams_with_star_out
        home_ppp, away_ppp, game_pace = adjust_for_injuries(
            home_ppp, away_ppp, game_pace,
            home_scorer_out=home_scorer_out,
            away_scorer_out=away_scorer_out,
        )
        if home_scorer_out:
            logger.info("Injury adjustment applied: %s PPP -= 0.04", home["full_name"])
        if away_scorer_out:
            logger.info("Injury adjustment applied: %s PPP -= 0.04", vis["full_name"])

        # --- Module 4: Total Calculation ---
        predicted_total = calculate_predicted_total(game_pace, home_ppp, away_ppp, live_total)

        # --- Module 5: Market Deviation Correction ---
        predicted_total = apply_market_calibration(predicted_total, live_total)

        # --- Debug output ---
        print("====== MODEL DEBUG ======")
        print("Game Pace:", round(game_pace, 2))
        print("Home PPP:", round(home_ppp, 4))
        print("Away PPP:", round(away_ppp, 4))
        print("Predicted Total:", round(predicted_total, 2))
        print("Closing Line:", live_total)
        print("=========================")

        logger.info("Game Pace: %.2f  Home PPP: %.4f  Away PPP: %.4f", game_pace, home_ppp, away_ppp)
        logger.info("Predicted Total: %.2f  Closing Line: %.1f", predicted_total, live_total)

        # --- Module 6: Formula-based probability (replaces Monte Carlo) ---
        diff = predicted_total - live_total
        over_probability = 0.5 + diff / 18.0
        over_probability = max(0.05, min(0.95, over_probability))
        under_probability = 1.0 - over_probability

        logger.info("Diff: %.2f  Over Prob: %.4f  Under Prob: %.4f", diff, over_probability, under_probability)

        progress.set_game_progress(
            f"⚙️ Game {idx + 1}/{len(games)}: {zh_name(vis['full_name'])} vs {zh_name(home['full_name'])} ✅"
        )

        # --- Module 7: Model Judgment (with diff filter) ---
        if abs(diff) < 2.5:
            total_pick = "PASS"
        elif max(over_probability, under_probability) < 0.58:
            total_pick = "PASS"
        elif diff > 0:
            total_pick = "大分"
        else:
            total_pick = "小分"

        # --- Save prediction to database ---
        prediction_row = {
            "game_id": game_id,
            "home_team": home.get("full_name", ""),
            "away_team": vis.get("full_name", ""),
            "total_pick": total_pick,
            "over_prob": over_probability,
            "under_prob": under_probability,
        }

        insert_prediction(snapshot_date=target_date, row=prediction_row)
        saved_count += 1

        # --- Supabase persistence (mandatory when configured) ---
        from .supabase_client import save_prediction, save_simulation_log
        save_prediction({
            **prediction_row,
            "game_date": target_date,
            "total_line": live_total,
        })
        save_simulation_log({
            "game_id": game_id,
            "model_version": model_bundle.version,
            "simulation_runs": 0,
            "over_probability": over_probability,
            "under_probability": under_probability,
        })

        # --- Collect game result for core pick selection ---
        game_results.append({
            "idx": len(game_results),
            "game_id": game_id,
            "home": home,
            "vis": vis,
            "live_total": live_total,
            "predicted_total": predicted_total,
            "over_probability": over_probability,
            "under_probability": under_probability,
            "odds_source": odds_source,
        })

    # --- Daily recommendation ---
    # Sort all games by confidence = abs(prob_over - 0.5) descending.
    # Core pick: the game with max confidence where probability >= 0.70.
    # Only 1 core pick allowed per day.
    if game_results:
        # Compute confidence for each game
        for gr in game_results:
            gr["confidence"] = abs(gr["over_probability"] - 0.5)
            gr["is_core"] = False

        sorted_results = sorted(game_results, key=lambda x: x["confidence"], reverse=True)

        # Core pick: max confidence game with max(over_prob, under_prob) >= 0.70
        for gr in sorted_results:
            max_prob = max(gr["over_probability"], gr["under_probability"])
            if max_prob >= 0.70:
                gr["is_core"] = True
                break  # Only 1 core pick
    else:
        sorted_results = []

    # --- Build table output for all games ---
    predictions = []
    for gr in sorted_results:
        prob_over = gr["over_probability"]
        prob_under = gr["under_probability"]
        game_diff = gr["predicted_total"] - gr["live_total"]

        if abs(game_diff) < 2.5:
            prediction = "PASS"
        elif max(prob_over, prob_under) < 0.58:
            prediction = "PASS"
        elif game_diff > 0:
            prediction = "大分"
        else:
            prediction = "小分"

        predictions.append({
            "away": zh_name(gr["vis"]["full_name"]),
            "home": zh_name(gr["home"]["full_name"]),
            "line": gr["live_total"],
            "model_total": round(gr["predicted_total"], 1),
            "over_prob": prob_over,
            "under_prob": prob_under,
            "prediction": prediction,
            "is_core": gr["is_core"],
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
    msg = f"\n🏀 NBA大小分预测｜{target_date}\n\n{table}\n"
    send_message(msg)
    logger.info("Telegram message sent successfully")

    # --- Completed ---
    progress.finish()


if __name__ == "__main__":
    run_prediction()
