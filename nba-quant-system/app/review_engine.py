from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import requests

from .api_client import BallDontLieClient
from .telegram_bot import send_message

logger = logging.getLogger(__name__)

BALLDONTLIE = "https://api.balldontlie.io/v1"
API_KEY = os.getenv("BALLDONTLIE_API_KEY", "")


def spread_hit(row: dict) -> bool:
    """Determine if a spread pick was correct.

    Uses ``final_home_score``, ``final_visitor_score``, ``spread``, and
    ``spread_pick`` fields from *row*.
    """
    actual_margin = row["final_home_score"] - row["final_visitor_score"]
    spread = row["spread"]
    if row["spread_pick"] == "home":
        return actual_margin > spread
    if row["spread_pick"] == "away":
        return actual_margin < spread
    return False


def total_hit(row: dict) -> bool:
    """Determine if a total (over/under) pick was correct.

    Uses ``final_home_score``, ``final_visitor_score``, ``total_pick``,
    and ``total_line`` fields from *row*.
    """
    actual_total = row["final_home_score"] + row["final_visitor_score"]
    if row["total_pick"] == "over":
        return actual_total > row["total_line"]
    if row["total_pick"] == "under":
        return actual_total < row["total_line"]
    return False


def calc_spread_hit(pred_pick, home_team, visitor_team,
                    final_home, final_visitor,
                    spread_line):
    """Unified betting spread logic.

    adjusted_margin = (home score - visitor score) + spread_line
    If picking home: hit when adjusted_margin > 0
    If picking away: hit when adjusted_margin < 0
    """
    actual_margin = final_home - final_visitor
    adjusted_margin = actual_margin + spread_line

    if pred_pick == "home":
        return adjusted_margin > 0
    if pred_pick == "away":
        return adjusted_margin < 0
    return False


def calc_total_hit(total_pick, final_home, final_visitor, total_line):
    """Over/under hit logic.

    Over: actual total > line
    Under: actual total < line
    """
    final_total = final_home + final_visitor

    if total_pick == "over":
        return final_total > total_line
    if total_pick == "under":
        return final_total < total_line
    return False


def zh_hit(flag):
    """Return Chinese hit/miss indicator."""
    return "✅命中" if flag else "❌未中"


def format_spread_text(pick, home_team, visitor_team, spread):
    """Format spread recommendation in Chinese."""
    if pick == "home":
        return f"{home_team} {spread}"
    else:
        sign = "+" if spread < 0 else "-"
        return f"{visitor_team} {sign}{abs(spread)}"


def format_total_text(total_pick, total_line):
    """Format total recommendation in Chinese."""
    direction = "大分" if total_pick == "over" else "小分"
    return f"{direction} {total_line}"


def build_review_message(result, pred, spread_result, total_result):
    """Build Chinese Telegram review message."""
    home = result["home_team"]
    away = result["visitor_team"]

    spread_text = format_spread_text(
        pred["spread_pick"], home, away, result["spread"]
    )
    total_text = format_total_text(
        pred["total_pick"], result["total"]
    )
    score = f"{result['home_score']} - {result['visitor_score']}"

    message = (
        "📊 NBA复盘结果\n"
        "\n"
        "🏀 对阵：\n"
        f"{home} vs {away}\n"
        "\n"
        "📉 让分推荐：\n"
        f"{spread_text}\n"
        f"结果：{zh_hit(spread_result)}\n"
        "\n"
        "📈 大小分推荐：\n"
        f"{total_text}\n"
        f"结果：{zh_hit(total_result)}\n"
        "\n"
        "🔢 最终比分：\n"
        f"{score}"
    )
    return message


def calculate_rates(rows: list[dict]) -> tuple[float, float, float]:
    """Compute spread, total, and overall hit rates from review result rows.

    Returns ``(spread_rate, total_rate, overall_rate)``.  All values are 0
    when *rows* is empty.
    """
    if not rows:
        return 0, 0, 0

    n = len(rows)
    spread_hits = sum(int(r["spread_hit"]) for r in rows)
    total_hits = sum(int(r["ou_hit"]) for r in rows)

    spread_rate = spread_hits / n
    total_rate = total_hits / n
    overall_rate = (spread_hits + total_hits) / (n * 2)

    return spread_rate, total_rate, overall_rate


def parse_prediction(row: dict) -> dict:
    """Extract prediction fields from a Supabase predictions row.

    All prediction data lives inside ``payload.details``.  This function
    normalises the nested structure into a flat dict suitable for the
    review pipeline.
    """
    payload = row.get("payload", {})
    details = payload.get("details", {})
    sim = details.get("simulation", {})
    total_rating = details.get("total_rating", {})

    predicted_margin = sim.get("predicted_margin")
    predicted_total = sim.get("predicted_total")

    spread_pick = (
        "home" if predicted_margin is not None and predicted_margin > 0 else "away"
    )

    total_pick = (
        "over" if predicted_total is not None and total_rating.get("total_confidence", 0) > 50
        else "under"
    )

    return {
        "game_id": row["game_id"],
        "spread_pick": spread_pick,
        "total_pick": total_pick,
        "predicted_margin": predicted_margin,
        "predicted_total": predicted_total,
    }


def extract_prediction_fields(p: dict) -> tuple[str, str]:
    """Extract spread and total picks from a raw Supabase prediction row.

    Returns ``(spread_pick, total_pick)`` derived from the nested
    ``payload.details.simulation`` structure.
    """
    payload = p.get("payload", {})
    details = payload.get("details", {})
    sim = details.get("simulation", {})

    predicted_margin = sim.get("predicted_margin")
    predicted_total = sim.get("predicted_total")

    spread_pick = "home" if predicted_margin and predicted_margin > 0 else "away"
    total_pick = "over" if predicted_total and predicted_total > 0 else "under"

    return spread_pick, total_pick


def _deduplicate_predictions(predictions: list[dict]) -> list[dict]:
    """Keep only the latest prediction per game_id based on created_at."""
    latest: dict = {}
    for row in predictions:
        gid = row.get("game_id")
        if gid not in latest:
            latest[gid] = row
        else:
            if row.get("created_at", "") > latest[gid].get("created_at", ""):
                latest[gid] = row
    return list(latest.values())


def load_latest_predictions() -> list[dict]:
    """Load the latest prediction per game from Supabase predictions."""
    from .supabase_client import _get_client

    client = _get_client()
    if client is None:
        return []

    res = (
        client.table("predictions")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )

    latest: dict = {}
    for row in res.data:
        gid = row["game_id"]
        if gid not in latest:
            latest[gid] = row

    return list(latest.values())


def fetch_game_result(game_id):
    """Fetch final scores for a game from the BallDontLie API."""
    url = f"{BALLDONTLIE}/games/{game_id}"
    headers = {
        "Authorization": API_KEY
    }

    try:
        r = requests.get(url, headers=headers, timeout=20)

        if r.status_code != 200:
            print("BALLDONTLIE BAD RESPONSE:", r.text)
            return None

        data = r.json()["data"]

        if data["status"] != "Final":
            print("GAME NOT FINISHED:", game_id)
            return None

        home_team_data = data.get("home_team", {})
        visitor_team_data = data.get("visitor_team", {})

        return {
            "home_team": home_team_data.get("full_name", ""),
            "visitor_team": visitor_team_data.get("full_name", ""),
            "home_score": data["home_team_score"],
            "visitor_score": data["visitor_team_score"],
            "spread": 0,
            "total": 0
        }

    except Exception as e:
        print("BALLDONTLIE ERROR:", e)
        return None


def run_review() -> None:
    from .supabase_client import save_review_result, fetch_recent_review_results

    predictions = load_latest_predictions()

    for p in predictions:
        game_id = p["game_id"]
        pred = parse_prediction(p)

        result = fetch_game_result(game_id)
        print("GAME RESULT:", game_id, result)

        if not result:
            print("NO RESULT FOUND:", game_id)
            continue

        final_home = result["home_score"]
        final_visitor = result["visitor_score"]

        spread_result = calc_spread_hit(
            pred["spread_pick"],
            result.get("home_team", ""),
            result.get("visitor_team", ""),
            final_home,
            final_visitor,
            result["spread"]
        )

        total_result = calc_total_hit(
            pred["total_pick"],
            final_home,
            final_visitor,
            result["total"]
        )

        record = {
            "game_id": game_id,
            "spread_pick": pred["spread_pick"],
            "total_pick": pred["total_pick"],
            "spread_hit": spread_result,
            "ou_hit": total_result,
            "final_home_score": final_home,
            "final_visitor_score": final_visitor,
            "reviewed_at": datetime.utcnow().isoformat(),
        }
        save_review_result(record)

        message = build_review_message(
            result, pred, spread_result, total_result
        )
        try:
            send_message(message)
        except Exception:
            logger.debug("Telegram send failed for game %s", game_id)

    review_rows = fetch_recent_review_results()
    n = len(review_rows)
    if n == 0:
        print("Review completed. No games to review.")
        return

    spread_rate = sum(int(r["spread_hit"]) for r in review_rows) / n
    total_rate = sum(int(r["ou_hit"]) for r in review_rows) / n

    report = {
        "games": n,
        "spread_hit_rate": spread_rate,
        "total_hit_rate": total_rate
    }

    with open("review_latest.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Review completed.")


def backfill_review_games() -> list[dict]:
    """Backfill game_date for existing predictions using the BallDontLie API.

    For each prediction stored in Supabase:
    1. Fetch the game via ``/games/{game_id}``.
    2. If the prediction has no ``game_date``, update it with the API value.
    3. Only "Final" games are included in the returned list so they can be
       reviewed without re-predicting.

    Returns a list of API game dicts that have status "Final".
    """
    import logging

    from .supabase_client import (
        fetch_all_predictions,
        update_prediction_game_date,
    )

    logger = logging.getLogger(__name__)
    client = BallDontLieClient()
    predictions = _deduplicate_predictions(fetch_all_predictions())

    if not predictions:
        logger.info("backfill: no predictions found")
        return []

    final_games: list[dict] = []

    for row in predictions:
        game_id = row.get("game_id")
        if not game_id:
            continue

        try:
            game = client.get_game(int(game_id))
        except Exception:
            logger.warning("backfill: could not fetch game %s", game_id)
            continue

        game_date = game.get("date", "")
        if isinstance(game_date, str) and "T" in game_date:
            game_date = game_date.split("T")[0]

        # Update game_date when missing
        if not row.get("game_date") and game_date:
            record_id = row.get("id")
            if record_id is not None:
                try:
                    update_prediction_game_date(int(record_id), game_date)
                except Exception:
                    logger.warning("backfill: could not update game_date for id=%s", record_id)

        # Collect Final games for review
        status = str(game.get("status", ""))
        if status.startswith("Final"):
            game["game_date"] = game_date
            game["home_score"] = game.get("home_team_score", 0)
            game["away_score"] = game.get("visitor_team_score", 0)
            final_games.append(game)

    logger.info("backfill: processed %d predictions, %d final games", len(predictions), len(final_games))
    return final_games


if __name__ == "__main__":
    print("Starting review process...")
    run_review()
