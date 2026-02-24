from __future__ import annotations

import json
import logging
import os

import requests

from .api_client import BallDontLieClient

logger = logging.getLogger(__name__)

BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY", "")


def spread_hit(row: dict) -> bool:
    """Determine if a spread pick was correct.

    Uses ``final_home_score``, ``final_visitor_score``, ``spread``, and
    ``recommended_side`` fields from *row*.
    """
    actual_margin = row["final_home_score"] - row["final_visitor_score"]
    spread = row["spread"]
    if row["recommended_side"] == "home":
        return actual_margin > spread
    if row["recommended_side"] == "away":
        return actual_margin < spread
    return False


def total_hit(row: dict) -> bool:
    """Determine if a total (over/under) pick was correct.

    Uses ``final_home_score``, ``final_visitor_score``, ``recommended_total``,
    and ``total_line`` fields from *row*.
    """
    actual_total = row["final_home_score"] + row["final_visitor_score"]
    if row["recommended_total"] == "over":
        return actual_total > row["total_line"]
    if row["recommended_total"] == "under":
        return actual_total < row["total_line"]
    return False


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
    """Load the latest prediction per game from Supabase predictions_snapshot."""
    from .supabase_client import _get_client

    client = _get_client()
    if client is None:
        return []

    res = client.table("predictions_snapshot") \
        .select("*") \
        .order("created_at", desc=True) \
        .execute()

    rows = res.data

    latest: dict = {}
    for r in rows:
        gid = r["game_id"]
        if gid not in latest:
            latest[gid] = r

    return list(latest.values())


def fetch_game_result(game_id):
    """Fetch final scores for a game from the BallDontLie API."""
    url = f"https://api.balldontlie.io/v1/games/{game_id}"
    headers = {"Authorization": BALLDONTLIE_API_KEY}
    r = requests.get(url, headers=headers)
    g = r.json()["data"]
    return {
        "home_score": g["home_team_score"],
        "visitor_score": g["visitor_team_score"]
    }


def run_review() -> None:
    predictions = load_latest_predictions()

    review_rows = []

    for p in predictions:
        result = fetch_game_result(p["game_id"])

        home = result["home_score"]
        away = result["visitor_score"]

        actual_margin = home - away
        actual_total = home + away

        spread_hit = (
            actual_margin > p["spread"]
            if p["recommended_side"] == "home"
            else actual_margin < p["spread"]
        )

        ou_hit = (
            actual_total > p["total_line"]
            if p["recommended_total"] == "over"
            else actual_total < p["total_line"]
        )

        review_rows.append({
            "game_id": p["game_id"],
            "spread_hit": spread_hit,
            "ou_hit": ou_hit
        })

    n = len(review_rows)
    spread_rate = sum(r["spread_hit"] for r in review_rows) / n
    total_rate = sum(r["ou_hit"] for r in review_rows) / n

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
