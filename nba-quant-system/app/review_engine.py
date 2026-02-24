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

    res = client.table("predictions") \
        .select("*") \
        .order("created_at", desc=True) \
        .execute()

    if not res.data:
        raise Exception("No predictions found")

    print("==== SAMPLE PREDICTION ROW ====")
    print(res.data[0])
    print("================================")

    rows = res.data

    latest: dict = {}
    for r in rows:
        gid = r["game_id"]
        if gid not in latest:
            latest[gid] = r

    return [parse_prediction(r) for r in latest.values()]


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
    from .supabase_client import save_review_result, fetch_recent_review_results

    predictions = load_latest_predictions()

    for p in predictions:
        result = fetch_game_result(p["game_id"])

        home = result["home_score"]
        away = result["visitor_score"]

        actual_margin = home - away
        actual_total = home + away

        s_hit = (
            actual_margin > 0
            if p["spread_pick"] == "home"
            else actual_margin < 0
        )

        predicted_total = p.get("predicted_total")
        if predicted_total is not None:
            o_hit = (
                actual_total > predicted_total
                if p["total_pick"] == "over"
                else actual_total < predicted_total
            )
        else:
            o_hit = False

        save_review_result({
            "game_id": p["game_id"],
            "spread_pick": p["spread_pick"],
            "total_pick": p["total_pick"],
            "spread_hit": s_hit,
            "ou_hit": o_hit,
            "final_home_score": home,
            "final_visitor_score": away,
        })

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
