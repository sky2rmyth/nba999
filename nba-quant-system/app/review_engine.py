from __future__ import annotations

import logging

from .api_client import BallDontLieClient
from .i18n_cn import cn
from .rating_engine import is_spread_correct, is_total_correct
from .retrain_engine import ensure_models
from .telegram_bot import send_message

logger = logging.getLogger(__name__)


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


def _rolling_performance(days: int = 30) -> dict:
    """Compute rolling N-day performance metrics from Supabase review_results."""
    from .supabase_client import fetch_recent_review_results

    rows = fetch_recent_review_results(days)

    if not rows:
        return {"spread_rate": 0.0, "total_rate": 0.0, "roi": 0.0, "count": 0}

    spread_hits = sum(1 for r in rows if r.get("spread_correct"))
    total_hits = sum(1 for r in rows if r.get("total_correct"))
    count = len(rows)
    total_bets = count * 2

    spread_rate = spread_hits / max(count, 1)
    total_rate = total_hits / max(count, 1)
    roi = ((spread_hits + total_hits) / max(total_bets, 1)) * 2 - 1

    return {"spread_rate": spread_rate, "total_rate": total_rate, "roi": roi, "count": count}


def run_review() -> None:
    from .supabase_client import fetch_all_predictions, save_review_result

    predictions = _deduplicate_predictions(fetch_all_predictions())

    if not predictions:
        send_message(cn("review_no_games"))
        return

    client = BallDontLieClient()

    # Extract unique game_ids
    game_ids = {row["game_id"] for row in predictions if row.get("game_id")}

    # Fetch game status for each game_id
    final_games: dict[int, dict] = {}
    for game_id in game_ids:
        try:
            game = client.get_game(int(game_id))
        except Exception:
            logger.warning("review: could not fetch game %s", game_id)
            continue
        if str(game.get("status", "")) == "Final":
            final_games[game_id] = game

    if not final_games:
        send_message(cn("review_no_games"))
        return

    reviewed = 0
    spread_hits = 0
    total_hits = 0

    for pred in predictions:
        gid = pred.get("game_id")
        if gid not in final_games:
            continue

        game = final_games[gid]
        payload = pred.get("payload") or {}

        home_score = int(game.get("home_team_score", 0))
        visitor_score = int(game.get("visitor_team_score", 0))
        margin = home_score - visitor_score
        total_pts = home_score + visitor_score

        spread_pick = payload.get("spread_pick", "")
        total_pick = payload.get("total_pick", "")
        live_spread = payload.get("live_spread")
        live_total = payload.get("live_total")

        spread_correct = is_spread_correct(spread_pick, margin, live_spread)
        total_correct = is_total_correct(total_pick, total_pts, live_total)

        spread_hits += int(spread_correct)
        total_hits += int(total_correct)
        reviewed += 1

        save_review_result({
            "game_id": gid,
            "home_team": payload.get("home_team"),
            "away_team": payload.get("away_team"),
            "spread_pick": spread_pick,
            "total_pick": total_pick,
            "spread_correct": bool(spread_correct),
            "total_correct": bool(total_correct),
            "final_home_score": home_score,
            "final_visitor_score": visitor_score,
        })

    stake = max(reviewed, 1)
    spread_rate = spread_hits / stake
    total_rate = total_hits / stake
    win_rate = (spread_hits + total_hits) / (2 * stake)

    # Rolling 30-day performance
    rolling = _rolling_performance(30)

    msg = (
        f"📊 复盘报告\n\n"
        f"让分命中率：{spread_rate:.1%}\n"
        f"大小命中率：{total_rate:.1%}\n"
        f"综合胜率：{win_rate:.1%}\n"
        f"复盘场次：{reviewed}\n"
        f"\n━━━━━━━━━━━━━━━━\n"
        f"📈 近30天滚动表现\n"
        f"让分命中率：{rolling['spread_rate']:.1%}\n"
        f"大小命中率：{rolling['total_rate']:.1%}\n"
        f"样本数：{rolling['count']}"
    )
    send_message(msg)
    ensure_models(force=reviewed >= 1)


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
