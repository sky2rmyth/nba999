from __future__ import annotations

import logging

from .api_client import BallDontLieClient
from .i18n_cn import cn
from .retrain_engine import ensure_models
from .telegram_bot import send_message

logger = logging.getLogger(__name__)


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


def _rolling_performance(days: int = 30) -> dict:
    """Compute rolling N-day performance metrics from Supabase review_results."""
    from .supabase_client import fetch_recent_review_results

    rows = fetch_recent_review_results(days)

    spread_rate, total_rate, overall_rate = calculate_rates(rows)
    count = len(rows)

    roi = overall_rate * 2 - 1 if count else 0.0

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

        spread_pick = payload.get("spread_pick", "")
        total_pick = payload.get("total_pick", "")
        live_spread = payload.get("live_spread")
        live_total = payload.get("live_total")

        # Map existing pick strings to the standard field names
        recommended_side = "away" if "受让" in spread_pick else "home"
        if total_pick == "大分":
            recommended_total = "over"
        elif total_pick == "小分":
            recommended_total = "under"
        else:
            recommended_total = ""

        hit_row = {
            "final_home_score": home_score,
            "final_visitor_score": visitor_score,
            "spread": live_spread if live_spread is not None else 0,
            "recommended_side": recommended_side,
            "recommended_total": recommended_total,
            "total_line": live_total if live_total is not None else 0,
        }

        s_hit = spread_hit(hit_row)
        t_hit = total_hit(hit_row)

        spread_hits += int(s_hit)
        total_hits += int(t_hit)
        reviewed += 1

        save_review_result({
            "game_id": gid,
            "home_team": payload.get("home_team"),
            "away_team": payload.get("away_team"),
            "spread_pick": spread_pick,
            "total_pick": total_pick,
            "spread_hit": bool(s_hit),
            "ou_hit": bool(t_hit),
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
