from __future__ import annotations

from datetime import datetime, timedelta

from .api_client import BallDontLieClient
from .database import get_conn, save_result
from .i18n_cn import cn
from .rating_engine import is_spread_correct, is_total_correct
from .retrain_engine import ensure_models
from .telegram_bot import send_message


def _rolling_performance(days: int = 30) -> dict:
    """Compute rolling N-day performance metrics."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.spread_pick, p.total_pick, p.live_spread, p.live_total,
                   r.final_home_score, r.final_visitor_score, p.snapshot_date
            FROM predictions_snapshot p
            JOIN results r ON p.game_id = r.game_id
            WHERE p.snapshot_date >= ? AND p.is_final_prediction = 1
            ORDER BY p.snapshot_date DESC
            """,
            (cutoff,),
        ).fetchall()

    if not rows:
        return {"spread_rate": 0.0, "total_rate": 0.0, "roi": 0.0, "count": 0}

    spread_hits = 0
    total_hits = 0
    total_bets = 0
    for r in rows:
        margin = r["final_home_score"] - r["final_visitor_score"]
        total_pts = r["final_home_score"] + r["final_visitor_score"]
        if r["live_spread"] is not None:
            spread_hits += int(is_spread_correct(r["spread_pick"], margin, r["live_spread"]))
            total_bets += 1
        if r["live_total"] is not None:
            total_hits += int(is_total_correct(r["total_pick"], total_pts, r["live_total"]))
            total_bets += 1

    count = len(rows)
    spread_rate = spread_hits / max(count, 1)
    total_rate = total_hits / max(count, 1)
    roi = ((spread_hits + total_hits) / max(total_bets, 1)) * 2 - 1

    return {"spread_rate": spread_rate, "total_rate": total_rate, "roi": roi, "count": count}


def run_review(target_date: str | None = None) -> None:
    target_date = target_date or datetime.utcnow().strftime("%Y-%m-%d")
    client = BallDontLieClient()

    try:
        games = client.games(**{"dates[]": [target_date], "per_page": 100})
    except Exception:
        games = []

    # --- Safety: only process finished games ---
    finished_games = [g for g in games if str(g.get("status", "")).startswith("Final")]

    if not finished_games:
        send_message(cn("review_no_games"))
        return

    for g in finished_games:
        save_result(g["id"], int(g.get("home_team_score", 0)), int(g.get("visitor_team_score", 0)), g)

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.*, r.final_home_score, r.final_visitor_score
            FROM predictions_snapshot p
            JOIN results r ON p.game_id=r.game_id
            WHERE p.snapshot_date=? AND p.is_final_prediction = 1
            """,
            (target_date,),
        ).fetchall()

    if not rows:
        send_message(cn("review_no_games"))
        return

    spread_hits = 0
    total_hits = 0
    clv_open = 0.0
    clv_live = 0.0
    stake = max(len(rows), 1)

    from .supabase_client import save_review_result

    for r in rows:
        margin = r["final_home_score"] - r["final_visitor_score"]
        total = r["final_home_score"] + r["final_visitor_score"]
        spread_correct = is_spread_correct(r["spread_pick"], margin, r["live_spread"] or 0)
        total_correct = is_total_correct(r["total_pick"], total, r["live_total"])
        spread_hits += int(spread_correct)
        total_hits += int(total_correct)
        clv_open += abs((r["opening_spread"] or 0) - (r["live_spread"] or 0))
        clv_live += abs((r["live_spread"] or 0) - margin)

        # --- Save review result to Supabase ---
        save_review_result({
            "game_id": r["game_id"],
            "game_date": target_date,
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "spread_pick": r["spread_pick"],
            "total_pick": r["total_pick"],
            "spread_correct": bool(spread_correct),
            "total_correct": bool(total_correct),
            "final_home_score": r["final_home_score"],
            "final_visitor_score": r["final_visitor_score"],
        })

    spread_rate = spread_hits / stake
    total_rate = total_hits / stake
    win_rate = (spread_hits + total_hits) / (2 * stake)
    roi = win_rate * 2 - 1

    # Rolling 30-day performance
    rolling = _rolling_performance(30)

    msg = (
        f"📊 昨日战绩｜{target_date}\n\n"
        f"让分命中率：{spread_rate:.1%}\n"
        f"大小命中率：{total_rate:.1%}\n"
        f"综合胜率：{win_rate:.1%}\n"
        f"ROI：{roi:.1%}\n"
        f"CLV(初盘)：{clv_open/stake:.2f}\n"
        f"CLV(即时)：{clv_live/stake:.2f}\n"
        f"\n━━━━━━━━━━━━━━━━\n"
        f"📈 近30天滚动表现\n"
        f"让分命中率：{rolling['spread_rate']:.1%}\n"
        f"大小命中率：{rolling['total_rate']:.1%}\n"
        f"ROI：{rolling['roi']:.1%}\n"
        f"样本数：{rolling['count']}"
    )
    send_message(msg)
    ensure_models(force=len(rows) >= 1)


if __name__ == "__main__":
    run_review()
