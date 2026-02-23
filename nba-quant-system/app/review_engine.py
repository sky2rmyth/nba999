from __future__ import annotations

from datetime import datetime

from .api_client import BallDontLieClient
from .database import get_conn, save_result
from .retrain_engine import ensure_models
from .telegram_bot import send_message


def run_review(target_date: str | None = None) -> None:
    target_date = target_date or datetime.utcnow().strftime("%Y-%m-%d")
    client = BallDontLieClient()
    games = client.games(**{"dates[]": [target_date], "per_page": 100})

    for g in games:
        if not str(g.get("status", "")).startswith("Final"):
            continue
        save_result(g["id"], int(g.get("home_team_score", 0)), int(g.get("visitor_team_score", 0)), g)

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.*, r.final_home_score, r.final_visitor_score
            FROM predictions_snapshot p
            JOIN results r ON p.game_id=r.game_id
            WHERE p.snapshot_date=?
            """,
            (target_date,),
        ).fetchall()

    spread_hits = 0
    total_hits = 0
    clv_open = 0.0
    clv_live = 0.0
    stake = max(len(rows), 1)

    for r in rows:
        margin = r["final_home_score"] - r["final_visitor_score"]
        total = r["final_home_score"] + r["final_visitor_score"]
        spread_correct = (r["spread_pick"].endswith("让分") and margin + (r["live_spread"] or 0) > 0) or (
            r["spread_pick"].endswith("受让") and margin + (r["live_spread"] or 0) <= 0
        )
        total_correct = (r["total_pick"] == "大分" and total > (r["live_total"] or 220)) or (
            r["total_pick"] == "小分" and total <= (r["live_total"] or 220)
        )
        spread_hits += int(spread_correct)
        total_hits += int(total_correct)
        clv_open += abs((r["opening_spread"] or 0) - (r["live_spread"] or 0))
        clv_live += abs((r["live_spread"] or 0) - margin)

    spread_rate = spread_hits / stake
    total_rate = total_hits / stake
    roi = ((spread_hits + total_hits) / (2 * stake)) * 2 - 1

    msg = (
        f"📘 NBA赛后复盘｜{target_date}\n\n"
        f"让分命中率：{spread_rate:.1%}\n"
        f"大小分命中率：{total_rate:.1%}\n"
        f"ROI：{roi:.1%}\n"
        f"CLV(初盘)：{clv_open/stake:.2f}\n"
        f"CLV(即时)：{clv_live/stake:.2f}"
    )
    send_message(msg)
    ensure_models(force=len(rows) >= 1)


if __name__ == "__main__":
    run_review()
