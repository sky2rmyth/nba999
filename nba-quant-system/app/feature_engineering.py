from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .database import DB_PATH


FEATURE_COLUMNS = [
    "home_recent_margin",
    "visitor_recent_margin",
    "margin_diff",
    "opening_spread",
    "live_spread",
    "opening_total",
    "live_total",
]


def _recent_margin(conn: sqlite3.Connection, team_id: int, before_date: str) -> float:
    rows = conn.execute(
        """
        SELECT home_team_id, visitor_team_id, home_score, visitor_score
        FROM games WHERE date < ? AND status LIKE 'Final%'
        AND (home_team_id=? OR visitor_team_id=?)
        ORDER BY date DESC LIMIT 10
        """,
        (before_date, team_id, team_id),
    ).fetchall()
    if not rows:
        return 0.0
    margins = []
    for r in rows:
        if r[0] == team_id:
            margins.append((r[2] or 0) - (r[3] or 0))
        else:
            margins.append((r[3] or 0) - (r[2] or 0))
    return float(sum(margins) / len(margins))


def build_training_frame(db_path: Path = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    games = pd.read_sql_query("SELECT * FROM games WHERE status LIKE 'Final%'", conn)
    if games.empty:
        return pd.DataFrame()

    odds = pd.read_sql_query(
        """
        SELECT game_id,
            MAX(CASE WHEN line_type='opening' THEN spread_home END) opening_spread,
            MAX(CASE WHEN line_type='live' THEN spread_home END) live_spread,
            MAX(CASE WHEN line_type='opening' THEN total_line END) opening_total,
            MAX(CASE WHEN line_type='live' THEN total_line END) live_total
        FROM odds_history GROUP BY game_id
        """,
        conn,
    )

    df = games.merge(odds, on="game_id", how="left")

    # Compute average total across all completed games for default odds
    valid_totals = games.loc[
        games["home_score"].notna() & games["visitor_score"].notna(),
    ]
    avg_total = float(
        (valid_totals["home_score"] + valid_totals["visitor_score"]).mean()
    ) if not valid_totals.empty else 215.0

    rows = []
    for r in df.to_dict("records"):
        hrm = _recent_margin(conn, int(r["home_team_id"]), str(r["date"]))
        vrm = _recent_margin(conn, int(r["visitor_team_id"]), str(r["date"]))
        margin = (r.get("home_score") or 0) - (r.get("visitor_score") or 0)
        total = (r.get("home_score") or 0) + (r.get("visitor_score") or 0)
        o_spread = r.get("opening_spread")
        o_total = r.get("opening_total")
        if not pd.notna(o_spread) or not pd.notna(o_total):
            o_spread = 0.0
            o_total = avg_total
        rows.append(
            {
                "game_id": r["game_id"],
                "home_recent_margin": hrm,
                "visitor_recent_margin": vrm,
                "margin_diff": hrm - vrm,
                "opening_spread": float(o_spread),
                "live_spread": float(r.get("live_spread") if pd.notna(r.get("live_spread")) else o_spread),
                "opening_total": float(o_total),
                "live_total": float(r.get("live_total") if pd.notna(r.get("live_total")) else o_total),
                "spread_label": 1 if margin + float(o_spread) > 0 else 0,
                "total_label": 1 if total > float(o_total) else 0,
            }
        )
    return pd.DataFrame(rows)
