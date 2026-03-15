from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "database.sqlite"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS games (
                game_id INTEGER PRIMARY KEY,
                season INTEGER,
                date TEXT,
                status TEXT,
                home_team_id INTEGER,
                visitor_team_id INTEGER,
                home_score INTEGER,
                visitor_score INTEGER,
                payload_json TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS odds_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                captured_at TEXT NOT NULL,
                line_type TEXT NOT NULL CHECK(line_type IN ('opening','live')),
                spread_home REAL,
                total_line REAL,
                bookmaker TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions_snapshot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT NOT NULL,
                game_id INTEGER NOT NULL,
                home_team TEXT,
                away_team TEXT,
                total_pick TEXT NOT NULL,
                over_prob REAL,
                under_prob REAL,
                opening_spread REAL,
                live_spread REAL,
                opening_total REAL,
                live_total REAL,
                simulation_runs INTEGER NOT NULL DEFAULT 10000,
                odds_source TEXT NOT NULL DEFAULT 'NONE',
                is_final_prediction BOOLEAN NOT NULL DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS results (
                game_id INTEGER PRIMARY KEY,
                final_home_score INTEGER NOT NULL,
                final_visitor_score INTEGER NOT NULL,
                total_points INTEGER NOT NULL,
                completed_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at TEXT NOT NULL,
                model_type TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                data_points INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                artifact_path TEXT NOT NULL
            );
            """
        )


def upsert_game(game: dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO games(game_id,season,date,status,home_team_id,visitor_team_id,home_score,visitor_score,payload_json)
            VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(game_id) DO UPDATE SET
            season=excluded.season,date=excluded.date,status=excluded.status,
            home_team_id=excluded.home_team_id,visitor_team_id=excluded.visitor_team_id,
            home_score=excluded.home_score,visitor_score=excluded.visitor_score,
            payload_json=excluded.payload_json,updated_at=CURRENT_TIMESTAMP
            """,
            (
                game["id"],
                game.get("season"),
                game.get("date"),
                game.get("status"),
                game.get("home_team", {}).get("id"),
                game.get("visitor_team", {}).get("id"),
                game.get("home_team_score"),
                game.get("visitor_team_score"),
                json.dumps(game, ensure_ascii=False),
            ),
        )


def insert_odds(game_id: int, line_type: str, payload: dict[str, Any], spread_home: float | None, total_line: float | None, bookmaker: str | None) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO odds_history(game_id,captured_at,line_type,spread_home,total_line,bookmaker,payload_json)
            VALUES(?,datetime('now'),?,?,?,?,?)""",
            (game_id, line_type, spread_home, total_line, bookmaker, json.dumps(payload, ensure_ascii=False)),
        )


def insert_prediction(snapshot_date: str, row: dict[str, Any]) -> None:
    with get_conn() as conn:
        # Mark any previous predictions for this game as non-final
        conn.execute(
            "UPDATE predictions_snapshot SET is_final_prediction = 0 WHERE game_id = ?",
            (row["game_id"],),
        )
        conn.execute(
            """
            INSERT INTO predictions_snapshot(
                snapshot_date,game_id,home_team,away_team,total_pick,
                over_prob,under_prob,is_final_prediction
            ) VALUES(?,?,?,?,?,?,?,1)
            """,
            (
                snapshot_date,
                row["game_id"],
                row.get("home_team"),
                row.get("away_team"),
                row["total_pick"],
                row["over_prob"],
                row["under_prob"],
            ),
        )


def save_result(game_id: int, home: int, visitor: int, payload: dict[str, Any]) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO results(game_id,final_home_score,final_visitor_score,total_points,completed_at,payload_json)
            VALUES(?,?,?,?,datetime('now'),?)
            ON CONFLICT(game_id) DO UPDATE SET
            final_home_score=excluded.final_home_score,final_visitor_score=excluded.final_visitor_score,
            total_points=excluded.total_points,completed_at=excluded.completed_at,payload_json=excluded.payload_json
            """,
            (game_id, home, visitor, home + visitor, json.dumps(payload, ensure_ascii=False)),
        )


def log_model(model_type: str, algorithm: str, data_points: int, metrics: dict[str, Any], artifact: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO model_history(trained_at,model_type,algorithm,data_points,metrics_json,artifact_path) VALUES(datetime('now'),?,?,?,?,?)",
            (model_type, algorithm, data_points, json.dumps(metrics), artifact),
        )
