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
                prediction_time TEXT NOT NULL,
                spread_pick TEXT NOT NULL,
                spread_prob REAL NOT NULL,
                total_pick TEXT NOT NULL,
                total_prob REAL NOT NULL,
                confidence_score REAL NOT NULL,
                star_rating INTEGER NOT NULL,
                recommendation_index REAL NOT NULL,
                expected_home_score REAL NOT NULL,
                expected_visitor_score REAL NOT NULL,
                simulation_variance REAL NOT NULL,
                opening_spread REAL,
                live_spread REAL,
                opening_total REAL,
                live_total REAL,
                details_json TEXT NOT NULL,
                UNIQUE(snapshot_date, game_id)
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
        conn.execute(
            """
            INSERT OR IGNORE INTO predictions_snapshot(
                snapshot_date,game_id,prediction_time,spread_pick,spread_prob,total_pick,total_prob,
                confidence_score,star_rating,recommendation_index,expected_home_score,expected_visitor_score,
                simulation_variance,opening_spread,live_spread,opening_total,live_total,details_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                snapshot_date,
                row["game_id"],
                row["prediction_time"],
                row["spread_pick"],
                row["spread_prob"],
                row["total_pick"],
                row["total_prob"],
                row["confidence_score"],
                row["star_rating"],
                row["recommendation_index"],
                row["expected_home_score"],
                row["expected_visitor_score"],
                row["simulation_variance"],
                row.get("opening_spread"),
                row.get("live_spread"),
                row.get("opening_total"),
                row.get("live_total"),
                json.dumps(row.get("details", {}), ensure_ascii=False),
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
