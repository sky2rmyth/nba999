"""Dataset builder for the NBA totals over/under classification model.

Builds a labelled dataset from historical game data and odds snapshots.

Label: ``1`` if ``final_total > closing_line`` else ``0``.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .database import DB_PATH
from .feature_engineering import _compute_team_features

logger = logging.getLogger(__name__)

# 28 features used by the totals classifier (matches TOTAL_FEATURE_COLUMNS).
CLASSIFIER_FEATURES: list[str] = [
    # Odds / line features
    "closing_total",
    "opening_total",
    "line_movement",
    # Pace
    "home_pace",
    "away_pace",
    "pace_avg",
    "pace_diff",
    # Offensive / defensive ratings
    "home_off_rating",
    "away_off_rating",
    "home_def_rating",
    "away_def_rating",
    # Shooting rates
    "home_3p_rate",
    "away_3p_rate",
    "home_ft_rate",
    "away_ft_rate",
    # Offensive rebound rate
    "home_off_reb_rate",
    "away_off_reb_rate",
    # Last-5 games pace
    "home_last5_pace",
    "away_last5_pace",
    # Last-5 games offensive rating
    "home_last5_off_rating",
    "away_last5_off_rating",
    # Rest / fatigue
    "home_rest_days",
    "away_rest_days",
    "home_back_to_back",
    "away_back_to_back",
    # Interaction features
    "pace_interaction",
    "off_vs_def_home",
    "off_vs_def_away",
]


def build_dataset(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Build labelled dataset for the totals classifier.

    Joins ``games`` with ``predictions_snapshot`` to obtain closing / opening
    lines.  Only final games that have both scores **and** stored odds are
    included.

    Returns a :class:`~pandas.DataFrame` with columns
    :data:`CLASSIFIER_FEATURES` + ``"label"``.
    """
    conn = sqlite3.connect(db_path)

    query = """
        SELECT g.game_id, g.home_team_id, g.visitor_team_id, g.date,
               g.home_score, g.visitor_score,
               p.opening_total, p.live_total
        FROM games g
        JOIN predictions_snapshot p
            ON g.game_id = p.game_id AND p.is_final_prediction = 1
        WHERE g.status LIKE 'Final%%'
          AND g.home_score IS NOT NULL AND g.visitor_score IS NOT NULL
          AND p.live_total IS NOT NULL
        ORDER BY g.date
    """

    games = pd.read_sql_query(query, conn)
    if games.empty:
        conn.close()
        logger.info("Dataset builder: 0 samples (no games with odds)")
        return pd.DataFrame()

    rows: list[dict] = []
    for rec in games.to_dict("records"):
        home_id = int(rec["home_team_id"])
        away_id = int(rec["visitor_team_id"])
        game_date = str(rec["date"])
        home_score = rec.get("home_score") or 0
        away_score = rec.get("visitor_score") or 0
        closing_total = float(rec["live_total"])
        opening_total_val = (
            float(rec["opening_total"])
            if rec.get("opening_total") is not None
            else closing_total
        )

        if home_score == 0 and away_score == 0:
            continue

        final_total = home_score + away_score
        label = 1 if final_total > closing_total else 0

        # Compute per-team features via existing helper.
        # _compute_team_features uses "last5_{prefix}_*" / "{prefix}_rest_days"
        # naming; we map those to the CLASSIFIER_FEATURES names below.
        home_feat = _compute_team_features(conn, home_id, away_id, game_date, "home")
        away_feat = _compute_team_features(conn, away_id, home_id, game_date, "away")

        hp = home_feat.get("home_pace", 98.0)
        ap = away_feat.get("away_pace", 98.0)

        row: dict = {
            "game_id": rec["game_id"],
            # Odds / line
            "closing_total": closing_total,
            "opening_total": opening_total_val,
            "line_movement": closing_total - opening_total_val,
            # Pace
            "home_pace": hp,
            "away_pace": ap,
            "pace_avg": (hp + ap) / 2.0,
            "pace_diff": hp - ap,
            # Ratings
            "home_off_rating": home_feat.get("home_off_rating", 0.0),
            "away_off_rating": away_feat.get("away_off_rating", 0.0),
            "home_def_rating": home_feat.get("home_def_rating", 0.0),
            "away_def_rating": away_feat.get("away_def_rating", 0.0),
            # Shooting rates
            "home_3p_rate": home_feat.get("home_3p_rate", 0.0),
            "away_3p_rate": away_feat.get("away_3p_rate", 0.0),
            "home_ft_rate": home_feat.get("home_ft_rate", 0.0),
            "away_ft_rate": away_feat.get("away_ft_rate", 0.0),
            # Offensive rebound rate
            "home_off_reb_rate": home_feat.get("home_off_reb_rate", 0.0),
            "away_off_reb_rate": away_feat.get("away_off_reb_rate", 0.0),
            # Last-5 (mapped from last5_{prefix}_* names)
            "home_last5_pace": home_feat.get("last5_home_pace", 0.0),
            "away_last5_pace": away_feat.get("last5_away_pace", 0.0),
            "home_last5_off_rating": home_feat.get("last5_home_off_rating", 0.0),
            "away_last5_off_rating": away_feat.get("last5_away_off_rating", 0.0),
            # Rest (mapped from {prefix}_rest_days / {prefix}_b2b)
            "home_rest_days": home_feat.get("home_rest_days", 0.0),
            "away_rest_days": away_feat.get("away_rest_days", 0.0),
            "home_back_to_back": home_feat.get("home_b2b", 0.0),
            "away_back_to_back": away_feat.get("away_b2b", 0.0),
            # Interaction features
            "pace_interaction": hp * ap,
            "off_vs_def_home": home_feat.get("home_off_rating", 0.0) - away_feat.get("away_def_rating", 0.0),
            "off_vs_def_away": away_feat.get("away_off_rating", 0.0) - home_feat.get("home_def_rating", 0.0),
            # Label
            "label": label,
        }
        rows.append(row)

    conn.close()
    result = pd.DataFrame(rows)

    if not result.empty:
        for col in CLASSIFIER_FEATURES:
            if col not in result.columns:
                result[col] = 0.0
        result[CLASSIFIER_FEATURES] = result[CLASSIFIER_FEATURES].fillna(0.0)

    logger.info("Dataset builder: %d features, %d samples", len(CLASSIFIER_FEATURES), len(result))
    return result
