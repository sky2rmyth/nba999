from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .database import DB_PATH
from .data_pipeline import (
    calculate_possessions,
    calculate_pace,
    offensive_rating,
    defensive_rating,
)

logger = logging.getLogger(__name__)

# League-average rate constants for shooting rate approximations
LEAGUE_AVG_3P_RATE = 0.37
LEAGUE_AVG_FT_RATE = 0.27
LEAGUE_AVG_ORB_RATE = 0.25


# ---------------------------------------------------------------------------
# Possession-model helper functions
# ---------------------------------------------------------------------------

def calculate_possessions_from_boxscore(boxscore: dict) -> float:
    """Calculate possessions from a box-score dict.

    Keys expected: ``fga``, ``oreb``, ``turnovers``, ``fta``.
    """
    fga = boxscore.get("fga", 0)
    orb = boxscore.get("oreb", 0)
    tov = boxscore.get("turnovers", 0)
    fta = boxscore.get("fta", 0)

    poss = fga - orb + tov + 0.44 * fta  # 0.44 = FT possession factor
    return max(poss, 1)


def calculate_game_pace(home_pace: float, away_pace: float) -> float:
    """Calculate expected game pace using matchup pace formula.

    The faster team is weighted slightly more (0.55) than the slower team (0.45).
    """
    fast = max(home_pace, away_pace)
    slow = min(home_pace, away_pace)
    return 0.55 * fast + 0.45 * slow


def calculate_ppp(off_rating: float) -> float:
    """Convert offensive rating to Points Per Possession."""
    return off_rating / 100


def calculate_three_point_rate(three_pa: float, fga: float) -> float:
    """Three-point attempt rate: 3PA / FGA."""
    if fga <= 0:
        return 0.0
    return three_pa / fga


def calculate_free_throw_rate(fta: float, fga: float) -> float:
    """Free-throw attempt rate: FTA / FGA."""
    if fga <= 0:
        return 0.0
    return fta / fga


def calculate_orb_rate(oreb: float, opp_dreb: float) -> float:
    """Offensive rebound rate: OREB / (OREB + OPP_DREB)."""
    total = oreb + opp_dreb
    if total <= 0:
        return 0.0
    return oreb / total


def calculate_tov_rate(turnovers: float, possessions: float) -> float:
    """Turnover rate: turnovers / possessions."""
    if possessions <= 0:
        return 0.0
    return turnovers / possessions


FEATURE_COLUMNS = [
    # Team offensive/defensive ratings
    "home_off_rating",
    "home_def_rating",
    "home_net_rating",
    "away_off_rating",
    "away_def_rating",
    "away_net_rating",
    # Pace
    "home_pace",
    "away_pace",
    "pace_interaction",
    # Rolling last-5 stats
    "home_avg_score_last5",
    "home_avg_allowed_last5",
    "home_margin_last5",
    "away_avg_score_last5",
    "away_avg_allowed_last5",
    "away_margin_last5",
    # Rolling last-10 stats
    "home_avg_score_last10",
    "home_avg_allowed_last10",
    "home_margin_last10",
    "away_avg_score_last10",
    "away_avg_allowed_last10",
    "away_margin_last10",
    # Home/away indicator
    "home_indicator",
    # Rest days
    "home_rest_days",
    "away_rest_days",
    # Back-to-back flag
    "home_b2b",
    "away_b2b",
    # Recent scoring variance
    "home_scoring_variance",
    "away_scoring_variance",
    # Opponent efficiency
    "opp_home_def_eff",
    "opp_away_def_eff",
    "opp_home_off_eff",
    "opp_away_off_eff",
    # Consistency and volatility
    "home_consistency",
    "away_consistency",
    "home_off_volatility",
    "away_off_volatility",
    "home_def_volatility",
    "away_def_volatility",
    # Recent margin trend
    "home_margin_trend",
    "away_margin_trend",
]


# Feature columns for the totals over/under classification model.
# These are the 28 input features used by GradientBoostingClassifier.
TOTAL_FEATURE_COLUMNS = [
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


def _get_team_games(conn: sqlite3.Connection, team_id: int, before_date: str, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        """
        SELECT home_team_id, visitor_team_id, home_score, visitor_score, date
        FROM games WHERE date < ? AND status LIKE 'Final%%'
        AND (home_team_id=? OR visitor_team_id=?)
        ORDER BY date DESC LIMIT ?
        """,
        (before_date, team_id, team_id, limit),
    ).fetchall()
    results = []
    for r in rows:
        is_home = r[0] == team_id
        scored = (r[2] or 0) if is_home else (r[3] or 0)
        allowed = (r[3] or 0) if is_home else (r[2] or 0)
        results.append({
            "scored": scored,
            "allowed": allowed,
            "margin": scored - allowed,
            "total": scored + allowed,
            "date": r[4],
            "is_home": is_home,
        })
    return results


def _compute_team_features(conn: sqlite3.Connection, team_id: int, opponent_id: int,
                           before_date: str, prefix: str) -> dict:
    games = _get_team_games(conn, team_id, before_date, limit=20)
    opp_games = _get_team_games(conn, opponent_id, before_date, limit=20)

    feat: dict = {}

    if not games:
        for col in FEATURE_COLUMNS:
            if col.startswith(prefix):
                feat[col] = 0.0
        feat[f"opp_{prefix}_def_eff"] = 0.0
        feat[f"opp_{prefix}_off_eff"] = 0.0
        # Total-classifier features
        feat[f"last5_{prefix}_off_rating"] = 0.0
        feat[f"last5_{prefix}_pace"] = 0.0
        feat[f"{prefix}_3p_rate"] = 0.0
        feat[f"{prefix}_ft_rate"] = 0.0
        feat[f"{prefix}_off_reb_rate"] = 0.0
        return feat

    scores = [g["scored"] for g in games]
    allowed = [g["allowed"] for g in games]
    margins = [g["margin"] for g in games]
    totals_vals = [g["total"] for g in games]

    # Offensive/defensive/net ratings (per-100 possessions approximation)
    avg_score = np.mean(scores)
    avg_allowed = np.mean(allowed)
    avg_total = np.mean(totals_vals) if totals_vals else 210.0

    # Estimate possessions from scores using league-average efficiency (~1.14 PPP).
    # This avoids the old ``avg_total / 2`` shortcut which inflates pace to ~110
    # and collapses offensive ratings to ~100.
    est_possessions = avg_total / 2.14  # ≈ 98-100 for a typical 210-214 total
    est_possessions = max(est_possessions, 1)
    pace = est_possessions

    off_rtg = offensive_rating(avg_score, est_possessions)
    # Clamp off_rating to reasonable NBA range [105, 120]
    off_rtg = max(105, min(off_rtg, 120))
    feat[f"{prefix}_off_rating"] = off_rtg
    feat[f"{prefix}_def_rating"] = defensive_rating(avg_allowed, est_possessions)
    feat[f"{prefix}_net_rating"] = feat[f"{prefix}_off_rating"] - feat[f"{prefix}_def_rating"]
    feat[f"{prefix}_pace"] = pace

    # Last 5 games
    last5 = games[:5] if len(games) >= 5 else games
    feat[f"{prefix}_avg_score_last5"] = np.mean([g["scored"] for g in last5])
    feat[f"{prefix}_avg_allowed_last5"] = np.mean([g["allowed"] for g in last5])
    feat[f"{prefix}_margin_last5"] = np.mean([g["margin"] for g in last5])

    # Last-5 offensive rating & pace (for total classifier).
    # 2.14 = avg PPP factor for possession estimation; 210.0 = approximate league-average total.
    last5_avg_score = np.mean([g["scored"] for g in last5])
    last5_avg_total = np.mean([g["total"] for g in last5]) if last5 else 210.0
    last5_est_poss = max(last5_avg_total / 2.14, 1)
    feat[f"last5_{prefix}_off_rating"] = offensive_rating(last5_avg_score, last5_est_poss)
    feat[f"last5_{prefix}_pace"] = last5_est_poss

    # Last 10 games
    last10 = games[:10] if len(games) >= 10 else games
    feat[f"{prefix}_avg_score_last10"] = np.mean([g["scored"] for g in last10])
    feat[f"{prefix}_avg_allowed_last10"] = np.mean([g["allowed"] for g in last10])
    feat[f"{prefix}_margin_last10"] = np.mean([g["margin"] for g in last10])

    # Rest days
    if len(games) >= 2:
        try:
            d0 = pd.Timestamp(before_date)
            d1 = pd.Timestamp(games[0]["date"])
            rest = max(0, (d0 - d1).days)
        except Exception:
            rest = 2
    else:
        rest = 3
    feat[f"{prefix}_rest_days"] = float(rest)
    feat[f"{prefix}_b2b"] = 1.0 if rest <= 1 else 0.0

    # Scoring variance
    feat[f"{prefix}_scoring_variance"] = float(np.var(scores)) if len(scores) >= 2 else 0.0

    # Consistency (inverse of coefficient of variation)
    std_score = float(np.std(scores)) if len(scores) >= 2 else 1.0
    feat[f"{prefix}_consistency"] = avg_score / max(std_score, 0.1)

    # Offensive/defensive volatility
    feat[f"{prefix}_off_volatility"] = float(np.std(scores)) if len(scores) >= 2 else 0.0
    feat[f"{prefix}_def_volatility"] = float(np.std(allowed)) if len(allowed) >= 2 else 0.0

    # Margin trend (last 5 vs last 10 margin)
    m5 = np.mean([g["margin"] for g in last5])
    m10 = np.mean([g["margin"] for g in last10])
    feat[f"{prefix}_margin_trend"] = m5 - m10

    # 3-point rate and free-throw rate approximations.
    # Box-score-level 3PA/FGA/FTA are not stored in the games table, so we
    # derive proxies from the team's scoring efficiency relative to league avg.
    _league_off = 110.0
    feat[f"{prefix}_3p_rate"] = LEAGUE_AVG_3P_RATE * (off_rtg / _league_off)
    feat[f"{prefix}_ft_rate"] = LEAGUE_AVG_FT_RATE * (off_rtg / _league_off)
    feat[f"{prefix}_off_reb_rate"] = LEAGUE_AVG_ORB_RATE * (off_rtg / _league_off)

    # Opponent efficiency
    if opp_games:
        opp_scores = [g["scored"] for g in opp_games]
        opp_allowed = [g["allowed"] for g in opp_games]
        opp_total = np.mean([g["total"] for g in opp_games])
        opp_possessions = opp_total / 2.14
        opp_possessions = max(opp_possessions, 1)
        feat[f"opp_{prefix}_def_eff"] = defensive_rating(np.mean(opp_allowed), opp_possessions)
        feat[f"opp_{prefix}_off_eff"] = offensive_rating(np.mean(opp_scores), opp_possessions)
    else:
        feat[f"opp_{prefix}_def_eff"] = 0.0
        feat[f"opp_{prefix}_off_eff"] = 0.0

    return feat


def build_training_frame(db_path: Path = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    games = pd.read_sql_query(
        "SELECT * FROM games WHERE status LIKE 'Final%' ORDER BY date", conn
    )
    if games.empty:
        return pd.DataFrame()

    rows = []
    for r in games.to_dict("records"):
        home_id = int(r["home_team_id"])
        away_id = int(r["visitor_team_id"])
        game_date = str(r["date"])
        home_score = r.get("home_score") or 0
        away_score = r.get("visitor_score") or 0

        if home_score == 0 and away_score == 0:
            continue

        home_feat = _compute_team_features(conn, home_id, away_id, game_date, "home")
        away_feat = _compute_team_features(conn, away_id, home_id, game_date, "away")

        row = {"game_id": r["game_id"]}
        row.update(home_feat)
        row.update(away_feat)

        # Home indicator is always 1 for training (home team perspective)
        row["home_indicator"] = 1.0

        # Pace interaction
        home_pace = row.get("home_pace", 98.0)
        away_pace = row.get("away_pace", 98.0)
        row["pace_interaction"] = home_pace * away_pace / 100.0

        # Targets: actual scores
        row["home_score"] = float(home_score)
        row["away_score"] = float(away_score)

        rows.append(row)

    conn.close()
    result = pd.DataFrame(rows)
    if not result.empty:
        # Fill missing features with 0
        for col in FEATURE_COLUMNS:
            if col not in result.columns:
                result[col] = 0.0
        result[FEATURE_COLUMNS] = result[FEATURE_COLUMNS].fillna(0.0)
    logger.info("Feature count: %d", len(FEATURE_COLUMNS))
    logger.info("Training samples: %d", len(result))
    return result


def build_total_training_frame(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Build training data for the totals over/under classification model.

    Joins ``games`` with ``predictions_snapshot`` to obtain ``closing_total``
    (``live_total``) and ``opening_total`` for each game.  Only games that have
    both a final score **and** stored odds are included.

    Label: 1 if total_score > closing_total, else 0.
    """
    conn = sqlite3.connect(db_path)

    # Join games with their most-recent (final) prediction to get odds lines.
    query = """
        SELECT g.game_id, g.home_team_id, g.visitor_team_id, g.date,
               g.home_score, g.visitor_score,
               p.opening_total, p.live_total
        FROM games g
        JOIN predictions_snapshot p ON g.game_id = p.game_id AND p.is_final_prediction = 1
        WHERE g.status LIKE 'Final%%'
          AND g.home_score IS NOT NULL AND g.visitor_score IS NOT NULL
          AND p.live_total IS NOT NULL
        ORDER BY g.date
    """
    games = pd.read_sql_query(query, conn)
    if games.empty:
        conn.close()
        logger.info("Total training frame: 0 samples (no games with odds)")
        return pd.DataFrame()

    rows: list[dict] = []
    for r in games.to_dict("records"):
        home_id = int(r["home_team_id"])
        away_id = int(r["visitor_team_id"])
        game_date = str(r["date"])
        home_score = r.get("home_score") or 0
        away_score = r.get("visitor_score") or 0
        closing_total = float(r["live_total"])
        opening_total_val = float(r["opening_total"]) if r.get("opening_total") is not None else closing_total

        if home_score == 0 and away_score == 0:
            continue

        total_score = home_score + away_score
        label = 1 if total_score > closing_total else 0

        home_feat = _compute_team_features(conn, home_id, away_id, game_date, "home")
        away_feat = _compute_team_features(conn, away_id, home_id, game_date, "away")

        row: dict = {"game_id": r["game_id"]}
        row.update(home_feat)
        row.update(away_feat)

        # Odds / line features
        row["closing_total"] = closing_total
        row["opening_total"] = opening_total_val
        row["line_movement"] = closing_total - opening_total_val

        # Pace derived features
        hp = row.get("home_pace", 98.0)
        ap = row.get("away_pace", 98.0)
        row["pace_avg"] = (hp + ap) / 2.0
        row["pace_diff"] = hp - ap

        # Map last5 column names to match TOTAL_FEATURE_COLUMNS
        row["home_last5_off_rating"] = row.get("last5_home_off_rating", 0.0)
        row["away_last5_off_rating"] = row.get("last5_away_off_rating", 0.0)
        row["home_last5_pace"] = row.get("last5_home_pace", 0.0)
        row["away_last5_pace"] = row.get("last5_away_pace", 0.0)

        # Back-to-back flag
        row["home_back_to_back"] = row.get("home_b2b", 0.0)
        row["away_back_to_back"] = row.get("away_b2b", 0.0)

        # Interaction features
        row["pace_interaction"] = hp * ap
        home_off = row.get("home_off_rating", 0.0)
        away_off = row.get("away_off_rating", 0.0)
        home_def = row.get("home_def_rating", 0.0)
        away_def = row.get("away_def_rating", 0.0)
        row["off_vs_def_home"] = home_off - away_def
        row["off_vs_def_away"] = away_off - home_def

        # Target label
        row["label"] = label
        rows.append(row)

    conn.close()
    result = pd.DataFrame(rows)
    if not result.empty:
        for col in TOTAL_FEATURE_COLUMNS:
            if col not in result.columns:
                result[col] = 0.0
        result[TOTAL_FEATURE_COLUMNS] = result[TOTAL_FEATURE_COLUMNS].fillna(0.0)
    logger.info("Total training feature count: %d", len(TOTAL_FEATURE_COLUMNS))
    logger.info("Total training samples: %d", len(result))
    return result
