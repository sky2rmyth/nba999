from __future__ import annotations

import logging
from datetime import date

from .api_client import BallDontLieClient
from .database import init_db, upsert_game

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Advanced metrics: possession-based calculations
# ---------------------------------------------------------------------------

def calculate_possessions(team_stats: dict) -> float:
    """Calculate possessions from box score stats.

    Formula: FGA + 0.44 * FTA - Offensive Rebounds + Turnovers
    """
    fga = team_stats.get("fga", 0)
    fta = team_stats.get("fta", 0)
    orb = team_stats.get("offensive_rebounds", 0)
    tov = team_stats.get("turnovers", 0)

    possessions = fga + 0.44 * fta - orb + tov

    return max(possessions, 1)


def calculate_pace(team_stats: dict) -> float:
    """Calculate pace (possessions per 48 minutes)."""
    possessions = calculate_possessions(team_stats)

    minutes = 48

    pace = possessions / minutes * 48

    return pace


def offensive_rating(points: float, possessions: float) -> float:
    """Calculate offensive rating (points per 100 possessions)."""
    if possessions == 0:
        return 0

    return (points / possessions) * 100


def defensive_rating(opp_points: float, possessions: float) -> float:
    """Calculate defensive rating (opponent points per 100 possessions)."""
    if possessions == 0:
        return 0

    return (opp_points / possessions) * 100


def _try_send_telegram(text: str) -> None:
    """Send Telegram message, silently ignore failures."""
    try:
        from .telegram_bot import send_message
        send_message(text)
    except Exception:
        logger.debug("Telegram send skipped: %s", text)


def bootstrap_historical_data() -> None:
    init_db()
    logger.info("Bootstrapping historical NBA data...")
    _try_send_telegram("📥 开始历史数据初始化")
    client = BallDontLieClient()
    current_season = date.today().year if date.today().month >= 8 else date.today().year - 1
    seasons = [current_season - 2, current_season - 1, current_season]
    total = 0
    for season in seasons:
        _try_send_telegram(f"赛季下载中... {season}-{season + 1}")
        games = client.games(**{"seasons[]": [season], "per_page": 100})
        for g in games:
            if str(g.get("status", "")).startswith("Final"):
                upsert_game(g)
                total += 1
    logger.info("Downloaded games: %d", total)
    _try_send_telegram(f"已下载比赛数量: {total}")


def sync_date_games(target_date: str) -> list[dict]:
    client = BallDontLieClient()
    games = client.games(**{"dates[]": [target_date], "per_page": 100})
    for g in games:
        upsert_game(g)
    return games
