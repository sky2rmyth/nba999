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


def fetch_team_season_stats(season: int) -> dict:
    """Fetch team season averages from the balldontlie GOAT endpoint.

    Returns a dict keyed by ``team_id`` with pace, off/def ratings, etc.
    """
    client = BallDontLieClient()
    data = client.team_season_averages(season=season)

    team_stats: dict = {}
    for team in data:
        team_stats[team["team_id"]] = {
            "pace": team["pace"],
            "off_rating": team["off_rating"],
            "def_rating": team["def_rating"],
            "ts_pct": team["ts_pct"],
            "ast_pct": team["ast_pct"],
            "reb_pct": team["reb_pct"],
            "tov_pct": team["tov_pct"],
        }
    return team_stats


def fetch_player_injuries() -> list[dict]:
    """Fetch current player injuries from the balldontlie injuries endpoint."""
    client = BallDontLieClient()
    return client.injuries(per_page=100)


def fetch_game_advanced_stats(game_id: int) -> dict:
    """Fetch game advanced stats from the balldontlie GOAT endpoint.

    Returns a dict with off_rating, def_rating, pace, ts_pct, efg_pct,
    ast_pct, reb_pct, tov_pct for the given game.
    """
    client = BallDontLieClient()
    data = client.game_advanced_stats(**{"game_ids[]": [game_id], "per_page": 100})

    stats: dict = {}
    for entry in data:
        team_id = entry.get("team_id")
        if team_id is not None:
            stats[team_id] = {
                "off_rating": entry.get("off_rating"),
                "def_rating": entry.get("def_rating"),
                "pace": entry.get("pace"),
                "ts_pct": entry.get("ts_pct"),
                "efg_pct": entry.get("efg_pct"),
                "ast_pct": entry.get("ast_pct"),
                "reb_pct": entry.get("reb_pct"),
                "tov_pct": entry.get("tov_pct"),
            }
    return stats


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
