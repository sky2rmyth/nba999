from __future__ import annotations

from datetime import date

from .api_client import BallDontLieClient
from .database import init_db, upsert_game


def bootstrap_historical_data() -> None:
    init_db()
    client = BallDontLieClient()
    current_season = date.today().year if date.today().month >= 8 else date.today().year - 1
    seasons = [current_season - 2, current_season - 1, current_season]
    for season in seasons:
        games = client.games(**{"seasons[]": [season], "per_page": 100})
        for g in games:
            upsert_game(g)


def sync_date_games(target_date: str) -> list[dict]:
    client = BallDontLieClient()
    games = client.games(**{"dates[]": [target_date], "per_page": 100})
    for g in games:
        upsert_game(g)
    return games
