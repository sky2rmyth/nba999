from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"


def fetch_today_odds() -> list[dict[str, Any]]:
    """Fetch NBA odds from the-odds-api.com (PRIMARY source)."""
    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        logger.warning("ODDS_API_KEY not set — skipping primary odds provider")
        return []
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "decimal",
    }
    try:
        resp = requests.get(ODDS_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        logger.info("Primary odds provider returned %d games", len(data))
        return data
    except Exception:
        logger.warning("Primary odds provider failed", exc_info=True)
        return []


def _find_spread_market(bookmaker: dict[str, Any]) -> dict[str, Any] | None:
    for m in bookmaker.get("markets", []):
        if m.get("key") == "spreads":
            return m
    return None


def _find_totals_market(bookmaker: dict[str, Any]) -> dict[str, Any] | None:
    for m in bookmaker.get("markets", []):
        if m.get("key") == "totals":
            return m
    return None


def _extract_line_by_timestamp(
    game: dict[str, Any], earliest: bool
) -> dict[str, Any]:
    """Extract spread/total from bookmakers by timestamp.

    earliest=True  → opening line (earliest last_update)
    earliest=False → live line (latest last_update)
    """
    bookmakers = [b for b in game.get("bookmakers", []) if b.get("last_update")]
    if not bookmakers:
        return {"home_spread": None, "away_spread": None, "total_points": None}

    # Sort bookmakers by last_update timestamp
    sorted_books = sorted(
        bookmakers,
        key=lambda b: b.get("last_update", ""),
        reverse=not earliest,
    )

    home_team = game.get("home_team", "")
    away_team = game.get("away_team", "")
    home_spread = None
    away_spread = None
    total_points = None

    for bk in sorted_books:
        spread_market = _find_spread_market(bk)
        if spread_market and home_spread is None:
            for outcome in spread_market.get("outcomes", []):
                if outcome.get("name") == home_team:
                    home_spread = outcome.get("point")
                elif outcome.get("name") == away_team:
                    away_spread = outcome.get("point")

        totals_market = _find_totals_market(bk)
        if totals_market and total_points is None:
            for outcome in totals_market.get("outcomes", []):
                if outcome.get("name") == "Over":
                    total_points = outcome.get("point")
                    break

        if home_spread is not None and total_points is not None:
            break

    return {
        "home_spread": home_spread,
        "away_spread": away_spread,
        "total_points": total_points,
    }


def extract_opening_line(game: dict[str, Any]) -> dict[str, Any]:
    """Extract opening line (earliest timestamp) from bookmakers data."""
    return _extract_line_by_timestamp(game, earliest=True)


def extract_live_line(game: dict[str, Any]) -> dict[str, Any]:
    """Extract live line (latest timestamp) from bookmakers data."""
    return _extract_line_by_timestamp(game, earliest=False)
