from __future__ import annotations

from typing import Any

from . import database


def parse_main_market(odds_payload: dict[str, Any]) -> tuple[float | None, float | None, str | None]:
    markets = odds_payload.get("markets") or odds_payload.get("data") or []
    if not markets:
        return None, None, None
    first = markets[0]
    spread = first.get("spread_home") or first.get("home_spread")
    total = first.get("total") or first.get("total_line")
    bookmaker = first.get("bookmaker") or odds_payload.get("bookmaker")
    try:
        spread = float(spread) if spread is not None else None
    except (TypeError, ValueError):
        spread = None
    try:
        total = float(total) if total is not None else None
    except (TypeError, ValueError):
        total = None
    return spread, total, bookmaker


def store_opening_and_live(game_id: int, opening_payload: dict[str, Any], live_payload: dict[str, Any]) -> None:
    o_spread, o_total, o_book = parse_main_market(opening_payload)
    l_spread, l_total, l_book = parse_main_market(live_payload)
    database.insert_odds(game_id, "opening", opening_payload, o_spread, o_total, o_book)
    database.insert_odds(game_id, "live", live_payload, l_spread, l_total, l_book)
