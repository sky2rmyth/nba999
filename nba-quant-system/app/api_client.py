from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class EndpointSpec:
    path: str
    allowed_params: set[str]


class BallDontLieClient:
    """Strict balldontlie client with endpoint and parameter validation."""

    ENDPOINTS: dict[str, EndpointSpec] = {
        "games": EndpointSpec("/games", {"dates[]", "seasons[]", "team_ids[]", "postseason", "per_page", "cursor", "start_date", "end_date"}),
        "teams": EndpointSpec("/teams", {"per_page", "cursor"}),
        "players": EndpointSpec("/players", {"search", "per_page", "cursor", "team_ids[]"}),
        "game_player_stats": EndpointSpec("/game_player_stats", {"game_ids[]", "player_ids[]", "team_ids[]", "per_page", "cursor", "start_date", "end_date"}),
        "season_averages": EndpointSpec("/season_averages", {"season", "player_ids[]"}),
        "team_season_averages": EndpointSpec("/team_season_averages", {"season", "team_ids[]"}),
        "box_scores": EndpointSpec("/box_scores", {"game_ids[]", "team_ids[]", "per_page", "cursor"}),
        "injuries": EndpointSpec("/injuries", {"team_ids[]", "player_ids[]", "per_page", "cursor"}),
        "betting_odds": EndpointSpec("/betting_odds", {"game_ids[]", "bookmaker", "per_page", "cursor", "start_date", "end_date"}),
    }

    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: int = 30) -> None:
        self.api_key = api_key or os.getenv("BALLDONTLIE_API_KEY", "")
        if not self.api_key:
            raise ValueError("BALLDONTLIE_API_KEY is required")
        self.base_url = (base_url or os.getenv("BALLDONTLIE_BASE_URL", "https://api.balldontlie.io/v1")).rstrip("/")
        self.timeout = timeout

    def _validate_params(self, endpoint: str, params: dict[str, Any]) -> None:
        allowed = self.ENDPOINTS[endpoint].allowed_params
        invalid = sorted(set(params) - allowed)
        if invalid:
            raise ValueError(f"Invalid params for {endpoint}: {invalid}")

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if endpoint not in self.ENDPOINTS:
            raise ValueError(f"Unsupported endpoint: {endpoint}")
        params = params or {}
        self._validate_params(endpoint, params)
        url = f"{self.base_url}{self.ENDPOINTS[endpoint].path}"
        headers = {"Authorization": self.api_key}
        resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def fetch_all_pages(self, endpoint: str, params: dict[str, Any] | None = None, page_limit: int = 100) -> list[dict[str, Any]]:
        params = dict(params or {})
        out: list[dict[str, Any]] = []
        cursor = None
        pages = 0
        while pages < page_limit:
            if cursor is not None:
                params["cursor"] = cursor
            payload = self._request(endpoint, params)
            out.extend(payload.get("data", []))
            meta = payload.get("meta", {})
            cursor = meta.get("next_cursor")
            pages += 1
            if not cursor:
                break
        return out

    def games(self, **params: Any) -> list[dict[str, Any]]:
        return self.fetch_all_pages("games", params)

    def teams(self, **params: Any) -> list[dict[str, Any]]:
        return self.fetch_all_pages("teams", params)

    def players(self, **params: Any) -> list[dict[str, Any]]:
        return self.fetch_all_pages("players", params)

    def game_player_stats(self, **params: Any) -> list[dict[str, Any]]:
        return self.fetch_all_pages("game_player_stats", params)

    def season_averages(self, **params: Any) -> list[dict[str, Any]]:
        return self._request("season_averages", params).get("data", [])

    def team_season_averages(self, **params: Any) -> list[dict[str, Any]]:
        return self._request("team_season_averages", params).get("data", [])

    def box_scores(self, **params: Any) -> list[dict[str, Any]]:
        return self.fetch_all_pages("box_scores", params)

    def injuries(self, **params: Any) -> list[dict[str, Any]]:
        return self.fetch_all_pages("injuries", params)

    def betting_odds(self, **params: Any) -> list[dict[str, Any]]:
        return self.fetch_all_pages("betting_odds", params)
