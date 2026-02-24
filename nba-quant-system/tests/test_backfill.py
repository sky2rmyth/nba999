"""Tests for backfill_review_games and supporting helpers."""
from __future__ import annotations

import os
from unittest import mock

import pytest

os.environ.pop("SUPABASE_URL", None)
os.environ.pop("SUPABASE_KEY", None)

from app import supabase_client


@pytest.fixture(autouse=True)
def _reset_client():
    """Reset module-level state between tests."""
    supabase_client._client = None
    supabase_client._available = None
    yield
    supabase_client._client = None
    supabase_client._available = None


# ---------- fetch_all_predictions ----------

class TestFetchAllPredictions:
    def test_returns_all_rows(self):
        fake_client = mock.MagicMock()
        fake_client.table.return_value.select.return_value.order.return_value.execute.return_value.data = [
            {"id": 1, "game_id": 100, "payload": {}, "game_date": None},
            {"id": 2, "game_id": 200, "payload": {}, "game_date": "2025-01-15"},
        ]
        supabase_client._client = fake_client
        supabase_client._available = True

        results = supabase_client.fetch_all_predictions()
        assert len(results) == 2
        assert results[0]["game_id"] == 100
        assert results[1]["game_id"] == 200

    def test_returns_empty_when_not_configured(self):
        supabase_client._available = False
        assert supabase_client.fetch_all_predictions() == []

    def test_returns_empty_on_error(self):
        fake_client = mock.MagicMock()
        fake_client.table.return_value.select.return_value.order.return_value.execute.side_effect = RuntimeError("fail")
        supabase_client._client = fake_client
        supabase_client._available = True
        assert supabase_client.fetch_all_predictions() == []


# ---------- update_prediction_game_date ----------

class TestUpdatePredictionGameDate:
    def test_updates_game_date(self):
        fake_client = mock.MagicMock()
        supabase_client._client = fake_client
        supabase_client._available = True

        supabase_client.update_prediction_game_date(1, "2025-02-01")

        fake_client.table.assert_called_with("predictions")
        fake_client.table.return_value.update.assert_called_once_with({"game_date": "2025-02-01"})
        fake_client.table.return_value.update.return_value.eq.assert_called_once_with("id", 1)

    def test_skips_when_not_configured(self):
        supabase_client._available = False
        supabase_client.update_prediction_game_date(1, "2025-02-01")  # should not raise


# ---------- BallDontLieClient.get_game ----------

class TestGetGame:
    def test_get_game_returns_data(self):
        from app.api_client import BallDontLieClient

        game_data = {
            "id": 42,
            "date": "2025-01-15T00:00:00.000Z",
            "status": "Final",
            "home_team_score": 110,
            "visitor_team_score": 105,
        }

        with mock.patch("app.api_client.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"data": game_data}
            mock_get.return_value.raise_for_status = mock.MagicMock()

            client = BallDontLieClient(api_key="test-key")
            result = client.get_game(42)

        assert result["id"] == 42
        assert result["status"] == "Final"
        assert result["home_team_score"] == 110

    def test_get_game_calls_correct_url(self):
        from app.api_client import BallDontLieClient

        with mock.patch("app.api_client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {"data": {"id": 99}}
            mock_get.return_value.raise_for_status = mock.MagicMock()

            client = BallDontLieClient(api_key="test-key", base_url="https://api.example.com/v1")
            client.get_game(99)

        call_args = mock_get.call_args
        assert call_args[0][0] == "https://api.example.com/v1/games/99"


# ---------- backfill_review_games ----------

class TestBackfillReviewGames:
    def test_backfill_updates_missing_game_date(self):
        """Predictions with game_date=None get updated from API."""
        fake_client = mock.MagicMock()
        fake_client.table.return_value.select.return_value.order.return_value.execute.return_value.data = [
            {"id": 1, "game_id": 42, "payload": {}, "game_date": None},
        ]
        supabase_client._client = fake_client
        supabase_client._available = True

        game_resp = {
            "id": 42,
            "date": "2025-01-15T00:00:00.000Z",
            "status": "Final",
            "home_team_score": 110,
            "visitor_team_score": 105,
        }

        with mock.patch("app.review_engine.BallDontLieClient") as MockClient:
            mock_api = MockClient.return_value
            mock_api.get_game.return_value = game_resp

            from app.review_engine import backfill_review_games
            result = backfill_review_games()

        # Verify update was called with correct args
        update_chain = fake_client.table.return_value.update
        update_chain.assert_called_with({"game_date": "2025-01-15"})
        update_chain.return_value.eq.assert_called_with("id", 1)

        assert len(result) == 1
        assert result[0]["status"] == "Final"

    def test_backfill_skips_update_when_game_date_exists(self):
        """Predictions with existing game_date are not updated."""
        fake_client = mock.MagicMock()
        fake_client.table.return_value.select.return_value.order.return_value.execute.return_value.data = [
            {"id": 1, "game_id": 42, "payload": {}, "game_date": "2025-01-15"},
        ]
        supabase_client._client = fake_client
        supabase_client._available = True

        game_resp = {
            "id": 42,
            "date": "2025-01-15T00:00:00.000Z",
            "status": "Final",
            "home_team_score": 110,
            "visitor_team_score": 105,
        }

        with mock.patch("app.review_engine.BallDontLieClient") as MockClient:
            mock_api = MockClient.return_value
            mock_api.get_game.return_value = game_resp

            from app.review_engine import backfill_review_games
            result = backfill_review_games()

        # update should NOT have been called for the game_date column
        update_calls = [
            c for c in fake_client.table.return_value.update.call_args_list
            if c[0][0].get("game_date")
        ]
        assert len(update_calls) == 0

        # But Final game should still be returned
        assert len(result) == 1

    def test_backfill_excludes_non_final_games(self):
        """Only Final games are returned."""
        fake_client = mock.MagicMock()
        fake_client.table.return_value.select.return_value.order.return_value.execute.return_value.data = [
            {"id": 1, "game_id": 42, "payload": {}, "game_date": "2025-01-15"},
        ]
        supabase_client._client = fake_client
        supabase_client._available = True

        game_resp = {
            "id": 42,
            "date": "2025-01-15T00:00:00.000Z",
            "status": "In Progress",
            "home_team_score": 55,
            "visitor_team_score": 50,
        }

        with mock.patch("app.review_engine.BallDontLieClient") as MockClient:
            mock_api = MockClient.return_value
            mock_api.get_game.return_value = game_resp

            from app.review_engine import backfill_review_games
            result = backfill_review_games()

        assert len(result) == 0

    def test_backfill_returns_empty_when_no_predictions(self):
        """Returns empty list when there are no predictions."""
        supabase_client._available = False

        with mock.patch("app.review_engine.BallDontLieClient"):
            from app.review_engine import backfill_review_games
            result = backfill_review_games()

        assert result == []

    def test_backfill_continues_on_api_error(self):
        """API errors for individual games don't crash the backfill."""
        fake_client = mock.MagicMock()
        fake_client.table.return_value.select.return_value.order.return_value.execute.return_value.data = [
            {"id": 1, "game_id": 42, "payload": {}, "game_date": None},
            {"id": 2, "game_id": 43, "payload": {}, "game_date": None},
        ]
        supabase_client._client = fake_client
        supabase_client._available = True

        game_resp = {
            "id": 43,
            "date": "2025-01-16T00:00:00.000Z",
            "status": "Final",
            "home_team_score": 100,
            "visitor_team_score": 98,
        }

        with mock.patch("app.review_engine.BallDontLieClient") as MockClient:
            mock_api = MockClient.return_value
            mock_api.get_game.side_effect = [RuntimeError("timeout"), game_resp]

            from app.review_engine import backfill_review_games
            result = backfill_review_games()

        # First game failed, second succeeded
        assert len(result) == 1
        assert result[0]["id"] == 43
