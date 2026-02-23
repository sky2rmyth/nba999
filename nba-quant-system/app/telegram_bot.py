from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def _bot_url(method: str) -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required")
    return f"https://api.telegram.org/bot{token}/{method}"


def send_message(text: str, reply_markup: dict | None = None) -> Optional[int]:
    """Send a Telegram message. Returns the message_id for later editing."""
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not chat_id:
        raise ValueError("TELEGRAM_CHAT_ID is required")
    payload: dict = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }
    if reply_markup is None:
        reply_markup = control_panel_markup()
    payload["reply_markup"] = reply_markup
    resp = requests.post(_bot_url("sendMessage"), json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data.get("result", {}).get("message_id")


def edit_message(message_id: int, text: str) -> None:
    """Edit an existing Telegram message by message_id."""
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not chat_id:
        raise ValueError("TELEGRAM_CHAT_ID is required")
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(_bot_url("editMessageText"), json=payload, timeout=20)
        resp.raise_for_status()
    except Exception:
        logger.debug("edit_message failed for message_id=%s", message_id, exc_info=True)


def control_panel_markup() -> dict:
    base_url = os.getenv("GITHUB_PAGES_URL", "https://skyzmyth.github.io/nba999/")
    return {
        "inline_keyboard": [
            [{"text": "📊 今日预测", "url": f"{base_url}?action=predict"}],
            [{"text": "🔁 复盘学习", "url": f"{base_url}?action=review"}],
            [{"text": "📈 模型状态", "url": f"{base_url}?action=model_status"}],
        ]
    }


def dispatch_workflow(workflow_file: str, ref: str = "main") -> None:
    token = os.getenv("GITHUB_TOKEN", "")
    repo = os.getenv("GITHUB_REPOSITORY", "")
    if not token or not repo:
        raise ValueError("GITHUB_TOKEN and GITHUB_REPOSITORY required for workflow dispatch")
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/dispatches"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    requests.post(url, headers=headers, json={"ref": ref}, timeout=20).raise_for_status()


class ProgressTracker:
    """Track and display real-time progress by editing a single Telegram message."""

    STAGES = [
        ("🟡", "System Starting"),
        ("🔵", "Fetching Games Data"),
        ("🧠", "Loading Models"),
        ("⚙️", "Running Monte Carlo Simulation"),
        ("💾", "Saving Results"),
        ("✅", "Completed"),
    ]

    def __init__(self) -> None:
        self.message_id: Optional[int] = None
        self.current_stage = -1
        self.game_progress: list[str] = []

    def _build_text(self) -> str:
        lines = ["<b>🏀 NBA Quant System</b>", ""]
        for i, (icon, label) in enumerate(self.STAGES):
            if i < self.current_stage:
                lines.append(f"✅ {label}")
            elif i == self.current_stage:
                lines.append(f"{icon} {label} ...")
            else:
                lines.append(f"⬜ {label}")
        if self.game_progress:
            lines.append("")
            lines.extend(self.game_progress)
        return "\n".join(lines)

    def start(self) -> None:
        self.current_stage = 0
        try:
            self.message_id = send_message(self._build_text(), reply_markup={})
        except Exception:
            logger.debug("ProgressTracker.start: send failed", exc_info=True)

    def advance(self, stage_index: int) -> None:
        self.current_stage = stage_index
        self._update()

    def set_game_progress(self, text: str) -> None:
        self.game_progress.append(text)
        self._update()

    def finish(self) -> None:
        self.current_stage = len(self.STAGES) - 1
        self._update()

    def _update(self) -> None:
        if self.message_id is None:
            return
        try:
            edit_message(self.message_id, self._build_text())
        except Exception:
            logger.debug("ProgressTracker._update: edit failed", exc_info=True)
