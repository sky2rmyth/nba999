from __future__ import annotations

import os

import requests


def send_message(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "reply_markup": control_panel_markup()}
    requests.post(url, json=payload, timeout=20).raise_for_status()


def control_panel_markup() -> dict:
    return {
        "inline_keyboard": [
            [{"text": "生成今日预测", "callback_data": "predict_today"}],
            [{"text": "复盘今日比赛", "callback_data": "review_today"}],
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
