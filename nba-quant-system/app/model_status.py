"""Model status reporter — sends current model state to Telegram."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .database import get_conn, init_db
from .prediction_models import MODEL_DIR, _current_version, load_models
from .feature_engineering import FEATURE_COLUMNS
from .telegram_bot import send_message

logger = logging.getLogger(__name__)


def run_model_status() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    init_db()

    version = _current_version()
    feature_count = len(FEATURE_COLUMNS)

    bundle = load_models()
    model_available = bundle is not None

    # Latest training info from database
    latest_metrics: dict = {}
    last_trained = "N/A"
    total_data_points = 0
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM model_history ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                last_trained = row["trained_at"]
                total_data_points = row["data_points"]
                latest_metrics = json.loads(row["metrics_json"])
    except Exception:
        logger.debug("Could not fetch model history")

    # Recent prediction performance
    perf_line = ""
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT p.spread_pick, p.total_pick, p.live_spread, p.live_total,
                       r.final_home_score, r.final_visitor_score
                FROM predictions_snapshot p
                JOIN results r ON p.game_id = r.game_id
                ORDER BY p.created_at DESC LIMIT 30
                """
            ).fetchall()
        if rows:
            s_hits = t_hits = 0
            for r in rows:
                margin = r["final_home_score"] - r["final_visitor_score"]
                total_pts = r["final_home_score"] + r["final_visitor_score"]
                if r["live_spread"] is not None:
                    s_ok = ("受让" not in r["spread_pick"] and margin + r["live_spread"] > 0) or (
                        "受让" in r["spread_pick"] and margin + r["live_spread"] <= 0
                    )
                    s_hits += int(s_ok)
                if r["live_total"] is not None:
                    t_ok = (r["total_pick"] == "大分" and total_pts > r["live_total"]) or (
                        r["total_pick"] == "小分" and total_pts <= r["live_total"]
                    )
                    t_hits += int(t_ok)
            n = len(rows)
            perf_line = (
                f"\n📊 最近 {n} 场表现\n"
                f"让分命中: {s_hits}/{n} ({s_hits/n:.1%})\n"
                f"大小命中: {t_hits}/{n} ({t_hits/n:.1%})"
            )
    except Exception:
        pass

    home_mae = latest_metrics.get("home_mae", "N/A")
    away_mae = latest_metrics.get("away_mae", "N/A")
    sc_acc = latest_metrics.get("spread_cover_accuracy", "N/A")
    to_acc = latest_metrics.get("total_over_accuracy", "N/A")

    if isinstance(home_mae, float):
        home_mae = f"{home_mae:.2f}"
    if isinstance(away_mae, float):
        away_mae = f"{away_mae:.2f}"
    if isinstance(sc_acc, float):
        sc_acc = f"{sc_acc:.1%}"
    if isinstance(to_acc, float):
        to_acc = f"{to_acc:.1%}"

    msg = (
        f"📈 模型状态报告\n\n"
        f"版本: {version}\n"
        f"模型可用: {'✅' if model_available else '❌'}\n"
        f"特征数量: {feature_count}\n"
        f"训练样本: {total_data_points}\n"
        f"最后训练: {last_trained}\n\n"
        f"🎯 模型指标\n"
        f"主队得分 MAE: {home_mae}\n"
        f"客队得分 MAE: {away_mae}\n"
        f"让分覆盖准确率: {sc_acc}\n"
        f"大小分准确率: {to_acc}"
        f"{perf_line}"
    )
    send_message(msg)
    logger.info("Model status sent to Telegram")


if __name__ == "__main__":
    run_model_status()
