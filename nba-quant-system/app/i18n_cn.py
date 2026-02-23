"""Centralized Chinese strings for Telegram messages.

All user-facing Telegram messages MUST use ``cn(key)`` to ensure
consistent Chinese output.  Internal logs may remain in English.
"""
from __future__ import annotations

_STRINGS: dict[str, str] = {
    # --- retrain_engine ---
    "model_cached": "📦 模型来源: 使用本地缓存模型",
    "model_loaded": "🧠 模型来源：Supabase已加载",
    "model_trained": "📦 模型来源: 训练新模型",
    "training_start": "🧠 系统正在学习历史比赛数据",
    "training_done": "✅ 学习完成",
    "building_dataset": "构建训练数据集...",
    "feature_count": "特征数量: {feature_count}",
    "sample_count": "训练样本数: {sample_count}",
    "training_report": (
        "📊 模型训练报告\n\n"
        "版本: {version}\n"
        "训练方式: Hybrid Architecture\n"
        "模型: LightGBM\n"
        "训练样本: {sample_count}\n"
        "特征数量: {feature_count}\n"
        "训练耗时: {duration:.1f} 秒\n"
        "主队得分模型: 完成\n"
        "客队得分模型: 完成\n"
        "让分覆盖模型: 完成 ({sc_acc:.1%})\n"
        "大小分模型: 完成 ({to_acc:.1%})"
    ),
    "upload_models": "上传模型到云端存储...",
    "upload_done": "模型上传完成",
    "skip_retrain_insufficient": "⏭ 新完成比赛不足 {min_games} 场，跳过重新训练",

    # --- model_status ---
    "model_status_report": (
        "📈 模型状态报告\n\n"
        "版本：{version}\n"
        "模型状态：{available}\n"
        "训练样本：{training_samples}\n"
        "平均误差：{mae_display}\n"
        "让分准确率：{sc_acc}\n"
        "大小分准确率：{to_acc}\n"
        "最后训练时间：{last_trained}"
    ),

    # --- review_engine ---
    "review_no_games": (
        "📊 复盘系统\n"
        "⚠️ 当前没有可复盘比赛\n"
        "比赛尚未结束或结果未更新"
    ),
    "review_header": "📊 昨日战绩｜{target_date}",
    "review_rolling": "📈 近30天滚动表现",
}


def cn(key: str, **kwargs: object) -> str:
    """Return a Chinese string by *key*, optionally formatted with *kwargs*.

    >>> cn("model_loaded")
    '📦 模型来源: 已加载历史学习成果'
    >>> cn("feature_count", feature_count=50)
    '特征数量: 50'
    """
    template = _STRINGS.get(key, key)
    if kwargs:
        return template.format(**kwargs)
    return template
