# NBA Quant System

生产级 NBA 预测与复盘系统（手动触发版）。

## 特性
- 严格 balldontlie API 客户端（限定端点和参数校验）。
- 首次运行自动拉取近 2 个赛季 + 当前赛季比赛数据入库。
- 本地 SQLite 持久化：`games`、`odds_history`、`predictions_snapshot`、`results`、`model_history`。
- 双模型：让分与大小分（LightGBM 优先，自动回退）。
- 每场独立 10000 次以上回合制蒙特卡洛模拟。
- 盘口行为分析：sharp movement / reverse line movement / market confidence。
- Telegram 中文高密度单页输出，包含控制台按钮。
- 仅 `workflow_dispatch` 手动触发，禁止自动定时。

## 目录
```
nba-quant-system/
  app/
  models/
  data/database.sqlite
  .github/workflows/predict.yml
  .github/workflows/review.yml
```

## 环境变量
复制 `.env.example` 并配置：
- `BALLDONTLIE_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

可选（如果你希望 Telegram 回调触发 GitHub workflow）：
- `GITHUB_TOKEN`
- `GITHUB_REPOSITORY`

## 本地运行
```bash
cd nba-quant-system
pip install -r requirements.txt
python -m app.prediction_engine
python -m app.review_engine
```

## GitHub Actions
- `predict.yml`：手动点击 **Run workflow** 后生成当日预测。
- `review.yml`：手动点击 **Run workflow** 后执行赛后复盘。

> 两个工作流都只支持 `workflow_dispatch`，不会自动定时运行。
