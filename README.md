# Swing Agent

A rules-based US-equity swing-trading research dashboard and paper-trading loop.
Python · Streamlit · yfinance · Alpaca (paper/live). Chinese/English UI, light/dark themes.

美股波段交易研究面板 + 模拟盘循环：规则化信号、板块轮动、末日期权与 LEAP 分析、底部确认、信号验证、回测。

## Features

- **Signals**: 8 rule-based strategies (trend pullback, breakout, post-earnings drift, oversold bounce, bottom recovery,
  relative momentum, 52-week-high breakout, main-wave trend template), a "resonance" count across strategies,
  main-wave stage labels, day-by-day signal history, click-to-detail (catalysts, fundamentals, chart, news).
- **Market / macro**: index snapshot, sector rotation (RRG quadrants with rollback), official macro calendar sync
  (FOMC / BEA / Census / BLS), earnings calendar.
- **Options**: 0DTE range/breakout probabilities, LEAP call comparison (delta, leverage, breakeven, implied vs
  historical win rate, contract price & IV history), underlying IV30 / HV30 / IV Rank (self-accumulated history).
- **Validation**: event studies (forward returns after a signal vs baseline, by year, by symbol, per-stock selection),
  a strategy × symbol win-rate matrix with best-strategy-per-stock / best-stock-per-strategy rankings, vectorised
  backtests with slot-based portfolio, ATR stops, market-regime filter, no-lookahead tests for every strategy.
- **Execution**: local paper broker or Alpaca paper/live via API (account, positions with P&L, orders, equity history in the UI;
  keys entered in a local-only settings form, never committed); risk layer (position caps, daily loss halt, drawdown kill switch);
  "fresh" entry mode that only follows new signals; running-strategy status card with next decision time.

Everything is historical statistics and rules. **Nothing here is investment advice.**

## Run locally

```bash
conda create -n trading python=3.12 -y
conda activate trading
pip install -r requirements.txt
cp .env.example .env            # edit: SYMBOLS, STRATEGY, MODE, keys
streamlit run dashboard.py      # dashboard
python run.py                   # paper-trading loop (weekdays 9:35 ET)
pytest -q tests
```

## Deploy the dashboard on Streamlit Community Cloud (read-only)

1. Push this repo to GitHub. Never commit `.env`; keys go to Streamlit **Secrets**.
2. New app → main file `dashboard.py`.
3. In *Settings → Secrets* add (TOML):

```toml
SWING_CLOUD = "1"                # read-only mode: hides file-writing features
SYMBOLS = "@mag7,@storage"
STRATEGY = "trend_pullback"
VIEW_GROUPS = "mag7,storage"
# ANTHROPIC_API_KEY = "..."      # optional: Claude catalysts / macro brief
```

Cloud notes: no persistent disk (watchlist / calendar edits are disabled; caches rebuild on cold start, first load ~1–2 min),
the trading loop does not run there, free-tier RAM is ~1 GB, and Yahoo may throttle cloud IPs.

## Layout

```
dashboard.py          Streamlit UI (12 tabs)
run.py                daily decision loop (paper / Alpaca)
backtest.py           CLI backtest
tradebot/             data, strategies, backtest, risk, execution, options, macro, i18n
tests/                pytest (incl. no-lookahead checks for every strategy)
```

## License

MIT. Market data via Yahoo Finance is subject to Yahoo's terms; use at your own risk.
