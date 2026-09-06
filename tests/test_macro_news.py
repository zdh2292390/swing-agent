import json

import pandas as pd

from tradebot.agent.research import CatalystReport, MacroBrief, has_credentials
from tradebot.macro import EVENT_COLUMNS, MACRO_TICKERS, compute_snapshot, load_manual_events, market_history, save_manual_events, upcoming_earnings, upcoming_events
from tradebot.news import format_info
from tradebot.earnings import EarningsCalendar
from tradebot.universe import GROUPS

NY = "America/New_York"


def test_storage_group_present():
    assert "storage" in GROUPS and "MU" in GROUPS["storage"] and "PSTG" not in GROUPS["storage"]


def test_manual_events_round_trip(tmp_path):
    path = tmp_path / "ev.json"
    assert load_manual_events(path).empty
    df = pd.DataFrame([
        {"日期": "2026-09-16", "事件": "FOMC 决议", "影响": "高", "备注": ""},
        {"日期": "2026-09-11", "事件": "CPI", "影响": "高", "备注": "8 月"},
        {"日期": "", "事件": "   ", "影响": "", "备注": "空行应被丢掉"},
    ])
    saved = save_manual_events(df, path)
    assert list(saved["事件"]) == ["CPI", "FOMC 决议"]
    loaded = load_manual_events(path)
    assert list(loaded.columns) == EVENT_COLUMNS and len(loaded) == 2
    up = upcoming_events(loaded, days=45, now=pd.Timestamp("2026-09-04", tz=NY))
    assert list(up["距今天"]) == [7, 12]
    assert upcoming_events(loaded, days=3, now=pd.Timestamp("2026-09-04", tz=NY)).empty


def test_upcoming_earnings_from_calendar():
    ev = pd.DataFrame({"surprise_pct": [1.0, 2.0]},
                      index=pd.DatetimeIndex([pd.Timestamp("2026-09-10 16:00", tz=NY), pd.Timestamp("2026-12-01 08:00", tz=NY)]))
    cal = EarningsCalendar(events={"AAPL": ev, "MSFT": ev.iloc[1:]})
    out = upcoming_earnings(["AAPL", "MSFT"], cal, groups={"mag7": ["AAPL", "MSFT"]}, days=30, now=pd.Timestamp("2026-09-04", tz=NY))
    assert list(out["标的"]) == ["AAPL"] and out.iloc[0]["盘前/盘后"] == "盘后" and out.iloc[0]["板块"] == "mag7"


def test_format_info_handles_missing_fields():
    rows = dict(format_info({"marketCap": 2.5e12, "trailingPE": 30.0, "sector": "Technology"}))
    assert rows["市值 ($)"] == "2.50 万亿"
    assert rows["PE（TTM / 前瞻）"] == "30.0 / n/a"
    assert "净利率" not in rows


def test_research_schemas_and_credentials(monkeypatch):
    rep = CatalystReport(symbol="NVDA", one_liner="x", bullish=[], bearish=[])
    assert rep.watch == []
    MacroBrief(as_of="2026-09-04", regime="r", upcoming_events=[], recent_news=[])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr("tradebot.agent.research.Path.home", lambda: __import__("pathlib").Path("/nonexistent"))
    assert has_credentials() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert has_credentials() is True


def test_snapshot_and_history_from_synthetic_closes():
    import numpy as np
    idx = pd.bdate_range("2025-06-02", periods=300)
    closes = pd.DataFrame({code: 100 * np.cumprod(1 + np.random.default_rng(i).normal(0.0005, 0.01, 300)) for i, (code, _, _) in enumerate(MACRO_TICKERS)}, index=idx)
    snap = compute_snapshot(closes)
    assert list(snap["代码"]) == [c for c, _, _ in MACRO_TICKERS]
    assert snap["名称"].iloc[1] == "纳指100" and snap["代码"].iloc[1] == "^NDX"
    assert snap["最新"].notna().all() and snap["趋势"].str.len().gt(0).all()
    hist = market_history(closes, ["^GSPC", "^NDX", "SMH"], days=63)
    assert list(hist.columns) == ["标普500", "纳指100", "半导体ETF"] and len(hist) == 63
    assert (hist.iloc[0].round(6) == 100).all()
