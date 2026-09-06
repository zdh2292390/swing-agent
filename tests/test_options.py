import datetime as dt

import numpy as np
import pandas as pd

from tradebot.earnings import EarningsCalendar
from tradebot.options import build_odte_table, event_day_moves, prob_within, realized_moves, trading_days_between

NY = "America/New_York"


def test_trading_days_between_skips_weekend_and_holiday():
    assert trading_days_between(dt.date(2026, 9, 4), dt.date(2026, 9, 4)) == 0
    assert trading_days_between(dt.date(2026, 9, 4), dt.date(2026, 9, 8)) == 1   # 周末 + 劳动节 9/7 休市
    assert trading_days_between(dt.date(2026, 9, 8), dt.date(2026, 9, 11)) == 3


def _closes(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-06-02", periods=n, tz=NY)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)


def test_realized_moves_and_probabilities():
    c = _closes()
    m1 = realized_moves(c, 1, lookback=250)
    assert len(m1) == 250 and (m1 >= 0).all()
    assert 0.55 < prob_within(m1, 1.0) < 0.85      # 日波动 σ≈1%，|r|<1% 大约 68%
    assert prob_within(m1, 100.0) == 1.0 and prob_within(m1, 0.0) == 0.0
    assert np.isnan(prob_within(np.array([]), 1.0))
    m5 = realized_moves(c, 5)
    assert np.median(m5) > np.median(m1)


def test_event_day_moves_uses_prev_close():
    c = _closes(50)
    c.iloc[10] = c.iloc[9] * 1.08
    mv = event_day_moves(c, [c.index[10], c.index[0], c.index[20]], horizon=1)
    assert len(mv) == 2 and abs(mv[0] - 8.0) < 1e-9  # 第 0 天没有前收盘被跳过


def test_build_table_with_fake_chain_and_events():
    today = dt.date(2026, 9, 4)
    c = _closes(400)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6})
    chains = {
        "AAA": {"symbol": "AAA", "expiry": "2026-09-08", "dte_cal": 4, "dte_trading": 1, "spot": float(c.iloc[-1]), "atm_strike": 100.0,
                "call_mid": 1.0, "put_mid": 1.0, "straddle": 2.0, "implied_move_pct": 2.0, "atm_iv": 30.0, "iv_move_pct": 1.6,
                "atm_oi": 500, "atm_volume": 100, "spread_pct": 5.0, "n_expiries_7d": 4, "is_daily": True},
        "BBB": None,
    }
    ev = pd.DataFrame({"surprise_pct": [1.0]}, index=pd.DatetimeIndex([pd.Timestamp("2026-09-07 16:00", tz=NY)]))
    cal = EarningsCalendar(events={"AAA": ev})
    macro = pd.DataFrame([{"日期": "2026-09-08", "事件": "CPI 通胀", "影响": "高", "备注": ""},
                          {"日期": "2026-09-08", "事件": "月度期权到期", "影响": "低", "备注": ""}])
    tbl = build_odte_table(["AAA", "BBB"], {"AAA": df}, chains, cal, macro, groups={"g": ["AAA"]}, today=today)
    assert list(tbl["标的"]) == ["AAA"]
    r = tbl.iloc[0]
    assert r["交易日"] == 1 and r["隐含波动%"] == 2.0 and r["流动性"] == "✓" and r["日频到期"] == "是"
    assert "财报 09-07" in r["事件"] and "CPI 通胀" in r["事件"] and "月度期权到期" not in r["事件"]
    assert 0 <= r["区间内概率"] <= 100 and abs(r["区间内概率"] + r["突破概率"] - 100) < 1e-9
    assert r["事件分布"] == ""  # 只有一次历史财报，样本不足不启用事件分布
