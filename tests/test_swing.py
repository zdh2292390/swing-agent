from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradebot.backtest.engine import extract_trades, run_backtest, trade_stats
from tradebot.earnings import EarningsCalendar
from tradebot.strategies import Breakout, PostEarningsMomentum, TrendPullback
from tradebot.strategies.swing import SwingStrategy

NY = "America/New_York"
IDX = pd.bdate_range("2024-01-02", periods=40, tz=NY)


def flat(n=40, close=100.0, spread=1.0):
    return pd.DataFrame({"open": close, "high": close + spread, "low": close - spread, "close": close, "volume": 1e6}, index=IDX[:n])


@dataclass
class Dummy(SwingStrategy):
    plan: dict = field(default_factory=dict)  # sym -> (entry_bars, exit_bars, score)
    name: str = "dummy"

    def compute(self, symbol, df):
        entries, exits, score = self.plan[symbol]
        e = pd.Series(False, index=df.index); e.iloc[list(entries)] = True
        x = pd.Series(False, index=df.index); x.iloc[list(exits)] = True
        return pd.DataFrame({"entry": e, "exit": x, "score": float(score)}, index=df.index)


def test_slots_respect_max_positions_and_rank_by_score():
    data = {s: flat() for s in "ABC"}
    st = Dummy(max_positions=2, atr_stop_mult=0.0, plan={"A": ([5], [], 1), "B": ([5], [], 3), "C": ([5], [], 2)})
    w = st.target_weights(data)
    assert w.iloc[4].sum() == 0
    assert w.loc[IDX[5], "B"] == 0.5 and w.loc[IDX[5], "C"] == 0.5 and w.loc[IDX[5], "A"] == 0
    assert w.loc[IDX[30], "B"] == 0.5  # 没有出场信号就一直持有
    assert (w.sum(axis=1) <= 1 + 1e-12).all()


def test_time_stop_and_exit_signal():
    data = {"A": flat(), "B": flat()}
    st = Dummy(max_positions=2, atr_stop_mult=0.0, max_hold_bars=3, plan={"A": ([5], [], 1), "B": ([5], [7], 1)})
    w = st.target_weights(data)
    assert w.loc[IDX[7], "A"] == 0.5 and w.loc[IDX[8], "A"] == 0  # 持有 3 根后时间止损
    assert w.loc[IDX[6], "B"] == 0.5 and w.loc[IDX[7], "B"] == 0  # 出场信号当根退出


def test_atr_stop_triggers_on_low():
    df = flat(spread=1.0)  # TR=2 -> ATR≈2
    df.loc[IDX[10], "low"] = 90.0
    st = Dummy(max_positions=1, atr_n=3, atr_stop_mult=1.0, trailing_stop=False, plan={"A": ([5], [], 1)})
    w = st.target_weights({"A": df})
    assert w.loc[IDX[9], "A"] == 1.0 and w.loc[IDX[10], "A"] == 0.0


def test_earnings_blackout_blocks_entry():
    ev = pd.DataFrame({"surprise_pct": [1.0]}, index=pd.DatetimeIndex([IDX[7] + pd.Timedelta(hours=16)]))
    cal = EarningsCalendar(events={"A": ev})
    st = Dummy(max_positions=1, atr_stop_mult=0.0, earnings_blackout_days=5, calendar=cal, plan={"A": ([5, 12], [], 1)})
    w = st.target_weights({"A": flat()})
    assert w.loc[IDX[5], "A"] == 0.0   # 财报前 2 天，禁止开仓
    assert w.loc[IDX[12], "A"] == 1.0  # 财报后正常


def test_post_earnings_reaction_day_mapping_and_entry():
    close = pd.Series(100.0, index=IDX)
    close.iloc[11:] = 105.0   # 事件 idx[10] 16:00 盘后 -> 反应日 idx[11] 涨 5%
    close.iloc[20:] = 105.0 * 1.01  # 事件 idx[20] 08:00 盘前 -> 反应日 idx[20] 只涨 1%
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1e6}, index=IDX)
    ev = pd.DataFrame({"surprise_pct": [2.0, 2.0]},
                      index=pd.DatetimeIndex([IDX[10] + pd.Timedelta(hours=16), IDX[20] + pd.Timedelta(hours=8)]))
    cal = EarningsCalendar(events={"A": ev})
    rd = cal.reaction_days("A", IDX)
    assert list(rd["reaction_day"]) == [IDX[11], IDX[20]]
    st = PostEarningsMomentum(min_reaction=0.03, calendar=cal, atr_stop_mult=0.0, max_hold_bars=5)
    sig = st.compute("A", df)
    assert sig["entry"].iloc[11] and not sig["entry"].iloc[20]
    w = st.target_weights({"A": df})
    assert w.loc[IDX[11], "A"] > 0 and w.loc[IDX[16], "A"] == 0  # 持有 5 根后退出


def _walk(seed, n=600):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n, tz=NY)
    c = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    o = np.r_[100, c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99, "close": c,
                         "volume": 1e6 * (1 + rng.random(n))}, index=idx)


def test_trend_pullback_and_breakout_produce_valid_weights():
    data = {s: _walk(i) for i, s in enumerate("ABCDE")}
    for strat in [TrendPullback(max_positions=3), Breakout(max_positions=3)]:
        res = run_backtest(data, strat)
        w = res.signals
        assert (w >= 0).all().all() and (w.sum(axis=1) <= 1 + 1e-9).all()
        assert (w.max(axis=1) <= 1 / 3 + 1e-9).all()
        assert res.metrics and "n_trades" in res.trade_stats


def test_extract_trades_uses_next_open_prices():
    df = _walk(1, n=10)
    w = pd.DataFrame({"A": [0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0]}, index=df.index)
    tr = extract_trades(w, {"A": df}, "next_open")
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t["entry_time"] == df.index[2] and t["exit_time"] == df.index[4]
    assert t["entry_px"] == df["open"].iloc[2] and t["exit_px"] == df["open"].iloc[4]
    assert abs(t["ret"] - (df["open"].iloc[4] / df["open"].iloc[2] - 1)) < 1e-12
    assert trade_stats(tr)["n_trades"] == 1
