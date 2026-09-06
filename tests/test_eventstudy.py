import numpy as np
import pandas as pd

from tradebot.backtest.eventstudy import dedupe_positions, event_study
from tradebot.strategies.base import Strategy
from tradebot.strategies.swing import SwingStrategy

NY = "America/New_York"


def _bars(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n, tz=NY)
    c = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.003, "low": np.minimum(o, c) * 0.997, "close": c, "volume": 1e6}, index=idx)


class Oracle(SwingStrategy):
    """作弊策略：偷看未来，只在接下来 5 根 bar 会涨 3% 以上的日子发信号。用来验证事件研究能识别出有信息量的信号。"""
    name = "oracle"

    def compute(self, symbol, df):
        c = df["close"]
        fut = c.shift(-5) / df["open"].shift(-1) - 1
        entry = (fut > 0.03).fillna(False)
        return pd.DataFrame({"entry": entry, "exit": False, "score": fut}, index=df.index)


def test_dedupe():
    assert list(dedupe_positions(np.array([1, 3, 5, 20, 21, 40]), 10)) == [1, 20, 40]


def test_oracle_signals_beat_baseline_and_random_signals_do_not():
    data = {f"S{i}": _bars(seed=i) for i in range(4)}
    res = event_study(Oracle(), data, horizons=(1, 5, 10), dedupe_bars=1)
    row5 = res.summary.set_index("持有bar").loc[5]
    assert row5["信号数"] > 20 and row5["信号平均%"] > 3.0 and row5["超额%"] > 2.0 and row5["信号胜率%"] > 95
    assert res.path_signal.loc[5] > res.path_baseline.loc[5]
    assert set(res.by_symbol["标的"]) == set(data) and len(res.by_year) >= 1

    class Coin(SwingStrategy):
        name = "coin"

        def compute(self, symbol, df):
            rng = np.random.default_rng(len(symbol))
            entry = pd.Series(rng.random(len(df)) < 0.05, index=df.index)
            return pd.DataFrame({"entry": entry, "exit": False, "score": 0.0}, index=df.index)

    res2 = event_study(Coin(), data, horizons=(5, 20), dedupe_bars=1)
    assert abs(res2.summary.set_index("持有bar").loc[20, "超额%"]) < 3.0  # 随机信号没有稳定超额


def test_date_range_and_single_symbol_strategy():
    from tradebot.strategies import MACross
    data = {"A": _bars(seed=7)}
    res = event_study(MACross(fast=5, slow=20), data, horizons=(5,), start="2024-06-01", end="2025-03-01")
    assert (res.signals["date"] >= pd.Timestamp("2024-06-01", tz=NY)).all() and (res.signals["date"] <= pd.Timestamp("2025-03-01", tz=NY)).all()
