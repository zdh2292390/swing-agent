"""泄漏检查：任何一天的信号，只用那天及之前的数据重算，必须和从完整序列里读出来的一致。对全部已注册策略生效。"""
import numpy as np
import pandas as pd
import pytest

from tradebot.earnings import EarningsCalendar
from tradebot.strategies import STRATEGIES, get_strategy
from tradebot.strategies.base import Strategy
from tradebot.strategies.swing import SwingStrategy

NY = "America/New_York"


def _bars(n=320, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-02", periods=n, tz=NY)
    c = 100 * np.cumprod(1 + rng.normal(0.0003, 0.02, n))
    o = c * (1 + rng.normal(0, 0.004, n))
    v = rng.integers(5e5, 3e6, n).astype(float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + abs(rng.normal(0, 0.006, n))),
                         "low": np.minimum(o, c) * (1 - abs(rng.normal(0, 0.006, n))), "close": c, "volume": v}, index=idx)


def _strategy(name):
    kw = {"swing_k": 3} if name == "bottom_recovery" else {}
    s = get_strategy(name, **kw)
    if name == "post_earnings":  # 注入一份财报日历，避免联网
        ev = pd.DataFrame({"surprise_pct": [5.0, 3.0, -2.0]},
                          index=pd.DatetimeIndex([pd.Timestamp("2025-04-24 16:00", tz=NY), pd.Timestamp("2025-07-24 16:00", tz=NY), pd.Timestamp("2025-10-23 16:00", tz=NY)]))
        s.calendar = EarningsCalendar(events={"X": ev, "Y": ev})
    return s


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_signals_do_not_change_when_future_bars_are_removed(name):
    strat = _strategy(name)
    data = {"X": _bars(seed=1), "Y": _bars(seed=2)}
    for sym, df in data.items():
        if isinstance(strat, SwingStrategy):
            full = strat.compute(sym, df)
            cols = ("entry", "exit")
        else:
            full = strat.target_weights(df).to_frame("w")
            cols = ("w",)
        for d in df.index[-15:]:
            trunc = df[df.index <= d]
            part = strat.compute(sym, trunc).iloc[-1] if isinstance(strat, SwingStrategy) else strat.target_weights(trunc).to_frame("w").iloc[-1]
            for col in cols:
                a, b = full.loc[d, col], part[col]
                if isinstance(a, (bool, np.bool_)):
                    assert bool(a) == bool(b), f"{name} {sym} {d.date()} {col}: full={a} truncated={b}"
                else:
                    assert (pd.isna(a) and pd.isna(b)) or abs(float(a) - float(b)) < 1e-9, f"{name} {sym} {d.date()} {col}"
