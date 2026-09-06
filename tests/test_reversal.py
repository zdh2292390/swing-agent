import numpy as np
import pandas as pd

from tradebot.backtest import run_backtest
from tradebot.strategies import BottomRecovery, OversoldBounce, get_strategy

NY = "America/New_York"


def _df(closes, vol=None):
    c = np.asarray(closes, dtype=float)
    idx = pd.bdate_range("2025-01-02", periods=len(c), tz=NY)
    o = np.r_[c[0], c[:-1]]
    v = np.asarray(vol, dtype=float) if vol is not None else np.full(len(c), 1e6)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.005, "low": np.minimum(o, c) * 0.995, "close": c, "volume": v}, index=idx)


def test_registry_has_reversal_strategies():
    assert isinstance(get_strategy("oversold_bounce"), OversoldBounce)
    assert isinstance(get_strategy("bottom_recovery", swing_k=3), BottomRecovery)


def test_oversold_bounce_enters_after_sharp_drop_and_up_bar():
    flat = np.full(60, 100.0) + np.sin(np.arange(60) / 2.0)   # 平稳
    crash = np.linspace(100, 84, 8)                            # 急跌 16%
    bounce = np.linspace(84, 97, 12)                           # 反弹
    df = _df(np.r_[flat, crash, bounce])
    sig = OversoldBounce().compute("X", df)
    first_entry = sig.index[sig["entry"]][0]
    assert first_entry >= df.index[len(flat) + len(crash)]        # 只在止跌（收盘 > 前收）之后进，不在下跌途中
    assert first_entry <= df.index[len(flat) + len(crash) + 2]
    assert sig.loc[df.index[-1], "exit"]                            # 回到均线上方后有出场信号
    res = run_backtest({"X": df}, OversoldBounce(), slippage_bps=0)
    assert res.weights["X"].max() > 0 and res.equity.iloc[-1] > res.equity.iloc[0]


def test_bottom_recovery_enters_after_w_bottom_breakout():
    down = np.linspace(140, 100, 40); bounce = np.linspace(100, 112, 8); pull = np.linspace(112, 103, 8); up = np.linspace(103, 125, 16)
    df = _df(np.r_[down, bounce, pull, up])
    strat = BottomRecovery(swing_k=3, drawdown_min=0.10)
    sig = strat.compute("X", df)
    entries = sig.index[sig["entry"]]
    assert len(entries) > 0
    assert entries[0] > df.index[len(down) + len(bounce) + len(pull)]   # 只在第二个低点确认并突破之后
    assert not sig["entry"].iloc[:len(down) + len(bounce)].any()         # 第一段反弹里不进（低点还没抬高 / 未确认）
    res = run_backtest({"X": df}, strat, slippage_bps=0)
    assert res.weights["X"].iloc[-1] > 0


def test_downtrend_generates_no_recovery_entry():
    df = _df(np.linspace(150, 90, 90))
    assert not BottomRecovery(swing_k=3).compute("X", df)["entry"].any()
