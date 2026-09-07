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


def test_double_bottom_enters_on_neckline_breakout_only_after_second_low_confirmed():
    from tradebot.strategies import DoubleBottom
    from tradebot.strategies.reversal import find_double_bottoms
    down = np.linspace(140, 100, 30)          # 跌 29%
    up1 = np.linspace(100, 110, 10)           # 反弹到 110（颈线）
    down2 = np.linspace(110, 101, 10)         # 回到 101（第二低点，差 1%）
    up2 = np.linspace(101, 109, 8)            # 颈线下方磨
    brk = np.linspace(109, 120, 10)           # 突破颈线
    df = _df(np.r_[down, up1, down2, up2, brk])
    pats = find_double_bottoms(df, swing_k=3, min_gap=10)
    assert len(pats) == 1
    p = pats[0]
    assert p["L0"] == len(down) - 1 and p["L1"] == len(down) + len(up1) + len(down2) - 1
    assert p["breakout_i"] is not None and p["breakout_i"] >= p["confirm_i"]
    assert df["close"].iloc[p["breakout_i"]] > p["neckline"] and df["close"].iloc[p["breakout_i"] - 1] <= p["neckline"]
    strat = DoubleBottom(swing_k=3, min_gap=10)
    sig = strat.compute("X", df)
    assert sig["entry"].sum() == 1 and sig.index[sig["entry"]][0] == df.index[p["breakout_i"]]
    res = run_backtest({"X": df}, strat, slippage_bps=0)
    assert res.weights["X"].iloc[-1] > 0


def test_double_bottom_rejects_unequal_lows_and_downtrend():
    from tradebot.strategies.reversal import find_double_bottoms
    down = np.linspace(140, 100, 30); up1 = np.linspace(100, 110, 10); down2 = np.linspace(110, 92, 10); up2 = np.linspace(92, 115, 15)
    assert find_double_bottoms(_df(np.r_[down, up1, down2, up2]), swing_k=3, min_gap=10) == []   # 第二低点低了 8%，不是 W
    assert find_double_bottoms(_df(np.linspace(150, 90, 90)), swing_k=3) == []
