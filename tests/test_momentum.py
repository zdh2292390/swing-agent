import numpy as np
import pandas as pd

from tradebot.backtest import event_study, run_backtest
from tradebot.strategies import High52Breakout, RelativeMomentum, TrendPullback, get_strategy

NY = "America/New_York"


def _df(closes, vol=None, seed=0):
    c = np.asarray(closes, dtype=float)
    idx = pd.bdate_range("2023-01-02", periods=len(c), tz=NY)
    o = np.r_[c[0], c[:-1]]
    v = np.asarray(vol, dtype=float) if vol is not None else np.full(len(c), 1e6)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c, "volume": v}, index=idx)


def test_relative_momentum_ranks_cross_sectionally():
    n = 400
    rng = np.random.default_rng(0)
    base = np.cumprod(1 + rng.normal(0.0003, 0.008, n))
    data = {f"S{i}": _df(100 * base * np.cumprod(1 + np.full(n, drift))) for i, drift in enumerate([0.004, 0.002, 0.0005, -0.0005, -0.002])}
    strat = RelativeMomentum(top_frac=0.2, exit_frac=0.4, regime_filter=False)
    W = strat.target_weights(data)
    last = W.iloc[-1]
    assert last["S0"] > 0 and last["S4"] == 0            # 最强的进，最弱的不进
    sig = strat.compute("S0", data["S0"])                 # prepare 后 compute 返回横截面版本
    assert sig["entry"].iloc[-1] and not strat.compute("S4", data["S4"])["entry"].iloc[-1]
    fresh = RelativeMomentum(regime_filter=False)
    fb = fresh.compute("S4", data["S4"])                   # 没 prepare：退化版，仍是因果的
    assert len(fb) == n


def test_high52_breakout_needs_tight_base_and_volume():
    n = 330
    flat = 100 + np.sin(np.arange(n - 30) / 5.0) * 2      # 窄幅整理，振幅 4%
    up = np.linspace(102, 118, 30)
    closes = np.r_[flat, up]
    vol = np.full(n, 1e6)
    vol[len(flat):] = 2.5e6                                # 突破放量
    vol[len(flat) - 20:len(flat)] = 6e5                    # 整理末期缩量
    sig = High52Breakout(regime_filter=False).compute("X", _df(closes, vol))
    entries = sig.index[sig["entry"]]
    assert len(entries) >= 1 and entries[0] >= _df(closes).index[len(flat)]
    no_vol = High52Breakout(regime_filter=False).compute("X", _df(closes, np.full(n, 1e6)))
    assert not no_vol["entry"].any()                       # 不放量不算突破


def test_regime_filter_blocks_new_entries_when_benchmark_below_sma():
    n = 400
    up = _df(100 * np.cumprod(1 + np.full(n, 0.002)))
    spy_down = _df(np.linspace(500, 300, n))               # 大盘一路跌，始终在 200 日线下方
    data = {"X": up, "SPY": spy_down}
    strat = TrendPullback(regime_filter=True, regime_symbol="SPY")
    W = strat.target_weights(data)
    assert (W["X"] == 0).all()                             # 大盘过滤开着：一次都不开仓
    strat_off = TrendPullback(regime_filter=False)
    W2 = strat_off.target_weights({"X": up})
    assert W2["X"].max() >= 0                              # 关掉过滤能正常运行


def test_event_study_prepares_cross_sectional_strategy():
    n = 400
    rng = np.random.default_rng(1)
    data = {f"S{i}": _df(100 * np.cumprod(1 + rng.normal(0.0003 + 0.001 * i, 0.01, n))) for i in range(4)}
    res = event_study(RelativeMomentum(regime_filter=False), data, horizons=(5, 20), dedupe_bars=5)
    assert "持有bar" in res.summary.columns and res.summary["信号数"].max() > 0


def test_relative_momentum_excludes_foreign_and_marks_state():
    n = 400
    rng = np.random.default_rng(3)
    data = {"AAA": _df(100 * np.cumprod(1 + np.full(n, 0.003))), "BBB": _df(100 * np.cumprod(1 + rng.normal(0, 0.01, n))),
            "005930.KS": _df(100 * np.cumprod(1 + np.full(n, 0.006)))}  # 海外涨得最猛，但不能占名额
    strat = RelativeMomentum(top_frac=0.5, regime_filter=False)
    W = strat.target_weights(data)
    assert (W["005930.KS"] == 0).all() and W["AAA"].iloc[-1] > 0
    assert not strat.compute("005930.KS", data["005930.KS"])["entry"].any()
    assert RelativeMomentum.signal_kind == "state" and TrendPullback.signal_kind == "event"
    res = event_study(strat, {k: v for k, v in data.items() if k != "005930.KS"}, horizons=(5,), dedupe_bars=1)
    # 状态型只统计“进入”事件：AAA 一直在列，进入事件远少于在列天数
    assert 0 < res.summary["信号数"].iloc[0] < 50
