import numpy as np
import pandas as pd

from tradebot.backtest import run_backtest
from tradebot.strategies import BuyAndHold
from tradebot.strategies.base import Strategy


def make_bars(n=200, seed=0, drift=0.0005):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-02 09:30", periods=n, freq="1h", tz="America/New_York")
    ret = rng.normal(drift, 0.003, n)
    close = 100 * np.cumprod(1 + ret)
    open_ = np.r_[100.0, close[:-1]] * (1 + rng.normal(0, 0.0005, n))
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) * 1.001, "low": np.minimum(open_, close) * 0.999,
         "close": close, "volume": 1e6},
        index=idx,
    )


class OneBar(Strategy):
    """只在第 k 根 bar 收盘时给出信号，用来验证成交时点。"""
    name = "one_bar"

    def __init__(self, k):
        self.k = k

    def target_weights(self, df):
        w = pd.Series(0.0, index=df.index)
        w.iloc[self.k] = 1.0
        return w


def test_signal_is_shifted_one_bar_and_filled_at_next_open():
    df = make_bars()
    k = 50
    res = run_backtest({"X": df}, OneBar(k), slippage_bps=0, commission_bps=0)
    nonzero = res.returns[res.returns != 0]
    assert list(nonzero.index) == [df.index[k + 1]]
    expected = df["open"].iloc[k + 2] / df["open"].iloc[k + 1] - 1
    assert abs(nonzero.iloc[0] - expected) < 1e-12


def test_buy_and_hold_matches_open_to_open_compounding():
    df = make_bars()
    res = run_backtest({"X": df}, BuyAndHold(), slippage_bps=0, commission_bps=0, initial_cash=1.0)
    # 第一根 bar 持仓为 0（信号后移），之后满仓；最后一根没有下一开盘价被丢掉
    expected = (df["open"].shift(-1) / df["open"]).iloc[1:-1].prod()
    assert abs(res.equity.iloc[-1] - expected) < 1e-9


def test_costs_reduce_returns_and_equal_weight_caps_exposure():
    data = {"A": make_bars(seed=1), "B": make_bars(seed=2)}
    free = run_backtest(data, BuyAndHold(), slippage_bps=0)
    costly = run_backtest(data, BuyAndHold(), slippage_bps=50)
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert abs(free.exposure.iloc[-1] - 1.0) < 1e-12
    assert (free.weights.max(axis=1) <= 0.5 + 1e-12).all()


def test_metrics_present():
    res = run_backtest({"X": make_bars()}, BuyAndHold())
    for k in ["cagr", "sharpe", "max_drawdown", "ann_turnover"]:
        assert k in res.metrics
    assert res.metrics["max_drawdown"] <= 0


def test_late_listing_symbol_does_not_truncate_history():
    a = make_bars(n=200, seed=3)
    b = make_bars(n=200, seed=4).iloc[120:]  # B 在第 120 根 bar 才"上市"
    res = run_backtest({"A": a, "B": b}, BuyAndHold(), slippage_bps=0)
    assert res.equity.index[0] == a.index[0]          # 起点仍是 A 的历史
    # B 上市前 A 独占额度 1.0；上市后（信号后移一根）各 0.5
    assert abs(res.weights["A"].iloc[50] - 1.0) < 1e-12
    assert abs(res.weights["A"].iloc[-1] - 0.5) < 1e-12
    assert abs(res.weights["B"].iloc[-1] - 0.5) < 1e-12
    assert res.weights["B"].iloc[:120].abs().sum() == 0


def test_slice_result_recomputes_equity_and_trades():
    from tradebot.backtest import slice_result
    df = make_bars(n=300)
    res = run_backtest({"X": df}, BuyAndHold(), slippage_bps=0)
    start, end = df.index[100], df.index[200]
    part = slice_result(res, start, end, initial_cash=1.0)
    assert part.equity.index[0] >= start and part.equity.index[-1] <= end
    expected = (1 + res.returns[(res.returns.index >= start) & (res.returns.index <= end)]).prod()
    assert abs(part.equity.iloc[-1] - expected) < 1e-12
    assert "cagr" in part.metrics
