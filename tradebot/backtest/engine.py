"""向量化回测引擎（多标的、只做多、支持日线和小时线、支持单标的策略和组合策略）。

成交假设（fill="next_open"，默认）：
  第 t-1 根 bar 收盘后算出信号 -> 第 t 根 bar 开盘价成交 -> 持有到第 t+1 根 bar 开盘。
  因此第 t 根 bar 的持仓 pos_t = target_{t-1}，赚取的是 open_{t+1}/open_t - 1。
  这样严格避免了"用收盘价算信号又按同一根收盘价成交"的未来函数。

fill="close"：pos_t = target_{t-1}，赚 close_t/close_{t-1} - 1，
  等价于在信号 bar 的收盘价成交，略偏乐观，仅供对比。

成本：每次仓位变动按 (commission_bps + slippage_bps) * |Δpos| 扣除。
单标的策略（Strategy）自动包成等权组合；组合策略（PortfolioStrategy）直接给组合权重。
引擎再套 max_position_pct 和总仓位 ≤ 1 两道硬约束。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..strategies.base import PortfolioStrategy, Strategy, as_portfolio, union_index
from .metrics import compute_metrics


@dataclass
class BacktestResult:
    strategy: str
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame  # 实际持有的组合权重（已后移一根 bar）
    signals: pd.DataFrame  # 策略给出的目标权重（未后移）
    turnover: pd.Series
    exposure: pd.Series
    per_symbol_returns: pd.DataFrame
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_stats: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def _bar_returns(df: pd.DataFrame, fill: str) -> pd.Series:
    if fill == "next_open":
        return df["open"].shift(-1) / df["open"] - 1
    if fill == "close":
        return df["close"] / df["close"].shift(1) - 1
    raise ValueError(f"未知成交假设 {fill!r}，可选 next_open / close")


def extract_trades(weights: pd.DataFrame, data: dict[str, pd.DataFrame], fill: str = "next_open") -> pd.DataFrame:
    """从持仓权重序列还原逐笔交易（权重 0 -> 正 为入场，正 -> 0 为出场），价格按成交假设取。"""
    rows = []
    for sym in weights.columns:
        w = weights[sym].values
        df = data[sym]
        if fill == "next_open":
            px = df["open"].reindex(weights.index).values  # 第 t 根持仓在 open_t 建立
        else:
            px = df["close"].shift(1).reindex(weights.index).values
        last_close = df["close"].reindex(weights.index).ffill().values
        entry_i = None
        for i in range(len(w)):
            if entry_i is None and w[i] > 0:
                entry_i = i
            elif entry_i is not None and w[i] == 0:
                rows.append(_trade_row(sym, weights.index, entry_i, i, px, w, open_trade=False))
                entry_i = None
        if entry_i is not None:  # 仍持有：按最新收盘估值
            rows.append(_trade_row(sym, weights.index, entry_i, len(w) - 1, px, w, open_trade=True, mark=last_close[-1]))
    cols = ["symbol", "entry_time", "exit_time", "entry_px", "exit_px", "ret", "bars_held", "weight", "open"]
    return pd.DataFrame(rows, columns=cols)


def _trade_row(sym, index, i0, i1, px, w, open_trade, mark=None):
    entry_px = px[i0]
    exit_px = mark if open_trade else px[i1]
    ret = exit_px / entry_px - 1 if entry_px and not np.isnan(entry_px) and exit_px and not np.isnan(exit_px) else np.nan
    return {
        "symbol": sym, "entry_time": index[i0], "exit_time": index[i1], "entry_px": entry_px, "exit_px": exit_px,
        "ret": ret, "bars_held": i1 - i0, "weight": float(np.nanmax(w[i0:i1 + 1])), "open": open_trade,
    }


def trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    closed = trades[~trades["open"]] if len(trades) else trades
    r = closed["ret"].dropna() if len(closed) else pd.Series(dtype=float)
    if len(r) == 0:
        return {"n_trades": 0}
    wins, losses = r[r > 0], r[r <= 0]
    return {
        "n_trades": int(len(r)),
        "open_trades": int(trades["open"].sum()),
        "win_rate": float((r > 0).mean()),
        "avg_ret": float(r.mean()),
        "median_ret": float(r.median()),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "payoff": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else float("nan"),
        "best": float(r.max()),
        "worst": float(r.min()),
        "avg_hold_bars": float(closed["bars_held"].mean()),
    }


def run_backtest(
    data: dict[str, pd.DataFrame],
    strategy: Strategy | PortfolioStrategy,
    initial_cash: float = 100_000.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 2.0,
    fill: str = "next_open",
    max_position_pct: float | None = None,
) -> BacktestResult:
    if not data:
        raise ValueError("没有可回测的标的")
    pstrat = as_portfolio(strategy)
    union = union_index(data)
    cost_rate = (commission_bps + slippage_bps) / 1e4

    target = pstrat.target_weights(data).reindex(index=union, columns=list(data)).ffill().fillna(0.0).clip(0.0, 1.0)
    if max_position_pct is not None:
        target = target.clip(upper=max_position_pct)
    gross = target.sum(axis=1)
    target = target.div(gross.where(gross > 1.0, 1.0), axis=0)
    weights = target.shift(1).fillna(0.0)  # 后移一根 bar 再持有

    bar_rets = pd.DataFrame({sym: _bar_returns(df, fill).reindex(union) for sym, df in data.items()})
    valid = bar_rets.notna().any(axis=1)  # next_open 模式下最后一根没有下一开盘价
    bar_rets = bar_rets.fillna(0.0)
    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    per_symbol = weights * bar_rets
    port_ret = per_symbol.sum(axis=1) - turnover * cost_rate

    port_ret, weights, turnover, per_symbol = port_ret[valid], weights[valid], turnover[valid], per_symbol[valid]
    equity = initial_cash * (1 + port_ret).cumprod()
    exposure = weights.sum(axis=1)

    trades = extract_trades(weights, data, fill)
    result = BacktestResult(
        strategy=pstrat.describe(),
        equity=equity,
        returns=port_ret,
        weights=weights,
        signals=target,
        turnover=turnover,
        exposure=exposure,
        per_symbol_returns=per_symbol,
        trades=trades,
        trade_stats=trade_stats(trades),
    )
    result.metrics = compute_metrics(port_ret, equity, turnover, exposure)
    return result


def slice_result(result: BacktestResult, start=None, end=None, initial_cash: float = 100_000.0) -> BacktestResult:
    """把回测结果裁到 [start, end]，重算净值、指标、交易统计。策略信号仍用全部历史算，所以预热期不受影响。"""
    idx = result.returns.index
    mask = pd.Series(True, index=idx)
    if start is not None:
        s_ = pd.Timestamp(start)
        s_ = s_.tz_localize(idx.tz) if idx.tz is not None and s_.tzinfo is None else s_
        mask &= idx >= s_
    if end is not None:
        e_ = pd.Timestamp(end)
        e_ = e_.tz_localize(idx.tz) if idx.tz is not None and e_.tzinfo is None else e_
        mask &= idx <= e_
    ret = result.returns[mask]
    if ret.empty:
        raise ValueError("所选区间内没有数据")
    weights = result.weights[mask]
    turnover = result.turnover[mask]
    per_symbol = result.per_symbol_returns[mask]
    equity = initial_cash * (1 + ret).cumprod()
    exposure = weights.sum(axis=1)
    trades = result.trades[(result.trades["entry_time"] >= ret.index[0]) & (result.trades["entry_time"] <= ret.index[-1])] if len(result.trades) else result.trades
    out = BacktestResult(result.strategy, equity, ret, weights, result.signals.loc[ret.index[0]:ret.index[-1]], turnover, exposure, per_symbol,
                         trades=trades.reset_index(drop=True), trade_stats=trade_stats(trades) if len(trades) else {"n_trades": 0})
    out.metrics = compute_metrics(ret, equity, turnover, exposure)
    return out
