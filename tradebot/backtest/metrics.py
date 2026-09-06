from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _span_years(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return float("nan")
    secs = (index[-1] - index[0]).total_seconds()
    return secs / (365.25 * 24 * 3600)


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
    exposure: pd.Series,
) -> dict[str, float]:
    """基于每根 bar 的组合收益计算常用指标。年化因子由数据实际跨度推算，
    不依赖"每年多少根 bar"的假设，小时线、日线都适用。"""
    returns = returns.dropna()
    years = _span_years(returns.index)
    n = len(returns)
    if n == 0 or not years or years <= 0:
        return {}
    periods_per_year = n / years

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0
    std = float(returns.std())
    vol = std * math.sqrt(periods_per_year)
    sharpe = float(returns.mean()) / std * math.sqrt(periods_per_year) if std > 0 else float("nan")
    downside = returns[returns < 0]
    dstd = float(downside.std()) if len(downside) > 1 else float("nan")
    sortino = float(returns.mean()) / dstd * math.sqrt(periods_per_year) if dstd and dstd > 0 else float("nan")

    dd = equity / equity.cummax() - 1
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")

    active = returns[returns != 0]
    hit_rate = float((active > 0).mean()) if len(active) else float("nan")
    gains = active[active > 0].sum()
    losses = -active[active < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else float("inf")

    return {
        "bars": n,
        "years": years,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
        "avg_exposure": float(exposure.mean()),
        "ann_turnover": float(turnover.sum() / years),
        "final_equity": float(equity.iloc[-1]),
    }


_PCT = {"total_return", "cagr", "ann_vol", "max_drawdown", "hit_rate", "avg_exposure"}
METRIC_KEYS = [
    "bars", "years", "total_return", "cagr", "ann_vol", "sharpe", "sortino",
    "max_drawdown", "calmar", "hit_rate", "profit_factor", "avg_exposure",
    "ann_turnover", "final_equity",
]
METRIC_LABELS = {
    "bars": "bar 数", "years": "年数", "total_return": "总收益", "cagr": "年化收益",
    "ann_vol": "年化波动", "sharpe": "Sharpe", "sortino": "Sortino", "max_drawdown": "最大回撤",
    "calmar": "Calmar", "hit_rate": "胜率(按 bar)", "profit_factor": "盈亏比",
    "avg_exposure": "平均仓位", "ann_turnover": "年换手", "final_equity": "期末净值",
}


def fmt_metric(key: str, v) -> str:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "n/a" if math.isnan(v) else "inf"
    if key in _PCT:
        return f"{v * 100:.2f}%"
    if key == "bars":
        return f"{int(v)}"
    if key == "final_equity":
        return f"{v:,.0f}"
    return f"{v:.2f}"


def metrics_table(rows: dict[str, dict[str, float]], chinese: bool = False) -> pd.DataFrame:
    """rows: {列名: metrics} -> DataFrame，行是指标、列是策略，值已格式化为字符串。"""
    table = pd.DataFrame({c: {k: fmt_metric(k, m.get(k, float("nan"))) for k in METRIC_KEYS} for c, m in rows.items()})
    if chinese:
        table.index = [METRIC_LABELS[k] for k in table.index]
    return table


def format_metrics(rows: dict[str, dict[str, float]]) -> str:
    """rows: {列名: metrics}，返回对齐的文本表。"""
    cols = list(rows)
    width = max(len(c) for c in cols + ["metric"]) + 2
    lines = ["metric".ljust(16) + "".join(c.rjust(width) for c in cols)]
    for k in METRIC_KEYS:
        cells = [fmt_metric(k, rows[c].get(k, float("nan"))).rjust(width) for c in cols]
        lines.append(k.ljust(16) + "".join(cells))
    return "\n".join(lines)
