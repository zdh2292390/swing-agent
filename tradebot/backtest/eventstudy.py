"""信号验证（事件研究）：不看组合、不看止损，只问"信号出现之后股价平均怎么走"。

口径：信号在 bar t 收盘出现，按 t+1 开盘价入场，持有 h 根 bar，以 t+h 收盘计算收益；
基准 = 同一批标的、同一时间段里"任意一天"的同口径收益，用来判断信号有没有信息量。
同一标的 dedupe_bars 根 bar 内的重复信号只算第一次，避免连续信号把样本灌水。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..strategies.base import Strategy
from ..strategies.swing import SwingStrategy

HORIZONS = (1, 5, 10, 20, 40, 60)


@dataclass
class EventStudyResult:
    strategy: str
    signals: pd.DataFrame  # 每个信号一行：symbol, date, fwd_h..., mae_20, mfe_20
    baseline: pd.DataFrame  # 基准样本
    summary: pd.DataFrame  # 按持有期汇总
    path_signal: pd.Series  # 信号后平均累计收益路径（bar 1..max_h）
    path_baseline: pd.Series
    by_year: pd.DataFrame
    by_symbol: pd.DataFrame
    n_symbols: int = 0
    notes: list[str] = field(default_factory=list)


def _forward_table(df: pd.DataFrame, positions: np.ndarray, horizons: tuple[int, ...], max_h: int) -> tuple[pd.DataFrame, np.ndarray]:
    """给定信号位置，算各持有期收益、20 bar 内最大不利/有利偏移，以及逐 bar 路径矩阵。"""
    o, c, hi, lo = df["open"].values, df["close"].values, df["high"].values, df["low"].values
    n = len(df)
    rows, paths = [], []
    for i in positions:
        if i + 1 >= n:
            continue
        entry = o[i + 1]
        if not entry or np.isnan(entry):
            continue
        row = {"date": df.index[i]}
        for h in horizons:
            row[f"fwd_{h}"] = (c[i + h] / entry - 1) * 100 if i + h < n else np.nan
        w_end = min(i + 20, n - 1)
        row["mae_20"] = (lo[i + 1:w_end + 1].min() / entry - 1) * 100 if w_end > i else np.nan
        row["mfe_20"] = (hi[i + 1:w_end + 1].max() / entry - 1) * 100 if w_end > i else np.nan
        rows.append(row)
        p = np.full(max_h, np.nan)
        k = min(max_h, n - 1 - i)
        if k > 0:
            p[:k] = (c[i + 1:i + 1 + k] / entry - 1) * 100
        paths.append(p)
    return pd.DataFrame(rows), (np.vstack(paths) if paths else np.empty((0, max_h)))


def dedupe_positions(positions: np.ndarray, min_gap: int) -> np.ndarray:
    kept, last = [], -10**9
    for i in positions:
        if i - last >= min_gap:
            kept.append(i)
            last = i
    return np.array(kept, dtype=int)


def event_study(
    strategy: Strategy | SwingStrategy,
    data: dict[str, pd.DataFrame],
    horizons: tuple[int, ...] = HORIZONS,
    dedupe_bars: int = 10,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    baseline_per_symbol: int = 1500,
    seed: int = 0,
    regime_only: bool = False,
) -> EventStudyResult:
    max_h = max(horizons)
    rng = np.random.default_rng(seed)
    sig_frames, base_frames, sig_paths, base_paths = [], [], [], []
    notes = []
    bench_sym = getattr(strategy, "regime_symbol", None) if regime_only else None
    signal_data = {k: v for k, v in data.items() if k != bench_sym}  # 基准只用来判断大盘状态，不参与信号统计
    if isinstance(strategy, SwingStrategy):
        strategy.prepare(signal_data)  # 横截面策略需要先看全部标的
    for sym, df in signal_data.items():
        if len(df) < max_h + 50:
            continue
        if isinstance(strategy, SwingStrategy):
            ent_s = strategy.compute(sym, df)["entry"].fillna(False).astype(bool)
            if getattr(strategy, "signal_kind", "event") == "state":  # 状态型：只统计“进入状态”那一天
                ent_s = ent_s & ~ent_s.shift(1).fillna(False).astype(bool)
            entry = ent_s.values
            if regime_only and getattr(strategy, "regime_filter", False):
                try:
                    from ..strategies.base import union_index
                    ok = strategy.regime_series(data, df.index).values.astype(bool)
                    entry = entry & ok
                except Exception:
                    pass
        else:  # 单标的策略：权重 0 -> 正 视为入场信号
            w = strategy.target_weights(df).fillna(0.0).values
            entry = (w > 0) & (np.r_[0.0, w[:-1]] == 0)
        idx = df.index
        mask = np.ones(len(df), dtype=bool)
        if start is not None:
            mask &= idx >= pd.Timestamp(start).tz_localize(idx.tz) if idx.tz is not None and pd.Timestamp(start).tzinfo is None else idx >= pd.Timestamp(start)
        if end is not None:
            mask &= idx <= pd.Timestamp(end).tz_localize(idx.tz) if idx.tz is not None and pd.Timestamp(end).tzinfo is None else idx <= pd.Timestamp(end)
        pos = np.where(entry & mask)[0]
        pos = dedupe_positions(pos, dedupe_bars)
        if len(pos):
            f, p = _forward_table(df, pos, horizons, max_h)
            f.insert(0, "symbol", sym)
            sig_frames.append(f)
            sig_paths.append(p)
        all_pos = np.where(mask)[0]
        all_pos = all_pos[all_pos + 1 < len(df)]
        if len(all_pos) > baseline_per_symbol:
            all_pos = np.sort(rng.choice(all_pos, baseline_per_symbol, replace=False))
        if len(all_pos):
            fb, pb = _forward_table(df, all_pos, horizons, max_h)
            fb.insert(0, "symbol", sym)
            base_frames.append(fb)
            base_paths.append(pb)

    signals = pd.concat(sig_frames, ignore_index=True) if sig_frames else pd.DataFrame(columns=["symbol", "date"] + [f"fwd_{h}" for h in horizons] + ["mae_20", "mfe_20"])
    baseline = pd.concat(base_frames, ignore_index=True) if base_frames else signals.iloc[0:0].copy()

    rows = []
    for h in horizons:
        s_ = signals[f"fwd_{h}"].dropna() if len(signals) else pd.Series(dtype=float)
        b_ = baseline[f"fwd_{h}"].dropna() if len(baseline) else pd.Series(dtype=float)
        rows.append({
            "持有bar": h, "信号数": int(len(s_)),
            "信号平均%": s_.mean() if len(s_) else np.nan, "信号中位%": s_.median() if len(s_) else np.nan,
            "信号胜率%": (s_ > 0).mean() * 100 if len(s_) else np.nan,
            "基准平均%": b_.mean() if len(b_) else np.nan, "基准胜率%": (b_ > 0).mean() * 100 if len(b_) else np.nan,
            "超额%": (s_.mean() - b_.mean()) if len(s_) and len(b_) else np.nan,
            "t值": (s_.mean() / s_.std() * np.sqrt(len(s_))) if len(s_) > 2 and s_.std() > 0 else np.nan,
        })
    summary = pd.DataFrame(rows)

    def _mean_path(paths):
        if not paths:
            return pd.Series(np.nan, index=range(1, max_h + 1))
        m = np.vstack(paths)
        return pd.Series(np.nanmean(m, axis=0), index=range(1, max_h + 1))

    path_signal, path_baseline = _mean_path(sig_paths), _mean_path(base_paths)

    ref_h = 20 if 20 in horizons else horizons[len(horizons) // 2]
    col = f"fwd_{ref_h}"
    if len(signals):
        signals["year"] = pd.to_datetime(signals["date"]).dt.year
        by_year = signals.groupby("year").agg(信号数=("symbol", "size"), 平均=(col, "mean"), 胜率=(col, lambda x: (x > 0).mean() * 100)).reset_index()
        by_year.columns = ["年份", "信号数", f"平均{ref_h}bar%", f"胜率{ref_h}bar%"]
        by_symbol = signals.groupby("symbol").agg(信号数=("date", "size"), 平均=(col, "mean"), 胜率=(col, lambda x: (x > 0).mean() * 100),
                                                    mae=("mae_20", "mean"), mfe=("mfe_20", "mean")).reset_index()
        by_symbol.columns = ["标的", "信号数", f"平均{ref_h}bar%", f"胜率{ref_h}bar%", "平均MAE20%", "平均MFE20%"]
        by_symbol = by_symbol.sort_values(f"平均{ref_h}bar%", ascending=False)
    else:
        by_year = pd.DataFrame(columns=["年份", "信号数", f"平均{ref_h}bar%", f"胜率{ref_h}bar%"])
        by_symbol = pd.DataFrame(columns=["标的", "信号数", f"平均{ref_h}bar%", f"胜率{ref_h}bar%", "平均MAE20%", "平均MFE20%"])
        notes.append("这段时间没有信号")
    return EventStudyResult(strategy.describe(), signals, baseline, summary, path_signal, path_baseline, by_year, by_symbol, len(signal_data), notes)
