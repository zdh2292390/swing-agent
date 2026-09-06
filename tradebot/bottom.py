"""底部确认筛选：用可复算的规则判断一只股票是否走完"下跌 → 企稳 → 确认反转"的结构。

规则（日线，默认参数）：
  1. 前期下跌   最近 lookback 根 bar 内，最近摆动低点相对之前最高价的跌幅 ≥ drawdown_min
  2. 低点抬高   最近一个摆动低点高于前一个摆动低点（摆动低点 = 左右各 swing_k 根 bar 内的最低点）
  3. 突破颈线   收盘高于两个低点之间的最高价（颈线）；只有一个低点时用低点后 10 根 bar 的最高价
  4. 收复均线   收盘 > SMA20 且 SMA20 比 5 根 bar 前高（拐头）
  5. 量能确认   突破日成交量 ≥ vol_mult × 20 日均量；还没突破就看低点之后最大的阳线量
  6. RSI       低点处 RSI 高于前低处 RSI 而价格不高于前低 1.02 倍（底背离），或当前 RSI > 50
阶段：无明显下跌 / 未见底 / 初步企稳 / 确认 / 确认后回踩 / 跌破低点
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategies.indicators import rsi, sma


@dataclass
class BottomParams:
    lookback: int = 120
    swing_k: int = 5
    drawdown_min: float = 0.12
    vol_mult: float = 1.3
    sma_n: int = 20
    pullback_tol: float = 0.03


def swing_lows(low: pd.Series, k: int) -> list[int]:
    """左右各 k 根 bar 内的严格最低点的位置。最后 k 根 bar 无法确认。"""
    v = low.values
    out = []
    for i in range(k, len(v) - k):
        left, right = v[i - k:i], v[i + 1:i + k + 1]
        if v[i] < left.min() and v[i] <= right.min():  # 左边严格更高，右边允许平（取第一个）
            out.append(i)
    return out


def analyze_bottom(df: pd.DataFrame, p: BottomParams = BottomParams()) -> dict:
    w = df.tail(p.lookback).copy()
    if len(w) < max(40, p.swing_k * 4):
        return {"阶段": "数据不足", "得分": 0}
    close, low, high, vol = w["close"], w["low"], w["high"], w["volume"]
    r = rsi(df["close"], 14).reindex(w.index)
    ma = sma(df["close"], p.sma_n).reindex(w.index)
    vol_avg = df["volume"].rolling(20).mean().reindex(w.index)

    lows = swing_lows(low, p.swing_k)
    if not lows:
        return {"阶段": "未见底", "得分": 0, "说明": "最近没有形成可确认的摆动低点"}
    L1 = lows[-1]
    L0 = lows[-2] if len(lows) >= 2 else None

    prior_high = float(high.iloc[: L1 + 1].max())
    low1 = float(low.iloc[L1])
    drawdown = 1 - low1 / prior_high if prior_high > 0 else 0.0
    has_decline = drawdown >= p.drawdown_min

    higher_low = L0 is not None and low1 > float(low.iloc[L0])
    if L0 is not None:
        neckline = float(high.iloc[L0:L1 + 1].max())
    else:
        seg = high.iloc[L1 + 1:L1 + 11]
        neckline = float(seg.max()) if len(seg) >= 3 else float("nan")

    px = float(close.iloc[-1])
    after = close.iloc[L1 + 1:]
    breakout_now = bool(not np.isnan(neckline) and px > neckline)
    breakout_any = bool(not np.isnan(neckline) and (after > neckline).any())
    breakout_i = int(np.argmax((after > neckline).values)) + L1 + 1 if breakout_any else None

    ma_now, ma_prev = float(ma.iloc[-1]), float(ma.iloc[-6]) if len(ma) >= 6 else float("nan")
    reclaimed = bool(px > ma_now and ma_now > ma_prev) if not np.isnan(ma_now) and not np.isnan(ma_prev) else False

    if breakout_i is not None and not np.isnan(vol_avg.iloc[breakout_i]) and vol_avg.iloc[breakout_i] > 0:
        vol_ratio = float(vol.iloc[breakout_i] / vol_avg.iloc[breakout_i])
    else:
        seg = w.iloc[L1 + 1:]
        ups = seg[seg["close"] > seg["open"]]
        if len(ups):
            j = ups["volume"].idxmax()
            vol_ratio = float(vol.loc[j] / vol_avg.loc[j]) if vol_avg.loc[j] and not np.isnan(vol_avg.loc[j]) else float("nan")
        else:
            vol_ratio = float("nan")
    vol_ok = bool(not np.isnan(vol_ratio) and vol_ratio >= p.vol_mult)

    rsi_div = bool(L0 is not None and low1 <= float(low.iloc[L0]) * 1.02 and float(r.iloc[L1]) > float(r.iloc[L0]) + 2)
    rsi_now = float(r.iloc[-1])
    rsi_ok = bool(rsi_div or rsi_now > 50)

    bars_since_low = len(w) - 1 - L1
    rebound = px / low1 - 1
    to_neck = px / neckline - 1 if not np.isnan(neckline) else float("nan")

    if not has_decline:
        stage = "无明显下跌"
    elif px < low1:
        stage = "跌破低点"
    elif breakout_now and reclaimed:
        stage = "确认" if (higher_low or L0 is None) else "确认（未抬高）"
    elif breakout_any and not breakout_now and px > ma_now * (1 - p.pullback_tol):
        stage = "确认后回踩"
    elif higher_low or reclaimed:
        stage = "初步企稳"
    else:
        stage = "未见底"

    score = int(has_decline) + int(higher_low) + int(breakout_now) + int(reclaimed) + int(vol_ok) + int(rsi_ok)
    return {
        "阶段": stage, "得分": score,
        "跌幅%": round(drawdown * 100, 1), "前高": round(prior_high, 2),
        "低点日": w.index[L1].strftime("%m-%d"), "低点价": round(low1, 2),
        "前低价": round(float(low.iloc[L0]), 2) if L0 is not None else None,
        "低点抬高": "✓" if higher_low else ("—" if L0 is None else "✗"),
        "颈线": round(neckline, 2) if not np.isnan(neckline) else None,
        "突破颈线": "✓" if breakout_now else ("回踩" if breakout_any else "✗"),
        "收复SMA20": "✓" if reclaimed else "✗",
        "量能": "✓" if vol_ok else "✗", "量比": round(vol_ratio, 2) if not np.isnan(vol_ratio) else None,
        "RSI背离": "✓" if rsi_div else "✗", "RSI": round(rsi_now, 0),
        "距低点%": round(rebound * 100, 1), "距颈线%": round(to_neck * 100, 1) if not np.isnan(to_neck) else None,
        "低点距今": bars_since_low,
        "_L0": int(L0) if L0 is not None else None, "_L1": int(L1), "_neckline": neckline, "_start": len(df) - len(w),
    }


def build_bottom_table(symbols: list[str], data: dict[str, pd.DataFrame], p: BottomParams = BottomParams(),
                       groups: dict[str, list[str]] | None = None) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        df = data.get(sym)
        if df is None or len(df) < 40:
            continue
        res = analyze_bottom(df, p)
        rows.append({"标的": sym, "板块": "/".join(g for g, m in (groups or {}).items() if sym in m), "现价": round(float(df["close"].iloc[-1]), 2), **res})
    out = pd.DataFrame(rows)
    if len(out):
        order = {"确认": 0, "确认后回踩": 1, "确认（未抬高）": 2, "初步企稳": 3, "未见底": 4, "跌破低点": 5, "无明显下跌": 6, "数据不足": 7}
        out["_o"] = out["阶段"].map(order).fillna(9)
        out = out.sort_values(["_o", "得分"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)
    return out
