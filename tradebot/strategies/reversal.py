"""抄底类策略：超跌反弹（均值回归）与底部回升（结构反转）。都是 SwingStrategy，共用槽位、止损、排序。

全部因果：只用截至当前 bar 的数据。摆动低点在其后 swing_k 根 bar 才被确认，信号会相应滞后。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from .indicators import atr, momentum, rolling_max_prev, rsi, sma
from .swing import BASE_HELP, SwingStrategy


@dataclass
class OversoldBounce(SwingStrategy):
    """超跌反弹：先超卖、再止跌确认才进，回到均值就走。

    超卖 = RSI(rsi_n) < rsi_max 或 收盘 < 布林下轨(bb_n, bb_k)，且收盘比过去 20 根 bar 最高价低 drop_from_high 以上
    entry = 前一根 bar 处于超卖 且 当前收盘 > 前收（止跌确认，可关）
    exit  = 收盘回到 bb_n 日均线上方 或 RSI > exit_rsi；另有 max_hold_bars 时间止损与 ATR 止损
    score = (均线 - 收盘) / ATR，偏离越大越优先
    """

    rsi_n: int = 14
    rsi_max: float = 30.0
    bb_n: int = 20
    bb_k: float = 2.0
    drop_from_high: float = 0.08
    confirm_up_bar: bool = True
    require_above_sma200: bool = False
    exit_rsi: float = 55.0
    exit_to_mean: bool = True
    max_hold_bars: int = 10
    atr_stop_mult: float = 2.0
    trailing_stop: bool = False
    name: str = "oversold_bounce"

    PARAM_HELP: ClassVar[dict[str, str]] = {
        **BASE_HELP,
        "rsi_n": "RSI 周期", "rsi_max": "RSI 低于此值算超卖", "bb_n": "布林/均线周期", "bb_k": "布林带宽（标准差倍数）",
        "drop_from_high": "相对 20 根 bar 最高价至少跌这么多（0.08 = 8%）", "confirm_up_bar": "要求当日收盘 > 前收再进（止跌确认）",
        "require_above_sma200": "只在 200 日线上方抄底（不接长期空头的刀）", "exit_rsi": "RSI 高于此值离场", "exit_to_mean": "收盘回到均线上方即离场",
    }

    def compute(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        c, h = df["close"], df["high"]
        r = rsi(c, self.rsi_n)
        ma = sma(c, self.bb_n)
        sd = c.rolling(self.bb_n).std()
        lower = ma - self.bb_k * sd
        hi20 = rolling_max_prev(h, 20)
        oversold = ((r < self.rsi_max) | (c < lower)) & (c <= hi20 * (1 - self.drop_from_high))
        entry = (oversold.shift(1).fillna(False) & (c > c.shift(1))) if self.confirm_up_bar else oversold
        if self.require_above_sma200:
            entry &= c > sma(c, 200)
        exit_ = (r > self.exit_rsi)
        if self.exit_to_mean:
            exit_ |= c > ma
        a = atr(df, self.atr_n)
        score = (ma - c) / a.replace(0, np.nan)
        return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False), "score": score}, index=df.index)

    def warmup_bars(self) -> int:
        return (200 if self.require_above_sma200 else max(self.bb_n, self.rsi_n)) + 25


@dataclass
class BottomRecovery(SwingStrategy):
    """底部回升：跌深 → 低点抬高 → 突破反弹高点 → 收复均线，四步齐了才进。

    摆动低点：左右各 swing_k 根 bar 内最低，且只在 k 根 bar 之后确认（因果）。
    entry = 最近低点相对之前 lookback 内最高价跌幅 ≥ drawdown_min
            且（可选）最近低点 > 前一个低点
            且 收盘 > 过去 breakout_n 根 bar 最高价（不含当前）
            且 收盘 > SMA(sma_n) 且 SMA 比 5 根 bar 前高
            且（可选）成交量 > vol_mult × 20 bar 均量
    exit  = 收盘 < 最近低点（结构失败） 或 收盘 < SMA - 1×ATR（明显跌回均线下）
    score = 20 根 bar 动量，反弹越强越优先
    """

    swing_k: int = 5
    lookback: int = 120
    drawdown_min: float = 0.12
    require_higher_low: bool = True
    breakout_n: int = 10
    sma_n: int = 20
    vol_mult: float = 0.0
    atr_stop_mult: float = 2.5
    name: str = "bottom_recovery"

    PARAM_HELP: ClassVar[dict[str, str]] = {
        **BASE_HELP,
        "swing_k": "摆动低点左右窗口（bar）", "lookback": "找前高的回看 bar 数", "drawdown_min": "最近低点相对前高的最小跌幅（0.12 = 12%）",
        "require_higher_low": "要求低点抬高（W 底）；关掉允许 V 底", "breakout_n": "收盘需突破过去 N 根 bar 最高价", "sma_n": "收复的均线周期",
        "vol_mult": "成交量须 > N×20bar 均量；0 关闭",
    }

    def compute(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        c, h, lo, v = df["close"], df["high"], df["low"], df["volume"]
        k = self.swing_k
        n = len(df)
        pos = pd.Series(np.arange(n), index=df.index, dtype=float)
        # 摆动低点：比之前 k 根都低、不高于之后 k 根（平价取第一个）；用到未来 k 根，再整体后移 k 根变成因果的“已确认”标记
        prev_min = lo.shift(1).rolling(k).min()
        next_min = lo[::-1].shift(1).rolling(k).min()[::-1]
        is_swing = (lo < prev_min) & (lo <= next_min)
        confirmed = is_swing.shift(k).fillna(False).astype(bool)          # 在 t 确认 t-k 是低点
        swing_price = lo.shift(k).where(confirmed)
        swing_pos = pos.shift(k).where(confirmed)
        prior_high = h.rolling(self.lookback, min_periods=20).max().shift(k).where(confirmed)  # 低点之前 lookback 内最高价
        last_low = swing_price.ffill()
        last_low_pos = swing_pos.ffill()
        prev_low = swing_price.dropna().shift(1).reindex(df.index).ffill()
        prior_high_at_low = prior_high.ffill()

        drawdown = 1 - last_low / prior_high_at_low
        has_decline = drawdown >= self.drawdown_min
        higher_low = last_low > prev_low
        breakout = c > rolling_max_prev(h, self.breakout_n)
        ma = sma(c, self.sma_n)
        reclaimed = (c > ma) & (ma > ma.shift(5))
        bars_since_low = pos - last_low_pos
        entry = has_decline & breakout & reclaimed & (bars_since_low >= 1) & (c > last_low)
        if self.require_higher_low:
            entry &= higher_low
        if self.vol_mult > 0:
            entry &= v > self.vol_mult * sma(v, 20)
        a = atr(df, self.atr_n)
        exit_ = (c < last_low) | (c < ma - a)
        return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False), "score": momentum(c, 20)}, index=df.index)

    def warmup_bars(self) -> int:
        return self.lookback + self.swing_k + 30
