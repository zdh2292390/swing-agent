"""上涨信号里证据最强的两类：横截面相对强度动量、52 周新高突破。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from ..universe import is_us_listed
from .indicators import momentum, rolling_max_prev, sma
from .swing import BASE_HELP, SwingStrategy


@dataclass
class RelativeMomentum(SwingStrategy):
    """相对强度动量（横截面）：过去 12 个月剔除最近 1 个月的涨幅，在标的池里排名靠前的才买。

    动量 = close[t-skip] / close[t-lookback] - 1（默认 12-1：lookback=252，skip=21）
    entry = 动量在当日全部有数据的标的里排前 top_frac，且动量 > 0，且收盘 > SMA(trend_sma)
    exit  = 排名跌出前 exit_frac（滞后带，减少来回换手）或 收盘 < SMA(trend_sma)
    score = 动量本身
    需要看全部标的，所以排名在 prepare() 里预算；单独对一只股票调用 compute 时退化为“动量 > 0 且在均线上”。
    """

    lookback: int = 252
    skip: int = 21
    top_frac: float = 0.2
    exit_frac: float = 0.4
    trend_sma: int = 200
    atr_stop_mult: float = 3.0
    name: str = "relative_momentum"
    _cs: dict = field(default_factory=dict, repr=False, compare=False)

    PARAM_HELP: ClassVar[dict[str, str]] = {
        **BASE_HELP,
        "lookback": "动量回看 bar 数（252 ≈ 12 个月）", "skip": "剔除最近多少根 bar（21 ≈ 1 个月，避开短期反转）",
        "top_frac": "动量排名进入前多少比例才入场（0.2 = 前 20%）", "exit_frac": "排名跌出前多少比例才出场（滞后带）",
        "trend_sma": "个股趋势过滤均线",
    }
    signal_kind: ClassVar[str] = "state"

    def _mom(self, df: pd.DataFrame) -> pd.Series:
        c = df["close"]
        return c.shift(self.skip) / c.shift(self.lookback) - 1

    def prepare(self, data: dict[str, pd.DataFrame]) -> None:
        # 只在可交易的美股里排名；海外“仅监控”标的不占名额，也不给信号
        pool = {sym: df for sym, df in data.items() if is_us_listed(sym)}
        self._cs = {}
        for sym, df in data.items():
            if sym not in pool:
                self._cs[sym] = pd.DataFrame({"entry": False, "exit": False, "score": np.nan}, index=df.index)
        if not pool:
            return
        moms = pd.DataFrame({sym: self._mom(df) for sym, df in pool.items()})
        n_avail = moms.notna().sum(axis=1)
        rank = moms.rank(axis=1, ascending=False)  # 1 = 最强
        top_n = np.ceil(n_avail * self.top_frac).clip(lower=1)
        exit_n = np.ceil(n_avail * self.exit_frac).clip(lower=1)
        for sym, df in pool.items():
            c = df["close"]
            m = moms[sym].reindex(df.index)
            r = rank[sym].reindex(df.index)
            tn = top_n.reindex(df.index)
            en = exit_n.reindex(df.index)
            trend = c > sma(c, self.trend_sma)
            entry = (r <= tn) & (m > 0) & trend
            exit_ = (r > en) | (~trend)
            self._cs[sym] = pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False), "score": m}, index=df.index)

    def compute(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        cached = self._cs.get(symbol)
        if cached is not None and len(cached) == len(df) and cached.index.equals(df.index):
            return cached
        # 退化版（没有横截面信息）：绝对动量 > 0 且在趋势均线上
        c = df["close"]
        m = self._mom(df)
        trend = c > sma(c, self.trend_sma)
        return pd.DataFrame({"entry": ((m > 0) & trend).fillna(False), "exit": (~trend).fillna(False), "score": m}, index=df.index)

    def warmup_bars(self) -> int:
        return max(self.lookback, self.trend_sma) + 30


@dataclass
class High52Breakout(SwingStrategy):
    """52 周新高突破：收盘创过去 high_n 根 bar 新高（不含当天），之前有一段窄幅缩量整理，突破当天放量。

    entry = 收盘 > 过去 high_n bar 最高价
            且 之前 base_n bar 的（最高 − 最低）/ 收盘 ≤ base_range_max（整理够窄）
            且（可选）整理期成交量萎缩：最近 20 bar 均量 < 之前 60 bar 均量
            且 突破日成交量 > vol_mult × 20 bar 均量
    exit  = 收盘 < SMA(exit_sma)
    score = 过去 126 bar 动量（相对强度）
    """

    high_n: int = 252
    base_n: int = 40
    base_range_max: float = 0.25
    dryup: bool = True
    vol_mult: float = 1.5
    exit_sma: int = 50
    atr_stop_mult: float = 3.0
    name: str = "high52_breakout"

    PARAM_HELP: ClassVar[dict[str, str]] = {
        **BASE_HELP,
        "high_n": "新高回看 bar 数（252 ≈ 52 周）", "base_n": "突破前整理期长度（bar）", "base_range_max": "整理期振幅上限（0.25 = 25%）",
        "dryup": "要求整理期成交量萎缩", "vol_mult": "突破日成交量 / 20 bar 均量 至少", "exit_sma": "跌破哪条均线出场",
    }

    def compute(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        c, h, lo, v = df["close"], df["high"], df["low"], df["volume"]
        new_high = c > rolling_max_prev(h, self.high_n)
        base_hi = h.shift(1).rolling(self.base_n).max()
        base_lo = lo.shift(1).rolling(self.base_n).min()
        tight = (base_hi - base_lo) / c <= self.base_range_max
        vol20 = sma(v, 20)
        entry = new_high & tight & (v > self.vol_mult * vol20.shift(1))
        if self.dryup:
            entry &= vol20.shift(1) < sma(v, 60).shift(1)
        exit_ = c < sma(c, self.exit_sma)
        return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False), "score": momentum(c, 126)}, index=df.index)

    def warmup_bars(self) -> int:
        return self.high_n + 30
