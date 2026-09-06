"""Swing 策略族：共用一个"槽位式"组合构建器，三个策略只定义入场 / 出场 / 排序分数。

组合构建器（SwingStrategy.target_weights）每根 bar 做三件事：
  1. 检查持仓的出场：策略出场信号、ATR 止损（可移动）、时间止损、财报前退出
  2. 用空余槽位接纳新入场：候选按 score 从高到低，最多 max_positions 个同时持仓
  3. 给持仓分配权重：等权 1/max_positions，或按波动率目标反比
输出的是"该 bar 收盘时希望持有的权重"，引擎会后移一根 bar 在下一开盘成交。

止损价以信号 bar 的收盘价为基准（实际成交在下一开盘，略有差异）；
止损触发用 bar 的最低价，触发后在下一根 bar 开盘卖出。
"""
from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from ..config import ROOT
from ..earnings import EarningsCalendar
from .base import PortfolioStrategy, union_index
from .indicators import (
    atr,
    infer_bars_per_year,
    momentum,
    realized_vol,
    rolling_max_prev,
    rolling_min_prev,
    rsi,
    sma,
)

BASE_HELP = {
    "max_positions": "最多同时持有几只",
    "atr_n": "ATR 周期（bar）",
    "atr_stop_mult": "止损 = 入场价 - N×ATR；0 关闭",
    "trailing_stop": "止损随最高收盘价上移",
    "max_hold_bars": "最长持有 bar 数；0 不限",
    "earnings_blackout_days": "财报前 N 个日历日不新开仓；0 关闭",
    "exit_before_earnings": "财报前 N 天内主动退出",
    "vol_sizing": "按波动率反比定仓位（否则等权）",
    "target_vol": "组合年化波动目标（vol_sizing 时）",
    "vol_lookback": "波动率回看 bar 数",
    "regime_filter": "大盘过滤：基准（SPY）前一日收盘在 regime_sma 日线下方时不开新仓",
    "regime_sma": "大盘过滤用的均线周期",
    "regime_exit": "大盘跌破均线时是否同时清仓（默认只停新仓）",
}


@dataclass
class SwingStrategy(PortfolioStrategy):
    max_positions: int = 5
    atr_n: int = 14
    atr_stop_mult: float = 2.5
    trailing_stop: bool = True
    max_hold_bars: int = 0
    earnings_blackout_days: int = 0
    exit_before_earnings: bool = False
    vol_sizing: bool = False
    target_vol: float = 0.20
    vol_lookback: int = 20
    regime_filter: bool = True
    regime_sma: int = 200
    regime_exit: bool = False
    regime_symbol: str = "SPY"
    name: str = "swing"
    calendar: EarningsCalendar | None = field(default=None, repr=False, compare=False)

    PARAM_HELP: ClassVar[dict[str, str]] = BASE_HELP
    signal_kind: ClassVar[str] = "event"  # event = 入场条件是事件（上穿/突破）；state = 入场条件是状态（如处于动量前 20%）

    # ---- 子类实现：每个标的的 entry / exit / score ----
    @abstractmethod
    def compute(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """返回与 df 同索引的 DataFrame，列 entry(bool) exit(bool) score(float)。"""

    def needs_calendar(self) -> bool:
        return self.earnings_blackout_days > 0 or self.exit_before_earnings

    def get_calendar(self) -> EarningsCalendar:
        if self.calendar is None:
            self.calendar = EarningsCalendar(cache_dir=ROOT / "data_cache")
        return self.calendar

    def describe(self) -> str:
        """只列出与默认值不同的参数，保持简短。"""
        import dataclasses

        parts = []
        for f in dataclasses.fields(self):
            if f.name in {"name", "calendar"} or f.name.startswith("_") or f.default is dataclasses.MISSING:
                continue
            v = getattr(self, f.name)
            if v != f.default:
                parts.append(f"{f.name}={v}")
        return f"{self.name}({', '.join(parts)})" if parts else self.name

    def prepare(self, data: dict[str, pd.DataFrame]) -> None:
        """需要横截面信息的策略在这里预算（见 RelativeMomentum）；默认什么都不做。"""
        return None

    def regime_series(self, data: dict[str, pd.DataFrame], union: pd.DatetimeIndex) -> pd.Series:
        """大盘过滤：基准（默认 SPY）前一日收盘是否在 regime_sma 日均线上方，对齐到 union。拿不到数据时全 True。"""
        if not self.regime_filter:
            return pd.Series(True, index=union)
        bench = data.get(self.regime_symbol)
        if bench is None:
            import os

            if os.environ.get("TRADEBOT_OFFLINE"):  # 测试 / 离线：不联网拉基准，视为过滤关闭
                return pd.Series(True, index=union)
            try:
                from ..data import fetch_bars

                days = int((union[-1] - union[0]).days) + int(self.regime_sma * 1.6) + 30
                bench = fetch_bars([self.regime_symbol], "1d", days, cache_dir=ROOT / "data_cache", max_cache_age_hours=12)[self.regime_symbol]
            except Exception:
                return pd.Series(True, index=union)
        c = bench["close"]
        ok = (c > sma(c, self.regime_sma)).shift(1).fillna(True)  # 用前一日收盘，日内 bar 也不会偷看当天收盘
        ok.index = pd.DatetimeIndex(ok.index).normalize()
        ok = ok[~ok.index.duplicated(keep="last")]
        aligned = ok.reindex(union.normalize(), method="ffill").fillna(True)
        aligned.index = union
        return aligned.astype(bool)

    # ---- 组合构建 ----
    def target_weights(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        self.prepare(data)
        symbols = list(data)
        union = union_index(data)
        n, m = len(union), len(symbols)
        bpy = infer_bars_per_year(union)
        regime_ok = self.regime_series(data, union).values.astype(bool)

        close = np.full((n, m), np.nan)
        low = np.full((n, m), np.nan)
        atr_a = np.full((n, m), np.nan)
        vol_a = np.full((n, m), np.nan)
        score = np.full((n, m), np.nan)
        entry = np.zeros((n, m), dtype=bool)
        exit_ = np.zeros((n, m), dtype=bool)
        blackout = np.zeros((n, m), dtype=bool)
        cal = self.get_calendar() if self.needs_calendar() else None

        for j, sym in enumerate(symbols):
            df = data[sym]
            sig = self.compute(sym, df)
            close[:, j] = df["close"].reindex(union).values
            low[:, j] = df["low"].reindex(union).values
            atr_a[:, j] = atr(df, self.atr_n).reindex(union).values
            if self.vol_sizing:
                vol_a[:, j] = realized_vol(df["close"], self.vol_lookback, bpy).reindex(union).values
            score[:, j] = sig["score"].reindex(union).values
            entry[:, j] = sig["entry"].reindex(union).fillna(False).astype(bool).values
            exit_[:, j] = sig["exit"].reindex(union).fillna(False).astype(bool).values
            if cal is not None:
                blackout[:, j] = cal.blackout_mask(sym, union, max(self.earnings_blackout_days, 1))

        W = np.zeros((n, m))
        held: dict[int, dict] = {}
        mult = self.atr_stop_mult
        for i in range(n):
            # 1. 出场
            for j in list(held):
                c = close[i, j]
                if math.isnan(c):
                    continue
                p = held[j]
                p["bars"] += 1  # 只在该标的自己有 bar 的日子计数，避免别的标的（如韩股）的交易日把持有天数数多
                a = atr_a[i, j]
                if self.trailing_stop and mult > 0 and not math.isnan(a):
                    p["high"] = max(p["high"], c)
                    p["stop"] = max(p["stop"], p["high"] - mult * a) if not math.isnan(p["stop"]) else p["high"] - mult * a
                hit_stop = mult > 0 and not math.isnan(p["stop"]) and low[i, j] <= p["stop"]
                timed_out = self.max_hold_bars > 0 and p["bars"] >= self.max_hold_bars
                earn_exit = self.exit_before_earnings and blackout[i, j]
                regime_out = self.regime_exit and not regime_ok[i]
                if exit_[i, j] or hit_stop or timed_out or earn_exit or regime_out:
                    del held[j]
            # 2. 入场（大盘过滤关着时不开新仓）
            free = self.max_positions - len(held)
            if free > 0 and regime_ok[i]:
                cands = [
                    j for j in range(m)
                    if j not in held and entry[i, j] and not math.isnan(close[i, j])
                    and not (self.earnings_blackout_days > 0 and blackout[i, j])
                ]
                cands.sort(key=lambda j: -(score[i, j] if not math.isnan(score[i, j]) else -math.inf))
                for j in cands[:free]:
                    c, a = close[i, j], atr_a[i, j]
                    stop = c - mult * a if (mult > 0 and not math.isnan(a)) else math.nan
                    w = 1.0 / self.max_positions
                    if self.vol_sizing:
                        v = vol_a[i, j]
                        if not math.isnan(v) and v > 0:
                            w = min((self.target_vol / self.max_positions) / v, 2.0 / self.max_positions, 1.0)
                    held[j] = {"entry_i": i, "stop": stop, "high": c, "w": w, "bars": 0}
            # 3. 权重
            for j, p in held.items():
                W[i, j] = p["w"]
            g = W[i].sum()
            if g > 1.0:
                W[i] /= g
        return pd.DataFrame(W, index=union, columns=symbols)


# =====================================================================
@dataclass
class TrendPullback(SwingStrategy):
    """趋势回调：长期上升趋势里，RSI 回落到阈值以下再重新向上穿越时入场。

    entry: close > SMA(trend_slow) 且 SMA(trend_fast) > SMA(trend_slow) 且 RSI 从 rsi_entry 下方上穿
    exit : RSI > rsi_exit（回调结束、涨势兑现）或 close 跌破 SMA(trend_slow)（趋势坏了）
    score: 过去 score_lookback 根 bar 的动量，趋势最强者优先
    """

    trend_fast: int = 50
    trend_slow: int = 200
    rsi_n: int = 14
    rsi_entry: float = 40.0
    rsi_exit: float = 65.0
    confirm_turn: bool = True
    score_lookback: int = 126
    name: str = "trend_pullback"

    PARAM_HELP: ClassVar[dict[str, str]] = {
        **BASE_HELP,
        "trend_fast": "快均线周期", "trend_slow": "慢均线周期（趋势过滤）",
        "rsi_n": "RSI 周期", "rsi_entry": "RSI 低于此值视为回调", "rsi_exit": "RSI 高于此值退出",
        "confirm_turn": "要求 RSI 重新上穿阈值再入场（否则回落即入）", "score_lookback": "排序用动量回看 bar 数",
    }

    def compute(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        fast, slow = sma(c, self.trend_fast), sma(c, self.trend_slow)
        r = rsi(c, self.rsi_n)
        uptrend = (c > slow) & (fast > slow)
        if self.confirm_turn:
            pullback_done = (r.shift(1) < self.rsi_entry) & (r >= self.rsi_entry)
        else:
            pullback_done = r < self.rsi_entry
        entry = (uptrend & pullback_done).fillna(False)
        exit_ = ((r > self.rsi_exit) | (c < slow)).fillna(False)
        return pd.DataFrame({"entry": entry, "exit": exit_, "score": momentum(c, self.score_lookback)}, index=df.index)

    def warmup_bars(self) -> int:
        return max(self.trend_slow, self.score_lookback) + 20


@dataclass
class Breakout(SwingStrategy):
    """突破：收盘创出过去 lookback 根 bar 新高（不含当前）入场，跌破 exit_lookback 新低出场。

    可选趋势过滤（trend_n>0：close > SMA(trend_n)）和成交量确认（vol_mult>0：volume > vol_mult × 20 bar 均量）。
    score: 过去 score_lookback 根 bar 的动量（相对强度），强者优先。
    """

    lookback: int = 55
    exit_lookback: int = 20
    trend_n: int = 100
    vol_mult: float = 1.2
    score_lookback: int = 126
    name: str = "breakout"

    PARAM_HELP: ClassVar[dict[str, str]] = {
        **BASE_HELP,
        "lookback": "突破回看 bar 数（唐奇安上轨）", "exit_lookback": "出场回看 bar 数（下轨）",
        "trend_n": "趋势过滤均线周期；0 关闭", "vol_mult": "成交量须 > N×20bar 均量；0 关闭",
        "score_lookback": "排序用动量回看 bar 数",
    }

    def compute(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        entry = c > rolling_max_prev(c, self.lookback)
        if self.trend_n > 0:
            entry &= c > sma(c, self.trend_n)
        if self.vol_mult > 0:
            entry &= df["volume"] > self.vol_mult * sma(df["volume"], 20)
        exit_ = c < rolling_min_prev(c, self.exit_lookback)
        return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False), "score": momentum(c, self.score_lookback)}, index=df.index)

    def warmup_bars(self) -> int:
        return max(self.lookback, self.exit_lookback, self.trend_n, self.score_lookback) + 20


@dataclass
class PostEarningsMomentum(SwingStrategy):
    """财报后动量（PEAD）：财报反应日收盘涨幅 >= min_reaction（可要求 EPS 超预期）时入场，持有 max_hold_bars 后退出。

    反应日 = 公布时间之后的第一个交易日收盘（盘后公布算下一交易日）。
    反应涨幅 = 反应日收盘 / 前一交易日收盘 - 1。score = 反应涨幅。
    没有策略出场信号，靠时间止损和 ATR 止损。
    """

    min_reaction: float = 0.03
    require_beat: bool = True
    max_hold_bars: int = 20
    name: str = "post_earnings"

    PARAM_HELP: ClassVar[dict[str, str]] = {
        **BASE_HELP,
        "min_reaction": "反应日涨幅阈值，0.03 = 3%",
        "require_beat": "要求 EPS 超预期（没有数据时不阻止）",
    }

    def compute(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        cal = self.get_calendar()
        entry = pd.Series(False, index=df.index)
        score = pd.Series(np.nan, index=df.index)
        days = df.index.normalize()
        last_pos_of_day = pd.Series(np.arange(len(df)), index=days).groupby(level=0).last()  # 每个交易日最后一根 bar 的位置
        day_list = last_pos_of_day.index
        rd = cal.reaction_days(symbol, day_list)
        closes = df["close"].values
        for _, row in rd.iterrows():
            d = row["reaction_day"]
            k = day_list.get_indexer([d])[0]
            if k <= 0:
                continue
            i, i_prev = int(last_pos_of_day.iloc[k]), int(last_pos_of_day.iloc[k - 1])
            ret = closes[i] / closes[i_prev] - 1
            beat_ok = (not self.require_beat) or pd.isna(row["surprise_pct"]) or row["surprise_pct"] > 0
            if ret >= self.min_reaction and beat_ok:
                entry.iloc[i] = True
                score.iloc[i] = ret
        return pd.DataFrame({"entry": entry, "exit": False, "score": score}, index=df.index)

    def warmup_bars(self) -> int:
        return self.max_hold_bars + self.atr_n + 10
