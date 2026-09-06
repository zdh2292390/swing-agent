"""常用指标。全部只用截至当前 bar 的数据，rolling/ewm 天然满足。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(avg_gain.notna(), np.nan)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR。"""
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rolling_max_prev(s: pd.Series, n: int) -> pd.Series:
    """前 n 根 bar（不含当前）的最高值，用于突破判断。"""
    return s.shift(1).rolling(n).max()


def rolling_min_prev(s: pd.Series, n: int) -> pd.Series:
    return s.shift(1).rolling(n).min()


def momentum(close: pd.Series, n: int) -> pd.Series:
    return close / close.shift(n) - 1


def infer_bars_per_year(index: pd.DatetimeIndex) -> float:
    """由 bar 间距推断：小时线约 7*252，日线 252。"""
    if len(index) < 3:
        return 252.0
    step = pd.Series(index[1:] - index[:-1]).median()
    return 7 * 252.0 if step <= pd.Timedelta(hours=2) else 252.0


def realized_vol(close: pd.Series, n: int, bars_per_year: float) -> pd.Series:
    return close.pct_change().rolling(n).std() * math.sqrt(bars_per_year)


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Wilder ADX。返回 DataFrame[adx, di_plus, di_minus]。"""
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = true_range(df)
    atr_w = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    di_plus = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_w
    di_minus = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_w
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    out = dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return pd.DataFrame({"adx": out, "di_plus": di_plus, "di_minus": di_minus}, index=df.index)
