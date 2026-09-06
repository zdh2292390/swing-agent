"""行情数据层：yfinance 日线 / 小时线 + parquet 缓存 + 未完成 bar 处理。

- 统一符号（BRK.B）在这里转成 yfinance 符号（BRK-B）下载，返回时再转回来。
- 索引统一为纽约时区的 bar 开始时间。日线是当天 00:00，小时线是 9:30, 10:30, ..., 15:30。
- yfinance 小时线最多回溯 729 天；日线可以拉到上市以来。
- 实盘时最后一根 bar 往往还没走完（10:31 拉到的 10:30 bar 只有一分钟数据），
  drop_incomplete_last_bar 会把它去掉，保证信号和回测一样只用完整 bar。
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .universe import from_yahoo, to_yahoo

COLUMNS = ["open", "high", "low", "close", "volume"]
NY_TZ = "America/New_York"
INTERVAL_INFO = {
    "1h": {"max_days": 729, "bars_per_day": 7},
    "1d": {"max_days": None, "bars_per_day": 1},
}


def bars_per_day(interval: str) -> int:
    return INTERVAL_INFO[interval]["bars_per_day"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    df = df[[c for c in COLUMNS if c in df.columns]].dropna(how="any")
    idx = pd.DatetimeIndex(df.index)
    # 日线是无时区的日期，直接按纽约本地时间标记；小时线带时区，转到纽约
    idx = idx.tz_localize(NY_TZ) if idx.tz is None else idx.tz_convert(NY_TZ)
    df.index = idx
    df.index.name = "time"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.astype(float)


def _download(symbols: list[str], interval: str, lookback_days: int) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    yahoo = [to_yahoo(s) for s in symbols]
    end = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=lookback_days + 1)
    raw = yf.download(
        tickers=yahoo,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance 没有返回数据: {symbols} interval={interval}")

    out: dict[str, pd.DataFrame] = {}
    for sym, ysym in zip(symbols, yahoo):
        if isinstance(raw.columns, pd.MultiIndex):
            if ysym in raw.columns.get_level_values(0):
                part = raw[ysym]
            elif ysym in raw.columns.get_level_values(1):
                part = raw.xs(ysym, axis=1, level=1)
            else:
                raise RuntimeError(f"yfinance 返回里找不到 {sym} ({ysym})")
        else:
            part = raw
        df = _normalize(part)
        if df.empty:
            raise RuntimeError(f"{sym} 数据为空")
        out[from_yahoo(ysym)] = df
    return out


def fetch_bars(
    symbols: list[str],
    interval: str = "1d",
    lookback_days: int | None = None,
    cache_dir: Path | None = None,
    max_cache_age_hours: float | None = 12.0,
) -> dict[str, pd.DataFrame]:
    """返回 {统一符号: OHLCV DataFrame}。

    cache_dir 为 None 不缓存（实盘用）；否则命中且未过期的缓存直接读取。
    max_cache_age_hours=0 强制重新下载。
    """
    if interval not in INTERVAL_INFO:
        raise ValueError(f"interval={interval!r} 不支持，可选 {list(INTERVAL_INFO)}")
    info = INTERVAL_INFO[interval]
    if lookback_days is None:
        lookback_days = info["max_days"] or 5000
    if info["max_days"]:
        lookback_days = min(int(lookback_days), info["max_days"])
    symbols = [s.upper() for s in symbols]

    if cache_dir is None:
        return _download(symbols, interval, lookback_days)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for sym in symbols:
        path = cache_dir / f"{sym.replace('.', '_')}_{interval}_{lookback_days}d.parquet"
        fresh = (
            path.exists()
            and max_cache_age_hours is not None
            and (time.time() - path.stat().st_mtime) < max_cache_age_hours * 3600
        )
        if fresh:
            result[sym] = pd.read_parquet(path)
        else:
            missing.append(sym)

    if missing:
        for sym, df in _download(missing, interval, lookback_days).items():
            df.to_parquet(cache_dir / f"{sym.replace('.', '_')}_{interval}_{lookback_days}d.parquet")
            result[sym] = df
    return {sym: result[sym] for sym in symbols}


def fetch_hourly(symbols, period_days: int = 729, cache_dir=None, max_cache_age_hours: float | None = 12.0):
    """兼容旧接口。"""
    return fetch_bars(symbols, "1h", period_days, cache_dir, max_cache_age_hours)


def drop_incomplete_last_bar(df: pd.DataFrame, interval: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """如果最后一根 bar 还没走完就去掉它。

    日线：当天 16:00 之后才算完整。小时线：bar 开始 + 1 小时，或当天 16:00，取早者。
    """
    if df.empty:
        return df
    now = pd.Timestamp.now(tz=NY_TZ) if now is None else pd.Timestamp(now).tz_convert(NY_TZ)
    last = df.index[-1]
    session_close = last.normalize() + pd.Timedelta(hours=16)
    if interval == "1d":
        bar_end = session_close
    elif interval == "1h":
        bar_end = min(last + pd.Timedelta(hours=1), session_close)
    else:
        raise ValueError(interval)
    return df if now >= bar_end else df.iloc[:-1]


def latest_prices(data: dict[str, pd.DataFrame]) -> dict[str, float]:
    return {sym: float(df["close"].iloc[-1]) for sym, df in data.items()}


def align(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """多标的对齐到共同时间戳（内连接）。上市晚的标的会把整体起点拉后，回测前留意。"""
    common = None
    for df in data.values():
        common = df.index if common is None else common.intersection(df.index)
    return {sym: df.loc[common] for sym, df in data.items()}
