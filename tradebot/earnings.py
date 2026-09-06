"""财报日历：yfinance 获取，parquet 缓存，支持注入测试数据。

反应日规则：公布时间戳 ts，反应日 = 第一个满足 (交易日 16:00 > ts) 的交易日。
  16:00 盘后公布 -> 下一交易日；08:00 盘前公布 -> 当天。
  yfinance 绝大多数记录是 16:00，少数时间未知记为 00:00 会被当成盘前，属于数据噪音。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .universe import to_yahoo

log = logging.getLogger("tradebot.earnings")
NY_TZ = "America/New_York"
COLUMNS = ["eps_estimate", "eps_actual", "surprise_pct"]


class EarningsCalendar:
    def __init__(
        self,
        cache_dir: Path | None = None,
        max_age_days: float = 7.0,
        events: dict[str, pd.DataFrame] | None = None,
    ):
        self.cache_dir = Path(cache_dir) / "earnings" if cache_dir else None
        self.max_age = max_age_days * 86400
        self._events: dict[str, pd.DataFrame] = {}
        for sym, df in (events or {}).items():
            self._events[sym.upper()] = self._clean(df)

    # ---------- 数据 ----------
    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], tz=NY_TZ))
        df = df.copy()
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_localize(NY_TZ) if idx.tz is None else idx.tz_convert(NY_TZ)
        df.index = idx
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        df = df[COLUMNS].astype(float)
        df = df[~df.index.isna()]
        df = df[~df.index.duplicated(keep="first")].sort_index()
        return df

    def _fetch(self, symbol: str) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.Ticker(to_yahoo(symbol)).get_earnings_dates(limit=100)
        if raw is None or raw.empty:
            return self._clean(None)
        raw = raw.rename(columns={"EPS Estimate": "eps_estimate", "Reported EPS": "eps_actual", "Surprise(%)": "surprise_pct"})
        return self._clean(raw)

    def get(self, symbol: str) -> pd.DataFrame:
        symbol = symbol.upper()
        if symbol in self._events:
            return self._events[symbol]
        path = self.cache_dir / f"{symbol.replace('.', '_')}.parquet" if self.cache_dir else None
        if path and path.exists() and (time.time() - path.stat().st_mtime) < self.max_age:
            df = self._clean(pd.read_parquet(path))
        else:
            try:
                df = self._fetch(symbol)
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(path)
            except Exception as e:  # 拿不到就当没有财报信息，不让整个回测挂掉
                log.warning("财报日历获取失败 %s: %s", symbol, e)
                df = self._clean(None)
        self._events[symbol] = df
        return df

    # ---------- 查询 ----------
    def next_after(self, symbol: str, ts: pd.Timestamp) -> pd.Timestamp | None:
        ev = self.get(symbol).index
        if len(ev) == 0:
            return None  # ETF 等没有财报
        t = pd.Timestamp(ts).tz_convert(NY_TZ)
        try:
            t = t.as_unit(ev.unit)  # 对齐时间精度，避免 pandas 3 的单位转换报错
        except Exception:
            pass
        pos = ev.searchsorted(t, side="right")
        return ev[pos] if pos < len(ev) else None

    def blackout_mask(self, symbol: str, index: pd.DatetimeIndex, days: int) -> np.ndarray:
        """每根 bar：之后 days 个日历日内（含当天）是否有财报。"""
        ev = self.get(symbol).index
        if len(ev) == 0 or days <= 0:
            return np.zeros(len(index), dtype=bool)
        ts = index.tz_convert(NY_TZ)
        pos = ev.searchsorted(ts, side="right")
        has_next = pos < len(ev)
        next_ev = ev[np.minimum(pos, len(ev) - 1)]
        within = (next_ev - ts) <= pd.Timedelta(days=days)
        return np.asarray(has_next & within)

    def reaction_days(self, symbol: str, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
        """返回 DataFrame[event, reaction_day, surprise_pct]，reaction_day 是 trading_days 里的元素。"""
        ev = self.get(symbol)
        days = pd.DatetimeIndex(trading_days).tz_convert(NY_TZ).normalize().unique().sort_values()
        if len(ev) == 0 or len(days) == 0:
            return pd.DataFrame(columns=["event", "reaction_day", "surprise_pct"])
        closes = days + pd.Timedelta(hours=16)
        pos = closes.searchsorted(ev.index, side="right")  # 第一个 close > ts 的交易日
        ok = pos < len(days)
        out = pd.DataFrame({
            "event": ev.index[ok],
            "reaction_day": days[pos[ok]],
            "surprise_pct": ev["surprise_pct"].values[ok],
        })
        return out.drop_duplicates("reaction_day", keep="first").reset_index(drop=True)
