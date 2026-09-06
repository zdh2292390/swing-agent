"""大盘与宏观：指数/利率/商品快照、用户维护的事件日历、标的池近期财报。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .earnings import EarningsCalendar

NY_TZ = "America/New_York"
MANUAL_EVENTS_PATH = ROOT / "macro_events.json"
EVENT_COLUMNS = ["日期", "事件", "影响", "备注"]

# (yfinance 代码, 名称, 类别) —— 这里直接用 yfinance 原生代码，不走标的池的符号映射。指数用指数本身，不用 ETF 代替。
MACRO_TICKERS: list[tuple[str, str, str]] = [
    ("^GSPC", "标普500", "指数"),
    ("^NDX", "纳指100", "指数"),
    ("^DJI", "道琼斯", "指数"),
    ("^RUT", "罗素2000", "指数"),
    ("SMH", "半导体ETF", "板块"),
    ("^VIX", "VIX", "波动"),
    ("^TNX", "美债10Y%", "利率"),
    ("TLT", "长债ETF", "利率"),
    ("DX-Y.NYB", "美元指数", "汇率"),
    ("GC=F", "黄金", "商品"),
    ("CL=F", "原油", "商品"),
    ("BTC-USD", "比特币", "加密"),
]
MACRO_NAME = {code: name for code, name, _ in MACRO_TICKERS}


def macro_closes(cache_dir: Path | None = None, ttl_hours: float = 0.25, days: int = 400) -> pd.DataFrame:
    """全部宏观标的的日收盘价，列 = yfinance 代码。parquet 缓存 ttl_hours。"""
    path = Path(cache_dir) / "macro_closes.parquet" if cache_dir else None
    if path and path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600:
        return pd.read_parquet(path)

    import yfinance as yf

    codes = [c for c, _, _ in MACRO_TICKERS]
    raw = yf.download(codes, period=f"{days}d", interval="1d", progress=False, group_by="ticker", auto_adjust=True, threads=True)
    cols = {}
    for code in codes:
        try:
            part = raw[code] if isinstance(raw.columns, pd.MultiIndex) else raw
            cols[code] = part["Close"]
        except Exception:
            cols[code] = pd.Series(dtype=float)
    closes = pd.DataFrame(cols)
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None) if getattr(closes.index, "tz", None) is not None else pd.DatetimeIndex(closes.index)
    closes = closes.sort_index()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        closes.to_parquet(path)
    return closes


def compute_snapshot(closes: pd.DataFrame) -> pd.DataFrame:
    """每个宏观标的的最新价、1/5/20 日涨跌（%）、趋势、距 52 周高。纯函数，便于测试。"""
    rows = []
    for code, name, kind in MACRO_TICKERS:
        c = closes[code].dropna() if code in closes.columns else pd.Series(dtype=float)
        if len(c) < 25:
            rows.append({"名称": name, "代码": code, "类别": kind, "最新": np.nan, "涨跌1日": np.nan, "涨跌5日": np.nan, "涨跌20日": np.nan, "趋势": "", "距52周高": np.nan})
            continue
        px = float(c.iloc[-1])

        def chg(k: int) -> float:
            return (px / float(c.iloc[-1 - k]) - 1) * 100 if len(c) > k else np.nan

        s50 = c.rolling(50).mean().iloc[-1]
        s200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else np.nan
        if np.isnan(s200):
            trend = "↑" if px > s50 else "↓"
        elif px > s200 and s50 > s200:
            trend = "↑ 多头"
        elif px < s200 and s50 < s200:
            trend = "↓ 空头"
        else:
            trend = "→ 震荡"
        rows.append({
            "名称": name, "代码": code, "类别": kind, "最新": px,
            "涨跌1日": chg(1), "涨跌5日": chg(5), "涨跌20日": chg(20),
            "趋势": trend, "距52周高": (px / float(c.tail(252).max()) - 1) * 100,
        })
    return pd.DataFrame(rows)


def market_snapshot(cache_dir: Path | None = None, ttl_hours: float = 0.25) -> pd.DataFrame:
    return compute_snapshot(macro_closes(cache_dir, ttl_hours))


def market_history(closes: pd.DataFrame, codes: list[str], days: int = 63) -> pd.DataFrame:
    """最近 days 个交易日的相对走势，起点 = 100，列名用中文名称。"""
    sub = closes[[c for c in codes if c in closes.columns]].dropna(how="all").tail(days).ffill()
    sub = sub.dropna(axis=1, how="any")
    return (sub / sub.iloc[0] * 100).rename(columns=MACRO_NAME)


def load_manual_events(path: Path = MANUAL_EVENTS_PATH) -> pd.DataFrame:
    """用户维护的宏观事件（FOMC、CPI、非农、大会等）。文件不存在返回空表。"""
    if not Path(path).exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        rows = []
    df = pd.DataFrame(rows)
    for c in EVENT_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[EVENT_COLUMNS].fillna("").astype(str)


def save_manual_events(df: pd.DataFrame, path: Path = MANUAL_EVENTS_PATH) -> pd.DataFrame:
    df = df.copy().fillna("").astype(str)
    df = df[df["事件"].str.strip() != ""]
    df = df.sort_values("日期").reset_index(drop=True)
    Path(path).write_text(json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=1), encoding="utf-8")
    return df


def upcoming_events(df: pd.DataFrame, days: int = 45, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """按日期筛出未来 days 天内的事件，附"距今天"。日期解析失败的行保留在最后。"""
    if df.empty:
        return df.assign(距今天=pd.Series(dtype="Int64"))
    now = (now or pd.Timestamp.now(tz=NY_TZ)).normalize()
    d = pd.to_datetime(df["日期"], errors="coerce")
    d = d.dt.tz_localize(NY_TZ) if d.dt.tz is None else d.dt.tz_convert(NY_TZ)
    delta = (d - now).dt.days
    out = df.assign(距今天=delta.astype("Int64"))
    keep = delta.isna() | ((delta >= -1) & (delta <= days))
    return out[keep].sort_values("距今天", na_position="last").reset_index(drop=True)


def upcoming_earnings(symbols: list[str], cal: EarningsCalendar, groups: dict[str, list[str]] | None = None,
                      days: int = 30, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """标的池里未来 days 天内有财报的标的。"""
    now = now or pd.Timestamp.now(tz=NY_TZ)
    rows = []
    for sym in symbols:
        nxt = cal.next_after(sym, now)
        if nxt is None:
            continue
        dd = (nxt - now).days
        if dd <= days:
            rows.append({
                "标的": sym,
                "板块": "/".join(g for g, m in (groups or {}).items() if sym in m),
                "日期": nxt.strftime("%m-%d %a"),
                "距今天": int(dd),
                "盘前/盘后": "盘后" if nxt.hour >= 16 else ("盘前" if 0 < nxt.hour < 9 or nxt.hour == 9 and nxt.minute < 30 else "待定"),
            })
    return pd.DataFrame(rows, columns=["标的", "板块", "日期", "距今天", "盘前/盘后"]).sort_values("距今天").reset_index(drop=True)
