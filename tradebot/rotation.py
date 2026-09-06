"""板块轮动分析：相对强度、RRG 象限、月度热力图、板块内部广度。

- 基准默认 SPY。相对强度 RS = 价格 / 基准。
- RS-Ratio = 100 × RS / RS 的 ratio_window 日均值（>100：近期相对基准强于自己过去 3 个月的平均关系）
- RS-Momentum = 100 × RS-Ratio / mom_window 日前的 RS-Ratio（>100：相对强度在改善）
  象限：领先(>100,>100) 走弱(>100,<100) 落后(<100,<100) 改善(<100,>100)。这是 JdK RRG 的常见近似，不是原版专有算法。
- 用户自己的板块按成员等权合成指数（只用美股成员），和 ETF 一起分析。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .strategies.indicators import rsi, sma

NY_TZ = "America/New_York"

# (代码, 名称, 分类)
ROTATION_TICKERS: list[tuple[str, str, str]] = [
    ("XLK", "科技", "行业"), ("XLC", "通讯", "行业"), ("XLY", "可选消费", "行业"), ("XLP", "必需消费", "行业"),
    ("XLV", "医疗", "行业"), ("XLF", "金融", "行业"), ("XLI", "工业", "行业"), ("XLE", "能源", "行业"),
    ("XLB", "材料", "行业"), ("XLU", "公用事业", "行业"), ("XLRE", "地产", "行业"),
    ("SMH", "半导体", "主题"), ("IGV", "软件", "主题"), ("XBI", "生物科技", "主题"), ("ITA", "航空国防", "主题"), ("URA", "铀/核能", "主题"),
    ("QQQ", "纳指100", "风格"), ("IWM", "小盘", "风格"), ("RSP", "标普等权", "风格"), ("MTUM", "动量", "风格"),
    ("IWF", "成长", "风格"), ("IWD", "价值", "风格"), ("USMV", "低波", "风格"),
    ("TLT", "长债", "跨资产"), ("GLD", "黄金", "跨资产"), ("BTC-USD", "比特币", "跨资产"),
]
BENCHMARKS = {"SPY": "标普500 (SPY)", "QQQ": "纳指100 (QQQ)", "RSP": "标普等权 (RSP)"}
HORIZONS = {"1周": 5, "1月": 21, "3月": 63, "6月": 126}
NAME = {c: n for c, n, _ in ROTATION_TICKERS}
KIND = {c: k for c, _, k in ROTATION_TICKERS}


def rotation_closes(cache_dir: Path | None = None, ttl_hours: float = 0.25, days: int = 500) -> pd.DataFrame:
    """ETF 与基准的日收盘价（列 = 代码），parquet 缓存。"""
    path = Path(cache_dir) / "rotation_closes.parquet" if cache_dir else None
    if path and path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600:
        return pd.read_parquet(path)
    import yfinance as yf

    codes = sorted({c for c, _, _ in ROTATION_TICKERS} | set(BENCHMARKS))
    raw = yf.download(codes, period=f"{days}d", interval="1d", progress=False, group_by="ticker", auto_adjust=True, threads=True)
    cols = {}
    for code in codes:
        try:
            part = raw[code] if isinstance(raw.columns, pd.MultiIndex) else raw
            cols[code] = part["Close"]
        except Exception:
            cols[code] = pd.Series(dtype=float)
    closes = pd.DataFrame(cols).sort_index()
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None) if getattr(closes.index, "tz", None) is not None else pd.DatetimeIndex(closes.index)
    # 比特币 7 天交易，对齐到股票交易日
    closes = closes.dropna(subset=["SPY"]) if "SPY" in closes.columns else closes
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        closes.to_parquet(path)
    return closes


def basket_index(members: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.Series:
    """成员等权合成指数（每日等权再平衡），起点 100，对齐到给定交易日索引。缺数据的成员当天不计。"""
    rets = []
    for sym, df in members.items():
        c = df["close"].copy()
        c.index = pd.DatetimeIndex(c.index).tz_localize(None) if getattr(c.index, "tz", None) is not None else pd.DatetimeIndex(c.index)
        c = c[~c.index.duplicated()].reindex(index).ffill()
        rets.append(c.pct_change())
    if not rets:
        return pd.Series(dtype=float, index=index)
    r = pd.concat(rets, axis=1).mean(axis=1, skipna=True).fillna(0.0)
    first_valid = pd.concat(rets, axis=1).notna().any(axis=1)
    r[~first_valid] = 0.0
    return 100 * (1 + r).cumprod()


def rrg_series(closes: pd.DataFrame, bench: str, ratio_window: int = 63, mom_window: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (RS-Ratio, RS-Momentum) 两个 DataFrame，列 = 标的。"""
    b = closes[bench]
    rs = closes.div(b, axis=0)
    ratio = 100 * rs / rs.rolling(ratio_window).mean()
    mom = 100 * ratio / ratio.shift(mom_window)
    return ratio.drop(columns=[bench], errors="ignore"), mom.drop(columns=[bench], errors="ignore")


def quadrant(ratio: float, mom: float) -> str:
    if np.isnan(ratio) or np.isnan(mom):
        return ""
    if ratio >= 100 and mom >= 100:
        return "领先"
    if ratio >= 100 and mom < 100:
        return "走弱"
    if ratio < 100 and mom < 100:
        return "落后"
    return "改善"


def relative_strength_table(closes: pd.DataFrame, bench: str, names: dict[str, str], kinds: dict[str, str],
                            ratio_window: int = 63, mom_window: int = 10) -> pd.DataFrame:
    ratio, mom = rrg_series(closes, bench, ratio_window, mom_window)
    rows = []
    b = closes[bench].dropna()
    for code in closes.columns:
        if code == bench:
            continue
        c = closes[code].dropna()
        if len(c) < 130:
            continue
        px = float(c.iloc[-1])
        row = {"名称": names.get(code, code), "代码": code, "分类": kinds.get(code, ""), "最新": px}
        for label, h in HORIZONS.items():
            r_self = (px / float(c.iloc[-1 - h]) - 1) * 100 if len(c) > h else np.nan
            r_b = (float(b.iloc[-1]) / float(b.iloc[-1 - h]) - 1) * 100 if len(b) > h else np.nan
            row[f"{label}%"] = r_self
            row[f"{label}超额"] = r_self - r_b
        s50, s200 = sma(c, 50).iloc[-1], sma(c, 200).iloc[-1] if len(c) >= 200 else np.nan
        row["趋势"] = ("↑ 多头" if px > s200 and s50 > s200 else ("↓ 空头" if px < s200 and s50 < s200 else "→ 震荡")) if not np.isnan(s200) else ("↑" if px > s50 else "↓")
        row["距52周高%"] = (px / float(c.tail(252).max()) - 1) * 100
        rr, mm = float(ratio[code].iloc[-1]), float(mom[code].iloc[-1])
        row["RS比率"] = rr
        row["RS动量"] = mm
        row["象限"] = quadrant(rr, mm)
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["3月超额排名"] = out["3月超额"].rank(ascending=False, method="min").astype("Int64")
        out = out.sort_values("3月超额", ascending=False).reset_index(drop=True)
    return out


def rrg_tail(closes: pd.DataFrame, bench: str, codes: list[str], ratio_window: int = 63, mom_window: int = 10,
             tail_points: int = 8, step: int = 5) -> dict[str, pd.DataFrame]:
    """每个标的最近 tail_points 个采样点（每 step 个交易日一个）的 (ratio, mom)。"""
    ratio, mom = rrg_series(closes, bench, ratio_window, mom_window)
    out = {}
    idx = list(range(len(ratio) - 1, -1, -step))[:tail_points][::-1]
    for code in codes:
        if code not in ratio.columns:
            continue
        pts = pd.DataFrame({"ratio": ratio[code].iloc[idx].values, "mom": mom[code].iloc[idx].values}, index=ratio.index[idx]).dropna()
        if len(pts):
            out[code] = pts
    return out


def monthly_returns(closes: pd.DataFrame, codes: list[str], months: int = 12) -> pd.DataFrame:
    """行 = 标的，列 = 月份（YYYY-MM），值 = 当月收益 %（最后一列是本月至今）。"""
    sub = closes[[c for c in codes if c in closes.columns]].dropna(how="all")
    m = sub.resample("ME").last()
    ret = (m / m.shift(1) - 1).tail(months) * 100
    ret.index = [d.strftime("%Y-%m") for d in ret.index]
    return ret.T


def group_breadth(groups: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """每个板块：成员数、站上 SMA50/SMA200 比例、RSI>50 比例、平均 20 日涨幅、平均距 52 周高。"""
    rows = []
    for g, members in groups.items():
        n = above50 = above200 = rsi_up = 0
        r20, dist = [], []
        for sym, df in members.items():
            c = df["close"]
            if len(c) < 60:
                continue
            n += 1
            px = float(c.iloc[-1])
            above50 += int(px > sma(c, 50).iloc[-1])
            if len(c) >= 200:
                above200 += int(px > sma(c, 200).iloc[-1])
            rsi_up += int(float(rsi(c, 14).iloc[-1]) > 50)
            r20.append((px / float(c.iloc[-21]) - 1) * 100 if len(c) > 21 else np.nan)
            dist.append((px / float(c.tail(252).max()) - 1) * 100)
        if n == 0:
            continue
        rows.append({"板块": g, "成员": n, ">SMA50%": above50 / n * 100, ">SMA200%": above200 / n * 100, "RSI>50%": rsi_up / n * 100,
                     "平均20日%": float(np.nanmean(r20)), "平均距52周高%": float(np.nanmean(dist))})
    return pd.DataFrame(rows)
