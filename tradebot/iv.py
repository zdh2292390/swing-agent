"""正股 30 天隐含波动率（IV30）快照与历史积累，用来算 IV Rank / IV Percentile。

- IV30：取最靠近 30 个日历日、分别在两侧的两个到期日，各取平值 call/put IV 的均值，按“方差 × 时间”线性插值到 30 天
  （和 VIX 的构造思路一致，只是用平值 IV 代替整条链）。只有一侧到期日时直接用最近的那一个。
- 历史：免费源没有 IV30 的历史序列，只能自己每天记一条到 data_cache/iv_history.csv。
  IV Rank = (今 − 区间最低) / (区间最高 − 区间最低)；IV Percentile = 区间内低于今天的天数占比。样本少于 60 天时只能参考。
- HV30：最近 30 个交易日收盘收益率的年化标准差，用来看 IV 相对真实波动贵不贵。
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .universe import to_yahoo

NY_TZ = "America/New_York"
HISTORY_FILE = "iv_history.csv"


def _atm_iv(chain_calls: pd.DataFrame, chain_puts: pd.DataFrame, spot: float) -> float | None:
    vals = []
    for df in (chain_calls, chain_puts):
        if df is None or df.empty or "impliedVolatility" not in df:
            continue
        d = df[(df["impliedVolatility"] > 0.01) & (df["bid"] > 0)] if "bid" in df else df[df["impliedVolatility"] > 0.01]
        if d.empty:
            continue
        near = d.iloc[(d["strike"] - spot).abs().argsort()[:2]]  # 平值上下各一档
        vals.extend(near["impliedVolatility"].tolist())
    return float(np.mean(vals)) if vals else None


def iv30_snapshot(symbol: str, target_days: int = 30, today: dt.date | None = None) -> dict | None:
    """返回 {symbol, date, spot, iv30(%), near_expiry, far_expiry, iv_near, iv_far}。失败返回 None。"""
    try:
        import yfinance as yf

        today = today or pd.Timestamp.now(tz=NY_TZ).date()
        t = yf.Ticker(to_yahoo(symbol))
        exps = sorted(dt.date.fromisoformat(e) for e in (t.options or ()) if dt.date.fromisoformat(e) > today)
        if not exps:
            return None
        spot = float(t.fast_info["last_price"])
        before = [e for e in exps if (e - today).days <= target_days and (e - today).days >= 5]
        after = [e for e in exps if (e - today).days > target_days]
        picks = ([before[-1]] if before else []) + ([after[0]] if after else [])
        if not picks:
            picks = [exps[0]]
        ivs, days = [], []
        for e in picks:
            ch = t.option_chain(e.isoformat())
            iv = _atm_iv(ch.calls, ch.puts, spot)
            if iv is not None:
                ivs.append(iv)
                days.append((e - today).days)
        if not ivs:
            return None
        if len(ivs) == 1:
            iv30 = ivs[0]
        else:
            (d1, v1), (d2, v2) = sorted(zip(days, ivs))
            if d1 == d2:
                iv30 = (v1 + v2) / 2
            else:  # 方差时间线性插值到 30 天
                var1, var2 = v1 * v1 * d1, v2 * v2 * d2
                w = (target_days - d1) / (d2 - d1)
                var30 = var1 + w * (var2 - var1)
                iv30 = math.sqrt(max(var30, 1e-9) / target_days)
        return {"symbol": symbol, "date": today.isoformat(), "spot": spot, "iv30": iv30 * 100,
                "near_expiry": picks[0].isoformat(), "far_expiry": picks[-1].isoformat(),
                "iv_near": ivs[0] * 100, "iv_far": ivs[-1] * 100, "days_near": days[0], "days_far": days[-1]}
    except Exception:
        return None


def hv30(close: pd.Series, n: int = 30) -> float:
    r = close.pct_change().dropna().tail(n)
    return float(r.std() * math.sqrt(252) * 100) if len(r) >= 10 else float("nan")


def load_history(cache_dir: Path) -> pd.DataFrame:
    p = Path(cache_dir) / HISTORY_FILE
    if not p.exists():
        return pd.DataFrame(columns=["date", "symbol", "spot", "iv30", "hv30"])
    df = pd.read_csv(p)
    return df


def record_snapshots(symbols: list[str], cache_dir: Path, closes: dict[str, pd.Series] | None = None, today: dt.date | None = None) -> pd.DataFrame:
    """给每只标的记一条今天的 IV30 / HV30；同一天重复调用会覆盖当天的记录。返回全部历史。"""
    today = today or pd.Timestamp.now(tz=NY_TZ).date()
    rows = []
    for sym in symbols:
        snap = iv30_snapshot(sym, today=today)
        if snap is None:
            continue
        hv = hv30(closes[sym]) if closes and sym in closes else float("nan")
        rows.append({"date": today.isoformat(), "symbol": sym, "spot": snap["spot"], "iv30": snap["iv30"], "hv30": hv})
    hist = load_history(cache_dir)
    if rows:
        new = pd.DataFrame(rows)
        hist = hist[~((hist["date"] == today.isoformat()) & (hist["symbol"].isin(new["symbol"])))]
        hist = pd.concat([hist, new], ignore_index=True).sort_values(["symbol", "date"])
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        hist.to_csv(Path(cache_dir) / HISTORY_FILE, index=False)
    return hist


def iv_rank_percentile(series: pd.Series, current: float, lookback_days: int = 252) -> dict:
    """series：某标的按日期排序的 iv30 历史。返回 rank / percentile / 样本数 / 区间高低。"""
    v = pd.Series(series).dropna().tail(lookback_days)
    n = int(len(v))
    if n < 2 or current is None or math.isnan(current):
        return {"iv_rank": float("nan"), "iv_pct": float("nan"), "n": n, "lo": float("nan"), "hi": float("nan")}
    lo, hi = float(v.min()), float(v.max())
    rank = (current - lo) / (hi - lo) * 100 if hi > lo else float("nan")
    pct = float((v < current).mean() * 100)
    return {"iv_rank": rank, "iv_pct": pct, "n": n, "lo": lo, "hi": hi}
