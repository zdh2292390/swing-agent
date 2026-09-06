"""LEAP call 分析：6 / 8 / 10 个月到期的看涨期权，按目标 delta 选行权价，算杠杆、盈亏平衡、两种胜率。

全部是可复算的量，不预测方向：
  - IV 来自期权链，delta / 概率用 Black-Scholes（连续股息 q，无风险利率 r）
  - 杠杆率 = delta × 现价 / 权利金（期权价值对正股涨跌的弹性）
  - 盈亏平衡 = 行权价 + 权利金；需涨幅 = 盈亏平衡 / 现价 - 1
  - 隐含胜率 = 风险中性下到期价 > 盈亏平衡的概率 N(d2)，用 IV
  - 历史胜率 = 过去全部同长度持有期里，正股涨幅 ≥ 需涨幅 的比例（重叠窗口，偏乐观，仅作参考）
  - 年化持有成本 = 时间价值 / 现价 / 年数，衡量拿 LEAP 代替正股每年付出的"租金"
"""
from __future__ import annotations

import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .universe import to_yahoo

NY_TZ = "America/New_York"
DEFAULT_MONTHS = (6, 8, 10)


# ---------------- Black-Scholes ----------------
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan"), float("nan")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_call_delta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    d1, _ = bs_d1_d2(S, K, T, r, q, sigma)
    return math.exp(-q * T) * _ncdf(d1) if not math.isnan(d1) else float("nan")


def bs_call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    if math.isnan(d1):
        return max(S - K, 0.0)
    return S * math.exp(-q * T) * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def prob_above(S: float, level: float, T: float, r: float, q: float, sigma: float) -> float:
    """风险中性下 P(S_T > level) = N(d2)，K 取 level。"""
    _, d2 = bs_d1_d2(S, level, T, r, q, sigma)
    return _ncdf(d2) if not math.isnan(d2) else float("nan")


# ---------------- 选到期、选行权 ----------------
def pick_expiry(expiries: list[dt.date], today: dt.date, months: int, tol_days: int = 45) -> dt.date | None:
    target = today + dt.timedelta(days=round(months * 30.44))
    cands = [(abs((e - target).days), e) for e in expiries if e > today]
    if not cands:
        return None
    gap, best = min(cands)
    return best if gap <= tol_days else None


def _mid(row) -> float:
    bid, ask, last = float(row.get("bid", 0) or 0), float(row.get("ask", 0) or 0), float(row.get("lastPrice", 0) or 0)
    return (bid + ask) / 2 if bid > 0 and ask > 0 else last


def pick_strike_by_delta(calls: pd.DataFrame, S: float, T: float, r: float, q: float, target_delta: float) -> tuple[pd.Series, float] | None:
    best = None
    for _, row in calls.iterrows():
        iv = float(row.get("impliedVolatility", 0) or 0)
        if iv <= 0.01 or _mid(row) <= 0:
            continue
        d = bs_call_delta(S, float(row["strike"]), T, r, q, iv)
        if math.isnan(d):
            continue
        gap = abs(d - target_delta)
        if best is None or gap < best[0]:
            best = (gap, row, d)
    return (best[1], best[2]) if best else None


# ---------------- 抓链 ----------------
def fetch_leap_candidates(symbol: str, spot: float | None = None, months: tuple[int, ...] = DEFAULT_MONTHS, target_delta: float = 0.7,
                          r: float = 0.04, q: float = 0.0, cache_dir: Path | None = None, ttl_hours: float = 1.0,
                          today: dt.date | None = None) -> list[dict]:
    """每个目标月份一张合约的摘要。没有期权 / 抓取失败返回 []。"""
    today = today or pd.Timestamp.now(tz=NY_TZ).date()
    path = None
    if cache_dir:
        p = Path(cache_dir) / "leaps"
        p.mkdir(parents=True, exist_ok=True)
        path = p / f"{symbol.replace('.', '_')}_{today.isoformat()}_{'-'.join(map(str, months))}_{int(target_delta * 100)}.json"
        if path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600:
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
    out: list[dict] = []
    try:
        import yfinance as yf

        t = yf.Ticker(to_yahoo(symbol))
        exps = [dt.date.fromisoformat(e) for e in (t.options or ())]
        if not exps:
            return []
        if spot is None:
            spot = float(t.fast_info["last_price"])
        used: set[dt.date] = set()
        for m in months:
            exp = pick_expiry(exps, today, m)
            if exp is None or exp in used:
                continue
            used.add(exp)
            calls = t.option_chain(exp.isoformat()).calls
            T = (exp - today).days / 365.0
            picked = pick_strike_by_delta(calls, spot, T, r, q, target_delta)
            if picked is None:
                continue
            row, delta = picked
            K = float(row["strike"])
            prem = _mid(row)
            iv = float(row["impliedVolatility"])
            intrinsic = max(spot - K, 0.0)
            extrinsic = max(prem - intrinsic, 0.0)
            be = K + prem
            bid, ask = float(row.get("bid", 0) or 0), float(row.get("ask", 0) or 0)
            out.append({
                "symbol": symbol, "months": m, "expiry": exp.isoformat(), "dte": (exp - today).days, "T": T,
                "contract": str(row.get("contractSymbol", "")),
                "spot": float(spot), "strike": K, "premium": prem, "bid": bid, "ask": ask,
                "spread_pct": (ask - bid) / prem * 100 if prem > 0 and ask > 0 else None,
                "iv": iv * 100, "delta": delta, "leverage": delta * spot / prem if prem > 0 else None,
                "intrinsic": intrinsic, "extrinsic": extrinsic, "extrinsic_pct": extrinsic / prem * 100 if prem > 0 else None,
                "annual_cost_pct": extrinsic / spot / T * 100 if T > 0 else None,
                "breakeven": be, "be_move_pct": (be / spot - 1) * 100,
                "p_itm_implied": prob_above(spot, K, T, r, q, iv) * 100,
                "p_be_implied": prob_above(spot, be, T, r, q, iv) * 100,
                "oi": int(row.get("openInterest") or 0), "volume": int(row.get("volume") or 0),
                "r": r, "q": q,
            })
        if path:
            path.write_text(json.dumps(out))
    except Exception:
        return out
    return out


# ---------------- 历史统计 ----------------
def horizon_stats(close: pd.Series, dte_days: int, required_move_pct: float) -> dict:
    """同长度持有期的历史分布：胜率（涨幅 ≥ 需涨幅）、上涨概率、中位涨幅、样本数、最近同长度真实波动率。"""
    h = max(int(round(dte_days * 252 / 365)), 1)
    ret = (close.shift(-h) / close - 1).dropna() * 100
    if len(ret) < 30:
        return {"p_hist_be": float("nan"), "p_hist_up": float("nan"), "median_ret": float("nan"), "n": int(len(ret)), "rv": float("nan")}
    daily = close.pct_change().dropna().tail(h)
    rv = float(daily.std() * math.sqrt(252) * 100) if len(daily) > 5 else float("nan")
    return {
        "p_hist_be": float((ret >= required_move_pct).mean() * 100),
        "p_hist_up": float((ret > 0).mean() * 100),
        "median_ret": float(ret.median()),
        "n": int(len(ret)),
        "rv": rv,
    }


def build_leap_table(candidates: dict[str, list[dict]], data: dict[str, pd.DataFrame], groups: dict[str, list[str]] | None = None) -> pd.DataFrame:
    rows = []
    for sym, lst in candidates.items():
        df = data.get(sym)
        for c in lst or []:
            hs = horizon_stats(df["close"], c["dte"], c["be_move_pct"]) if df is not None and len(df) > 60 else {}
            rows.append({
                "标的": sym, "板块": "/".join(g for g, m in (groups or {}).items() if sym in m),
                "月数": c["months"], "到期": c["expiry"], "天数": c["dte"], "现价": round(c["spot"], 2), "行权": c["strike"],
                "权利金": round(c["premium"], 2), "Delta": round(c["delta"], 2), "杠杆": round(c["leverage"], 1) if c["leverage"] else None,
                "IV%": round(c["iv"], 1), "RV%": round(hs.get("rv", float("nan")), 1) if hs and not math.isnan(hs.get("rv", float("nan"))) else None,
                "IV/RV": round(c["iv"] / hs["rv"], 2) if hs and hs.get("rv") and not math.isnan(hs["rv"]) and hs["rv"] > 0 else None,
                "时间价值%": round(c["extrinsic_pct"], 1) if c["extrinsic_pct"] is not None else None,
                "年化成本%": round(c["annual_cost_pct"], 1) if c["annual_cost_pct"] is not None else None,
                "盈亏平衡": round(c["breakeven"], 2), "需涨%": round(c["be_move_pct"], 1),
                "隐含胜率": round(c["p_be_implied"], 0), "隐含ITM": round(c["p_itm_implied"], 0),
                "历史胜率": round(hs["p_hist_be"], 0) if hs and not math.isnan(hs.get("p_hist_be", float("nan"))) else None,
                "历史上涨率": round(hs["p_hist_up"], 0) if hs and not math.isnan(hs.get("p_hist_up", float("nan"))) else None,
                "历史中位涨%": round(hs["median_ret"], 1) if hs and not math.isnan(hs.get("median_ret", float("nan"))) else None,
                "样本": hs.get("n") if hs else None,
                "OI": c["oi"], "价差%": round(c["spread_pct"], 1) if c["spread_pct"] is not None else None,
            })
    return pd.DataFrame(rows)


def payoff_curve(c: dict, moves: np.ndarray) -> np.ndarray:
    """到期时，正股涨跌 moves（小数）对应的期权收益率（相对权利金）。"""
    st = c["spot"] * (1 + moves)
    return (np.maximum(st - c["strike"], 0.0) - c["premium"]) / c["premium"] * 100


# ---------------- 合约历史价格 ----------------
def fetch_option_history(contract: str, period: str = "1y", cache_dir: Path | None = None, ttl_hours: float = 6.0) -> pd.DataFrame:
    """某张期权合约的日收盘与成交量（Yahoo 按最后成交价记，成交稀疏的日子会有跳空）。失败返回空表。"""
    path = None
    if cache_dir and contract:
        p = Path(cache_dir) / "option_history"
        p.mkdir(parents=True, exist_ok=True)
        path = p / f"{contract}_{period}.parquet"
        if path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600:
            return pd.read_parquet(path)
    try:
        import yfinance as yf

        h = yf.Ticker(contract).history(period=period)
        if h is None or h.empty:
            return pd.DataFrame(columns=["close", "volume"])
        out = pd.DataFrame({"close": h["Close"].astype(float), "volume": h["Volume"].astype(float)})
        idx = pd.DatetimeIndex(out.index)
        out.index = (idx.tz_convert(NY_TZ) if idx.tz is not None else idx.tz_localize(NY_TZ)).normalize()
        out = out[~out.index.duplicated(keep="last")].sort_index()
        if path:
            out.to_parquet(path)
        return out
    except Exception:
        return pd.DataFrame(columns=["close", "volume"])


def bs_price_path(underlying_close: pd.Series, strike: float, expiry: dt.date, r: float, q: float, sigma: float) -> pd.Series:
    """用固定 IV 的 Black-Scholes 反推这张合约在过去每一天“理论上”值多少钱，作为实际成交价的参照。"""
    idx = pd.DatetimeIndex(underlying_close.index)
    days_left = np.array([(expiry - d.date()).days for d in idx], dtype=float)
    vals = [bs_call_price(float(S_), strike, max(dl, 0.5) / 365.0, r, q, sigma) if dl > 0 else max(float(S_) - strike, 0.0)
            for S_, dl in zip(underlying_close.values, days_left)]
    return pd.Series(vals, index=underlying_close.index)


# ---------------- 从历史成交价反推隐含波动率 ----------------
def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float, lo: float = 0.01, hi: float = 5.0, tol: float = 1e-4) -> float:
    """二分法求 call 的隐含波动率。价格不高于内在价值或期限已尽返回 nan。"""
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return float("nan")
    intrinsic = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-6:
        return float("nan")
    if bs_call_price(S, K, T, r, q, hi) < price:
        return float("nan")
    for _ in range(80):
        mid = (lo + hi) / 2
        if bs_call_price(S, K, T, r, q, mid) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def implied_vol_series(option_hist: pd.DataFrame, underlying_close: pd.Series, strike: float, expiry: dt.date,
                       r: float, q: float, min_volume: float = 1) -> pd.Series:
    """按日反推 IV（%）。只用当天有成交（volume ≥ min_volume）且价格高于内在价值的日子，其余为 nan。"""
    if option_hist is None or option_hist.empty:
        return pd.Series(dtype=float)
    u = underlying_close.copy()
    u.index = pd.DatetimeIndex(u.index).normalize()
    u = u[~u.index.duplicated(keep="last")]
    oh = option_hist.copy()
    oh.index = pd.DatetimeIndex(oh.index).normalize()
    joined = oh.join(u.rename("S"), how="inner")
    out = []
    for d, row in joined.iterrows():
        T = (expiry - d.date()).days / 365.0
        if row.get("volume", 0) < min_volume or pd.isna(row["S"]):
            out.append(float("nan"))
            continue
        iv = implied_vol(float(row["close"]), float(row["S"]), strike, T, r, q)
        out.append(iv * 100 if not math.isnan(iv) else float("nan"))
    return pd.Series(out, index=joined.index, name="iv")


def iv_stats(iv: pd.Series, current_iv: float | None) -> dict:
    """区间 IV 的中位 / 最低 / 最高，以及当前 IV 在历史里的百分位（0~100，越高越贵）。"""
    v = iv.dropna()
    if len(v) < 5:
        return {"iv_median": float("nan"), "iv_min": float("nan"), "iv_max": float("nan"), "iv_pct": float("nan"), "n": int(len(v))}
    cur = current_iv if current_iv is not None else float(v.iloc[-1])
    return {"iv_median": float(v.median()), "iv_min": float(v.min()), "iv_max": float(v.max()),
            "iv_pct": float((v < cur).mean() * 100), "n": int(len(v))}
