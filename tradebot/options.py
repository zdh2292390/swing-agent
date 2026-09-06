"""短期 / 末日期权面板的数据层。

思路（全部是可复算的统计量，不做方向预测）：
  1. 取最近到期（默认 7 个日历日内）的期权链，用平值跨式的中间价算"市场隐含的到期波动幅度"。
  2. 用该标的过去 lookback 个样本的同持有期真实 |涨跌幅| 分布，算 P(|真实波动| < 隐含幅度)：
     这是"区间内概率"，对卖方（卖跨式 / 铁鹰）有利；1 - 它是"突破概率"，对买方有利。
  3. 窗口内有财报或高重要性宏观事件时，换成该标的在历史财报日 / 事件日的波动分布重新算，
     因为事件日的波动分布和平时完全不同。
  4. 流动性：平值持仓量与买卖价差占比。
"""
from __future__ import annotations

import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .earnings import EarningsCalendar
from .econ_calendar import nyse_holidays
from .universe import to_yahoo

NY_TZ = "America/New_York"


# ---------------- 日期 ----------------
def trading_days_between(start: dt.date, end: dt.date) -> int:
    """(start, end] 之间的交易日数，排除周末和 NYSE 休市。start == end 返回 0。"""
    if end <= start:
        return 0
    hol = nyse_holidays(start.year) | nyse_holidays(end.year)
    d, n = start, 0
    while d < end:
        d += dt.timedelta(days=1)
        if d.weekday() < 5 and d not in hol:
            n += 1
    return n


# ---------------- 期权链 ----------------
def _mid(row) -> float:
    bid, ask, last = float(row.get("bid", 0) or 0), float(row.get("ask", 0) or 0), float(row.get("lastPrice", 0) or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return last


def fetch_chain_summary(symbol: str, spot: float | None = None, max_dte: int = 7, cache_dir: Path | None = None,
                        ttl_hours: float = 0.25, today: dt.date | None = None) -> dict | None:
    """最近到期链的平值摘要。没有期权或抓取失败返回 None。"""
    today = today or pd.Timestamp.now(tz=NY_TZ).date()
    path = None
    if cache_dir:
        p = Path(cache_dir) / "options"
        p.mkdir(parents=True, exist_ok=True)
        path = p / f"{symbol.replace('.', '_')}_{today.isoformat()}_{max_dte}.json"
        if path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600:
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
    try:
        import yfinance as yf

        t = yf.Ticker(to_yahoo(symbol))
        exps = [dt.date.fromisoformat(e) for e in (t.options or ())]
        exps = [e for e in exps if e >= today]
        if not exps:
            return None
        within = [e for e in exps if (e - today).days <= max_dte]
        expiry = within[0] if within else exps[0]
        chain = t.option_chain(expiry.isoformat())
        calls, puts = chain.calls, chain.puts
        if spot is None:
            spot = float(t.fast_info["last_price"])
        strikes = sorted(set(calls["strike"]).intersection(set(puts["strike"])))
        if not strikes or not spot:
            return None
        atm = min(strikes, key=lambda k: abs(k - spot))
        c = calls[calls["strike"] == atm].iloc[0]
        p = puts[puts["strike"] == atm].iloc[0]
        call_mid, put_mid = _mid(c), _mid(p)
        straddle = call_mid + put_mid
        if straddle <= 0:
            return None
        iv = float(np.nanmean([c.get("impliedVolatility", np.nan), p.get("impliedVolatility", np.nan)]))
        dte_cal = (expiry - today).days
        t_years = max(dte_cal, 0.5) / 365.0
        spread = (float(c.get("ask", 0) or 0) - float(c.get("bid", 0) or 0)) + (float(p.get("ask", 0) or 0) - float(p.get("bid", 0) or 0))
        out = {
            "symbol": symbol, "expiry": expiry.isoformat(), "dte_cal": dte_cal,
            "dte_trading": trading_days_between(today, expiry), "spot": float(spot), "atm_strike": float(atm),
            "call_mid": call_mid, "put_mid": put_mid, "straddle": straddle,
            "implied_move_pct": straddle / spot * 100,
            "atm_iv": iv * 100 if not math.isnan(iv) else None,
            "iv_move_pct": (iv * math.sqrt(t_years) * 100) if not math.isnan(iv) else None,
            "atm_oi": int((c.get("openInterest") or 0) + (p.get("openInterest") or 0)),
            "atm_volume": int((c.get("volume") or 0) + (p.get("volume") or 0)),
            "spread_pct": spread / straddle * 100 if straddle else None,
            "n_expiries_7d": sum(1 for e in exps if (e - today).days <= 7),
        }
        out["is_daily"] = out["n_expiries_7d"] >= 4
        if path:
            path.write_text(json.dumps(out))
        return out
    except Exception:
        return None


# ---------------- 历史波动分布 ----------------
def realized_moves(close: pd.Series, horizon: int, lookback: int = 250) -> np.ndarray:
    """过去 lookback 个样本的 |horizon 日涨跌幅|（%）。"""
    h = max(int(horizon), 1)
    r = (close / close.shift(h) - 1).dropna().abs().tail(lookback) * 100
    return r.values


def prob_within(moves: np.ndarray, threshold_pct: float) -> float:
    moves = np.asarray(moves, dtype=float)
    moves = moves[~np.isnan(moves)]
    if len(moves) == 0 or threshold_pct is None or np.isnan(threshold_pct):
        return float("nan")
    return float((moves < threshold_pct).mean())


def event_day_moves(close: pd.Series, event_days: Iterable[pd.Timestamp], horizon: int = 1) -> np.ndarray:
    """事件日 t 的 |涨跌幅|（%）：close[t + h - 1] / close[t - 1] - 1。event_days 是反应日（盘后公布已映射到下一交易日）。"""
    idx = pd.DatetimeIndex(close.index).normalize()
    h = max(int(horizon), 1)
    out = []
    for d in event_days:
        d = pd.Timestamp(d)
        d = d.tz_convert(idx.tz) if d.tzinfo is not None and idx.tz is not None else d
        pos = idx.get_indexer([d.normalize()])[0]
        if pos < 1 or pos + h - 1 >= len(close):
            continue
        out.append(abs(close.iloc[pos + h - 1] / close.iloc[pos - 1] - 1) * 100)
    return np.array(out)


# ---------------- 汇总表 ----------------
def build_odte_table(
    symbols: list[str],
    data: dict[str, pd.DataFrame],
    chains: dict[str, dict | None],
    calendar: EarningsCalendar,
    macro_events: pd.DataFrame,
    groups: dict[str, list[str]] | None = None,
    signal_table: pd.DataFrame | None = None,
    strategy_col: str | None = None,
    today: dt.date | None = None,
    lookback: int = 250,
) -> pd.DataFrame:
    today = today or pd.Timestamp.now(tz=NY_TZ).date()
    today_ts = pd.Timestamp(today).tz_localize(NY_TZ)
    hi_macro = macro_events[macro_events["影响"] == "高"] if len(macro_events) else macro_events
    macro_dates = pd.to_datetime(hi_macro["日期"], errors="coerce") if len(hi_macro) else pd.Series(dtype="datetime64[ns]")
    past_macro_days = [pd.Timestamp(d).tz_localize(NY_TZ) for d in macro_dates.dropna() if d.date() <= today]

    rows = []
    for sym in symbols:
        ch = chains.get(sym)
        df = data.get(sym)
        if not ch or df is None or len(df) < 60:
            continue
        close = df["close"]
        expiry = dt.date.fromisoformat(ch["expiry"])
        horizon = max(ch["dte_trading"], 1)
        implied = ch["implied_move_pct"]
        moves = realized_moves(close, horizon, lookback)
        med_real = float(np.median(moves)) if len(moves) else float("nan")
        p_in = prob_within(moves, implied)

        # 窗口内事件
        events = []
        nxt = calendar.next_after(sym, today_ts - pd.Timedelta(hours=1))
        earn_in = nxt is not None and nxt.tz_convert(NY_TZ).date() <= expiry + dt.timedelta(days=1) and nxt.tz_convert(NY_TZ).date() >= today
        if earn_in:
            events.append(f"财报 {nxt.tz_convert(NY_TZ):%m-%d}")
        macro_in = []
        if len(hi_macro):
            for _, r in hi_macro.iterrows():
                d = pd.to_datetime(r["日期"], errors="coerce")
                if pd.notna(d) and today <= d.date() <= expiry:
                    macro_in.append(r["事件"])
        events += macro_in[:3]

        # 事件日分布
        ev_kind, ev_med, ev_p_in, ev_n = "", float("nan"), float("nan"), 0
        if earn_in:
            rd = calendar.reaction_days(sym, df.index)
            hist = event_day_moves(close, rd["reaction_day"], horizon=1) if len(rd) else np.array([])
            if len(hist) >= 4:
                ev_kind, ev_med, ev_p_in, ev_n = "财报日", float(np.median(hist)), prob_within(hist, implied), len(hist)
        elif macro_in and past_macro_days:
            hist = event_day_moves(close, past_macro_days, horizon=1)
            if len(hist) >= 3:
                ev_kind, ev_med, ev_p_in, ev_n = "宏观日", float(np.median(hist)), prob_within(hist, implied), len(hist)

        p_used = ev_p_in if not math.isnan(ev_p_in) else p_in
        liquid = (ch.get("spread_pct") is not None and ch["spread_pct"] <= 15) and ch.get("atm_oi", 0) >= 100
        row = {
            "标的": sym,
            "板块": "/".join(g for g, m in (groups or {}).items() if sym in m),
            "到期": ch["expiry"][5:], "交易日": ch["dte_trading"], "日频到期": "是" if ch.get("is_daily") else "",
            "现价": round(ch["spot"], 2), "平值": ch["atm_strike"], "跨式": round(ch["straddle"], 2),
            "隐含波动%": round(implied, 2), "历史中位%": round(med_real, 2) if not math.isnan(med_real) else None,
            "隐含/历史": round(implied / med_real, 2) if med_real and not math.isnan(med_real) else None,
            "区间内概率": round(p_used * 100, 0) if not math.isnan(p_used) else None,
            "突破概率": round((1 - p_used) * 100, 0) if not math.isnan(p_used) else None,
            "平时区间概率": round(p_in * 100, 0) if not math.isnan(p_in) else None,
            "样本": int(len(moves)),
            "事件": "、".join(events),
            "事件分布": ev_kind, "事件日中位%": round(ev_med, 2) if not math.isnan(ev_med) else None, "事件样本": ev_n or None,
            "ATM持仓量": ch.get("atm_oi"), "价差%": round(ch["spread_pct"], 1) if ch.get("spread_pct") is not None else None,
            "流动性": "✓" if liquid else "差",
            "IV%": round(ch["atm_iv"], 1) if ch.get("atm_iv") else None,
        }
        if signal_table is not None and sym in set(signal_table["标的"]):
            srow = signal_table[signal_table["标的"] == sym].iloc[0]
            row["趋势"] = srow.get("趋势", "")
            row["RSI14"] = round(float(srow.get("RSI14", float("nan"))), 0) if pd.notna(srow.get("RSI14")) else None
            if strategy_col and strategy_col in signal_table.columns:
                row["策略"] = srow.get(strategy_col, "")
        rows.append(row)
    return pd.DataFrame(rows)


def macro_in_window(macro_events: pd.DataFrame, today: dt.date, days: int) -> pd.DataFrame:
    if not len(macro_events):
        return macro_events
    d = pd.to_datetime(macro_events["日期"], errors="coerce")
    keep = d.notna() & (d.dt.date >= today) & (d.dt.date <= today + dt.timedelta(days=days))
    return macro_events[keep].sort_values("日期")
