"""Streamlit 可视化面板：回测 / 模拟盘 / 决策日志 / 行情。

  conda activate trading
  streamlit run dashboard.py
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import inspect
import json
import math
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import numpy as np

from tradebot.backtest import event_study, run_backtest, slice_result
from tradebot.backtest.metrics import metrics_table
from tradebot.config import DEFAULT_BACKTEST_DAYS, DEFAULT_LOOKBACK_DAYS, ROOT, load_settings
from tradebot.config import ROOT as ROOT_DIR
from tradebot.data import INTERVAL_INFO, drop_incomplete_last_bar, fetch_bars, latest_prices
from tradebot.earnings import EarningsCalendar
from tradebot.execution import make_broker
from tradebot.agent.research import has_credentials
from tradebot.econ_calendar import market_session_open, sync_official_events
from tradebot.macro import EVENT_COLUMNS, compute_snapshot, load_manual_events, macro_closes, market_history, save_manual_events, upcoming_earnings, upcoming_events
from tradebot.news import fetch_info, fetch_news, format_info
from tradebot.options import build_odte_table, fetch_chain_summary, macro_in_window
from tradebot.bottom import BottomParams, analyze_bottom, build_bottom_table
from tradebot.leaps import bs_price_path, build_leap_table, fetch_leap_candidates, fetch_option_history, implied_vol_series, iv_stats, payoff_curve
from tradebot.iv import hv30, iv30_snapshot, iv_rank_percentile, load_history as load_iv_history
from tradebot.rotation import BENCHMARKS, KIND, NAME, basket_index, group_breadth, monthly_returns, relative_strength_table, rotation_closes, rrg_tail
from tradebot.strategies import STRATEGIES, BuyAndHold, as_portfolio, get_strategy
from tradebot.strategies.indicators import atr, rolling_max_prev, rsi, sma
from tradebot.strategies.swing import SwingStrategy
from tradebot.strategies.composite import checklist as wave_checklist
from tradebot.universe import display_name, expand, is_us_listed, market_of
from tradebot import i18n
from tradebot.i18n import THEMES, THEME_NAMES_EN, tr, tr_text
from tradebot.watchlist import canonical

import os as _os
try:  # Streamlit Cloud：把 secrets 里的键放进环境变量，config.load_settings 就能照常读
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            _os.environ.setdefault(str(_k), str(_v))
except Exception:
    pass
CLOUD = _os.environ.get("SWING_CLOUD", "").lower() in {"1", "true", "yes"}  # 云端只读：没有持久文件系统，关掉写文件的功能

st.set_page_config(page_title="Swing Agent", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
S = load_settings()
NY = "America/New_York"

# ---------------- 语言 / 主题偏好（存 ui_prefs.json）----------------
PREFS_PATH = ROOT / "ui_prefs.json"
_prefs = i18n.load_prefs(PREFS_PATH)
LANG = st.session_state.get("ui_lang", _prefs.get("lang", "zh"))
THEME_NAME = st.session_state.get("ui_theme", _prefs.get("theme", "浅色"))
if THEME_NAME not in THEMES:
    THEME_NAME = "浅色"
i18n.set_lang(LANG)
i18n.install(st)
if st.session_state.get("_theme_applied") != THEME_NAME:  # 运行时改主题配置（Streamlit 无官方接口，单人使用足够）
    i18n.apply_theme(st, THEME_NAME)
    st.session_state["_theme_applied"] = THEME_NAME
    st.rerun()
DARK = i18n.is_dark(THEME_NAME)
TEMPLATE = "plotly_dark" if DARK else "plotly_white"
PLOTLY_CONFIG = {"displayModeBar": False}
MAX_DAYS = {"1h": 729, "1d": 7000}

CSS = """
<style>
:root { --tb-border: rgba(120,130,150,.22); --tb-muted: rgba(100,110,130,.95); --tb-chip: rgba(120,130,150,.10); }
.block-container { padding-top: 3.2rem; padding-bottom: 2.5rem; max-width: 1650px; }
[data-testid="stAppDeployButton"], .stAppDeployButton { display: none; }
footer { visibility: hidden; }
/* 指标卡 */
[data-testid="stMetric"] { background: var(--secondary-background-color); border: 1px solid var(--tb-border); border-radius: 10px; padding: 10px 14px 8px; }
[data-testid="stMetricLabel"] p { font-size: .78rem; color: var(--tb-muted); }
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 650; line-height: 1.2; }
[data-testid="stMetricDelta"] { font-size: .8rem; }
/* 标签页 */
button[data-baseweb="tab"] { padding: 6px 16px; font-weight: 600; }
[data-testid="stExpander"] details { border-radius: 10px; }
/* 页头 */
.tb-header { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; padding: 10px 16px; margin-bottom: 8px;
             border: 1px solid var(--tb-border); border-radius: 12px; background: linear-gradient(90deg, rgba(37,99,235,.10), rgba(37,99,235,0) 60%); }
.tb-header .t { font-size: 1.2rem; font-weight: 750; letter-spacing: .2px; white-space: nowrap; }
.tb-header .s { font-size: .78rem; color: var(--tb-muted); margin-top: 2px; }
.tb-kv { display:flex; gap: 6px 14px; flex-wrap: wrap; font-size: .8rem; color: var(--tb-muted); }
.tb-kv b { color: var(--text-color); font-weight: 650; }
.tb-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px; vertical-align: 1px; }
/* 分组标题 */
.tb-section { display:flex; align-items:baseline; gap:10px; margin: 16px 0 6px; padding-left: 10px; border-left: 4px solid var(--primary-color); }
.tb-section .h { margin:0; font-size: 1.05rem; font-weight: 700; }
.tb-badge { display:inline-block; padding: 1px 9px; border-radius: 999px; font-size: .72rem; background: var(--tb-chip); color: var(--tb-muted); }
.tb-badge.ok { background: rgba(22,163,74,.12); color: #15803d; }
.tb-badge.warn { background: rgba(217,119,6,.12); color: #b45309; }
/* 芯片 */
.tb-chip { display:inline-block; padding: 2px 8px; margin: 2px 4px 2px 0; border-radius: 6px; font-size: .78rem;
           font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: var(--tb-chip); }
.tb-chip.pos { background: rgba(22,163,74,.13); color: #15803d; font-weight: 600; }
.tb-chip.queue { background: rgba(37,99,235,.10); color: #1d4ed8; }
/* 大盘小卡 */
.tb-tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(112px, 1fr)); gap: 8px; margin: 2px 0 10px; }
.tb-tile { border: 1px solid var(--tb-border); border-radius: 10px; padding: 8px 10px; background: var(--secondary-background-color); }
.tb-tile .n { font-size: .72rem; color: var(--tb-muted); white-space: nowrap; }
.tb-tile .v { font-size: 1.02rem; font-weight: 650; margin-top: 2px; white-space: nowrap; }
.tb-tile .d { font-size: .76rem; font-weight: 600; white-space: nowrap; }
.tb-tile .m { font-size: .68rem; color: var(--tb-muted); line-height: 1.3; }
.up { color: #15803d; } .down { color: #b91c1c; } .flat { color: var(--tb-muted); }
/* 表格更紧凑一点 */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
if DARK:  # 深色下把辅助色调亮一点，否则灰字和边框看不清
    st.markdown("<style>:root { --tb-border: rgba(170,180,200,.28); --tb-muted: rgba(175,185,205,.95); --tb-chip: rgba(170,180,200,.16); }"
                ".tb-header { background: linear-gradient(90deg, rgba(96,165,250,.16), rgba(96,165,250,0) 60%); }"
                ".tb-chip.pos { background: rgba(34,197,94,.22); color: #86efac; } .tb-chip.queue { background: rgba(96,165,250,.20); color: #bfdbfe; }"
                ".tb-badge.ok { background: rgba(34,197,94,.20); color: #86efac; } .tb-badge.warn { background: rgba(245,158,11,.22); color: #fcd34d; }"
                ".up { color: #4ade80; } .down { color: #f87171; }</style>", unsafe_allow_html=True)


def market_is_open(now: pd.Timestamp) -> bool:
    return market_session_open(now)  # 含休市日与 13:00 提前收盘


def _regime_text() -> str:
    r = regime_status()
    if r.get("error"):
        return "未知"
    try:
        live_on = bool(getattr(get_strategy(S.strategy, **S.strategy_params), "regime_filter", False))
    except Exception:
        live_on = False
    state = f"SPY {r['close']:.0f} {'>' if r['ok'] else '<'} 200日线 {r['sma']:.0f}"
    if not live_on:
        return f"实盘策略未启用（{state}）"
    return f"放行（{state}）" if r["ok"] else f"拦截新仓（{state}）"


def prefs_bar() -> None:
    """页面最上面一行：语言与主题切换。改了就写进 ui_prefs.json 并重跑。"""
    c_sp, c_lang, c_theme = st.columns([7, 1.1, 1.3])
    lang_opts = ["中文", "English"]
    new_lang = c_lang.segmented_control("语言", lang_opts, default="English" if LANG == "en" else "中文", key="ui_lang_ctl", label_visibility="collapsed") if hasattr(st, "segmented_control") \
        else c_lang.radio("语言", lang_opts, index=1 if LANG == "en" else 0, horizontal=True, key="ui_lang_ctl", label_visibility="collapsed")
    theme_labels = {k: (THEME_NAMES_EN[k] if LANG == "en" else k) for k in THEMES}
    new_theme = c_theme.selectbox("主题", list(THEMES), index=list(THEMES).index(THEME_NAME), key="ui_theme_ctl", label_visibility="collapsed",
                                  format_func=lambda k: "🎨 " + theme_labels[k])
    lang_code = "en" if new_lang == "English" else "zh"
    if lang_code != LANG or new_theme != THEME_NAME:
        st.session_state["ui_lang"], st.session_state["ui_theme"] = lang_code, new_theme
        i18n.save_prefs(PREFS_PATH, {"lang": lang_code, "theme": new_theme})
        st.rerun()


def page_header() -> None:
    now = pd.Timestamp.now(tz=NY)
    is_open = market_is_open(now)
    dot = f"<span class='tb-dot' style='background:{'#16a34a' if is_open else '#9ca3af'}'></span>"
    st.markdown(
        f"<div class='tb-header'><div><div class='t'>📈 Swing Agent</div>"
        f"<div class='s'>{tr('美股 swing · 只做多')} · {tr_text(S.watchlist.describe())}</div></div>"
        f"<div class='tb-kv'><span>{dot}{tr('市场')} <b>{tr('开盘中') if is_open else tr('休市')}</b></span>"
        f"<span>{tr('纽约')} <b>{now:%m-%d %H:%M}</b></span><span>{tr('模式')} <b>{S.mode}</b></span><span>{tr('周期')} <b>{S.interval}</b></span>"
        f"<span>{tr('策略')} <b>{tr(STRAT_LABEL.get(S.strategy, S.strategy)) if 'STRAT_LABEL' in globals() else S.strategy}</b></span>"
        f"<span>{tr('起步')} <b>{tr('只跟新入场') if S.entry_mode == 'fresh' else tr('对齐策略')}</b></span>"
        f"<span>{tr('大盘过滤')} <b>{tr_text(_regime_text())}</b></span>"
        f"<span>{tr('Claude 复核')} <b>{tr('开') if S.agent_enabled else tr('关')}</b></span></div></div>",
        unsafe_allow_html=True,
    )


def section_header(title: str, badges: list[tuple[str, str]]) -> None:
    """分组标题：左侧色条 + 圆角徽章。badges = [(文字, 样式类 '' / 'ok' / 'warn')]"""
    b = "".join(f"<span class='tb-badge {cls}'>{tr_text(txt)}</span>" for txt, cls in badges)
    st.markdown(f"<div class='tb-section'><span class='h'>{tr_text(title)}</span>{b}</div>", unsafe_allow_html=True)


def chips(items: list[tuple[str, str]]) -> str:
    return "".join(f"<span class='tb-chip {cls}'>{txt}</span>" for txt, cls in items)


def fmt_price(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    return f"{v:,.0f}" if v >= 1000 else (f"{v:,.1f}" if v >= 100 else f"{v:.2f}")


def market_tiles(snap: pd.DataFrame) -> None:
    cards = []
    for _, r in snap.iterrows():
        d1 = r["涨跌1日"]
        cls = "flat" if pd.isna(d1) or abs(d1) < 0.005 else ("up" if d1 > 0 else "down")
        arrow = "" if pd.isna(d1) else ("▲" if d1 > 0 else ("▼" if d1 < 0 else "•"))
        d1_txt = "n/a" if pd.isna(d1) else f"{arrow} {abs(d1):.2f}%"
        meta = "" if pd.isna(r["涨跌5日"]) else f"5日 {r['涨跌5日']:+.1f}% · 20日 {r['涨跌20日']:+.1f}% · {r['趋势']}"
        cards.append(f"<div class='tb-tile' title='距52周高 {r['距52周高']:+.1f}%'><div class='n'>{r['名称']} <span style='opacity:.55;font-size:.66rem'>{r['代码']}</span></div><div class='v'>{fmt_price(r['最新'])}</div>"
                     f"<div class='d {cls}'>{d1_txt}</div><div class='m'>{meta}</div></div>")
    st.markdown(f"<div class='tb-tiles'>{''.join(cards)}</div>", unsafe_allow_html=True)


# ---------------- 数据与工具 ----------------
@st.cache_data(ttl=900, show_spinner="拉取行情 ...")
def load_data(symbols: tuple[str, ...], interval: str, days: int) -> dict[str, pd.DataFrame]:
    # 日线文件缓存 12 小时（收盘后才会有新 bar），小时线 30 分钟
    return fetch_bars(list(symbols), interval, days, cache_dir=S.cache_dir, max_cache_age_hours=12 if interval == "1d" else 0.5)


def refresh_data(symbols: tuple[str, ...], interval: str, days: int) -> None:
    fetch_bars(list(symbols), interval, days, cache_dir=S.cache_dir, max_cache_age_hours=0)
    load_data.clear()


def read_decisions() -> list[dict]:
    p = S.log_dir / "decisions.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def strategy_param_inputs(cls, defaults: dict, key_prefix: str) -> dict:
    """根据策略 dataclass 的 int/float/bool 字段自动生成输入控件，新策略无需改界面。"""
    params: dict = {}
    help_map = getattr(cls, "PARAM_HELP", {})
    for f in dataclasses.fields(cls):
        t = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
        if f.name == "name" or t not in {"int", "float", "bool"}:
            continue
        default = defaults.get(f.name, f.default)
        key, h = f"{key_prefix}_{cls.__name__}_{f.name}", help_map.get(f.name)
        if t == "bool":
            params[f.name] = bool(st.checkbox(f.name, value=bool(default), key=key, help=h))
        elif t == "int":
            params[f.name] = int(st.number_input(f.name, value=int(default), step=1, key=key, help=h))
        else:
            params[f.name] = float(st.number_input(f.name, value=float(default), step=0.01, format="%.3f", key=key, help=h))
    return params


def pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def default_days(interval: str) -> int:
    return S.backtest_days if interval == S.interval else DEFAULT_BACKTEST_DAYS[interval]


SWING_NAMES = ["trend_pullback", "breakout", "post_earnings", "oversold_bounce", "bottom_recovery", "relative_momentum", "high52_breakout", "main_wave"]
STRAT_LABEL = {"trend_pullback": "趋势回调", "breakout": "突破", "post_earnings": "财报动量", "oversold_bounce": "超跌反弹",
               "bottom_recovery": "底部回升", "relative_momentum": "相对强度", "high52_breakout": "52周新高", "main_wave": "主升浪",
               "ma_cross": "均线交叉", "buy_and_hold": "满仓"}


@st.cache_data(ttl=900, show_spinner=False)
def regime_status() -> dict:
    """大盘过滤状态：SPY 前一日收盘 vs 200 日线。"""
    try:
        spy = load_data(("SPY",), "1d", 400)["SPY"]
        spy = drop_incomplete_last_bar(spy, "1d")
        c = spy["close"]
        ma = c.rolling(200).mean()
        return {"ok": bool(c.iloc[-1] > ma.iloc[-1]), "close": float(c.iloc[-1]), "sma": float(ma.iloc[-1]), "date": spy.index[-1]}
    except Exception as e:
        return {"ok": True, "error": str(e)}


def strat_params(name: str) -> dict:
    """配置里选中的策略用 .env 参数，其他用默认。"""
    return dict(S.strategy_params) if name == S.strategy else {}


def strat_specs(names: list[str]) -> tuple:
    return tuple((n, json.dumps(strat_params(n), sort_keys=True)) for n in names)


def lookback_for(interval: str) -> int:
    return S.lookback_days if interval == S.interval else DEFAULT_LOOKBACK_DAYS[interval]


@st.cache_data(ttl=900, show_spinner="计算信号 ...")
def compute_signals(symbols: tuple[str, ...], groups_key: str, interval: str, days: int, specs: tuple) -> dict:
    """全部标的的指标快照 + 每个策略在最后一根完整 bar 的纯信号（入场 / 出场条件），不涉及任何仓位。"""
    groups: dict[str, list[str]] = json.loads(groups_key)
    raw = load_data(symbols, interval, days)
    data = {sym: drop_incomplete_last_bar(df, interval) for sym, df in raw.items()}
    cal = EarningsCalendar(cache_dir=S.cache_dir)
    now = pd.Timestamp.now(tz=NY)
    unit = "日" if interval == "1d" else "小时"

    rows = []
    for sym, df in data.items():
        c = df["close"]
        px = float(c.iloc[-1])
        last_raw = float(raw[sym]["close"].iloc[-1])

        def ret(k: int) -> float:
            return (px / float(c.iloc[-1 - k]) - 1) * 100 if len(c) > k else np.nan

        last_ts = df.index[-1]

        def lo_hi(days_: int) -> tuple[float, float]:
            w = df[df.index >= last_ts - pd.Timedelta(days=days_)]
            return float(w["low"].min()), float(w["high"].max())

        lo2, hi2 = lo_hi(61)
        lo5, hi5 = lo_hi(152)
        s50, s200 = sma(c, 50).iloc[-1], sma(c, 200).iloc[-1]
        if px > s200 and s50 > s200:
            trend = "↑ 多头"
        elif px < s200 and s50 < s200:
            trend = "↓ 空头"
        else:
            trend = "→ 震荡"
        nxt = cal.next_after(sym, now)
        days_to = int((nxt - now).days) if nxt is not None else None
        rows.append({
            "标的": sym, "市场": market_of(sym) + ("" if is_us_listed(sym) else " 仅监控"),
            "组": "/".join(g for g, m in groups.items() if sym in m), "现价": round(px, 2),
            "最新价": round(last_raw, 2), "盘中%": (last_raw / px - 1) * 100,
            f"涨跌1{unit}": ret(1), f"涨跌5{unit}": ret(5), f"涨跌20{unit}": ret(20),
            "趋势": trend, "RSI14": float(rsi(c, 14).iloc[-1]),
            "距55高": (px / float(rolling_max_prev(c, 55).iloc[-1]) - 1) * 100,
            "2月最低": round(lo2, 2), "2月最高": round(hi2, 2), "5月最低": round(lo5, 2), "5月最高": round(hi5, 2),
            "ATR%": float(atr(df, 14).iloc[-1]) / px * 100,
            "财报": (nxt.strftime("%m-%d") + f"（{days_to}天）") if nxt is not None else "",
            "距财报天": days_to,
        })
    table = pd.DataFrame(rows)

    weights, today, in_state, hist_entry, hist_exit, hist_state = {}, {}, {}, {}, {}, {}
    KEEP = 20  # 逐日历史保留的 bar 数
    for name, pjson in specs:
        strat = get_strategy(name, **json.loads(pjson))
        W = as_portfolio(strat).target_weights(data)  # 只给行情页画持仓区间用，表里不显示
        weights[name] = W
        label = STRAT_LABEL[name]
        col, exits, today_list, state_list = [], [], [], []
        ent_h, exit_h, state_h = {}, {}, {}
        for sym, df in data.items():
            txt, exit_flag = "", False
            if isinstance(strat, SwingStrategy):
                sig = strat.compute(sym, df)
                ent = sig["entry"].fillna(False).astype(bool)
                ex_s = sig["exit"].fillna(False).astype(bool)
                exit_flag = bool(ex_s.iloc[-1])
                if getattr(strat, "signal_kind", "event") == "state":
                    # 状态型（如相对强度）：▲ 新进 = 今天刚进入状态；● 在列 = 持续处于状态
                    onset = ent & ~ent.shift(1).fillna(False).astype(bool)
                    ent_h[sym], exit_h[sym] = onset.iloc[-KEEP:], ex_s.iloc[-KEEP:]
                    state_h.setdefault(sym, ent.iloc[-KEEP:])
                    if bool(onset.iloc[-1]):
                        txt = "▲ 新进"
                        today_list.append(sym)
                    elif bool(ent.iloc[-1]):
                        txt = "● 在列"
                        state_list.append(sym)
                    else:
                        recent = onset.iloc[-6:-1]
                        hits = np.where(recent.values)[0]
                        if len(hits):
                            txt = f"▲ {len(recent) - int(hits[-1])}{unit}前"
                else:
                    ent_h[sym], exit_h[sym] = ent.iloc[-KEEP:], ex_s.iloc[-KEEP:]
                    if bool(ent.iloc[-1]):
                        txt = "▲ 今日"
                        today_list.append(sym)
                    else:
                        recent = ent.iloc[-6:-1]
                        hits = np.where(recent.values)[0]
                        if len(hits):
                            txt = f"▲ {len(recent) - int(hits[-1])}{unit}前"
            else:  # 单标的策略（均线交叉 / 满仓）：用权重 0->正 当入场，正->0 当出场
                ws = W[sym]
                pos = (ws > 0).astype(int)
                chg = pos.diff().fillna(pos)
                w, w0 = float(ws.iloc[-1]), float(ws.iloc[-2]) if len(W) > 1 else 0.0
                if w > 0 and w0 == 0:
                    txt = "▲ 今日"
                    today_list.append(sym)
                exit_flag = w == 0 and w0 > 0
                ent_h[sym], exit_h[sym] = (chg == 1).iloc[-KEEP:], (chg == -1).iloc[-KEEP:]
            col.append(txt)
            exits.append(exit_flag)
        table[label] = col
        table[f"{label}·出场"] = exits
        today[name] = today_list
        in_state[name] = state_list
        hist_entry[name] = pd.DataFrame(ent_h).fillna(False).astype(bool)
        hist_exit[name] = pd.DataFrame(exit_h).fillna(False).astype(bool)
        hist_state[name] = pd.DataFrame(state_h).fillna(False).astype(bool) if state_h else None
    # 聚合：共振 = 当天有多少个策略给出信号（▲ 或 ●）；主升浪阶段
    strat_labels = [STRAT_LABEL[n] for n, _ in specs]
    # 只数“今天有效”的信号：▲ 今日 / ▲ 新进 / ● 在列；几天前的信号不算
    _active = lambda v: isinstance(v, str) and (v.startswith("▲ 今日") or v.startswith("▲ 新进") or v.startswith("● 在列"))
    table["共振"] = table[strat_labels].map(_active).sum(axis=1).astype(int) if strat_labels else 0
    stages = []
    for sym, df in data.items():
        try:
            stages.append(wave_checklist(df)["stage"] if len(df) >= 260 else "—")
        except Exception:
            stages.append("—")
    table["阶段"] = stages
    return {"table": table, "weights": weights, "today": today, "in_state": in_state, "hist_entry": hist_entry, "hist_exit": hist_exit,
            "hist_state": hist_state, "last_bar": max(df.index[-1] for df in data.values()), "unit": unit}


def signals_for(interval: str, names: list[str]) -> dict:
    wl = S.watchlist
    return compute_signals(tuple(S.symbols), json.dumps(wl.groups, ensure_ascii=False, sort_keys=True), interval, lookback_for(interval), strat_specs(names))


def validate_symbols(symbols: list[str]) -> list[str]:
    """返回拿不到行情的代码。"""
    bad = []
    for sym in symbols:
        try:
            fetch_bars([sym], "1d", 10, cache_dir=None)
        except Exception:
            bad.append(sym)
    return bad


def render_signal_table(df: pd.DataFrame, strat_cols: list[str], unit: str, hide_group: bool = False, key: str | None = None):
    """画信号表；给了 key 就允许单击选中一行（返回选择事件），用于打开标的详情。"""
    show = df.drop(columns=["组"]) if hide_group and "组" in df.columns else df
    pct_cols = [c for c in ["盘中%", f"涨跌1{unit}", f"涨跌5{unit}", f"涨跌20{unit}"] if c in show.columns]
    strat_cols = [c for c in strat_cols if c in show.columns]
    styler = show.style
    if strat_cols:
        styler = styler.map(_color_status, subset=strat_cols)
    if "阶段" in show.columns:
        styler = styler.map(_color_stage, subset=["阶段"])
    if "共振" in show.columns:
        styler = styler.map(lambda v: "color:#15803d;font-weight:700" if isinstance(v, (int, np.integer)) and v >= 2 else "", subset=["共振"])
    if pct_cols:
        styler = styler.map(lambda v: "color:#15803d" if isinstance(v, float) and v > 0 else ("color:#b91c1c" if isinstance(v, float) and v < 0 else ""), subset=pct_cols)
    select_kw = {"on_select": "rerun", "selection_mode": "single-row", "key": key} if key else {}
    return st.dataframe(
        styler, width="stretch", hide_index=True, height=min(42 + 35 * len(show), 900), **select_kw,
        column_config={
            **{c: st.column_config.NumberColumn(format="%+.2f%%") for c in pct_cols},
            "最新价": st.column_config.NumberColumn(format="%.2f", help="含未走完的当前 bar 的最新收盘；“现价”是信号用的最后一根完整 bar 的收盘"),
            "2月最低": st.column_config.NumberColumn(format="%.2f", help="最近 61 个日历日内 bar 的最低价"),
            "2月最高": st.column_config.NumberColumn(format="%.2f", help="最近 61 个日历日内 bar 的最高价"),
            "5月最低": st.column_config.NumberColumn(format="%.2f", help="最近 152 个日历日内 bar 的最低价"),
            "5月最高": st.column_config.NumberColumn(format="%.2f", help="最近 152 个日历日内 bar 的最高价"),
            "距55高": st.column_config.NumberColumn(format="%.1f%%"),
            "ATR%": st.column_config.NumberColumn(format="%.2f%%"),
            "RSI14": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f", width="small"),
            "现价": st.column_config.NumberColumn(format="%.2f", width="small"),
            "标的": st.column_config.TextColumn(width="small"),
            "趋势": st.column_config.TextColumn(width="small"),
            "距财报天": st.column_config.NumberColumn(format="%d"),
            "财报": st.column_config.TextColumn(width="small"),
            "共振": st.column_config.NumberColumn(format="%d", width="small", help="当天有几个策略同时给出信号（▲ 新入场 / ● 在列）"),
            "阶段": st.column_config.TextColumn(width="small", help="主升浪阶段：蓄势 → 启动 → 主升 / 整理 → 过热 → 衰竭；详情弹窗里有 8 条逐项体检"),
        },
    )


STAGE_COLOR = {"主升": "color:#15803d;font-weight:700", "启动": "color:#2563eb;font-weight:600", "整理": "color:#0f766e", "蓄势": "color:#b45309", "过热": "color:#dc2626;font-weight:600", "衰竭": "color:#7f1d1d"}


def _color_stage(v: str) -> str:
    return STAGE_COLOR.get(v, "") if isinstance(v, str) else ""


def _color_status(v: str) -> str:
    if not v:
        return ""
    if v.startswith("▲ 今日") or v.startswith("▲ 新进"):
        return "color:#15803d;font-weight:700"
    if v.startswith("● 在列"):
        return "color:#15803d;opacity:.55"
    if v.startswith("▲"):
        return "color:#15803d;opacity:.65"
    if "▼" in v:
        return "color:#b91c1c;font-weight:600"
    return ""


# ---------------- 个股详情 / 图表 / 宏观 ----------------
@st.cache_data(ttl=3 * 3600, show_spinner=False)
def cached_news(sym: str) -> list[dict]:
    return fetch_news(sym, 15, S.cache_dir)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_info(sym: str) -> dict:
    return fetch_info(sym, S.cache_dir)


@st.cache_data(ttl=6 * 3600, show_spinner="Claude 正在整理利好 / 利空 ...")
def cached_catalysts(sym: str, titles_key: str, price_note: str) -> dict:
    try:
        from tradebot.agent.research import Researcher

        return Researcher(model=S.agent_model).catalysts(sym, cached_news(sym), cached_info(sym), price_note).model_dump()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=6 * 3600, show_spinner="Claude 正在联网搜索宏观事件（约一分钟）...")
def cached_macro_brief(focus_key: str, day_key: str) -> dict:
    try:
        from tradebot.agent.research import Researcher

        return Researcher(model=S.agent_model).macro_brief(json.loads(focus_key)).model_dump()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=900, show_spinner=False)
def cached_macro_closes() -> pd.DataFrame:
    return macro_closes(S.cache_dir)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_upcoming_earnings(symbols: tuple[str, ...], groups_key: str) -> pd.DataFrame:
    return upcoming_earnings(list(symbols), EarningsCalendar(cache_dir=S.cache_dir), json.loads(groups_key), days=30)


STRAT_COLOR = {"trend_pullback": "#2563eb", "breakout": "#7c3aed", "post_earnings": "#f59e0b", "oversold_bounce": "#0891b2",
               "bottom_recovery": "#16a34a", "relative_momentum": "#db2777", "high52_breakout": "#b45309", "main_wave": "#dc2626",
               "ma_cross": "#6b7280", "buy_and_hold": "#9ca3af"}


def build_price_fig(sym: str, interval: str, days: int, overlays, height: int = 560,
                    show_raw: bool = True, show_trades: bool = True, show_holding: bool = True):
    """K 线 + SMA50/200 + 成交量；overlays 是策略名列表（或单个名 / "无"），每个策略一种颜色：
    △ 空心 = 那天满足入场条件；▲ 实心 = 策略回放中入场；▼ = 回放中退出；底色 = 回放中的模拟持仓期。"""
    if isinstance(overlays, str):
        overlays = [] if overlays == "无" else [overlays]
    overlays = [o for o in (overlays or []) if o in STRATEGIES]
    full = load_data((sym,), interval, default_days(interval))[sym]
    df = full[full.index >= full.index[-1] - pd.Timedelta(days=days)]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name=sym), 1, 1)
    for n_, color in ((50, "#2563eb"), (200, "#f59e0b")):
        fig.add_trace(go.Scatter(x=df.index, y=full["close"].rolling(n_).mean().reindex(df.index), name=f"SMA{n_}", line=dict(color=color, width=1, dash="dot")), 1, 1)
    notes = []
    if overlays:
        res = signals_for(interval, overlays)
        for i, ov in enumerate(overlays):
            lab, col = STRAT_LABEL[ov], STRAT_COLOR.get(ov, "#6b7280")
            strat = get_strategy(ov, **strat_params(ov))
            if show_raw and isinstance(strat, SwingStrategy):
                sig = strat.compute(sym, full).reindex(df.index)
                ent = sig["entry"].fillna(False).astype(bool)
                if getattr(strat, "signal_kind", "event") == "state":
                    ent = ent & ~ent.shift(1).fillna(False).astype(bool)  # 状态型只标“新进”
                raw_entry = df[ent]
                fig.add_trace(go.Scatter(x=raw_entry.index, y=raw_entry["low"] * (0.985 - 0.012 * i), mode="markers", name=f"△ {lab}·满足入场条件",
                                         marker=dict(symbol="triangle-up-open", color=col, size=9),
                                         hovertemplate=f"%{{x|%m-%d}} {lab}：满足入场条件<extra></extra>"), 1, 1)
            W = res["weights"][ov]
            if sym in W.columns and (show_trades or show_holding):
                h = (W[sym].reindex(df.index).fillna(0.0) > 0).astype(int)
                d = h.diff()
                starts = list(df.index[(d == 1) | (d.isna() & (h == 1))])
                real_ends = list(df.index[d == -1])
                spans_end = real_ends + ([df.index[-1]] if len(starts) > len(real_ends) else [])
                if show_holding:
                    for a_, b_ in zip(starts, spans_end):
                        fig.add_vrect(x0=a_, x1=b_, fillcolor=col, opacity=0.12, line_width=0, row=1, col=1)
                    if starts:
                        fig.add_trace(go.Scatter(x=[starts[0]], y=[df["low"].min()], mode="markers", name=f"▮ {lab}·模拟持仓期",
                                                 marker=dict(symbol="square", color=col, opacity=0.35, size=10), hoverinfo="skip"), 1, 1)
                if show_trades:
                    ent_, ex_ = df.loc[starts], df.loc[real_ends]
                    fig.add_trace(go.Scatter(x=ent_.index, y=ent_["low"] * (0.965 - 0.012 * i), mode="markers", name=f"▲ {lab}·模拟入场",
                                             marker=dict(symbol="triangle-up", color=col, size=12),
                                             hovertemplate=f"%{{x|%m-%d}} {lab}：回放入场<extra></extra>"), 1, 1)
                    if len(ex_):
                        fig.add_trace(go.Scatter(x=ex_.index, y=ex_["high"] * (1.035 + 0.012 * i), mode="markers", name=f"▼ {lab}·模拟退出",
                                                 marker=dict(symbol="triangle-down", color=col, size=12),
                                                 hovertemplate=f"%{{x|%m-%d}} {lab}：回放退出<extra></extra>"), 1, 1)
        notes.append("叠加 " + "、".join(STRAT_LABEL[o] for o in overlays) + "：△ 空心 = 那天收盘满足入场条件（信号，相对强度只标新进）；▲ 实心 = 策略回放中真的进了（有空槽位才会进）；"
                     "▼ = 回放中退出；底色 = 回放中的模拟持仓期；每个策略一种颜色。成交按下一根开盘。这是策略回放，不是你的账户。")
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="成交量", marker_color="#9ca3af"), 2, 1)
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=40, r=20, t=10, b=30), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.06, font=dict(size=10)))
    breaks = [dict(bounds=["sat", "mon"])]
    if interval == "1h":
        breaks.append(dict(bounds=[16, 9.5], pattern="hour"))
    fig.update_xaxes(rangebreaks=breaks)
    return fig, df, ("  · " + notes[0]) if notes else ""


def overlay_controls(prefix: str, compact: bool = False):
    """叠加哪些策略 + 显示什么。选择记在会话里，切换标的时保持。"""
    default = [S.strategy] if S.strategy in SWING_NAMES else []
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1]) if not compact else st.columns([3, 1, 1, 1])
    ovs = c1.multiselect("叠加策略信号", SWING_NAMES, default=default, key=f"{prefix}_overlays",
                         format_func=lambda n: STRAT_LABEL[n] + ("（实盘）" if n == S.strategy else ""))
    show_raw = c2.checkbox("△ 满足条件", value=True, key=f"{prefix}_raw", help="那天收盘满足入场条件的 bar（空心三角）")
    show_trades = c3.checkbox("▲▼ 回放进出", value=True, key=f"{prefix}_trades", help="策略回放里真实发生的入场 / 退出（实心三角）")
    show_holding = c4.checkbox("▮ 持仓底色", value=True, key=f"{prefix}_hold", help="回放中的模拟持仓期")
    return ovs, show_raw, show_trades, show_holding


IMPORTANCE_COLOR = {"高": "#b91c1c", "中": "#b45309", "低": "#6b7280"}
IMPACT_COLOR = {"利好": "#15803d", "利空": "#b91c1c", "中性": "#6b7280"}


def _badge(text: str, color: str) -> str:
    return f"<span style='background:{color};color:white;border-radius:4px;padding:0 6px;font-size:0.8em'>{text}</span>"


def render_headlines(news: list[dict], with_summary: bool = True) -> None:
    if not news:
        st.caption("没有拿到新闻。")
        return
    for n in news:
        t = (n.get("time") or "")[:16].replace("T", " ")
        title = f"[{n['title']}]({n['url']})" if n.get("url") else n["title"]
        st.markdown(f"- {title}  <span style='color:#6b7280;font-size:0.85em'>{t} · {n.get('source', '')}</span>", unsafe_allow_html=True)
        if with_summary and n.get("summary"):
            st.caption(n["summary"][:180] + ("…" if len(n["summary"]) > 180 else ""))


def render_catalysts(rep: dict) -> None:
    st.markdown(f"**{rep.get('one_liner', '')}**")
    c1, c2 = st.columns(2, gap="large")
    for col, title, key in ((c1, "🟢 利好", "bullish"), (c2, "🔴 利空", "bearish")):
        with col:
            st.markdown(f"##### {title}")
            items = rep.get(key) or []
            if not items:
                st.caption("无")
            for c in items:
                meta = " · ".join(x for x in [c.get("date", ""), c.get("source", "")] if x)
                st.markdown(f"- {_badge(c['importance'], IMPORTANCE_COLOR.get(c['importance'], '#6b7280'))} **{c['headline']}**  \n"
                            f"  <span style='color:#6b7280'>{c['why']}{(' · ' + meta) if meta else ''}</span>", unsafe_allow_html=True)
    if rep.get("watch"):
        st.markdown("##### 👀 接下来盯住")
        for w in rep["watch"]:
            st.markdown(f"- {w}")


def render_macro_brief(b: dict) -> None:
    st.markdown(f"**环境判断**（{b.get('as_of', '')}）：{b.get('regime', '')}")
    m1, m2 = st.columns([1, 1.2], gap="large")
    with m1:
        st.markdown("**未来事件**")
        ev = b.get("upcoming_events") or []
        if ev:
            st.dataframe(pd.DataFrame(ev).rename(columns={"date": "日期", "event": "事件", "importance": "重要性", "why_it_matters": "为什么重要"}),
                         hide_index=True, width="stretch", height=min(42 + 35 * len(ev), 320))
        else:
            st.caption("无")
    with m2:
        st.markdown("**近期要闻**")
        for n in b.get("recent_news") or []:
            meta = " · ".join(x for x in [n.get("date", ""), n.get("source", "")] if x)
            st.markdown(f"- {_badge(n['impact'], IMPACT_COLOR.get(n['impact'], '#6b7280'))} **{n['title']}**  \n"
                        f"  <span style='color:#6b7280'>{n['why']} · 影响：{'、'.join(n.get('affected') or [])}{(' · ' + meta) if meta else ''}</span>",
                        unsafe_allow_html=True)


def _close_detail() -> None:
    """弹窗被关闭（X / ESC）时清掉状态，否则下次重跑会再弹出来。
    同时给信号表换一个新的 key，让表格里的选中行清零，下次点任何一行（包括同一行）都能直接弹。"""
    st.session_state["detail_symbol"] = None
    st.session_state["detail_pick"] = "—"
    st.session_state["last_click"] = None
    st.session_state["tbl_version"] = st.session_state.get("tbl_version", 0) + 1


def request_detail(sym: str) -> None:
    """任何页签都可以请求打开详情；真正的弹窗在脚本末尾统一渲染（一次运行只能开一个）。"""
    st.session_state["detail_symbol"] = sym


@st.dialog("标的详情", width="large", on_dismiss=_close_detail)
def show_detail(sym: str, row: dict | None) -> None:
    info = cached_info(sym)
    groups = "/".join(S.watchlist.group_of(sym))
    cn = display_name(sym)
    tag = "" if is_us_listed(sym) else f" · {market_of(sym)}股，仅监控不交易"
    st.markdown(f"### {sym} · {cn + ' · ' if cn else ''}{info.get('longName', '')}  <span style='color:#6b7280;font-size:0.7em'>{groups}{tag}</span>", unsafe_allow_html=True)
    price_note = ""
    if row:
        unit_cols = [c for c in row if c.startswith("涨跌1")]
        u = unit_cols[0] if unit_cols else None
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("现价", f"{row['现价']:.2f}", f"{row[u]:+.2f}%" if u else None)
        k2.metric("20 根 bar", f"{row.get(u.replace('1', '20'), float('nan')):+.1f}%" if u else "n/a")
        k3.metric("趋势", row.get("趋势", ""))
        k4.metric("RSI14", f"{row.get('RSI14', float('nan')):.0f}")
        earn_txt = (row.get("财报") or "").split("（")[0]
        k5.metric("下次财报", earn_txt or "—", f"{int(row['距财报天'])} 天" if row.get("距财报天") is not None and pd.notna(row.get("距财报天")) else None, delta_color="off")
        price_note = f"Price {row['现价']:.2f}, 1-bar {row[u]:+.2f}%, trend {row.get('趋势', '')}, RSI14 {row.get('RSI14', float('nan')):.0f}, next earnings {earn_txt or 'n/a'}" if u else ""
    try:
        _df_full = load_data((sym,), "1d", 750)[sym]
        _df_full = drop_incomplete_last_bar(_df_full, "1d")
        ck = wave_checklist(_df_full) if len(_df_full) >= 260 else None
    except Exception:
        ck = None
    if ck:
        stage_style = STAGE_COLOR.get(ck["stage"], "color:#6b7280")
        items = "".join(f"<span class='tb-chip' style='{'background:rgba(22,163,74,.13);color:#15803d' if ok else 'background:rgba(185,28,28,.10);color:#b91c1c'}'>{'✓' if ok else '✗'} {lab}</span>"
                        for lab, ok in ck["items"])
        adx_txt = f"ADX {ck['adx']:.0f}{'↑' if ck['adx_up'] else '↓'}" if pd.notna(ck["adx"]) else "ADX n/a"
        st.markdown(f"<div style='margin:4px 0 6px'><b>主升浪体检</b>　阶段 <span style='{stage_style}'>{ck['stage']}</span>　模板 <b>{ck['score']}/8</b>　{adx_txt}　"
                    f"乖离 SMA50 {ck['ext_pct']:+.1f}%　RSI {ck['rsi']:.0f}</div>{items}", unsafe_allow_html=True)
    news = cached_news(sym)
    t1, t2, t3, t4 = st.tabs(["催化剂", "基本面", "走势", "新闻原文"])
    with t1:
        if has_credentials():
            rep = cached_catalysts(sym, json.dumps([n["title"] for n in news[:15]]), price_note)
            if rep.get("error"):
                st.warning(f"Claude 调用失败：{rep['error']}")
                render_headlines(news, with_summary=False)
            else:
                render_catalysts(rep)
                st.caption("由 Claude 根据下面“新闻原文”和基本面整理，只是研究摘要，不是买卖建议。")
        else:
            st.info("未检测到 Anthropic 凭证。设置 ANTHROPIC_API_KEY 或运行 `ant auth login` 后，这里会由 Claude 把新闻分成利好 / 利空并标注重要性。下面先看原始新闻。")
            render_headlines(news, with_summary=False)
    with t2:
        rows = format_info(info)
        if rows:
            st.dataframe(pd.DataFrame(rows, columns=["指标", "值"]), hide_index=True, width="stretch", height=42 + 35 * len(rows))
        else:
            st.caption("没有拿到基本面数据。")
        if info.get("longBusinessSummary"):
            with st.expander("公司简介"):
                st.write(info["longBusinessSummary"])
    with t3:
        ovs, show_raw, show_trades, show_holding = overlay_controls("detail")
        days_ov = st.select_slider("显示区间", options=[90, 180, 365, 730], value=180 if S.interval == "1d" else 90, key="detail_days",
                                   format_func=lambda d: f"{d} 天")
        try:
            fig, _, note = build_price_fig(sym, S.interval, days_ov, ovs, height=430, show_raw=show_raw, show_trades=show_trades, show_holding=show_holding)
            if note:
                st.caption(note.strip(" ·"))
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
        except Exception as e:
            st.error(f"图表失败：{e}")
    with t4:
        render_headlines(news)


def row_for(sym: str) -> dict | None:
    table = st.session_state.get("signal_table")
    if table is None:
        return None
    hit = table[table["标的"] == sym]
    return hit.iloc[0].to_dict() if len(hit) else None


# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.markdown("### 📈 Swing Agent")
    st.caption("美股 · swing · 只做多")
    if st.button("刷新行情缓存", width="stretch"):
        refresh_data(tuple(S.symbols), S.interval, default_days(S.interval))
        st.toast("已重新下载行情")
    auto = st.selectbox("自动刷新", ["关", "5 分钟", "15 分钟", "60 分钟"], index=0, key="auto_refresh",
                        help="到点后重新拉行情、重算信号并刷新整页；日线信号本身每天收盘后才会变")
    if auto != "关":
        _minutes = int(auto.split()[0])

        @st.fragment(run_every=dt.timedelta(minutes=_minutes))
        def _auto_refresh_tick():
            last = st.session_state.get("_auto_refresh_ts")
            now_ts = time.time()
            if last is None:
                st.session_state["_auto_refresh_ts"] = now_ts
            elif now_ts - last >= _minutes * 60 - 10:
                st.session_state["_auto_refresh_ts"] = now_ts
                load_data.clear()
                compute_signals.clear()
                st.rerun(scope="app")
            st.caption(f"自动刷新 {auto} · 上次 {pd.Timestamp.fromtimestamp(st.session_state['_auto_refresh_ts'], tz=NY):%H:%M:%S}")

        _auto_refresh_tick()
    if CLOUD:
        st.info("云端只读版：标的池、事件日历、模拟盘按钮已关闭，数据每次冷启动重新拉取。完整功能请在本机运行。", icon="☁️")
    with st.expander(f"编辑标的池 · {S.watchlist.describe()}", expanded=False) if not CLOUD else st.container():
        wl = S.watchlist
        if CLOUD:
            st.caption(f"标的池（只读）：{S.watchlist.describe()}")
        st.caption("已保存到 watchlist.json，.env 的 SYMBOLS 不再生效" if wl.source == "file" else "当前来自 .env 的 SYMBOLS；第一次保存后写入 watchlist.json，之后以文件为准")
        grp = st.selectbox("板块", list(wl.groups) or ["（无）"], key="edit_group") if not CLOUD else None
        new_grp = st.text_input("新建板块", placeholder="例如 能源 / 自选", key="new_group") if not CLOUD else ""
        if not CLOUD and st.button("新建板块", disabled=not new_grp.strip(), width="stretch"):
            try:
                wl.add_group(new_grp)
                wl.save()
                st.rerun()
            except ValueError as e:
                st.error(str(e))
        if grp in wl.groups and not CLOUD:
            members = wl.groups[grp]
            keep = st.multiselect("成员（取消勾选即删除）", members, default=members, key=f"members_{grp}_{len(members)}_{hash(tuple(members))}")
            add_text = st.text_input("添加标的", placeholder="逗号分隔，BRK-B 会转成 BRK.B", key=f"add_{grp}")
            cur_on = wl.enabled.get(grp, True)
            on = st.checkbox("启用该板块", value=cur_on, key=f"on_{grp}_{cur_on}")
            if st.button("保存该板块", type="primary", width="stretch", key=f"save_{grp}"):
                new_syms = [canonical(t) for t in add_text.split(",") if t.strip()]
                new_syms = [x for x in new_syms if x not in keep]
                with st.spinner("检查新标的行情 ..."):
                    bad = validate_symbols(new_syms)
                if bad:
                    st.error(f"拿不到行情，未保存：{', '.join(bad)}")
                else:
                    wl.set_members(grp, list(keep) + new_syms)
                    wl.set_enabled(grp, on)
                    wl.save()
                    st.rerun()
            del_ok = st.checkbox("确认删除该板块", key=f"del_{grp}")
            if st.button("删除板块", disabled=not del_ok, width="stretch"):
                wl.remove_group(grp)
                wl.save()
                st.rerun()
    if not CLOUD:
        with st.expander("连接设置（券商 / 密钥）", expanded=False):
            from tradebot.config import update_env

            st.caption("密钥只写进本机项目目录的 .env，不回显、不上传。模拟盘和实盘的 key 是两组，别混。")
            mode_pick = st.selectbox("执行模式", ["local", "alpaca"], index=["local", "alpaca"].index(S.mode if S.mode in ("local", "alpaca") else "local"), key="conn_mode",
                                     format_func=lambda m: {"local": "local · 本地模拟撮合（无需账户）", "alpaca": "alpaca · Alpaca 模拟盘 / 实盘"}[m])
            has_key = bool(S.alpaca_api_key) and bool(S.alpaca_secret_key)
            st.markdown(f"当前 Alpaca 密钥：{'已填（••••' + S.alpaca_api_key[-4:] + '）' if has_key else '未填'}")
            new_key = st.text_input("Alpaca API Key", type="password", key="conn_key", placeholder="留空则保留现有")
            new_secret = st.text_input("Alpaca Secret Key", type="password", key="conn_secret", placeholder="留空则保留现有")
            paper_pick = st.checkbox("模拟盘（ALPACA_PAPER=true）", value=S.alpaca_paper, key="conn_paper", help="取消勾选 = 实盘，会用真钱下单")
            if not paper_pick:
                st.warning("你选择了实盘。请确认填的是 Live 的 key，并且模拟盘已经跑过足够长时间。")
            if st.button("保存并测试连接", type="primary", width="stretch", key="conn_save"):
                vals = {"MODE": mode_pick, "ALPACA_PAPER": "true" if paper_pick else "false"}
                if new_key.strip():
                    vals["ALPACA_API_KEY"] = new_key.strip()
                if new_secret.strip():
                    vals["ALPACA_SECRET_KEY"] = new_secret.strip()
                update_env(vals)
                ok_msg, err_msg = None, None
                if mode_pick == "alpaca":
                    try:
                        from tradebot.config import load_settings as _ls
                        from tradebot.execution import make_broker as _mb

                        _b = _mb(_ls())
                        _a = _b.account()
                        ok_msg = f"连接成功：{_b.name} · 净值 {_a.equity:,.0f} · 现金 {_a.cash:,.0f} · 持仓 {len(_b.positions())} 只"
                    except Exception as _e:
                        err_msg = f"连接失败：{_e}"
                else:
                    ok_msg = "已切回本地模拟撮合。"
                st.session_state["conn_result"] = (ok_msg, err_msg)
                load_data.clear()
                st.rerun()
            res_conn = st.session_state.get("conn_result")
            if res_conn:
                ok_msg, err_msg = res_conn
                if ok_msg:
                    st.success(ok_msg)
                if err_msg:
                    st.error(err_msg)
            if st.button("重启循环进程（让新配置生效）", width="stretch", key="conn_restart_loop"):
                import subprocess

                r = subprocess.run(["./manage.sh", "restart", "loop"], cwd=str(ROOT_DIR), capture_output=True, text=True)
                st.code((r.stdout + r.stderr).strip()[-600:] or "已执行")
    with st.expander("配置（来自 .env）", expanded=False):
        st.markdown(
            f"**模式** `{S.mode}` · **周期** `{S.interval}` · **起步** `{S.entry_mode}`（{'只跟新入场' if S.entry_mode == 'fresh' else '对齐策略仓位'}）  \n"
            f"**策略** `{S.strategy}` {S.strategy_params or ''}  \n"
            f"**标的池** {S.watchlist.describe()}  \n**Claude 复核** {'开' if S.agent_enabled else '关'}"
        )
        for g, members in S.watchlist.groups.items():
            flag = "" if S.watchlist.enabled.get(g, True) else "（停用）"
            st.markdown(f"**{g}**{flag} {', '.join(members) or '空'}")
        st.caption("模式、周期、策略改 .env 后重启面板生效；标的池在“信号”页编辑。")

prefs_bar()
page_header()
tab_sig, tab_pos, tab_macro, tab_rot, tab_odte, tab_leap, tab_bottom, tab_bt, tab_verify, tab_paper, tab_log, tab_data, tab_help = st.tabs(["信号", "策略仓位", "大盘宏观", "板块轮动", "末日期权", "LEAP Call", "底部确认", "回测", "信号验证", "模拟盘", "决策日志", "行情", "说明"])


# ================= 信号 =================
with tab_sig:
    wl = S.watchlist
    c1, c2, c3 = st.columns([1, 3, 1])
    sig_interval = c1.radio("周期", ["1d", "1h"], index=0 if S.interval == "1d" else 1, horizontal=True, key="sig_interval")
    chosen = c2.multiselect("策略", list(STRATEGIES), default=SWING_NAMES, format_func=lambda n: f"{STRAT_LABEL[n]} ({n})", key="sig_strats")
    if c3.button("重新计算", help="强制重新下载行情并重算（不等缓存过期）"):
        refresh_data(tuple(S.symbols), sig_interval, lookback_for(sig_interval))
        compute_signals.clear()

    with st.expander("信号怎么读（完整规则见最后的“说明”页签）", expanded=False):
        st.markdown(
            "- **▲ 今日**：今天收盘满足该策略的入场条件；**▲ n天前**：最近 5 天内出现过；空白：没有。勾“显示出场条件”后 **▼** 表示今天满足出场规则。\n"
            "- **趋势回调**：上升趋势里 RSI 跌破 40 又站回 40。**突破**：收盘创 55 日新高且放量、在 100 日线上。**财报动量**：财报反应日涨 ≥ 3% 且超预期。"
            "**超跌反弹**：RSI < 30 或跌破布林下轨、比 20 日高点跌 8% 以上，且今天止跌。**底部回升**：跌深、低点抬高、突破近 10 日高、收复 20 日线。\n"
            "- **趋势**：收盘与 50 / 200 日线的关系。**RSI14**：14 日相对强弱，30 以下超卖、70 以上超买。**距55高**：现价相对过去 55 根 bar 最高收盘。\n"
            "- 全部只用已走完的 bar，按规则计算，不看任何账户持仓，也不是预测。"
        )
    if not S.symbols:
        st.info("标的池是空的，先在侧边栏“编辑标的池”里添加。")
    else:
        try:
            res = signals_for(sig_interval, chosen)
            table, unit = res["table"], res["unit"]
            st.session_state["signal_table"] = table
            strat_cols = [STRAT_LABEL[n] for n in chosen]

            # ---------- 今日信号一览 ----------
            lines = []
            for n in chosen:
                hits = res["today"].get(n, [])
                line = (f"<span class='tb-badge {'ok' if hits else ''}'>{STRAT_LABEL[n]} {len(hits)}</span> "
                        + ("、".join(hits) if hits else "<span style='color:var(--tb-muted)'>无</span>"))
                held_state = res.get("in_state", {}).get(n, [])
                if getattr(STRATEGIES[n], "signal_kind", "event") == "state":
                    line += f"  <span style='color:var(--tb-muted);font-size:.85em'>· 在列 {len(held_state)}：{'、'.join(held_state) if held_state else '无'}</span>"
                lines.append(line)
            multi = table[table["共振"] >= 2].sort_values("共振", ascending=False) if "共振" in table.columns else table.iloc[0:0]
            lines.append(f"<span class='tb-badge {'ok' if len(multi) else ''}'>共振 ≥ 2 · {len(multi)}</span> "
                         + ("、".join(f"{r['标的']}×{int(r['共振'])}" for _, r in multi.iterrows()) if len(multi) else "<span style='color:var(--tb-muted)'>无</span>")
                         + "  <span style='color:var(--tb-muted);font-size:.85em'>· 同一天被多个策略选中</span>")
            waves = table[table["阶段"].isin(["主升", "启动"])] if "阶段" in table.columns else table.iloc[0:0]
            lines.append(f"<span class='tb-badge {'ok' if len(waves) else ''}'>主升浪阶段</span> "
                         + ("、".join(f"{r['标的']}·{r['阶段']}" for _, r in waves.iterrows()) if len(waves) else "<span style='color:var(--tb-muted)'>无</span>")
                         + "  <span style='color:var(--tb-muted);font-size:.85em'>· 趋势模板 ≥ 6 条且 ADX ≥ 20 为主升，4~5 条且转强为启动</span>")
            section_header("今日入场信号", [(f"截至 {res['last_bar']:%m-%d} 收盘", ""), ("只用走完的 bar", "")])
            st.markdown("<br>".join(lines), unsafe_allow_html=True)
            rg = regime_status()
            if not rg.get("ok", True):
                st.warning(f"大盘过滤已触发：SPY 收盘 {rg['close']:.0f} 低于 200 日线 {rg['sma']:.0f}。所有 swing 策略暂停开新仓，信号照常显示但策略仓位页和模拟盘不会买。")

            with st.expander("最近几天每天的信号", expanded=False):
                e1, e2, e3 = st.columns([2, 1, 1.5])
                n_days = e1.slider("显示最近几根 bar", 3, 20, 7, key="hist_days")
                hist_exit_on = e2.checkbox("含出场条件", value=False, key="hist_exit_on")
                view_mode = e3.radio("视图", ["按日期", "按标的"], horizontal=True, key="hist_view")
                any_hist = next(iter(res["hist_entry"].values()), None)
                if any_hist is None or any_hist.empty:
                    st.caption("没有历史信号数据。")
                else:
                    dates = list(any_hist.index[-n_days:])
                    if view_mode == "按日期":
                        rows_h = []
                        for d in reversed(dates):
                            row = {"日期": d.strftime("%m-%d %a")}
                            for n in chosen:
                                he, hx = res["hist_entry"][n], res["hist_exit"][n]
                                ent_syms = [c for c in he.columns if d in he.index and bool(he.loc[d, c])]
                                cell = "、".join(ent_syms)
                                hs = res.get("hist_state", {}).get(n)
                                if hs is not None and d in hs.index:
                                    n_in = int(hs.loc[d].sum())
                                    cell = (cell + ("  " if cell else "") + f"(在列 {n_in})") if n_in else cell
                                if hist_exit_on:
                                    ex_syms = [c for c in hx.columns if d in hx.index and bool(hx.loc[d, c])]
                                    if ex_syms:
                                        cell += ("  " if cell else "") + "▼ " + "、".join(ex_syms)
                                row[STRAT_LABEL[n]] = cell
                            rows_h.append(row)
                        df_h = pd.DataFrame(rows_h)
                        st.dataframe(df_h.style.map(lambda v: "color:#15803d;font-weight:600" if isinstance(v, str) and v and not v.startswith("▼") else ("color:#b91c1c" if isinstance(v, str) and v.startswith("▼") else ""),
                                                    subset=[STRAT_LABEL[n] for n in chosen]),
                                     hide_index=True, width="stretch", height=42 + 35 * len(df_h),
                                     column_config={"日期": st.column_config.TextColumn(width="small")})
                        st.caption("每格是当天收盘满足该策略入场条件的标的；勾“含出场条件”后 ▼ 后面是满足出场规则的标的。"
                                   "相对强度这类状态型策略只列当天“新进”前 20% 的标的，括号里是当天在列的数量。")
                    else:
                        pick_s = st.selectbox("策略", chosen, format_func=lambda n: STRAT_LABEL[n], key="hist_strat")
                        he, hx = res["hist_entry"][pick_s], res["hist_exit"][pick_s]
                        syms_h = [c for c in he.columns if c in set(view["标的"])]
                        mat = pd.DataFrame(index=syms_h, columns=[d.strftime("%m-%d") for d in dates], dtype=object).fillna("")
                        hs = res.get("hist_state", {}).get(pick_s)
                        for d in dates:
                            col_ = d.strftime("%m-%d")
                            for c in syms_h:
                                if bool(he.loc[d, c]):
                                    mat.loc[c, col_] = "▲"
                                elif hs is not None and c in hs.columns and d in hs.index and bool(hs.loc[d, c]):
                                    mat.loc[c, col_] = "●"
                                elif hist_exit_on and bool(hx.loc[d, c]):
                                    mat.loc[c, col_] = "▼"
                        mat = mat[(mat != "").any(axis=1)]
                        if mat.empty:
                            st.caption("这几天没有信号。")
                        else:
                            mat.insert(0, "标的", mat.index)
                            st.dataframe(mat.style.map(lambda v: "color:#15803d;font-weight:700" if v == "▲" else ("color:#15803d;opacity:.5" if v == "●" else ("color:#b91c1c" if v == "▼" else ""))),
                                         hide_index=True, width="stretch", height=42 + 35 * len(mat))
                            st.caption("行 = 标的，列 = 日期；▲ 当天新出信号，● 状态型策略持续在列，只显示这几天有信号的标的。")

            # ---------- 筛选 ----------
            f1, f2, f3, f4, f5, f6 = st.columns([2.4, 1, 1, 1.4, 0.7, 1])
            active_groups = [g for g in wl.groups if wl.enabled.get(g, True)]
            default_groups = [g for g in active_groups if g.lower() in S.view_groups] or active_groups
            pick_groups = f1.multiselect("板块", active_groups, default=default_groups, key="sig_groups")
            only_sig = f2.checkbox("只看有信号", value=False, key="sig_only")
            show_exit = f3.checkbox("显示出场条件", value=False, key="sig_exit", help="▼ = 今天满足该策略的出场条件（不考虑是否持有）")
            sort_opts = ["标的", "共振", f"涨跌1{unit}", f"涨跌5{unit}", f"涨跌20{unit}", "RSI14", "距55高", "距财报天", "2月最低", "2月最高", "5月最低", "5月最高"]
            sort_col = f4.selectbox("排序", sort_opts, index=1, key="sig_sort")
            desc = f5.checkbox("降序", value=True, key="sig_desc")
            more_cols = f6.checkbox("更多列", value=False, key="sig_more", help="市场、组、最新价、盘中%、ATR%")

            view = table[table["标的"].isin({s_ for g in pick_groups for s_ in wl.groups[g]})].copy()
            for lab in strat_cols:  # 出场条件叠加显示
                ex_col = f"{lab}·出场"
                if show_exit and ex_col in view.columns:
                    view[lab] = [t if t else ("▼ 出场条件" if e else "") for t, e in zip(view[lab], view[ex_col])]
            if only_sig and strat_cols:
                view = view[(view[strat_cols] != "").any(axis=1)]
            view = view.sort_values(sort_col, ascending=not desc, na_position="last")

            base_cols = ["标的", "现价", f"涨跌1{unit}", f"涨跌5{unit}", f"涨跌20{unit}", "趋势", "阶段", "RSI14", "距55高",
                         "2月最低", "2月最高", "5月最低", "5月最高", "财报"]
            if more_cols:
                base_cols = ["标的", "市场", "组", "现价", "最新价", "盘中%", f"涨跌1{unit}", f"涨跌5{unit}", f"涨跌20{unit}", "趋势", "阶段", "RSI14", "距55高", "ATR%",
                             "2月最低", "2月最高", "5月最低", "5月最高", "财报"]
            show_cols = base_cols + ["共振"] + strat_cols

            st.caption("点任意一行打开该标的的详情：催化剂、基本面、走势、新闻。")
            clicked = None

            def _handle(ev, key, frame):
                global clicked
                if ev is None:
                    return
                rows_sel = list(ev.selection.rows)
                if rows_sel:
                    clicked = (key, rows_sel[0], frame.iloc[rows_sel[0]]["标的"])
                elif (st.session_state.get("last_click") or (None,))[0] == key:
                    st.session_state["last_click"] = None

            for g in pick_groups:
                rows = view[view["标的"].isin(wl.groups[g])]
                if rows.empty:
                    continue
                n_up = int((rows["趋势"] == "↑ 多头").sum())
                n_sig = int((rows[strat_cols] != "").any(axis=1).sum()) if strat_cols else 0
                section_header(g, [(f"{len(rows)} 只", ""), (f"多头 {n_up}", "ok" if n_up else ""), (f"有信号 {n_sig}", "warn" if n_sig else "")])
                tkey = f"sigtbl_{g}_v{st.session_state.get('tbl_version', 0)}"  # 弹窗关闭时 tbl_version 加一，选中行随之清零，不会再自动弹回
                _handle(render_signal_table(rows[show_cols], strat_cols, unit, key=tkey), tkey, rows)

            st.caption("共振 = 当天同时给出信号的策略个数（≥ 2 标绿）；阶段 = 主升浪阶段（蓄势 → 启动 → 主升 → 过热 → 衰竭，详情弹窗里有 8 条体检）。"
                       "▲ 今日 = 今天收盘满足该策略的入场条件；▲ n天前 = 最近 5 天内出现过入场条件；▼ 出场条件 = 勾选后显示，表示今天满足出场规则。"
                       "相对强度是状态型：▲ 新进 = 今天刚进入动量前 20%，● 在列 = 持续在前 20%（只在美股里排名，海外标的不占名额）。"
                       "全部是规则计算，不看任何账户持仓。距55高 = 现价相对过去 55 根 bar 最高收盘；财报 = 下次财报日期（距今天数）。"
                       "海外上市标的（如海力士）只出信号不交易，价格是当地货币。")

            if clicked and st.session_state.get("last_click") != clicked[:2]:
                st.session_state["last_click"] = clicked[:2]
                request_detail(clicked[2])
        except Exception as e:
            st.error(f"信号计算失败：{e}")


# ================= 策略仓位 =================
def strategy_positions(name: str, W: pd.DataFrame, data: dict[str, pd.DataFrame], strat) -> dict:
    """从策略回放出的权重序列还原：当前模拟持仓（入场日、入场价、浮动）、排队候选、最近退出。"""
    last = W.iloc[-1]
    held_rows, exits, queue = [], [], []
    for sym in W.columns:
        w = W[sym]
        pos = (w > 0).astype(int)
        change = pos.diff().fillna(pos)
        if float(last[sym]) > 0:
            start = w.index[change == 1][-1]
            c = data[sym]["close"]
            entry_px = float(c.loc[start]) if start in c.index else float("nan")
            px = float(c.iloc[-1])
            held_rows.append({"标的": sym, "权重": float(last[sym]) * 100, "入场日": start.strftime("%m-%d"),
                              "持有bar": int(len(w.loc[start:]) - 1), "入场价": round(entry_px, 2), "现价": round(px, 2),
                              "浮动%": (px / entry_px - 1) * 100 if entry_px else float("nan")})
        else:
            recent_exit = w.index[-10:][(change.iloc[-10:] == -1).values]
            if len(recent_exit):
                d = recent_exit[-1]
                held_before = w.index[(change == 1) & (w.index < d)]
                start = held_before[-1] if len(held_before) else None
                c = data[sym]["close"]
                ret = (float(c.loc[d]) / float(c.loc[start]) - 1) * 100 if start is not None else float("nan")
                exits.append({"标的": sym, "退出日": d.strftime("%m-%d"), "入场日": start.strftime("%m-%d") if start is not None else "",
                              "持有bar": int(len(w.loc[start:d]) - 1) if start is not None else None, "区间%": ret})
            if isinstance(strat, SwingStrategy):
                sig = strat.compute(sym, data[sym]).iloc[-1]
                if bool(sig["entry"]):
                    queue.append({"标的": sym, "分数": float(sig["score"]) if pd.notna(sig["score"]) else float("nan")})
    held_df = pd.DataFrame(held_rows).sort_values("权重", ascending=False) if held_rows else pd.DataFrame(columns=["标的", "权重", "入场日", "持有bar", "入场价", "现价", "浮动%"])
    queue_df = pd.DataFrame(queue).sort_values("分数", ascending=False) if queue else pd.DataFrame(columns=["标的", "分数"])
    exits_df = pd.DataFrame(exits).sort_values("退出日", ascending=False) if exits else pd.DataFrame(columns=["标的", "退出日", "入场日", "持有bar", "区间%"])
    return {"held": held_df, "queue": queue_df, "exits": exits_df}


with tab_pos:
    q1, q2 = st.columns([1, 4])
    pos_interval = q1.radio("周期", ["1d", "1h"], index=0 if S.interval == "1d" else 1, horizontal=True, key="pos_interval")
    pos_chosen = q2.multiselect("策略", list(STRATEGIES), default=SWING_NAMES, format_func=lambda n: f"{STRAT_LABEL[n]} ({n})", key="pos_strats")
    st.caption("这里是每个策略用最近的历史数据**从头回放**出来的模拟组合：如果一直严格按该策略执行，现在会持有什么、什么在排队、最近退出了什么。"
               "它不是你的账户，账户持仓只看“模拟盘”页；实盘循环按 .env 的起步方式（当前：" + ("只跟新入场" if S.entry_mode == "fresh" else "对齐策略仓位") + "）决定是否跟进这些仓位。")
    if not S.symbols or not pos_chosen:
        st.info("先选策略。")
    else:
        try:
            res_p = signals_for(pos_interval, pos_chosen)
            raw_p = load_data(tuple(S.symbols), pos_interval, lookback_for(pos_interval))
            data_p = {sym: drop_incomplete_last_bar(df, pos_interval) for sym, df in raw_p.items()}
            for name in pos_chosen:
                strat = get_strategy(name, **strat_params(name))
                W = res_p["weights"][name]
                info = strategy_positions(name, W, data_p, strat)
                held, queue, exits = info["held"], info["queue"], info["exits"]
                cap = f"/{strat.max_positions}" if isinstance(strat, SwingStrategy) else ""
                badges = [(f"模拟仓位 {len(held)}{cap}", "ok" if len(held) else ""), (f"排队 {len(queue)}", "warn" if len(queue) else ""), (f"最近 10 bar 退出 {len(exits)}", "")]
                if name == S.strategy:
                    badges.insert(0, ("实盘策略", "ok"))
                section_header(STRAT_LABEL[name], badges)
                h1, h2 = st.columns([1.6, 1], gap="large")
                with h1:
                    if len(held):
                        st.dataframe(held.style.map(lambda v: "color:#15803d" if isinstance(v, float) and v > 0 else ("color:#b91c1c" if isinstance(v, float) and v < 0 else ""), subset=["浮动%"]),
                                     hide_index=True, width="stretch", height=42 + 35 * len(held),
                                     column_config={"权重": st.column_config.NumberColumn(format="%.0f%%"), "浮动%": st.column_config.NumberColumn(format="%+.1f%%", help="按决策口径：入场日收盘 → 最新完整 bar 收盘"),
                                                    "入场价": st.column_config.NumberColumn(format="%.2f"), "现价": st.column_config.NumberColumn(format="%.2f"), "持有bar": st.column_config.NumberColumn(format="%d")})
                    else:
                        st.caption("空仓")
                with h2:
                    st.markdown("**排队（满足入场但没槽位或分数排后）**")
                    if len(queue):
                        st.markdown(chips([(f"{r['标的']}" + (f" {r['分数']:+.2f}" if pd.notna(r["分数"]) else ""), "queue") for _, r in queue.iterrows()]), unsafe_allow_html=True)
                    else:
                        st.caption("无")
                    st.markdown("**最近退出**")
                    if len(exits):
                        st.dataframe(exits, hide_index=True, width="stretch", height=42 + 35 * len(exits),
                                     column_config={"区间%": st.column_config.NumberColumn(format="%+.1f%%"), "持有bar": st.column_config.NumberColumn(format="%d")})
                    else:
                        st.caption("无")
            st.caption("入场价和浮动按决策口径：信号 bar 的收盘价；实际成交在下一根开盘。分数是策略的排序依据（趋势回调 / 突破 / 底部回升 = 动量，超跌反弹 = 偏离均线的 ATR 倍数，财报动量 = 反应日涨幅）。")
        except Exception as e:
            st.error(f"策略仓位失败：{e}")


# ================= 大盘宏观 =================
with tab_macro:
    wl_m = S.watchlist
    groups_key_m = json.dumps(wl_m.groups, ensure_ascii=False, sort_keys=True)
    try:
        closes_m = cached_macro_closes()
        snap = compute_snapshot(closes_m)
        section_header("大盘快照", [(f"数据截至 {closes_m.index[-1]:%m-%d}", ""), ("指数用指数本身，不是 ETF", "")])
        market_tiles(snap)
        g1, g2 = st.columns([1.5, 1], gap="large")
        with g1:
            st.markdown("**主要指数与半导体近 3 个月相对走势**（起点 = 100）")
            hist = market_history(closes_m, ["^GSPC", "^NDX", "^RUT", "SMH"], days=63)
            figm = go.Figure()
            for col_ in hist.columns:
                figm.add_trace(go.Scatter(x=hist.index, y=hist[col_], name=col_, line=dict(width=1.6)))
            figm.add_hline(y=100, line=dict(color="#9ca3af", width=1, dash="dot"))
            figm.update_layout(template=TEMPLATE, height=320, margin=dict(l=40, r=20, t=10, b=30), legend=dict(orientation="h", y=1.08),
                               yaxis_title="相对表现")
            figm.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(figm, width="stretch", config=PLOTLY_CONFIG)
        with g2:
            st.markdown("**明细**")
            st.dataframe(snap[["名称", "代码", "最新", "涨跌1日", "涨跌5日", "涨跌20日", "距52周高", "趋势"]].style.map(
                lambda v: "color:#15803d" if isinstance(v, float) and v > 0 else ("color:#b91c1c" if isinstance(v, float) and v < 0 else ""),
                subset=["涨跌1日", "涨跌5日", "涨跌20日"]),
                hide_index=True, width="stretch", height=42 + 35 * len(snap),
                column_config={"最新": st.column_config.NumberColumn(format="%.2f"),
                               **{c: st.column_config.NumberColumn(format="%+.2f%%") for c in ["涨跌1日", "涨跌5日", "涨跌20日", "距52周高"]}})
    except Exception as e:
        st.warning(f"大盘快照获取失败：{e}")

    # ---------- 事件 ----------
    events = load_manual_events()
    up45 = upcoming_events(events, days=45)
    n_hi = int((up45["影响"] == "高").sum()) if len(up45) else 0
    section_header("宏观事件", [(f"未来 45 天 {len(up45)} 条", ""), (f"高重要性 {n_hi}", "warn" if n_hi else "")])
    m1, m2 = st.columns([1.5, 1], gap="large")
    with m1:
        h1, h2 = st.columns([3, 1.2])
        h1.markdown(f"**事件日历**（共 {len(events)} 条：FOMC / CPI / 非农 / PCE / GDP / PPI / JOLTS / 零售 / ISM / 期权到期 / 休市，可自己加）")
        if not CLOUD and h2.button("同步官方日程", help="从美联储、BEA、普查局官网抓最新日程，BLS 用内置 2026 日程；按 日期+事件 去重，不动你手写的行"):
            with st.spinner("抓取官方日程 ..."):
                r = sync_official_events()
            msg = f"新增 {r['added']} 条，共 {r['total']} 条（{'；'.join(r['sources'])}）"
            if r["errors"]:
                st.warning(msg + "。部分来源失败：" + "；".join(r["errors"]))
            else:
                st.toast(msg)
            st.rerun()
        if len(up45):
            show = up45.copy()
            show["星期"] = pd.to_datetime(show["日期"], errors="coerce").dt.strftime("%a")
            show["日期"] = show["日期"].str[5:]  # 45 天窗口内只显示 月-日，省宽度
            show = show[["日期", "星期", "距今天", "事件", "影响", "备注"]]
            st.dataframe(
                show.style.map(lambda v: {"高": "color:#b91c1c;font-weight:600", "中": "color:#b45309"}.get(v, ""), subset=["影响"]),
                hide_index=True, width="stretch", height=min(42 + 35 * len(show), 520),
                column_config={"距今天": st.column_config.NumberColumn(format="%d 天", width="small"), "日期": st.column_config.TextColumn(width="small"),
                               "星期": st.column_config.TextColumn(width="small"), "影响": st.column_config.TextColumn(width="small")},
            )
            st.caption("ISM、密歇根、再融资的日期是按惯例推算的，其余来自官方日程。")
        else:
            st.caption("未来 45 天没有事件。点“同步官方日程”导入。")
        with st.expander(f"编辑全部事件（{len(events)} 条）", expanded=False):
            edited = st.data_editor(
                events, num_rows="dynamic", hide_index=True, width="stretch", height=380,
                key=f"macro_events_editor_{len(events)}_{hash(tuple(events['事件']))}",
                column_config={
                    "日期": st.column_config.TextColumn(help="YYYY-MM-DD", width="small"),
                    "事件": st.column_config.TextColumn(width="medium"),
                    "影响": st.column_config.SelectboxColumn(options=["高", "中", "低"], width="small"),
                    "备注": st.column_config.TextColumn(width="large"),
                },
            )
            if not CLOUD and st.button("保存事件日历"):
                save_manual_events(edited)
                st.rerun()
    with m2:
        st.markdown("**标的池未来 30 天财报**")
        try:
            ue = cached_upcoming_earnings(tuple(S.symbols), groups_key_m)
            if len(ue):
                st.dataframe(ue, hide_index=True, width="stretch", height=min(42 + 35 * len(ue), 400))
            else:
                st.caption("30 天内没有财报。")
        except Exception as e:
            st.warning(f"财报日历失败：{e}")

    # ---------- Claude 宏观简报 ----------
    section_header("宏观简报", [("Claude 联网搜索未来两周大事件与近一周要闻", "")])
    if has_credentials():
        if st.button("生成 / 刷新宏观简报", key="macro_brief_btn"):
            cached_macro_brief.clear()
            st.session_state["macro_brief_requested"] = True
        if st.session_state.get("macro_brief_requested"):
            brief = cached_macro_brief(json.dumps(list(wl_m.groups), ensure_ascii=False), pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d"))
            if brief.get("error"):
                st.warning(f"Claude 调用失败：{brief['error']}")
            else:
                render_macro_brief(brief)
                st.caption("结果缓存 6 小时；只是研究摘要，不是买卖建议。")
        else:
            st.caption("点上面的按钮生成，约一分钟，结果缓存 6 小时。")
    else:
        st.caption("未检测到 Anthropic 凭证。设置 ANTHROPIC_API_KEY 或 `ant auth login` 后，这里可以一键让 Claude 联网整理未来两周的宏观事件和近期要闻。")


# ================= 板块轮动 =================
@st.cache_data(ttl=900, show_spinner="拉取行业 / 主题 / 风格 ETF ...")
def cached_rotation_closes() -> pd.DataFrame:
    return rotation_closes(S.cache_dir)


QUAD_COLOR = {"领先": "#15803d", "走弱": "#b45309", "落后": "#b91c1c", "改善": "#1d4ed8"}
QUAD_DESC = {"领先": "比基准强，而且还在变强", "走弱": "比基准强，但势头在减弱", "落后": "比基准弱，而且还在变弱", "改善": "比基准弱，但势头在转强"}
QUAD_HINT = {"领先": "顺势持有 / 回调找机会", "走弱": "别追高，注意兑现", "落后": "回避，等它转到“改善”", "改善": "最值得盯的候选：从弱变强的早期"}

with tab_rot:
    with st.expander("这页怎么读", expanded=not st.session_state.get("rot_seen", False)):
        st.session_state["rot_seen"] = True
        st.markdown(
            "- **问题**：钱在往哪些板块流？哪些板块在变强、哪些在变弱？用每个板块 ETF **相对基准（标普 500）** 的强弱来回答，而不是看绝对涨跌。\n"
            "- **横轴 RS 比率**：板块价格 / 基准价格，再除以它自己过去 3 个月的平均值。>100 = 最近比自己平时更强于基准；<100 = 更弱。\n"
            "- **纵轴 RS 动量**：RS 比率相对 10 个交易日前的变化。>100 = 相对强度在上升；<100 = 在下降。\n"
            "- **四个象限**：右上 **领先**（强且更强）→ 右下 **走弱**（强但减弱）→ 左下 **落后**（弱且更弱）→ 左上 **改善**（弱但转强）→ 再回到领先。"
            "板块通常**顺时针**转，所以“改善”是发现下一个领先板块的地方，“走弱”是该减仓的提醒。\n"
            "- **轨迹**：每条线是最近几周的路径，箭头是当前位置和运动方向。**回溯**滑块可以把整页拉回到 N 个交易日前，看当时的格局和后来的演变。\n"
            "- **我的板块**：你自己的四个板块按美股成员等权合成，和 ETF 放在同一张图里比。\n"
            "- 这是历史统计的描述，不是预测；相对强度告诉你“谁在变强”，不告诉你“会强多久”。"
        )
    r1, r2, r3, r4, r5 = st.columns([1.3, 2.2, 1.6, 1, 0.7])
    bench = r1.selectbox("基准", list(BENCHMARKS), format_func=lambda k: BENCHMARKS[k], key="rot_bench")
    kinds_all = ["行业", "主题", "风格", "跨资产", "我的板块"]
    kinds_pick = r2.multiselect("范围", kinds_all, default=["行业", "主题", "我的板块"], key="rot_kinds")
    back = r3.slider("回溯（交易日）", 0, 120, 0, 1, key="rot_back", help="0 = 最新；拉到 20 就是看 20 个交易日前那天的格局")
    with r4.popover("RRG 参数"):
        rw = st.number_input("RS 比率窗口（日）", 20, 200, 63, 1, key="rot_rw", help="RS 相对自己多长时间均值")
        mw = st.number_input("RS 动量窗口（日）", 3, 40, 10, 1, key="rot_mw", help="RS 比率相对多少天前")
        tail_n = st.number_input("轨迹点数（周）", 2, 16, 5, 1, key="rot_tail")
        show_tail = st.checkbox("显示轨迹", value=True, key="rot_show_tail")
    if r5.button("刷新", key="rot_refresh"):
        cached_rotation_closes.clear()
        load_data.clear()
    try:
        closes_r = cached_rotation_closes().copy()
        names_r, kinds_r = dict(NAME), dict(KIND)
        my_groups = {}
        for g, members in S.watchlist.groups.items():
            us = [x for x in members if is_us_listed(x)]
            if S.watchlist.enabled.get(g, True) and us:
                my_groups[g] = us
        member_data = None
        if "我的板块" in kinds_pick and my_groups:
            member_data = load_data(tuple(sorted({x for m in my_groups.values() for x in m})), "1d", 750)
            for g, us in my_groups.items():
                code = f"我的·{g}"
                closes_r[code] = basket_index({x: member_data[x] for x in us if x in member_data}, closes_r.index)
                names_r[code], kinds_r[code] = f"我的·{g}", "我的板块"
        codes_r = [c for c in closes_r.columns if c != bench and kinds_r.get(c, "") in kinds_pick]
        if not codes_r:
            st.info("先选范围。")
        else:
            sub_full = closes_r[[bench] + codes_r]
            cut = len(sub_full) - int(back)
            sub = sub_full.iloc[:cut]                       # 回溯：只看到那一天为止
            sub_prev = sub_full.iloc[:max(cut - 5, 130)]     # 一周前，用来算象限变化
            as_of = sub.index[-1]
            tbl_r = relative_strength_table(sub, bench, names_r, kinds_r, int(rw), int(mw))
            tbl_prev = relative_strength_table(sub_prev, bench, names_r, kinds_r, int(rw), int(mw))[["代码", "象限", "RS比率", "RS动量"]].rename(
                columns={"象限": "一周前象限", "RS比率": "RS比率_前", "RS动量": "RS动量_前"})
            tbl_r = tbl_r.merge(tbl_prev, on="代码", how="left")
            tbl_r["象限变化"] = [("" if a_ == b_ else f"{a_}→{b_}") if isinstance(a_, str) and a_ else "" for a_, b_ in zip(tbl_r["一周前象限"], tbl_r["象限"])]
            tails = rrg_tail(sub, bench, codes_r, int(rw), int(mw), int(tail_n), 5)

            # ---------- 一句话结论 ----------
            by_q = {q: tbl_r[tbl_r["象限"] == q].sort_values("RS动量", ascending=False) for q in QUAD_COLOR}
            movers_up = tbl_r[tbl_r["象限变化"].str.endswith("领先") | tbl_r["象限变化"].str.endswith("改善")]
            movers_dn = tbl_r[tbl_r["象限变化"].str.endswith("走弱") | tbl_r["象限变化"].str.endswith("落后")]
            mine = tbl_r[tbl_r["分类"] == "我的板块"]
            summary = (f"**截至 {as_of:%m-%d}**（相对 {BENCHMARKS[bench]}）：领先 {len(by_q['领先'])} 个"
                       + (f"（{'、'.join(by_q['领先']['名称'].head(4))}）" if len(by_q["领先"]) else "")
                       + f"，改善 {len(by_q['改善'])} 个" + (f"（{'、'.join(by_q['改善']['名称'].head(4))}）" if len(by_q["改善"]) else "") + "。")
            if len(movers_up):
                summary += f" 一周内转强：{'、'.join(f'{r.名称}（{r.象限变化}）' for r in movers_up.itertuples())}。"
            if len(movers_dn):
                summary += f" 转弱：{'、'.join(f'{r.名称}（{r.象限变化}）' for r in movers_dn.itertuples())}。"
            if len(mine):
                summary += " 我的板块：" + "、".join(f"{r.名称.replace('我的·', '')} 在 **{r.象限}**" for r in mine.itertuples()) + "。"
            st.markdown(summary)

            section_header("相对轮动图（RRG）", [(f"基准 {BENCHMARKS[bench]}", ""), (f"{len(codes_r)} 个板块", ""), (f"截至 {as_of:%m-%d}" + (f"（回溯 {back} 日）" if back else ""), "warn" if back else "")])
            c1, c2 = st.columns([1.5, 1], gap="large")
            with c1:
                fig_r = go.Figure()
                allpts = pd.concat(list(tails.values())) if tails else pd.DataFrame({"ratio": [100], "mom": [100]})
                xr = [min(allpts["ratio"].min(), 100) - 1.5, max(allpts["ratio"].max(), 100) + 1.5]
                yr = [min(allpts["mom"].min(), 100) - 1, max(allpts["mom"].max(), 100) + 1]
                for (x0, x1, y0, y1, col, lab) in [(100, xr[1], 100, yr[1], "rgba(22,163,74,.07)", "领先"), (100, xr[1], yr[0], 100, "rgba(217,119,6,.07)", "走弱"),
                                                    (xr[0], 100, yr[0], 100, "rgba(185,28,28,.07)", "落后"), (xr[0], 100, 100, yr[1], "rgba(29,78,216,.07)", "改善")]:
                    fig_r.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=col, line_width=0, layer="below")
                    fig_r.add_annotation(x=(x0 + x1) / 2, y=y1 - (y1 - y0) * 0.06, text=f"<b>{lab}</b><br><span style='font-size:10px'>{QUAD_DESC[lab]}</span>",
                                         showarrow=False, font=dict(color=QUAD_COLOR[lab], size=12), opacity=0.85)
                fig_r.add_hline(y=100, line=dict(color="#9ca3af", width=1))
                fig_r.add_vline(x=100, line=dict(color="#9ca3af", width=1))
                palette = ["#2563eb", "#7c3aed", "#f59e0b", "#16a34a", "#dc2626", "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4b5563", "#0d9488", "#9333ea", "#ca8a04", "#1d4ed8", "#be123c", "#15803d", "#6d28d9", "#c2410c", "#0369a1", "#a16207"]
                for i, code in enumerate(codes_r):
                    pts = tails.get(code)
                    if pts is None or pts.empty:
                        continue
                    col = palette[i % len(palette)]
                    label = names_r.get(code, code)
                    if show_tail and len(pts) > 1:
                        fig_r.add_trace(go.Scatter(x=pts["ratio"], y=pts["mom"], mode="lines+markers", name=label, line=dict(color=col, width=1), opacity=0.45,
                                                   marker=dict(size=4, color=col), text=[f"{label} {d:%m-%d}" for d in pts.index],
                                                   hovertemplate="%{text}<br>RS比率 %{x:.1f} · RS动量 %{y:.1f}<extra></extra>"))
                        # 方向箭头：最后一段
                        fig_r.add_trace(go.Scatter(x=pts["ratio"].iloc[-2:], y=pts["mom"].iloc[-2:], mode="markers", name=label, showlegend=False, hoverinfo="skip",
                                                   marker=dict(symbol="arrow", angleref="previous", size=14, color=col)))
                    fig_r.add_trace(go.Scatter(x=[pts["ratio"].iloc[-1]], y=[pts["mom"].iloc[-1]], mode="markers", name=label,
                                               marker=dict(size=11, color=col, line=dict(color="white", width=1.5)),
                                               text=[f"{label} {pts.index[-1]:%m-%d}"], hovertemplate="%{text}<br>RS比率 %{x:.1f} · RS动量 %{y:.1f}<extra></extra>"))
                    fig_r.add_annotation(x=pts["ratio"].iloc[-1], y=pts["mom"].iloc[-1], text=label, showarrow=False, yshift=12, font=dict(size=10, color=col))
                fig_r.update_layout(template=TEMPLATE, height=580, margin=dict(l=40, r=20, t=10, b=40), showlegend=False,
                                    xaxis_title="← 比基准弱　　RS 比率（相对基准的强弱）　　比基准强 →", yaxis_title="← 在变弱　　RS 动量　　在变强 →",
                                    xaxis_range=xr, yaxis_range=yr)
                st.plotly_chart(fig_r, width="stretch", config=PLOTLY_CONFIG)
                st.caption("箭头 = 当前位置与最近一周的运动方向；顺时针轮动是常态：改善 → 领先 → 走弱 → 落后 → 改善。")
            with c2:
                st.markdown("**象限分布**（括号里是一周前的象限，有变化才显示）")
                for q in ["领先", "改善", "走弱", "落后"]:
                    rows_q = by_q[q]
                    items = []
                    for r in rows_q.itertuples():
                        chg = f" <span style='color:#6b7280;font-size:.8em'>({r.一周前象限}→)</span>" if r.象限变化 else ""
                        items.append(f"{r.名称}{chg}")
                    st.markdown(f"<span class='tb-badge' style='background:{QUAD_COLOR[q]}22;color:{QUAD_COLOR[q]}'>{q} {len(rows_q)}</span> "
                                f"<span style='color:#6b7280;font-size:.8em'>{QUAD_HINT[q]}</span><br>"
                                + ("、".join(items) if items else "<span style='color:var(--tb-muted)'>无</span>"), unsafe_allow_html=True)
                if my_groups and "我的板块" in kinds_pick and member_data is not None and not back:
                    st.markdown("**我的板块内部广度**（成员里多少比例站在均线上）")
                    br = group_breadth({g: {x: member_data[x] for x in us if x in member_data} for g, us in my_groups.items()})
                    if len(br):
                        st.dataframe(br, hide_index=True, width="stretch",
                                     column_config={c: st.column_config.NumberColumn(format="%.0f%%") for c in [">SMA50%", ">SMA200%", "RSI>50%"]}
                                     | {"平均20日%": st.column_config.NumberColumn(format="%+.1f%%"), "平均距52周高%": st.column_config.NumberColumn(format="%+.1f%%")})

            section_header("相对强度表", [("按 3 月超额收益排序", ""), (f"截至 {as_of:%m-%d}", "")])
            pct_r = [f"{h}%" for h in ["1周", "1月", "3月", "6月"]] + [f"{h}超额" for h in ["1周", "1月", "3月", "6月"]] + ["距52周高%"]
            show_r = tbl_r[["名称", "代码", "分类", "象限", "象限变化", "RS比率", "RS动量", "1周%", "1月%", "3月%", "6月%", "1周超额", "1月超额", "3月超额", "6月超额", "3月超额排名", "趋势", "距52周高%", "最新"]]
            sty_r = show_r.style.map(lambda v: "color:#15803d" if isinstance(v, float) and v > 0 else ("color:#b91c1c" if isinstance(v, float) and v < 0 else ""), subset=pct_r)
            sty_r = sty_r.map(lambda v: f"color:{QUAD_COLOR.get(v, '#6b7280')};font-weight:600" if v else "", subset=["象限"])
            sty_r = sty_r.map(lambda v: ("color:#15803d;font-weight:600" if isinstance(v, str) and (v.endswith("领先") or v.endswith("改善")) else ("color:#b91c1c" if isinstance(v, str) and v else "")), subset=["象限变化"])
            st.dataframe(sty_r, hide_index=True, width="stretch", height=min(42 + 35 * len(show_r), 700),
                         column_config={**{c: st.column_config.NumberColumn(format="%+.1f%%") for c in pct_r},
                                        "最新": st.column_config.NumberColumn(format="%.2f"), "RS比率": st.column_config.NumberColumn(format="%.1f", help=">100 比基准强于自己过去 N 日的平均"),
                                        "RS动量": st.column_config.NumberColumn(format="%.1f", help=">100 相对强度在上升"), "3月超额排名": st.column_config.NumberColumn(format="%d"),
                                        "象限变化": st.column_config.TextColumn(help="一周前象限 → 现在象限，没变就空")})

            section_header("月度收益热力图", [("最后一列是本月至今" if not back else f"截至 {as_of:%m-%d}", "")])
            mr = monthly_returns(sub, codes_r, months=12)
            mr.index = [names_r.get(c, c) for c in mr.index]
            mr = mr.loc[mr.iloc[:, -1].sort_values(ascending=False).index]
            fig_h = go.Figure(go.Heatmap(z=mr.values, x=list(mr.columns), y=list(mr.index), colorscale="RdYlGn", zmid=0,
                                         text=np.round(mr.values, 1), texttemplate="%{text}", textfont=dict(size=10),
                                         hovertemplate="%{y} · %{x}<br>%{z:.1f}%<extra></extra>", colorbar=dict(title="%")))
            fig_h.update_layout(template=TEMPLATE, height=max(320, 26 * len(mr) + 80), margin=dict(l=40, r=20, t=10, b=30), yaxis_autorange="reversed")
            st.plotly_chart(fig_h, width="stretch", config=PLOTLY_CONFIG)
            st.caption("每格是该板块当月的绝对收益；一行从左到右看它自己的节奏，一列从上到下看当月谁强谁弱，连续几个月的颜色带就是轮动。"
                       "RS 比率 = 100 × (价格/基准) / 其过去 N 日均值；RS 动量 = 100 × RS 比率 / M 日前的 RS 比率。RRG 的常见近似，不是 JdK 原版算法。"
                       "“我的板块”按美股成员每日等权合成，海外成员不计。全是历史统计，不预测方向。")
    except Exception as e:
        st.error(f"板块轮动失败：{e}")


# ================= 末日期权 =================
@st.cache_data(ttl=900, show_spinner="拉取期权链（每个标的一两次请求）...")
def cached_chains(symbols: tuple[str, ...], max_dte: int, day_key: str) -> dict:
    return {sym: fetch_chain_summary(sym, None, max_dte, cache_dir=S.cache_dir) for sym in symbols}


with tab_odte:
    us_syms = [x for x in S.symbols if is_us_listed(x)]
    o1, o2, o3, o4, o5 = st.columns([2, 1.6, 1.2, 1.2, 1])
    wl_groups_us = [g for g in S.watchlist.groups if S.watchlist.enabled.get(g, True) and any(is_us_listed(x) for x in S.watchlist.groups[g])]
    default_g = [g for g in wl_groups_us if g.lower() in S.view_groups] or wl_groups_us
    pick_g = o1.multiselect("板块", wl_groups_us, default=default_g, key="odte_groups", help="默认勾选由 .env 的 VIEW_GROUPS 决定")
    mode = o2.radio("排序口径", ["卖方：区间内概率", "买方：突破概率"], key="odte_mode")
    max_dte = o3.selectbox("最长到期", [1, 2, 3, 5, 7], index=3, key="odte_dte", help="只看这么多个日历日内到期的链；SPY/QQQ 等有每日到期，个股多为周三/周五")
    liquid_only = o4.checkbox("只看流动性达标", value=True, key="odte_liquid", help="平值持仓量 ≥ 100 且跨式买卖价差 ≤ 15%")
    if o5.button("刷新链", key="odte_refresh"):
        cached_chains.clear()
    today = pd.Timestamp.now(tz=NY).date()
    syms = [x for x in us_syms if any(x in S.watchlist.groups[g] for g in pick_g)]
    if not syms:
        st.info("先选板块。")
    else:
        try:
            events_df = load_manual_events()
            win = macro_in_window(events_df, today, max_dte)
            hi_mid = [r for _, r in win.iterrows() if r["影响"] in ("高", "中")] if len(win) else []
            if hi_mid:
                st.markdown("**窗口内宏观事件**：" + "　".join(f"{r['日期'][5:]} {r['事件']}" + ("（高）" if r["影响"] == "高" else "") for r in hi_mid))
            chains = cached_chains(tuple(syms), max_dte, today.isoformat())
            daily = load_data(tuple(syms), "1d", 750)
            sig = st.session_state.get("signal_table")
            tbl = build_odte_table(syms, daily, chains, EarningsCalendar(cache_dir=S.cache_dir), events_df, S.watchlist.groups,
                                   signal_table=sig, strategy_col=STRAT_LABEL.get(S.strategy), today=today)
            missing = [x for x in syms if not chains.get(x)]
            if tbl.empty:
                st.warning("没有拿到任何期权链。" + (f"失败：{', '.join(missing)}" if missing else ""))
            else:
                if liquid_only:
                    tbl = tbl[tbl["流动性"] == "✓"]
                key = "区间内概率" if mode.startswith("卖方") else "突破概率"
                tbl = tbl.sort_values([key, "隐含/历史"], ascending=[False, mode.startswith("买方")]).reset_index(drop=True)
                tbl.insert(0, "排名", range(1, len(tbl) + 1))
                cols = ["排名", "标的", "板块", "到期", "交易日", "日频到期", "现价", "平值", "跨式", "隐含波动%", "历史中位%", "隐含/历史",
                        "区间内概率", "突破概率", "平时区间概率", "样本", "事件", "事件分布", "事件日中位%", "事件样本", "ATM持仓量", "价差%", "IV%", "流动性",
                        "趋势", "RSI14", "策略"]
                show = tbl[[c for c in cols if c in tbl.columns]]
                prob_style = lambda v: ("color:#15803d;font-weight:600" if isinstance(v, (int, float)) and v >= 70 else ("color:#b91c1c" if isinstance(v, (int, float)) and v <= 40 else ""))
                styler = show.style.map(prob_style, subset=[key]).map(lambda v: "color:#b45309;font-weight:600" if isinstance(v, str) and v else "", subset=["事件"])
                ev = st.dataframe(styler, hide_index=True, width="stretch", height=min(42 + 35 * len(show), 900),
                                  on_select="rerun", selection_mode="single-row", key=f"odte_tbl_v{st.session_state.get('tbl_version', 0)}",
                                  column_config={
                                      "区间内概率": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
                                      "突破概率": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
                                      "平时区间概率": st.column_config.NumberColumn(format="%.0f%%", help="不看事件、只用平时分布算出的区间内概率"),
                                      "隐含波动%": st.column_config.NumberColumn(format="%.2f%%", help="平值跨式中间价 / 现价，市场定价的到期波动幅度"),
                                      "历史中位%": st.column_config.NumberColumn(format="%.2f%%", help="过去一年同持有期真实 |涨跌幅| 的中位数"),
                                      "事件日中位%": st.column_config.NumberColumn(format="%.2f%%"),
                                      "价差%": st.column_config.NumberColumn(format="%.1f%%"),
                                      "IV%": st.column_config.NumberColumn(format="%.0f%%"),
                                      "隐含/历史": st.column_config.NumberColumn(format="%.2f", help=">1 期权相对历史偏贵（利于卖方），<1 偏便宜（利于买方）"),
                                      "现价": st.column_config.NumberColumn(format="%.2f"),
                                      "平值": st.column_config.NumberColumn(format="%.2f"),
                                      "跨式": st.column_config.NumberColumn(format="%.2f", help="平值 call 中间价 + put 中间价"),
                                      "交易日": st.column_config.NumberColumn(format="%d", help="到期前剩余交易日"),
                                  })
                if ev is not None and ev.selection.rows:
                    picked = show.iloc[ev.selection.rows[0]]["标的"]
                    ck = ("odte", picked)
                    if st.session_state.get("last_click") != ck:
                        st.session_state["last_click"] = ck
                        request_detail(picked)
                if missing:
                    st.caption(f"没有期权链或抓取失败：{', '.join(missing)}")
                st.caption(
                    "口径：隐含波动% = 最近到期平值跨式中间价 / 现价。区间内概率 = 该标的过去一年同持有期真实 |涨跌幅| 小于隐含幅度的比例，"
                    "对卖跨式 / 铁鹰有利；突破概率 = 1 − 区间内概率，对买方有利。窗口内有财报（历史样本 ≥ 4）或高重要性宏观事件（样本 ≥ 3）时，"
                    "改用该标的在历史财报日 / 事件日的波动分布。这些都是历史统计，不预测方向；末日期权时间价值衰减极快、滑点大，仓位请按可全亏来控制。"
                )
        except Exception as e:
            st.error(f"期权面板失败：{e}")


# ================= LEAP Call =================
@st.cache_data(ttl=3600, show_spinner=False)
def cached_risk_free() -> float:
    """13 周国债收益率（^IRX，百分数）作为无风险利率，拿不到用 4%。"""
    try:
        import yfinance as yf

        h = yf.Ticker("^IRX").history(period="10d")["Close"].dropna()
        return float(h.iloc[-1]) / 100 if len(h) else 0.04
    except Exception:
        return 0.04


def dividend_yield(sym: str) -> float:
    q = (cached_info(sym) or {}).get("dividendYield")
    if q is None:
        return 0.0
    q = float(q)
    return q / 100 if q > 0.2 else q  # yfinance 有的版本给百分数


@st.cache_data(ttl=6 * 3600, show_spinner="拉取合约历史价格 ...")
def cached_option_history(contract: str, day_key: str) -> pd.DataFrame:
    return fetch_option_history(contract, "1y", cache_dir=S.cache_dir)


@st.cache_data(ttl=3600, show_spinner="计算正股 IV30（每个标的两条期权链）...")
def cached_iv30(symbols: tuple[str, ...], day_key: str) -> dict:
    return {sym: iv30_snapshot(sym) for sym in symbols}


@st.cache_data(ttl=3600, show_spinner="拉取 LEAP 期权链（每个标的三四次请求）...")
def cached_leaps(symbols: tuple[str, ...], months: tuple[int, ...], target_delta: float, r: float, day_key: str) -> dict:
    return {sym: fetch_leap_candidates(sym, None, months, target_delta, r, dividend_yield(sym), cache_dir=S.cache_dir) for sym in symbols}


with tab_leap:
    l1, l2, l3, l4, l5 = st.columns([2, 1.4, 1.4, 1.3, 0.9])
    wl_groups_us_l = [g for g in S.watchlist.groups if S.watchlist.enabled.get(g, True) and any(is_us_listed(x) for x in S.watchlist.groups[g])]
    default_lg = [g for g in wl_groups_us_l if g.lower() in S.view_groups] or wl_groups_us_l
    pick_lg = l1.multiselect("板块", wl_groups_us_l, default=default_lg, key="leap_groups")
    months_pick = l2.multiselect("到期月数", [4, 6, 8, 10, 12, 18, 24], default=[4, 6, 8, 10], key="leap_months", help="取最接近该月数的到期日，容差 45 天；4 个月严格说不算 LEAP，放这里方便对比时间价值")
    target_delta = l3.select_slider("目标 Delta", options=[0.5, 0.6, 0.7, 0.8, 0.9], value=0.7, key="leap_delta",
                                    help="按 Black-Scholes delta 最接近目标值选行权价；0.7~0.8 是常见的“代替正股”深度实值区")
    sort_key = l4.selectbox("排序", ["历史胜率", "隐含胜率", "杠杆", "年化成本%", "IV/RV", "需涨%"], key="leap_sort")
    if l5.button("刷新链", key="leap_refresh"):
        cached_leaps.clear()
    syms_l = [x for x in S.symbols if is_us_listed(x) and any(x in S.watchlist.groups[g] for g in pick_lg)]
    if not syms_l or not months_pick:
        st.info("先选板块和月数。")
    else:
        try:
            r_free = cached_risk_free()
            today_l = pd.Timestamp.now(tz=NY).date()
            cands = cached_leaps(tuple(syms_l), tuple(sorted(months_pick)), float(target_delta), r_free, today_l.isoformat())
            hist_l = load_data(tuple(syms_l), "1d", default_days("1d"))
            tbl_l = build_leap_table(cands, hist_l, S.watchlist.groups)
            missing_l = [x for x in syms_l if not cands.get(x)]
            if tbl_l.empty:
                st.warning("没有拿到任何 LEAP 链。" + (f"失败：{', '.join(missing_l)}" if missing_l else ""))
            else:
                asc = sort_key in ("年化成本%", "需涨%", "IV/RV")
                tbl_l = tbl_l.sort_values([sort_key, "标的"], ascending=[asc, True], na_position="last").reset_index(drop=True)
                # ---------- 正股 IV30 / HV30 / IV Rank ----------
                ivh = load_iv_history(S.cache_dir)
                n_days_hist = int(ivh["date"].nunique()) if len(ivh) else 0
                section_header("正股隐含波动率", [("IV30 = 30 天平值隐含波动率，真实报价", ""), (f"IV Rank 历史 {n_days_hist} 天（自己每天攒）", "warn" if n_days_hist < 60 else "ok")])
                snaps = cached_iv30(tuple(syms_l), today_l.isoformat())
                try:
                    vix_now = float(cached_macro_closes()["^VIX"].dropna().iloc[-1])
                except Exception:
                    vix_now = float("nan")
                rows_iv = []
                for sym in syms_l:
                    sn = snaps.get(sym)
                    if not sn:
                        continue
                    hv = hv30(hist_l[sym]["close"]) if sym in hist_l else float("nan")
                    hs = ivh[ivh["symbol"] == sym].sort_values("date")["iv30"] if len(ivh) else pd.Series(dtype=float)
                    rk = iv_rank_percentile(hs, sn["iv30"])
                    rows_iv.append({"标的": sym, "现价": round(sn["spot"], 2), "IV30%": round(sn["iv30"], 1), "HV30%": round(hv, 1) if not math.isnan(hv) else np.nan,
                                    "IV/HV": round(sn["iv30"] / hv, 2) if hv and not math.isnan(hv) else np.nan,
                                    "IV Rank": round(rk["iv_rank"], 0) if not math.isnan(rk["iv_rank"]) else np.nan,
                                    "IV Percentile": round(rk["iv_pct"], 0) if not math.isnan(rk["iv_pct"]) else np.nan,
                                    "历史天数": rk["n"], "区间低%": round(rk["lo"], 1) if not math.isnan(rk["lo"]) else np.nan, "区间高%": round(rk["hi"], 1) if not math.isnan(rk["hi"]) else np.nan,
                                    "近/远IV": round(sn["iv_near"] / sn["iv_far"], 2) if sn.get("iv_far") else np.nan,
                                    "IV/VIX": round(sn["iv30"] / vix_now, 2) if not math.isnan(vix_now) and vix_now > 0 else np.nan,
                                    "近月到期": sn["near_expiry"][5:], "远月到期": sn["far_expiry"][5:]})
                if rows_iv:
                    df_iv = pd.DataFrame(rows_iv).sort_values("IV/HV", ascending=False, na_position="last")
                    st.dataframe(df_iv.style.map(lambda v: "color:#b91c1c;font-weight:600" if isinstance(v, float) and v >= 1.3 else ("color:#15803d;font-weight:600" if isinstance(v, float) and v <= 0.8 else ""), subset=["IV/HV"]),
                                 hide_index=True, width="stretch", height=min(42 + 35 * len(df_iv), 520),
                                 column_config={"IV30%": st.column_config.NumberColumn(format="%.1f%%", help="最靠近 30 天的两个到期日的平值 IV，按方差时间插值到 30 天"),
                                                "HV30%": st.column_config.NumberColumn(format="%.1f%%", help="最近 30 个交易日收益率的年化标准差（真实波动）"),
                                                "IV/HV": st.column_config.NumberColumn(format="%.2f", help=">1.3 期权相对真实波动偏贵（标红），<0.8 偏便宜（标绿）"),
                                                "IV Rank": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f", help="(今 − 区间最低)/(区间最高 − 区间最低)，用自己攒的 IV30 历史"),
                                                "IV Percentile": st.column_config.NumberColumn(format="%.0f%%", help="历史里低于今天的天数占比"),
                                                "区间低%": st.column_config.NumberColumn(format="%.1f%%"), "区间高%": st.column_config.NumberColumn(format="%.1f%%"),
                                                "近/远IV": st.column_config.NumberColumn(format="%.2f", help="近月 IV / 远月 IV：>1.05 近月更贵（事件或恐慌），<0.95 正常的期限结构"),
                                                "IV/VIX": st.column_config.NumberColumn(format="%.2f", help="个股 IV30 / VIX：这只股票的期权相对大盘贵多少倍；对比自己平时的倍数看高低"),
                                                "现价": st.column_config.NumberColumn(format="%.2f"), "历史天数": st.column_config.NumberColumn(format="%d")})
                    st.caption("IV30 和 HV30 是真实数据。IV Rank / Percentile 需要正股 IV30 的历史序列，免费源没有，系统从今天起由模拟盘循环每个交易日记一条到 data_cache/iv_history.csv，"
                               "历史不到 60 天时只能参考；想要完整一年历史需要 ORATS、Barchart 之类的付费数据。下面 LEAP 表里的“合约IV百分位”是单张合约用成交价反推的，是另一回事。")

                section_header("LEAP Call 对比", [(f"{tbl_l['标的'].nunique()} 只 · {len(tbl_l)} 张", ""), (f"目标 Delta {target_delta}", ""),
                                                  (f"无风险利率 {r_free * 100:.2f}%", "")])
                cols_l = ["标的", "板块", "月数", "到期", "天数", "现价", "行权", "权利金", "Delta", "杠杆", "IV%", "RV%", "IV/RV", "时间价值%", "年化成本%",
                          "盈亏平衡", "需涨%", "隐含胜率", "隐含ITM", "历史胜率", "历史上涨率", "历史中位涨%", "样本", "OI", "价差%"]
                show_l = tbl_l[[c for c in cols_l if c in tbl_l.columns]]
                prob_style = lambda v: ("color:#15803d;font-weight:600" if isinstance(v, (int, float)) and not pd.isna(v) and v >= 60
                                        else ("color:#b91c1c" if isinstance(v, (int, float)) and not pd.isna(v) and v <= 40 else ""))
                styler_l = show_l.style.map(prob_style, subset=["隐含胜率", "历史胜率"]).map(
                    lambda v: "color:#b91c1c;font-weight:600" if isinstance(v, (int, float)) and not pd.isna(v) and v < 500 else "", subset=["样本"])
                ev_l = st.dataframe(styler_l, hide_index=True, width="stretch", height=min(42 + 35 * len(show_l), 760),
                                    on_select="rerun", selection_mode="single-row", key=f"leap_tbl_v{st.session_state.get('tbl_version', 0)}",
                                    column_config={
                                        "隐含胜率": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%", help="风险中性下到期价 > 盈亏平衡的概率，用 IV 算"),
                                        "历史胜率": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%", help="历史同长度持有期里涨幅 ≥ 需涨幅 的比例（重叠窗口，偏乐观）"),
                                        "隐含ITM": st.column_config.NumberColumn(format="%.0f%%", help="到期时价内（> 行权价）的风险中性概率"),
                                        "历史上涨率": st.column_config.NumberColumn(format="%.0f%%", help="历史同长度持有期里正股上涨的比例"),
                                        "历史中位涨%": st.column_config.NumberColumn(format="%+.1f%%"),
                                        "IV%": st.column_config.NumberColumn(format="%.1f%%", help="该合约隐含波动率（年化）"),
                                        "RV%": st.column_config.NumberColumn(format="%.1f%%", help="最近同长度窗口的真实年化波动率"),
                                        "IV/RV": st.column_config.NumberColumn(format="%.2f", help=">1 期权相对近期真实波动偏贵"),
                                        "时间价值%": st.column_config.NumberColumn(format="%.0f%%", help="权利金里时间价值的占比"),
                                        "年化成本%": st.column_config.NumberColumn(format="%.1f%%", help="时间价值 / 现价 / 年数：拿 LEAP 代替正股每年付的“租金”"),
                                        "需涨%": st.column_config.NumberColumn(format="%+.1f%%", help="到期不亏需要正股涨多少"),
                                        "价差%": st.column_config.NumberColumn(format="%.1f%%"),
                                        "杠杆": st.column_config.NumberColumn(format="%.1fx", help="delta × 现价 / 权利金"),
                                        "Delta": st.column_config.NumberColumn(format="%.2f"),
                                        "现价": st.column_config.NumberColumn(format="%.2f"), "行权": st.column_config.NumberColumn(format="%.2f"),
                                        "权利金": st.column_config.NumberColumn(format="%.2f"), "盈亏平衡": st.column_config.NumberColumn(format="%.2f"),
                                        "天数": st.column_config.NumberColumn(format="%d"),
                                    })
                sel_l = show_l.iloc[ev_l.selection.rows[0]]["标的"] if (ev_l is not None and ev_l.selection.rows) else show_l.iloc[0]["标的"]

                # ---- 选中标的：三张合约的到期盈亏图 ----
                cs = sorted(cands.get(sel_l) or [], key=lambda c: c["months"])
                if cs:
                    p1, p2 = st.columns([1.6, 1], gap="large")
                    with p1:
                        st.markdown(f"**{sel_l} · 到期时收益率 vs 正股涨跌**（相对权利金；灰线 = 直接持有正股）")
                        moves = np.linspace(-0.5, 0.8, 131)
                        figl = go.Figure()
                        figl.add_trace(go.Scatter(x=moves * 100, y=moves * 100, name="正股", line=dict(color="#9ca3af", width=1.5, dash="dot")))
                        palette = ["#2563eb", "#7c3aed", "#f59e0b", "#16a34a", "#dc2626", "#0891b2"]
                        for i, c in enumerate(cs):
                            figl.add_trace(go.Scatter(x=moves * 100, y=payoff_curve(c, moves), name=f"{c['months']}月 K{c['strike']:g} @{c['premium']:.2f}",
                                                      line=dict(color=palette[i % len(palette)], width=1.8)))
                            figl.add_vline(x=c["be_move_pct"], line=dict(color=palette[i % len(palette)], width=1, dash="dash"))
                        figl.add_hline(y=0, line=dict(color="#6b7280", width=1))
                        figl.update_layout(template=TEMPLATE, height=380, margin=dict(l=40, r=20, t=10, b=30), legend=dict(orientation="h", y=1.08),
                                           xaxis_title="到期时正股涨跌 %", yaxis_title="期权收益率 %", yaxis_range=[-110, 400])
                        st.plotly_chart(figl, width="stretch", config=PLOTLY_CONFIG)
                    with p2:
                        st.markdown("**合约明细**")
                        det = pd.DataFrame([{
                            "月数": c["months"], "到期": c["expiry"], "行权": c["strike"], "权利金": round(c["premium"], 2),
                            "内在": round(c["intrinsic"], 2), "时间价值": round(c["extrinsic"], 2), "Delta": round(c["delta"], 2),
                            "杠杆": round(c["leverage"], 1) if c["leverage"] else None, "盈亏平衡": round(c["breakeven"], 2),
                            "OI": c["oi"], "买/卖": f"{c['bid']:.2f} / {c['ask']:.2f}",
                        } for c in cs])
                        st.dataframe(det, hide_index=True, width="stretch")
                        if st.button("打开详情", key="leap_detail"):
                            request_detail(sel_l)

                    # ---- 合约历史价格：实际成交 vs 固定 IV 的理论价 vs 正股 ----
                    section_header(f"{sel_l} · 合约历史价格", [("Yahoo 最后成交价", ""), ("虚线 = 用当前 IV 反推的理论价", "")])
                    hc1, hc2 = st.columns([3, 1])
                    pick_cs = hc1.multiselect("合约", [c["contract"] for c in cs if c.get("contract")],
                                              default=[c["contract"] for c in cs if c.get("contract")], key=f"leap_hist_{sel_l}",
                                              format_func=lambda k: next((f"{c['months']}月 K{c['strike']:g} ({c['expiry']})" for c in cs if c.get("contract") == k), k))
                    show_theory = hc2.checkbox("理论价", value=True, key="leap_hist_theory", help="Black-Scholes，用该合约现在的 IV 固定不变反推历史，只是参照")
                    show_under = hc2.checkbox("正股", value=True, key="leap_hist_under")
                    if pick_cs:
                        under = hist_l[sel_l]["close"]
                        under.index = pd.DatetimeIndex(under.index).normalize()
                        figh = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06,
                                             specs=[[{"secondary_y": True}], [{"secondary_y": False}]], subplot_titles=("合约价格（右轴：正股）", "反推的隐含波动率 IV%"))
                        rows_h = []
                        for i, c in enumerate([c for c in cs if c.get("contract") in pick_cs]):
                            col = palette[i % len(palette)]
                            lab = f"{c['months']}月 K{c['strike']:g}"
                            oh = cached_option_history(c["contract"], today_l.isoformat())
                            if len(oh):
                                figh.add_trace(go.Scatter(x=oh.index, y=oh["close"], name=f"{lab} 成交价", line=dict(color=col, width=1.8),
                                                          hovertemplate="%{x|%Y-%m-%d} " + lab + " %{y:.2f}<extra></extra>"), row=1, col=1, secondary_y=False)
                                ivs = implied_vol_series(oh, under, c["strike"], dt.date.fromisoformat(c["expiry"]), c.get("r", 0.04), c.get("q", 0.0))
                                ivs_plot = ivs.dropna()
                                if len(ivs_plot):
                                    figh.add_trace(go.Scatter(x=ivs_plot.index, y=ivs_plot.values, name=f"{lab} IV", mode="lines+markers",
                                                              line=dict(color=col, width=1.2), marker=dict(size=3),
                                                              hovertemplate="%{x|%Y-%m-%d} " + lab + " IV %{y:.0f}%<extra></extra>"), row=2, col=1)
                                stt = iv_stats(ivs, c.get("iv"))
                                first, last_px = float(oh["close"].iloc[0]), float(oh["close"].iloc[-1])
                                rows_h.append({"合约": lab, "首次成交": oh.index[0].strftime("%Y-%m-%d"), "有成交天数": int((oh["volume"] > 0).sum()),
                                               "起始价": round(first, 2), "最新价": round(last_px, 2), "区间涨跌%": round((last_px / first - 1) * 100, 1) if first else None,
                                               "区间最高": round(float(oh["close"].max()), 2), "区间最低": round(float(oh["close"].min()), 2),
                                               "当前IV%": round(c["iv"], 1) if c.get("iv") else None,
                                               "IV中位%": round(stt["iv_median"], 1) if not math.isnan(stt["iv_median"]) else None,
                                               "IV最低%": round(stt["iv_min"], 1) if not math.isnan(stt["iv_min"]) else None,
                                               "IV最高%": round(stt["iv_max"], 1) if not math.isnan(stt["iv_max"]) else None,
                                               "合约IV百分位": round(stt["iv_pct"], 0) if not math.isnan(stt["iv_pct"]) else None})
                            else:
                                rows_h.append({"合约": lab, "首次成交": "无数据", "有成交天数": 0})
                            if show_theory and c.get("iv"):
                                start = oh.index[0] if len(oh) else under.index[-min(len(under), 250)]
                                u = under[under.index >= start]
                                th = bs_price_path(u, c["strike"], dt.date.fromisoformat(c["expiry"]), c.get("r", 0.04), c.get("q", 0.0), c["iv"] / 100)
                                figh.add_trace(go.Scatter(x=th.index, y=th.values, name=f"{lab} 理论价", line=dict(color=col, width=1, dash="dash"), opacity=0.7,
                                                          hovertemplate="%{x|%Y-%m-%d} 理论 %{y:.2f}<extra></extra>"), row=1, col=1, secondary_y=False)
                        if show_under:
                            u2 = under.tail(260)
                            figh.add_trace(go.Scatter(x=u2.index, y=u2.values, name=f"{sel_l} 正股", line=dict(color="#9ca3af", width=1.2)), row=1, col=1, secondary_y=True)
                        figh.update_layout(template=TEMPLATE, height=560, margin=dict(l=40, r=20, t=30, b=30), legend=dict(orientation="h", y=1.08, font=dict(size=10)))
                        figh.update_yaxes(title_text="期权价格", row=1, col=1, secondary_y=False)
                        figh.update_yaxes(title_text="正股", row=1, col=1, secondary_y=True)
                        figh.update_yaxes(title_text="IV %", row=2, col=1)
                        figh.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                        st.plotly_chart(figh, width="stretch", config=PLOTLY_CONFIG)
                        if rows_h:
                            st.dataframe(pd.DataFrame(rows_h), hide_index=True, width="stretch",
                                         column_config={"区间涨跌%": st.column_config.NumberColumn(format="%+.1f%%"),
                                                        "合约IV百分位": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%",
                                                                                                   help="这张合约当前 IV 在它自己（用成交价反推的）历史 IV 里的位置，不是正股的 IV Rank"),
                                                        **{c_: st.column_config.NumberColumn(format="%.1f%%") for c_ in ["当前IV%", "IV中位%", "IV最低%", "IV最高%"]}})
                        st.caption("成交价来自 Yahoo，按每天最后一笔成交记，成交少的日子会跳空或滞后；有些合约是几个月前才挂牌的，所以起点不一样。"
                                   "历史 IV 是用每天的成交价、正股收盘、剩余期限反推的 Black-Scholes 隐含波动率，只取当天有成交且价格高于内在价值的日子，所以有断点；"
                                   "IV 百分位高说明这张合约现在相对自己的历史偏贵，买 LEAP 一般希望在 IV 低位。理论价虚线假设 IV 固定为当前值。")
                if missing_l:
                    st.caption(f"没有 LEAP 链或抓取失败：{', '.join(missing_l)}")
                st.caption(
                    "口径：按目标 delta 在每个到期月选一档行权价（Black-Scholes，用链上 IV、13 周国债利率、股息率）。杠杆 = delta × 现价 / 权利金。"
                    "隐含胜率 = 风险中性下到期价高于盈亏平衡的概率；历史胜率 = 过去十几年全部同长度持有期里，正股涨幅达到需涨幅的比例，窗口重叠且这批股票过去十年表现极强，会偏乐观。"
                    "年化成本% 是拿 LEAP 代替正股每年付出的时间价值。样本少于 500（上市不久，如 SNDK）的历史胜率标红，不要当真。"
                    "LEAP 到期日稀疏（一般是 1/3/6/9 月和次年 1 月），8 个月和 10 个月可能落到同一个到期日，只保留一张。周末拿到的是周五收盘报价。全是历史与定价统计，不是建议。"
                )
        except Exception as e:
            st.error(f"LEAP 面板失败：{e}")


# ================= 底部确认 =================
def bottom_chart(sym: str, df: pd.DataFrame, res: dict, p: BottomParams):
    w = df.tail(p.lookback)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=w.index, open=w["open"], high=w["high"], low=w["low"], close=w["close"], name=sym), 1, 1)
    fig.add_trace(go.Scatter(x=w.index, y=df["close"].rolling(p.sma_n).mean().reindex(w.index), name=f"SMA{p.sma_n}", line=dict(color="#2563eb", width=1.2)), 1, 1)
    fig.add_trace(go.Scatter(x=w.index, y=df["close"].rolling(50).mean().reindex(w.index), name="SMA50", line=dict(color="#f59e0b", width=1)), 1, 1)
    for key, label, color in (("_L0", "前低", "#6b7280"), ("_L1", "最近低点", "#dc2626")):
        i = res.get(key)
        if i is not None:
            t = w.index[i]
            fig.add_trace(go.Scatter(x=[t], y=[w["low"].iloc[i] * 0.985], mode="markers+text", text=[label], textposition="bottom center",
                                     marker=dict(symbol="triangle-up", color=color, size=12), name=label, showlegend=False), 1, 1)
    neck = res.get("_neckline")
    if neck is not None and not (isinstance(neck, float) and np.isnan(neck)):
        fig.add_hline(y=neck, line=dict(color="#7c3aed", width=1.2, dash="dash"), annotation_text=f"颈线 {neck:.2f}", annotation_position="top left", row=1, col=1)
    colors = ["#16a34a" if c >= o else "#dc2626" for c, o in zip(w["close"], w["open"])]
    fig.add_trace(go.Bar(x=w.index, y=w["volume"], name="成交量", marker_color=colors, opacity=0.6), 2, 1)
    fig.add_trace(go.Scatter(x=w.index, y=df["volume"].rolling(20).mean().reindex(w.index), name="20日均量", line=dict(color="#6b7280", width=1)), 2, 1)
    fig.update_layout(template=TEMPLATE, height=520, margin=dict(l=40, r=20, t=10, b=30), xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.05))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


with tab_bottom:
    b1, b2, b3, b4 = st.columns([2, 1.6, 1.2, 1])
    wl_groups_all = [g for g in S.watchlist.groups if S.watchlist.enabled.get(g, True)]
    default_bg = [g for g in wl_groups_all if g.lower() in S.view_groups] or wl_groups_all
    pick_bg = b1.multiselect("板块", wl_groups_all, default=default_bg, key="bottom_groups")
    stage_filter = b2.multiselect("阶段", ["确认", "确认后回踩", "确认（未抬高）", "初步企稳", "未见底", "跌破低点", "无明显下跌"],
                                  default=["确认", "确认后回踩", "确认（未抬高）", "初步企稳"], key="bottom_stages")
    with b3.popover("参数"):
        bp_lookback = st.number_input("回看 bar 数", 60, 400, 120, 10, key="bp_lookback")
        bp_k = st.number_input("摆动窗口 k", 2, 15, 5, 1, key="bp_k", help="低点左右各 k 根 bar 内最低才算摆动低点")
        bp_dd = st.number_input("最小跌幅 %", 3.0, 60.0, 12.0, 1.0, key="bp_dd", help="低点相对之前最高价的跌幅至少这么多，才算有底可确认")
        bp_vol = st.number_input("量能倍数", 1.0, 4.0, 1.3, 0.1, key="bp_vol", help="突破日成交量 / 20 日均量")
    if b4.button("重算", key="bottom_refresh"):
        load_data.clear()
    bp = BottomParams(lookback=int(bp_lookback), swing_k=int(bp_k), drawdown_min=float(bp_dd) / 100, vol_mult=float(bp_vol))
    syms_b = [x for x in S.symbols if any(x in S.watchlist.groups[g] for g in pick_bg)]
    if not syms_b:
        st.info("先选板块。")
    else:
        try:
            daily_b = load_data(tuple(syms_b), "1d", 750)
            tbl_b = build_bottom_table(syms_b, daily_b, bp, S.watchlist.groups)
            n_all = len(tbl_b)
            if stage_filter:
                tbl_b = tbl_b[tbl_b["阶段"].isin(stage_filter)]
            counts = build_bottom_table(syms_b, daily_b, bp, S.watchlist.groups)["阶段"].value_counts().to_dict() if n_all else {}
            section_header("底部确认筛选", [(f"{len(tbl_b)}/{n_all} 只", ""), *[(f"{k} {v}", "ok" if k.startswith("确认") else ("warn" if k == "初步企稳" else "")) for k, v in counts.items()]])
            if tbl_b.empty:
                st.caption("没有符合所选阶段的标的。")
            else:
                cols_b = ["标的", "板块", "阶段", "得分", "现价", "跌幅%", "低点日", "低点价", "前低价", "低点抬高", "颈线", "突破颈线", "距颈线%",
                          "收复SMA20", "量能", "量比", "RSI背离", "RSI", "距低点%", "低点距今"]
                show_b = tbl_b[[c for c in cols_b if c in tbl_b.columns]].reset_index(drop=True)
                tick = lambda v: "color:#15803d;font-weight:700" if v == "✓" else ("color:#b91c1c" if v == "✗" else ("color:#b45309" if v == "回踩" else ""))
                stage_style = lambda v: ("color:#15803d;font-weight:700" if isinstance(v, str) and v.startswith("确认") else
                                         ("color:#b45309;font-weight:600" if v == "初步企稳" else ("color:#b91c1c" if v == "跌破低点" else "")))
                styler_b = show_b.style.map(tick, subset=[c for c in ["低点抬高", "突破颈线", "收复SMA20", "量能", "RSI背离"] if c in show_b.columns]).map(stage_style, subset=["阶段"])
                ev_b = st.dataframe(styler_b, hide_index=True, width="stretch", height=min(42 + 35 * len(show_b), 700),
                                    on_select="rerun", selection_mode="single-row", key=f"bottom_tbl_v{st.session_state.get('tbl_version', 0)}",
                                    column_config={
                                        "得分": st.column_config.ProgressColumn(min_value=0, max_value=6, format="%d/6", help="六条规则满足的条数"),
                                        "跌幅%": st.column_config.NumberColumn(format="%.1f%%", help="最近低点相对之前最高价的跌幅"),
                                        "距颈线%": st.column_config.NumberColumn(format="%+.1f%%"),
                                        "距低点%": st.column_config.NumberColumn(format="%+.1f%%"),
                                        "量比": st.column_config.NumberColumn(format="%.2f"),
                                        "RSI": st.column_config.NumberColumn(format="%.0f"),
                                        "低点距今": st.column_config.NumberColumn(format="%d bar"),
                                        "现价": st.column_config.NumberColumn(format="%.2f"), "低点价": st.column_config.NumberColumn(format="%.2f"),
                                        "前低价": st.column_config.NumberColumn(format="%.2f"), "颈线": st.column_config.NumberColumn(format="%.2f"),
                                    })
                sel_sym = show_b.iloc[ev_b.selection.rows[0]]["标的"] if (ev_b is not None and ev_b.selection.rows) else show_b.iloc[0]["标的"]
                res_b = analyze_bottom(daily_b[sel_sym], bp)
                st.markdown(f"**{sel_sym}** · 阶段 **{res_b['阶段']}** · 得分 {res_b.get('得分', 0)}/6" + (f" · {res_b['说明']}" if res_b.get("说明") else ""))
                st.plotly_chart(bottom_chart(sel_sym, daily_b[sel_sym], res_b, bp), width="stretch", config=PLOTLY_CONFIG)
                c_b1, c_b2 = st.columns([3, 1])
                c_b1.caption("点表格任意一行切换图。规则：前期跌幅 ≥ 阈值 → 低点抬高 → 收盘突破两个低点之间的颈线 → 收复 SMA20 且均线拐头 → 突破日放量 → RSI 底背离或回到 50 上方。"
                             "“确认后回踩”= 曾突破颈线、现在回落到颈线下方但仍在 SMA20 附近。最后 k 根 bar 的低点还无法确认，所以刚见底的股票会晚几天才出现。")
                if c_b2.button("打开详情", key="bottom_detail"):
                    request_detail(sel_sym)
        except Exception as e:
            st.error(f"底部确认失败：{e}")


# ================= 回测 =================
with tab_bt:
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.subheader("参数")
        names = list(STRATEGIES)
        strat_name = st.selectbox("策略", names, index=names.index(S.strategy) if S.strategy in names else 0)
        params = strategy_param_inputs(STRATEGIES[strat_name], S.strategy_params if strat_name == S.strategy else {}, "bt")
        symbols_text = st.text_input("标的（逗号分隔，可用 @组名）", S.symbols_raw)
        include_foreign = st.checkbox("包含海外上市（仅监控）标的", value=False, help="三星、海力士等 Alpaca 交易不了，默认不进回测")
        interval = st.radio("周期", ["1d", "1h"], index=0 if S.interval == "1d" else 1, horizontal=True)
        days = st.slider("回溯天数", 30, MAX_DAYS[interval], min(default_days(interval), MAX_DAYS[interval]), key=f"bt_days_{interval}",
                         help="yfinance 小时线最多 729 天；日线可到上市以来")
        fill = st.radio("成交假设", ["next_open", "close"], horizontal=True, help="next_open: 信号后下一根开盘成交；close: 信号 bar 收盘成交（偏乐观）")
        cost_bps = st.number_input("单边成本 bps", value=float(S.commission_bps + S.slippage_bps), step=0.5)
        max_pos = st.slider("单标的仓位上限", 0.05, 1.0, float(S.max_position_pct), 0.05)
        use_range = st.checkbox("只统计某个区间", value=False, help="信号仍用全部历史算，只把统计裁到区间内，用来做样本内 / 样本外对比")
        bt_start = bt_end = None
        if use_range:
            d1, d2 = st.columns(2)
            bt_start = d1.date_input("起", value=dt.date(2022, 1, 1), key="bt_start")
            bt_end = d2.date_input("止", value=dt.date.today(), key="bt_end")
        run_bt = st.button("运行回测", type="primary", width="stretch")

    with right:
        if run_bt:
            try:
                symbols = expand(symbols_text, S.watchlist.groups)
                if not include_foreign:
                    symbols = [x for x in symbols if is_us_listed(x)]
                data = load_data(tuple(symbols), interval, days)
                strategy = get_strategy(strat_name, **params)
                common = dict(initial_cash=S.initial_cash, commission_bps=0.0, slippage_bps=cost_bps, fill=fill)
                res = run_backtest(data, strategy, max_position_pct=max_pos, **common)
                bench = run_backtest(data, BuyAndHold(), **common)
                if use_range and (bt_start or bt_end):
                    res = slice_result(res, pd.Timestamp(bt_start), pd.Timestamp(bt_end) + pd.Timedelta(days=1), S.initial_cash)
                    bench = slice_result(bench, pd.Timestamp(bt_start), pd.Timestamp(bt_end) + pd.Timedelta(days=1), S.initial_cash)
                st.session_state["bt"] = (res, bench, symbols, interval)
            except Exception as e:
                st.error(f"回测失败：{e}")

        if "bt" not in st.session_state:
            st.info("左侧设置参数后点“运行回测”。")
        else:
            res, bench, symbols, bt_interval = st.session_state["bt"]
            m, b = res.metrics, bench.metrics
            st.subheader(f"{res.strategy}  vs  buy_and_hold")
            st.caption(f"{bt_interval} · {len(symbols)} 个标的 · {res.equity.index[0]:%Y-%m-%d} 至 {res.equity.index[-1]:%Y-%m-%d}")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("总收益", f"{m['total_return'] * 100:.1f}%", pct(m["total_return"] - b["total_return"]) + " vs 基准")
            k2.metric("年化收益", f"{m['cagr'] * 100:.1f}%", pct(m["cagr"] - b["cagr"]) + " vs 基准")
            k3.metric("Sharpe", f"{m['sharpe']:.2f}", f"{m['sharpe'] - b['sharpe']:+.2f} vs 基准")
            k4.metric("最大回撤", f"{m['max_drawdown'] * 100:.1f}%", pct(m["max_drawdown"] - b["max_drawdown"]) + " vs 基准")

            dd_s = res.equity / res.equity.cummax() - 1
            dd_b = bench.equity / bench.equity.cummax() - 1
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.25, 0.2], vertical_spacing=0.04,
                                subplot_titles=("净值", "回撤", "仓位"))
            fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity, name=res.strategy, line=dict(color="#2563eb", width=1.6)), 1, 1)
            fig.add_trace(go.Scatter(x=bench.equity.index, y=bench.equity, name="buy_and_hold", line=dict(color="#f59e0b", width=1.2)), 1, 1)
            fig.add_trace(go.Scatter(x=dd_s.index, y=dd_s, name="策略回撤", fill="tozeroy", line=dict(color="#2563eb", width=0.8), showlegend=False), 2, 1)
            fig.add_trace(go.Scatter(x=dd_b.index, y=dd_b, name="基准回撤", line=dict(color="#f59e0b", width=0.8), showlegend=False), 2, 1)
            fig.add_trace(go.Scatter(x=res.exposure.index, y=res.exposure, name="仓位", fill="tozeroy", line=dict(color="#6b7280", width=0.8), showlegend=False), 3, 1)
            fig.update_yaxes(tickformat=".0%", row=2, col=1)
            fig.update_yaxes(tickformat=".0%", row=3, col=1)
            fig.update_layout(template=TEMPLATE, height=620, margin=dict(l=40, r=20, t=40, b=30), legend=dict(orientation="h", y=1.06))
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

            c1, c2 = st.columns([1, 1], gap="large")
            with c1:
                st.markdown("**指标**")
                st.dataframe(metrics_table({res.strategy: m, "buy_and_hold": b}, chinese=True), width="stretch")
            with c2:
                st.markdown("**各标的累计贡献**")
                contrib = res.per_symbol_returns.cumsum()
                fig2 = go.Figure()
                for sym in contrib.columns:
                    fig2.add_trace(go.Scatter(x=contrib.index, y=contrib[sym], name=sym, line=dict(width=1.2)))
                fig2.update_layout(template=TEMPLATE, height=330, margin=dict(l=40, r=20, t=10, b=30), yaxis_tickformat=".1%")
                st.plotly_chart(fig2, width="stretch", config=PLOTLY_CONFIG)
                st.markdown("**最新目标权重**（策略在最后一根 bar 想持有的比例）")
                latest = res.signals.iloc[-1]
                st.dataframe(latest[latest > 0].rename("weight").to_frame().T if (latest > 0).any() else pd.DataFrame({"空仓": [""]}),
                             width="stretch", hide_index=True)

            ts = res.trade_stats
            if ts.get("n_trades"):
                st.markdown("**逐笔交易**（成交口径：信号后下一根开盘进出）")
                t1, t2, t3, t4, t5, t6 = st.columns(6)
                t1.metric("已平仓", ts["n_trades"], f"持有中 {ts.get('open_trades', 0)}", delta_color="off")
                t2.metric("胜率", f"{ts['win_rate'] * 100:.0f}%")
                t3.metric("平均收益", f"{ts['avg_ret'] * 100:+.2f}%", f"中位 {ts['median_ret'] * 100:+.2f}%", delta_color="off")
                t4.metric("均赢 / 均亏", f"{ts['avg_win'] * 100:+.1f}% / {ts['avg_loss'] * 100:+.1f}%")
                t5.metric("盈亏比", f"{ts['payoff']:.2f}")
                t6.metric("平均持有", f"{ts['avg_hold_bars']:.0f} bar", f"最好 {ts['best'] * 100:+.0f}% 最差 {ts['worst'] * 100:+.0f}%", delta_color="off")
                show = res.trades.sort_values("entry_time", ascending=False).copy()
                show["entry_time"] = pd.to_datetime(show["entry_time"]).dt.strftime("%Y-%m-%d %H:%M")
                show["exit_time"] = pd.to_datetime(show["exit_time"]).dt.strftime("%Y-%m-%d %H:%M")
                show["ret"] = show["ret"].map(lambda x: f"{x * 100:+.2f}%")
                show["weight"] = show["weight"].map(lambda x: f"{x * 100:.0f}%")
                show = show.rename(columns={"symbol": "标的", "entry_time": "入场", "exit_time": "出场", "entry_px": "入场价", "exit_px": "出场价",
                                            "ret": "收益", "bars_held": "持有 bar", "weight": "权重", "open": "持有中"})
                st.dataframe(show.round(2), width="stretch", hide_index=True, height=320)
                st.download_button("下载逐笔交易 CSV", res.trades.to_csv(index=False).encode("utf-8-sig"), "trades.csv", "text/csv")


# ================= 信号验证 =================
@st.cache_data(ttl=1800, show_spinner="统计历史信号之后的走势 ...")
def cached_event_study(name: str, pjson: str, symbols: tuple[str, ...], start: str, end: str, dedupe: int, regime_only: bool = False):
    strat = get_strategy(name, **json.loads(pjson))
    data = load_data(symbols, "1d", default_days("1d"))
    if regime_only:
        data = dict(data)
        data.setdefault(getattr(strat, "regime_symbol", "SPY"), load_data((getattr(strat, "regime_symbol", "SPY"),), "1d", default_days("1d"))[getattr(strat, "regime_symbol", "SPY")])
    return event_study(strat, data, dedupe_bars=dedupe, start=start or None, end=end or None, regime_only=regime_only)


@st.cache_data(ttl=1800, show_spinner="逐个策略统计每只标的的胜率 ...")
def cached_matrix(names: tuple[str, ...], symbols: tuple[str, ...], start: str, end: str, dedupe: int, regime_only: bool, h: int = 20) -> pd.DataFrame:
    """策略 × 标的：信号数、胜率、平均、基准胜率/平均、超额。长表，每行一个 (策略, 标的)。"""
    data = load_data(symbols, "1d", default_days("1d"))
    rows = []
    for name in names:
        strat = get_strategy(name, **strat_params(name))
        d = dict(data)
        if regime_only:
            rs = getattr(strat, "regime_symbol", "SPY")
            d.setdefault(rs, load_data((rs,), "1d", default_days("1d"))[rs])
        try:
            res = event_study(strat, d, dedupe_bars=dedupe, start=start or None, end=end or None, regime_only=regime_only)
        except Exception:
            continue
        col = f"fwd_{h}"
        sig, base = res.signals, res.baseline
        if col not in sig.columns:
            continue
        b_by = base.groupby("symbol")[col].agg(基准平均=lambda x: x.mean(), 基准胜率=lambda x: (x > 0).mean() * 100) if len(base) else pd.DataFrame()
        for sym, g in sig.groupby("symbol"):
            v = g[col].dropna()
            if len(v) == 0:
                continue
            bm = float(b_by.loc[sym, "基准平均"]) if sym in b_by.index else np.nan
            bw = float(b_by.loc[sym, "基准胜率"]) if sym in b_by.index else np.nan
            rows.append({"策略": STRAT_LABEL[name], "策略key": name, "标的": sym, "信号数": int(len(v)),
                         "胜率%": float((v > 0).mean() * 100), "平均%": float(v.mean()), "基准胜率%": bw, "基准平均%": bm,
                         "超额胜率": float((v > 0).mean() * 100 - bw) if not np.isnan(bw) else np.nan, "超额平均%": float(v.mean() - bm) if not np.isnan(bm) else np.nan})
    return pd.DataFrame(rows)


with tab_verify:
    v1, v2, v3, v4 = st.columns([1.6, 2, 1.6, 1])
    v_name = v1.selectbox("策略", list(STRATEGIES), index=list(STRATEGIES).index(S.strategy) if S.strategy in STRATEGIES else 0,
                          format_func=lambda n: f"{STRAT_LABEL[n]} ({n})", key="ev_strat")
    wl_groups_v = [g for g in S.watchlist.groups if S.watchlist.enabled.get(g, True) and any(is_us_listed(x) for x in S.watchlist.groups[g])]
    pick_v = v2.multiselect("板块", wl_groups_v, default=wl_groups_v, key="ev_groups", help="验证时默认用全部美股标的，样本更多")
    with v3:
        r1, r2 = st.columns(2)
        ev_start = r1.date_input("起", value=dt.date(2015, 1, 1), key="ev_start")
        ev_end = r2.date_input("止", value=dt.date.today(), key="ev_end")
    dedupe = v4.number_input("去重bar", 1, 60, 10, 1, key="ev_dedupe", help="同一标的多少根 bar 内的重复信号只算第一次")
    regime_only = st.checkbox("只统计大盘在 200 日线上方时的信号", value=False, key="ev_regime", help="对比勾选前后的超额，就能看到大盘过滤有没有用")
    pool_v = [x for x in S.symbols if is_us_listed(x) and any(x in S.watchlist.groups[g] for g in pick_v)]
    syms_v = st.multiselect("标的（默认板块内全部，可只留几只）", pool_v, default=pool_v, key=f"ev_symbols_{hash(tuple(pool_v))}",
                            format_func=lambda x: x + (f" {display_name(x)}" if display_name(x) else ""))
    if not syms_v:
        st.info("先选板块或标的。")
    else:
        try:
            with st.expander("策略参数（默认用 .env / 默认值，可改后重算）", expanded=False):
                v_params = strategy_param_inputs(STRATEGIES[v_name], strat_params(v_name), "ev")
            res_v = cached_event_study(v_name, json.dumps(v_params, sort_keys=True), tuple(syms_v), str(ev_start), str(ev_end), int(dedupe), bool(regime_only))
            summ = res_v.summary
            n_sig = int(summ["信号数"].max()) if len(summ) else 0
            section_header(f"{STRAT_LABEL[v_name]} 信号出现之后", [(f"{n_sig} 个信号", "ok" if n_sig else ""), (f"{len(syms_v)} 只标的", ""), (f"{ev_start} 至 {ev_end}", "")])
            if n_sig == 0:
                st.warning("这段时间没有信号。")
            else:
                k1, k2, k3, k4 = st.columns(4)
                row20 = summ.set_index("持有bar").loc[20] if 20 in summ["持有bar"].values else summ.iloc[-1]
                k1.metric("持有 20 bar 平均", f"{row20['信号平均%']:+.2f}%", f"{row20['超额%']:+.2f}% vs 基准")
                k2.metric("持有 20 bar 胜率", f"{row20['信号胜率%']:.0f}%", f"{row20['信号胜率%'] - row20['基准胜率%']:+.0f}% vs 基准")
                k3.metric("t 值（20 bar）", f"{row20['t值']:.1f}", "大于 2 才算有点意思" , delta_color="off")
                k4.metric("样本", f"{n_sig}", f"去重 {dedupe} bar", delta_color="off")

                c1, c2 = st.columns([1.4, 1], gap="large")
                with c1:
                    st.markdown("**信号后平均累计收益路径** vs 基准（同一批标的任意一天）")
                    figv = go.Figure()
                    figv.add_trace(go.Scatter(x=res_v.path_signal.index, y=res_v.path_signal.values, name="信号后", line=dict(color="#2563eb", width=2)))
                    figv.add_trace(go.Scatter(x=res_v.path_baseline.index, y=res_v.path_baseline.values, name="基准（任意一天）", line=dict(color="#9ca3af", width=1.5, dash="dot")))
                    figv.add_hline(y=0, line=dict(color="#6b7280", width=1))
                    figv.update_layout(template=TEMPLATE, height=340, margin=dict(l=40, r=20, t=10, b=30), legend=dict(orientation="h", y=1.08),
                                       xaxis_title="信号后第几根 bar", yaxis_title="平均累计收益 %")
                    st.plotly_chart(figv, width="stretch", config=PLOTLY_CONFIG)
                with c2:
                    st.markdown("**按持有期汇总**")
                    st.dataframe(summ, hide_index=True, width="stretch", height=42 + 35 * len(summ),
                                 column_config={**{c: st.column_config.NumberColumn(format="%+.2f%%") for c in ["信号平均%", "信号中位%", "基准平均%", "超额%"]},
                                                **{c: st.column_config.NumberColumn(format="%.0f%%") for c in ["信号胜率%", "基准胜率%"]},
                                                "t值": st.column_config.NumberColumn(format="%.1f"), "持有bar": st.column_config.NumberColumn(format="%d")})

                h_pick = st.select_slider("分布图持有期", options=[int(h) for h in summ["持有bar"]], value=20 if 20 in summ["持有bar"].values else int(summ["持有bar"].iloc[-1]), key="ev_h")
                d1, d2 = st.columns([1.4, 1], gap="large")
                with d1:
                    st.markdown(f"**持有 {h_pick} bar 收益分布**：信号 vs 基准")
                    sv = res_v.signals[f"fwd_{h_pick}"].dropna()
                    bv = res_v.baseline[f"fwd_{h_pick}"].dropna()
                    figd = go.Figure()
                    figd.add_trace(go.Histogram(x=bv, name="基准", histnorm="probability", opacity=0.45, marker_color="#9ca3af", nbinsx=60))
                    figd.add_trace(go.Histogram(x=sv, name="信号", histnorm="probability", opacity=0.6, marker_color="#2563eb", nbinsx=60))
                    figd.add_vline(x=0, line=dict(color="#6b7280", width=1))
                    figd.update_layout(template=TEMPLATE, barmode="overlay", height=300, margin=dict(l=40, r=20, t=10, b=30), legend=dict(orientation="h", y=1.08),
                                       xaxis_title="收益 %", yaxis_title="占比")
                    st.plotly_chart(figd, width="stretch", config=PLOTLY_CONFIG)
                with d2:
                    st.markdown("**按年份**（稳定性：是不是只有某几年好）")
                    by = res_v.by_year
                    st.dataframe(by, hide_index=True, width="stretch", height=min(42 + 35 * len(by), 300),
                                 column_config={c: st.column_config.NumberColumn(format="%+.1f%%" if "平均" in c else "%.0f%%") for c in by.columns if "bar" in c})
                st.markdown("**按标的**（MAE = 20 bar 内最大不利偏移，用来定止损；MFE = 最大有利偏移）")
                bs = res_v.by_symbol
                st.dataframe(bs.style.map(lambda v: "color:#15803d" if isinstance(v, float) and v > 0 else ("color:#b91c1c" if isinstance(v, float) and v < 0 else ""),
                                          subset=[c for c in bs.columns if "平均" in c and "MAE" not in c and "MFE" not in c]),
                             hide_index=True, width="stretch", height=min(42 + 35 * len(bs), 600),
                             column_config={c: st.column_config.NumberColumn(format="%+.1f%%" if ("平均" in c or "MAE" in c or "MFE" in c) else "%.0f%%") for c in bs.columns if c not in ("标的", "信号数")})
                with st.expander("全部信号明细"):
                    sig_show = res_v.signals.copy()
                    sig_show["date"] = pd.to_datetime(sig_show["date"]).dt.strftime("%Y-%m-%d")
                    st.dataframe(sig_show.drop(columns=["year"], errors="ignore").sort_values("date", ascending=False), hide_index=True, width="stretch", height=400)
                st.caption(
                    "怎么读：看“超额%”和胜率差是不是明显大于 0、t 值是否大于 2、路径曲线是否持续高于基准、按年份是否大多数年份都为正。"
                    "口径：信号 bar 收盘出现，下一根开盘入场，不含成本、不含止损、不限仓位，测的是信号本身有没有信息量。"
                    "注意：这批标的是今天挑出来的赢家，过去十几年几乎一路向上，所以基准本身就是正的；重叠窗口会让 t 值偏大；参数改到刚好好看叫过拟合。"
                )
        except Exception as e:
            st.error(f"信号验证失败：{e}")

    # ---------- 策略 × 标的 ----------
    if syms_v:
        section_header("策略 × 标的 胜率", [("持有 20 bar", ""), (f"{len(syms_v)} 只 · {len(SWING_NAMES)} 个策略", ""), ("样本少的格子别当真", "warn")])
        m1, m2, m3 = st.columns([1.2, 1, 1.2])
        min_n = m1.slider("至少几个信号才计入", 1, 30, 5, key="mx_min_n")
        metric = m2.selectbox("矩阵显示", ["胜率%", "超额胜率", "超额平均%", "平均%", "信号数"], key="mx_metric")
        try:
            mx = cached_matrix(tuple(SWING_NAMES), tuple(syms_v), str(ev_start), str(ev_end), int(dedupe), bool(regime_only))
            if mx.empty:
                st.caption("这段时间没有信号。")
            else:
                mx_ok = mx[mx["信号数"] >= min_n]
                # ---- 热力矩阵：标的 × 策略 ----
                piv = mx_ok.pivot(index="标的", columns="策略", values=metric).reindex(columns=[STRAT_LABEL[n] for n in SWING_NAMES])
                cnt = mx_ok.pivot(index="标的", columns="策略", values="信号数").reindex(columns=piv.columns)
                piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
                cnt = cnt.reindex(index=piv.index, columns=piv.columns)  # 行序跟着胜率矩阵走，否则样本数会错位
                zmid = {"胜率%": 50, "基准胜率%": 50}.get(metric, 0)
                # 用 st.dataframe 的单格选择：点格子 → rerun，原生高亮，不用 plotly 的点击事件（那个在 Streamlit 里会慢一步）
                from plotly.colors import sample_colorscale

                rows_l, cols_l = list(piv.index), list(piv.columns)
                vals_all = piv.values[~np.isnan(piv.values)]
                if metric in ("胜率%", "基准胜率%"):
                    lo_v, hi_v = 0.0, 100.0
                else:
                    span = float(np.nanmax(np.abs(vals_all))) if len(vals_all) else 1.0
                    lo_v, hi_v = -span, span
                txt = pd.DataFrame("", index=rows_l, columns=cols_l)
                css = pd.DataFrame("", index=rows_l, columns=cols_l)
                for sym_i in rows_l:
                    for lab_j in cols_l:
                        v_, n_ = piv.loc[sym_i, lab_j], cnt.loc[sym_i, lab_j]
                        if pd.isna(v_) or pd.isna(n_):
                            css.loc[sym_i, lab_j] = "background-color:rgba(0,0,0,0.03);color:#9ca3af"
                            continue
                        t_ = 0.0 if hi_v == lo_v else min(max((float(v_) - lo_v) / (hi_v - lo_v), 0.0), 1.0)
                        color = sample_colorscale("RdYlGn", [t_])[0]
                        txt.loc[sym_i, lab_j] = f"{v_:.0f} ({int(n_)})"
                        css.loc[sym_i, lab_j] = f"background-color:{color};color:#111827;text-align:center"
                txt.index.name = "标的"
                ev_m = st.dataframe(txt.style.apply(lambda _df: css, axis=None), width="stretch", height=min(40 + 35 * len(rows_l), 900),
                                    on_select="rerun", selection_mode="single-cell", key="mx_heat",
                                    column_config={c: st.column_config.TextColumn(c, width="small") for c in cols_l})
                st.caption(f"每格 = {metric}（括号里是信号数）。点任意格子，下面自动显示那个标的 × 策略的行情图和信号明细。")

                # ---- 点击格子：行情图 + 该格子的信号明细 ----
                click_cell = None
                try:
                    cells_sel = list(ev_m.selection.cells) if ev_m is not None else []
                    if cells_sel:
                        r_pos, c_name = cells_sel[0]
                        c_sym, c_lab = rows_l[int(r_pos)], str(c_name)
                        c_key = next((k for k, v in STRAT_LABEL.items() if v == c_lab), None)
                        if c_key and c_sym in syms_v and txt.loc[c_sym, c_lab] != "":
                            click_cell = (c_sym, c_key)
                except Exception:
                    click_cell = None
                # 保底：也可以用“查看格子”两个下拉框手动选
                cc1, cc2, cc3 = st.columns([1, 1, 2])
                pick_cell_sym = cc1.selectbox("查看格子 · 标的", sorted(mx_ok["标的"].unique()), key="mx_cell_sym")
                pick_cell_strat = cc2.selectbox("查看格子 · 策略", SWING_NAMES, format_func=lambda n: STRAT_LABEL[n], key="mx_cell_strat")
                cc3.caption("点上面矩阵的格子会自动切换到那一格；也可以在这里手动选。谁最后改动听谁的。")
                # 谁最后变了听谁的：点击变了用点击，下拉变了用下拉，否则沿用上次
                box_cell = (pick_cell_sym, pick_cell_strat)
                prev_click, prev_box = st.session_state.get("mx_prev_click"), st.session_state.get("mx_prev_box")
                cur = st.session_state.get("mx_cur")
                if click_cell and click_cell != prev_click:
                    cur = click_cell
                elif box_cell != prev_box:
                    cur = box_cell
                elif cur is None:
                    cur = click_cell or box_cell
                st.session_state["mx_prev_click"], st.session_state["mx_prev_box"], st.session_state["mx_cur"] = click_cell, box_cell, cur
                sel_sym, sel_key = cur
                if sel_sym not in syms_v:
                    sel_sym, sel_key = box_cell
                sel_lab = STRAT_LABEL[sel_key]
                if True:
                    if sel_key and sel_sym in syms_v:
                        cell = mx[(mx["标的"] == sel_sym) & (mx["策略key"] == sel_key)]
                        c_txt = (f"{int(cell.iloc[0]['信号数'])} 个信号 · 胜率 {cell.iloc[0]['胜率%']:.0f}%（基准 {cell.iloc[0]['基准胜率%']:.0f}%）· "
                                 f"平均 {cell.iloc[0]['平均%']:+.2f}%（基准 {cell.iloc[0]['基准平均%']:+.2f}%）") if len(cell) else "该格子样本不足"
                        section_header(f"{sel_sym} · {sel_lab}", [(c_txt, "")])
                        g1, g2 = st.columns([1.7, 1], gap="large")
                        with g1:
                            days_sel = st.select_slider("显示区间", options=[180, 365, 730, 1500, 3000], value=730, key="mx_days", format_func=lambda d: f"{d} 天")
                            try:
                                res_cell = cached_event_study(sel_key, json.dumps(strat_params(sel_key), sort_keys=True), (sel_sym,), str(ev_start), str(ev_end), int(dedupe), bool(regime_only))
                                fig_c, df_c, note_c = build_price_fig(sel_sym, "1d", days_sel, [sel_key], height=520, show_raw=False, show_trades=True, show_holding=True)
                                sg = res_cell.signals.copy()
                                if len(sg):
                                    sg["date"] = pd.to_datetime(sg["date"])
                                    sg = sg[(sg["date"] >= df_c.index[0]) & (sg["date"] <= df_c.index[-1])]
                                    lows = df_c["low"]
                                    for part, colr, nm in ((sg[sg["fwd_20"] > 0], "#15803d", "信号→20bar 后赢"), (sg[sg["fwd_20"] <= 0], "#dc2626", "信号→20bar 后亏")):
                                        if len(part):
                                            fig_c.add_trace(go.Scatter(x=part["date"], y=lows.reindex(part["date"]).values * 0.94, mode="markers", name=nm,
                                                                       marker=dict(symbol="circle", size=9, color=colr, line=dict(color="white", width=1)),
                                                                       text=[f"{d:%Y-%m-%d} 20bar {r:+.1f}%" for d, r in zip(part["date"], part["fwd_20"])],
                                                                       hovertemplate="%{text}<extra></extra>"), 1, 1)
                                st.plotly_chart(fig_c, width="stretch", config=PLOTLY_CONFIG)
                                st.caption("圆点 = 这个格子统计里用到的信号（已去重），绿 = 持有 20 bar 收益为正，红 = 为负；三角和底色是策略回放的实际进出与持仓。" + note_c)
                            except Exception as e:
                                st.error(f"图表失败：{e}")
                        with g2:
                            st.markdown("**信号明细**（持有 5 / 20 / 60 bar 收益，MAE = 20 bar 内最大回撤）")
                            try:
                                sg2 = res_cell.signals.copy()
                                if len(sg2):
                                    sg2["date"] = pd.to_datetime(sg2["date"]).dt.strftime("%Y-%m-%d")
                                    cols_show = [c for c in ["date", "fwd_5", "fwd_20", "fwd_60", "mae_20", "mfe_20"] if c in sg2.columns]
                                    show_sg = sg2[cols_show].rename(columns={"date": "信号日", "fwd_5": "5bar%", "fwd_20": "20bar%", "fwd_60": "60bar%", "mae_20": "MAE20%", "mfe_20": "MFE20%"}).sort_values("信号日", ascending=False)
                                    st.dataframe(show_sg.style.map(lambda v: "color:#15803d" if isinstance(v, float) and v > 0 else ("color:#b91c1c" if isinstance(v, float) and v < 0 else ""),
                                                                   subset=[c for c in show_sg.columns if c != "信号日"]),
                                                 hide_index=True, width="stretch", height=min(42 + 35 * len(show_sg), 560),
                                                 column_config={c: st.column_config.NumberColumn(format="%+.1f%%") for c in show_sg.columns if c != "信号日"})
                                else:
                                    st.caption("这段时间没有信号。")
                            except Exception as e:
                                st.error(f"明细失败：{e}")

                r1c, r2c = st.columns(2, gap="large")
                fmt = {"胜率%": st.column_config.NumberColumn(format="%.0f%%"), "基准胜率%": st.column_config.NumberColumn(format="%.0f%%"),
                       "超额胜率": st.column_config.NumberColumn(format="%+.0f", help="信号胜率 − 该标的任意一天的基准胜率，百分点"),
                       "平均%": st.column_config.NumberColumn(format="%+.2f%%"), "基准平均%": st.column_config.NumberColumn(format="%+.2f%%"),
                       "超额平均%": st.column_config.NumberColumn(format="%+.2f%%"), "信号数": st.column_config.NumberColumn(format="%d")}
                color_pos = lambda v: "color:#15803d;font-weight:600" if isinstance(v, float) and v > 0 else ("color:#b91c1c" if isinstance(v, float) and v < 0 else "")
                with r1c:
                    st.markdown("**每只个股：哪个策略最灵**（按胜率排）")
                    sym_pick = st.selectbox("个股", sorted(mx_ok["标的"].unique()), key="mx_sym")
                    t1 = mx_ok[mx_ok["标的"] == sym_pick].sort_values(["胜率%", "信号数"], ascending=[False, False])[["策略", "信号数", "胜率%", "基准胜率%", "超额胜率", "平均%", "超额平均%"]]
                    st.dataframe(t1.style.map(color_pos, subset=["超额胜率", "超额平均%"]), hide_index=True, width="stretch", height=42 + 35 * len(t1), column_config=fmt)
                with r2c:
                    st.markdown("**每个策略：在哪些个股上最灵**（按胜率排）")
                    st_pick = st.selectbox("策略", SWING_NAMES, format_func=lambda n: STRAT_LABEL[n], key="mx_strat")
                    t2 = mx_ok[mx_ok["策略key"] == st_pick].sort_values(["胜率%", "信号数"], ascending=[False, False])[["标的", "信号数", "胜率%", "基准胜率%", "超额胜率", "平均%", "超额平均%"]]
                    st.dataframe(t2.style.map(color_pos, subset=["超额胜率", "超额平均%"]), hide_index=True, width="stretch", height=min(42 + 35 * len(t2), 520), column_config=fmt)
                st.caption("胜率 = 信号后持有 20 根 bar 收益为正的比例；基准胜率 = 同一只股票任意一天持有 20 根 bar 的正收益比例；超额胜率 = 两者之差（百分点）。"
                           "这批股票过去十几年基准胜率本来就在 55%~65%，所以看超额比看绝对胜率有意义。信号少于所选下限的格子不显示；"
                           "一只股票十几个信号算出来的胜率抖动很大，别拿单格下结论。")
        except Exception as e:
            st.error(f"矩阵计算失败：{e}")


# ================= 模拟盘 =================
def loop_status() -> dict:
    """循环进程是否在跑、上次决策、下次决策时间。"""
    import os as _os2
    pid_file = S.log_dir / "run.pid"
    alive, pid = False, None
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            _os2.kill(pid, 0)
            alive = True
        except Exception:
            alive = False
    decs = read_decisions()
    last = pd.Timestamp(decs[-1]["time"]).tz_convert(NY) if decs else None
    now = pd.Timestamp.now(tz=NY)
    # 下次决策：日线 = 下一个交易日 9:35；小时线 = 下一个 10:31~15:31 的整点 31 分
    d = now
    for _ in range(10):
        candidate_day = d.normalize()
        is_trading = candidate_day.weekday() < 5 and candidate_day.date() not in nyse_holidays_set(candidate_day.year)
        if is_trading:
            slots = [candidate_day + pd.Timedelta(hours=9, minutes=35)] if S.interval == "1d" else [candidate_day + pd.Timedelta(hours=h, minutes=31) for h in range(10, 16)]
            future = [t for t in slots if t > now]
            if future:
                return {"alive": alive, "pid": pid, "last": last, "next": future[0]}
        d = candidate_day + pd.Timedelta(days=1)
    return {"alive": alive, "pid": pid, "last": last, "next": None}


def nyse_holidays_set(year: int):
    from tradebot.econ_calendar import nyse_holidays
    return nyse_holidays(year)


def running_strategy_card() -> None:
    ls = loop_status()
    strat = get_strategy(S.strategy, **S.strategy_params)
    params = ", ".join(f"{k}={v}" for k, v in S.strategy_params.items()) or "默认参数"
    rg = regime_status()
    dot = "#16a34a" if ls["alive"] else "#dc2626"
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 1.3, 1.3, 1.3])
        c1.markdown(f"**正在运行的策略**<br><span style='font-size:1.15rem;font-weight:700'>{STRAT_LABEL.get(S.strategy, S.strategy)}</span> "
                    f"<span style='color:var(--tb-muted);font-size:.85em'>{params}</span><br>"
                    f"<span style='color:var(--tb-muted);font-size:.85em'>周期 {S.interval} · 起步 {'只跟新入场' if S.entry_mode == 'fresh' else '对齐策略仓位'} · "
                    f"最多 {getattr(strat, 'max_positions', '-')} 只 · ATR 止损 {getattr(strat, 'atr_stop_mult', '-')}× · 大盘过滤 {'开' if getattr(strat, 'regime_filter', False) else '关'}"
                    f"{'' if rg.get('ok', True) else '（当前拦截新仓）'}</span>", unsafe_allow_html=True)
        c2.markdown(f"**循环进程**<br><span class='tb-dot' style='background:{dot}'></span>{'运行中' if ls['alive'] else '未运行'}"
                    + (f" <span style='color:var(--tb-muted);font-size:.8em'>pid {ls['pid']}</span>" if ls["pid"] else ""), unsafe_allow_html=True)
        c3.markdown(f"**上次决策**<br>{ls['last']:%m-%d %H:%M}" if ls["last"] is not None else "**上次决策**<br>—", unsafe_allow_html=True)
        c4.markdown(f"**下次决策**<br>{ls['next']:%m-%d %H:%M} ET" if ls["next"] is not None else "**下次决策**<br>—", unsafe_allow_html=True)
        st.caption("风控：单标的 " + f"{S.max_position_pct:.0%}" + f" · 总仓位 {S.max_gross_exposure:.0%} · 当日亏损 {S.daily_loss_limit_pct:.0%} 熔断 · 回撤 {S.max_drawdown_pct:.0%} 停机 · 决策用最后一根完整 bar，下一根开盘成交")


with tab_paper:
    try:
        broker = make_broker(S)
        if S.mode == "local":
            live = load_data(tuple(S.symbols), S.interval, S.lookback_days)
            broker.set_prices(latest_prices(live))
        acct = broker.account()
        positions = broker.positions()
        risk_state = read_json(S.log_dir / "risk_state.json")
        day_start = risk_state.get("day_start_equity", acct.equity)
        peak = risk_state.get("peak_equity", acct.equity)

        running_strategy_card()
        is_alpaca = broker.name.startswith("alpaca")
        h1, h2 = st.columns([3, 1])
        h1.subheader(f"账户 · {broker.name}" + ("（Alpaca 模拟盘，虚拟资金）" if broker.name == "alpaca_paper" else ("（Alpaca 实盘）" if broker.name == "alpaca_live" else "（本地模拟撮合）")))
        with h2:
            b1, b2 = st.columns(2)
            if not CLOUD and b1.button("立刻运行一次", type="primary", help="忽略开盘时间检查，跑一遍 拉数据→信号→风控→下单"):
                from run import build, run_once
                from tradebot.logging_utils import setup_logging

                log = setup_logging(S.log_dir)
                br, strat, risk, advisor = build(S, log)
                with st.spinner("运行中 ..."):
                    run_once(S, br, strat, risk, advisor, log, force=True)
                load_data.clear()
                st.rerun()
            if S.mode == "local" and not CLOUD:
                confirm = b2.checkbox("确认重置", key="confirm_reset")
                if b2.button("重置模拟盘", disabled=not confirm):
                    for name in ["paper_state.json", "risk_state.json", "decisions.jsonl"]:
                        (S.log_dir / name).unlink(missing_ok=True)
                    st.rerun()

        det = broker.account_details() if is_alpaca else None
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        if det:
            today_pl = (det["equity"] - det["last_equity"]) if det.get("last_equity") else None
            k1.metric("净值", f"{det['equity']:,.0f}", (f"{today_pl:+,.0f}（{today_pl / det['last_equity'] * 100:+.2f}%）今日" if today_pl is not None and det["last_equity"] else None))
            k2.metric("现金", f"{det['cash']:,.0f}", f"购买力 {det['buying_power']:,.0f}", delta_color="off")
            k3.metric("持仓市值", f"{sum(p.market_value for p in positions.values()):,.0f}", f"{len(positions)} 只", delta_color="off")
            k4.metric("距高点回撤", pct(acct.equity / peak - 1), f"停机线 -{S.max_drawdown_pct:.0%}", delta_color="off")
            k5.metric("账户状态", det["status"], f"日内交易 {det['daytrade_count']} 次" + ("，PDT" if det["pattern_day_trader"] else ""), delta_color="off")
            k6.metric("杠杆倍数", f"{det['multiplier']:.0f}x" if det.get("multiplier") else "—", "模拟盘" if det["paper"] else "实盘", delta_color="off")
        else:
            k1.metric("净值", f"{acct.equity:,.0f}", (pct(acct.equity / day_start - 1) if day_start else "n/a") + " 今日")
            k2.metric("现金", f"{acct.cash:,.0f}", f"{acct.cash / acct.equity * 100:.0f}% 仓外", delta_color="off")
            k3.metric("持仓数", f"{len(positions)}")
            k4.metric("距高点回撤", pct(acct.equity / peak - 1), f"停机线 -{S.max_drawdown_pct:.0%}", delta_color="off")
            k5.metric("当日熔断线", f"-{S.daily_loss_limit_pct:.0%}", f"单标的上限 {S.max_position_pct:.0%}", delta_color="off")
            k6.metric("模式", "本地模拟", "无真实账户", delta_color="off")

        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("**持仓**")
            if is_alpaca and positions:
                pdz = pd.DataFrame(broker.positions_detail())
                pdz = pdz.rename(columns={"symbol": "标的", "qty": "数量", "avg_price": "成本", "current_price": "现价", "market_value": "市值",
                                          "unrealized_pl": "浮动盈亏$", "unrealized_plpc": "浮动盈亏%", "change_today": "今日%"})
                pdz["占净值%"] = pdz["市值"] / acct.equity * 100
                st.dataframe(pdz[["标的", "数量", "成本", "现价", "市值", "浮动盈亏$", "浮动盈亏%", "今日%", "占净值%"]].style.map(
                    lambda v: "color:#15803d" if isinstance(v, float) and v > 0 else ("color:#b91c1c" if isinstance(v, float) and v < 0 else ""), subset=["浮动盈亏$", "浮动盈亏%", "今日%"]),
                    width="stretch", hide_index=True,
                    column_config={"成本": st.column_config.NumberColumn(format="%.2f"), "现价": st.column_config.NumberColumn(format="%.2f"), "市值": st.column_config.NumberColumn(format="%,.0f"),
                                   "浮动盈亏$": st.column_config.NumberColumn(format="%+,.0f"), "浮动盈亏%": st.column_config.NumberColumn(format="%+.2f%%"),
                                   "今日%": st.column_config.NumberColumn(format="%+.2f%%"), "占净值%": st.column_config.NumberColumn(format="%.1f%%")})
            elif positions:
                rows = []
                for sym, p in positions.items():
                    px = p.market_value / p.qty if p.qty else 0.0
                    rows.append({"标的": sym, "数量": p.qty, "成本": round(p.avg_price, 2), "现价": round(px, 2),
                                 "市值": round(p.market_value, 0), "浮动盈亏": pct(px / p.avg_price - 1) if p.avg_price else "n/a",
                                 "占净值": f"{p.market_value / acct.equity * 100:.1f}%"})
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.caption("空仓")
        with c2:
            if is_alpaca:
                st.markdown("**净值曲线**（Alpaca 账户历史）")
                try:
                    ph = broker.portfolio_history("3M", "1D")
                    ph = ph.dropna(subset=["equity"])
                    if len(ph) > 1:
                        figp = go.Figure(go.Scatter(x=ph.index, y=ph["equity"], mode="lines", line=dict(color="#2563eb", width=1.8), fill="tozeroy", fillcolor="rgba(37,99,235,.08)"))
                        figp.update_layout(template=TEMPLATE, height=260, margin=dict(l=40, r=20, t=10, b=30), yaxis_title="净值")
                        st.plotly_chart(figp, width="stretch", config=PLOTLY_CONFIG)
                    else:
                        st.caption("账户还没有足够的历史。")
                except Exception as e:
                    st.caption(f"净值历史获取失败：{e}")
                oo = broker.orders("open")
                st.markdown(f"**未成交订单** {len(oo)}")
                if oo:
                    st.dataframe(pd.DataFrame(oo)[["symbol", "side", "type", "qty", "filled_qty", "limit_price", "status", "submitted_at"]], hide_index=True, width="stretch")
                co = broker.orders("closed", limit=20, days=30)
                st.markdown(f"**最近 30 天成交 / 结束订单** {len(co)}")
                if co:
                    cdf = pd.DataFrame(co)
                    cdf["filled_at"] = pd.to_datetime(cdf["filled_at"], utc=True, errors="coerce").dt.tz_convert(NY).dt.strftime("%m-%d %H:%M")
                    st.dataframe(cdf[["symbol", "side", "qty", "filled_qty", "filled_avg_price", "status", "filled_at"]], hide_index=True, width="stretch",
                                 column_config={"filled_avg_price": st.column_config.NumberColumn(format="%.2f")})
                decs = read_decisions()
            else:
                st.markdown("**净值曲线**（每次决策前记录）")
                decs = read_decisions()
            if not is_alpaca and decs:
                eq = pd.DataFrame({"time": [d["time"] for d in decs], "equity": [d["equity_before"] for d in decs]})
                eq["time"] = pd.to_datetime(eq["time"], utc=True).dt.tz_convert(NY)
                fig = go.Figure(go.Scatter(x=eq["time"], y=eq["equity"], mode="lines+markers", line=dict(color="#2563eb")))
                fig.update_layout(template=TEMPLATE, height=260, margin=dict(l=40, r=20, t=10, b=30))
                st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
            elif not is_alpaca:
                st.caption("还没有运行记录")

        if decs:
            last = decs[-1]
            title = f"**最近一次决策** · {pd.Timestamp(last['time']).tz_convert(NY):%Y-%m-%d %H:%M} · {last['strategy']}"
            if last.get("last_bar"):
                title += f" · 信号截至 {pd.Timestamp(last['last_bar']).tz_convert(NY):%m-%d %H:%M}"
            if last.get("entry_mode"):
                title += f" · 起步 {'只跟新入场' if last['entry_mode'] == 'fresh' else '对齐策略'}"
            if last.get("halted"):
                title += "  · :red[已停机]"
            st.markdown(title)
            d1, d2 = st.columns([1, 1], gap="large")
            with d1:
                sig = pd.DataFrame({"策略权重": last["raw_targets"], "风控后权重": last["approved_targets"]})
                fig = go.Figure()
                fig.add_trace(go.Bar(x=sig.index, y=sig["策略权重"], name="策略权重", marker_color="#93c5fd"))
                fig.add_trace(go.Bar(x=sig.index, y=sig["风控后权重"], name="风控后权重", marker_color="#2563eb"))
                fig.update_layout(template=TEMPLATE, barmode="group", height=260, margin=dict(l=40, r=20, t=10, b=30),
                                  yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.15))
                st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
                if last.get("orders"):
                    od = pd.DataFrame(last["orders"])
                    cols = [c for c in ["side", "symbol", "qty", "est_price", "status", "error"] if c in od.columns]
                    st.dataframe(od[cols], width="stretch", hide_index=True)
                else:
                    st.caption("本次无订单")
            with d2:
                for r in last.get("risk_reasons") or []:
                    st.warning(r)
                rep = last.get("agent_report")
                if rep:
                    st.markdown(f"**Claude 复核**：{rep.get('market_note', '')}")
                    views = pd.DataFrame(rep["views"])
                    views["risk_flags"] = views["risk_flags"].apply(lambda x: "、".join(x))
                    st.dataframe(views.rename(columns={"symbol": "标的", "stance": "观点", "confidence": "置信度", "reasoning": "理由", "risk_flags": "风险标记"}),
                                 width="stretch", hide_index=True)
                elif last.get("agent_notes"):
                    st.info("\n".join(last["agent_notes"]))
                else:
                    st.caption("Claude 复核未启用（.env 里 AGENT_ENABLED=true 开启）")
    except Exception as e:
        st.error(f"读取账户失败：{e}")
        if S.mode == "alpaca":
            st.caption("检查 .env 里的 ALPACA_API_KEY / ALPACA_SECRET_KEY。")


# ================= 决策日志 =================
with tab_log:
    decs = read_decisions()
    if not decs:
        st.info("logs/decisions.jsonl 还没有记录。跑一次 `python run.py --once --force` 或在模拟盘页点“立刻运行一次”。")
    else:
        rows = []
        for d in decs:
            rows.append({
                "时间": pd.Timestamp(d["time"]).tz_convert(NY).strftime("%Y-%m-%d %H:%M"),
                "周期": d.get("interval", ""),
                "净值": round(d["equity_before"], 0),
                "现金": round(d["cash_before"], 0),
                "持有": " ".join(k for k, v in d["approved_targets"].items() if v > 0) or "空仓",
                "订单数": len(d.get("orders") or []),
                "停机": "是" if d.get("halted") else "",
                "风控原因": "; ".join(d.get("risk_reasons") or []),
                "Claude": "; ".join(d.get("agent_notes") or []),
            })
        table = pd.DataFrame(rows).iloc[::-1]
        st.dataframe(table, width="stretch", hide_index=True, height=420)
        st.download_button("下载 CSV", table.to_csv(index=False).encode("utf-8-sig"), "decisions.csv", "text/csv")
        _labels = [r["时间"] for r in rows]  # 绑定当前列表，避免闭包引用到后面被复用的 rows 变量
        idx = st.selectbox("查看某次的完整记录", list(range(len(decs)))[::-1], key=f"log_pick_{len(decs)}",
                           format_func=lambda i, _l=_labels: _l[i] if 0 <= i < len(_l) else str(i))
        if 0 <= idx < len(decs):
            st.json(decs[idx], expanded=False)


# ================= 行情 =================
with tab_data:
    c1, c2, c3, c5 = st.columns([1.4, 1, 2.4, 0.7])
    sym = c1.selectbox("标的", S.symbols, format_func=lambda x: f"{x}{(' ' + display_name(x)) if display_name(x) else ''} · {'/'.join(S.watchlist.group_of(x))}")
    d_interval = c2.radio("周期", ["1d", "1h"], index=0 if S.interval == "1d" else 1, horizontal=True, key="data_interval")
    days_d = c3.slider("显示天数", 5, MAX_DAYS[d_interval], 365 if d_interval == "1d" else 60, key=f"data_days_{d_interval}")
    c5.markdown("<div style='height:1.9em'></div>", unsafe_allow_html=True)
    if c5.button("详情", key="chart_detail_btn"):
        request_detail(sym)
    ovs_d, raw_d, trades_d, hold_d = overlay_controls("chart")
    try:
        fig, df, note = build_price_fig(sym, d_interval, days_d, ovs_d, show_raw=raw_d, show_trades=trades_d, show_holding=hold_d)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
        st.caption(f"{len(df)} 根 {d_interval} bar · {df.index[0]:%Y-%m-%d} 至 {df.index[-1]:%Y-%m-%d %H:%M} · 时区 {NY}" + note)
    except Exception as e:
        st.error(f"读取行情失败：{e}")


# ================= 详情弹窗（全局唯一，任何页签请求后在这里打开） =================
if st.session_state.get("detail_symbol"):
    _sym = st.session_state["detail_symbol"]
    show_detail(_sym, row_for(_sym))


# ================= 说明 =================
with tab_help:
    section_header("信号符号", [])
    st.markdown(
        "| 显示 | 含义 |\n|---|---|\n"
        "| ▲ 今日 | 今天（最后一根走完的 bar）收盘满足该策略的入场条件 |\n"
        "| ▲ n天前 | 最近 5 根 bar 内出现过入场条件，n 是距今根数 |\n"
        "| ▼ 出场条件 | 勾选“显示出场条件”后显示：今天满足该策略的出场规则（不考虑是否持有） |\n"
        "| ▲ 新进 / ● 在列 | 状态型策略（相对强度）专用：今天刚进入动量前 20% / 持续在前 20%。排名只在可交易的美股里做 |\n"
        "| 空白 | 今天既不满足入场也没有近期信号 |\n"
        "| 共振 | 聚合信号：当天同时给出信号的策略个数，≥ 2 标绿。多个独立规则同时看多，比单一信号可信一点，但也可能是同一波涨势被重复计数 |\n"
        "| 阶段 | 主升浪阶段：蓄势（贴近 52 周高的窄幅整理）→ 启动（模板 4~5 条且转强，或刚创新高）→ 主升（模板 ≥ 6 条且 ADX ≥ 20）→ 整理（模板 ≥ 6 但 ADX < 20，趋势中休整）→ 过热（乖离 SMA50 > 30% 或 RSI > 80）→ 衰竭（曾主升、现跌破 SMA50 且均线下行） |\n\n"
        "信号只用已经收盘的完整 bar：日线每天收盘后更新一次，小时线每小时一次。信号是规则的输出，不是预测，也不看任何账户持仓。"
    )

    section_header("指标列", [])
    st.markdown(
        "| 列 | 含义 |\n|---|---|\n"
        "| 现价 | 最后一根完整 bar 的收盘价（信号用的价格） |\n"
        "| 最新价 / 盘中% | 含未走完的当前 bar 的最新价，以及相对“现价”的变化（“更多列”里） |\n"
        "| 涨跌 1 / 5 / 20 | 现价相对 1、5、20 根 bar 前收盘的涨跌幅 |\n"
        "| 趋势 | ↑ 多头：收盘 > 200 日线且 50 日线 > 200 日线；↓ 空头：两者都在下方；→ 震荡：其余 |\n"
        "| RSI14 | 14 根 bar 的相对强弱指数，0 到 100；30 以下常称超卖，70 以上超买 |\n"
        "| 距55高 | 现价相对过去 55 根 bar（不含当前）最高收盘的百分比，0 附近即接近突破位 |\n"
        "| 2月 / 5月 最低最高 | 最近 61 / 152 个日历日内 bar 的最低价与最高价 |\n"
        "| ATR% | 14 根 bar 平均真实波幅 / 现价，衡量日常波动大小（“更多列”里） |\n"
        "| 财报 | 下次财报日期与距今天数，来自 yfinance 财报日历 |\n"
        "| 市场 / 组 | 上市地（海外只监控不交易）与所属板块（“更多列”里） |"
    )

    section_header("策略规则", [("参数：实盘策略用 .env 的值，其他用默认", "")])
    for name in SWING_NAMES + ["ma_cross", "buy_and_hold"]:
        cls = STRATEGIES[name]
        params_now = strat_params(name)
        live = " · 实盘策略" if name == S.strategy else ""
        with st.expander(f"{STRAT_LABEL[name]} ({name}){live}", expanded=(name == S.strategy)):
            doc = inspect.getdoc(cls) or ""
            st.markdown("\n".join(line.rstrip() for line in doc.splitlines()))
            rows = []
            for f in dataclasses.fields(cls):
                t = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
                if f.name == "name" or t not in {"int", "float", "bool"}:
                    continue
                val = params_now.get(f.name, f.default)
                rows.append({"参数": f.name, "当前值": val, "含义": getattr(cls, "PARAM_HELP", {}).get(f.name, "")})
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=42 + 35 * len(rows))

    section_header("共用机制", [])
    st.markdown(
        "- **槽位式组合**：swing 策略最多同时持有 `max_positions` 只（默认 5），每只等权 1/5 = 20%。候选多于空槽时按各策略的“排序分数”从高到低进，分数低的排队。\n"
        "- **止损**：`atr_stop_mult` 倍 ATR 止损，默认随最高收盘价上移（移动止损）；`max_hold_bars` 时间止损；可选财报前 N 天不新开仓 / 财报前退出。\n"
        "- **成交口径**：信号在 bar 收盘产生，成交按下一根 bar 开盘；回测与实盘一致。\n"
        "- **策略仓位页**：策略从头回放出来的模拟组合，不是账户。**模拟盘页**才是账户。\n"
        f"- **起步方式**（.env `ENTRY_MODE`，当前 `{S.entry_mode}`）：`fresh` 空仓起步只跟“▲ 新入场”，不追策略的老仓位；`align` 直接对齐策略当前仓位。\n"
        f"- **风控层**（所有策略共用）：单标的上限 {S.max_position_pct:.0%}，总仓位上限 {S.max_gross_exposure:.0%}，当日亏损 {S.daily_loss_limit_pct:.0%} 熔断清仓，回撤 {S.max_drawdown_pct:.0%} 停机。"
    )

    section_header("其他页面的口径", [])
    st.markdown(
        "- **末日期权**：隐含波动% = 最近到期平值跨式中间价 / 现价；区间内概率 = 过去一年同持有期真实 |涨跌幅| 小于隐含幅度的比例（卖方口径）；突破概率 = 1 − 区间内概率（买方口径）；窗口内有财报 / 高重要性宏观事件时改用事件日分布。\n"
        "- **LEAP Call**：按目标 delta 选行权；杠杆 = delta × 现价 / 权利金；隐含胜率 = Black-Scholes 下到期价 > 盈亏平衡的概率；历史胜率 = 历史同长度持有期涨幅 ≥ 需涨幅的比例（重叠窗口、偏乐观）；年化成本 = 时间价值 / 现价 / 年数。\n"
        "- **底部确认**：六条规则各一分：前期跌幅 ≥ 阈值、低点抬高、突破颈线、收复 SMA20 且拐头、突破日放量、RSI 底背离或 > 50；阶段 = 确认 / 确认后回踩 / 初步企稳 / 未见底 / 跌破低点 / 无明显下跌。\n"
        "- **板块轮动**：RS 比率 = 100 × (价格/基准) / 其 63 日均值；RS 动量 = 100 × RS 比率 / 10 日前值；象限：领先(>100,>100)、走弱(>100,<100)、落后(<100,<100)、改善(<100,>100)。\n"
        "- **大盘宏观**：指数用指数本身（^GSPC、^NDX、^DJI、^RUT），不是 ETF；事件日历来自美联储 / BEA / 普查局官网与 BLS 日程，ISM 等按惯例推算。"
    )
    st.caption("以上全部是规则与历史统计的说明，不构成任何投资建议。")
