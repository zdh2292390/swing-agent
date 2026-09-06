"""主升浪：趋势模板打分 + 阶段标签 + 策略。

趋势模板（改编自 Minervini / Weinstein 第二阶段，日线，8 条）：
  1 收盘 > SMA150 且 > SMA200          2 SMA150 > SMA200
  3 SMA200 比 21 根 bar 前高（在上行）  4 SMA50 > SMA150 且 > SMA200
  5 收盘 > SMA50                        6 收盘 ≥ 52 周最低 × 1.25（脱离底部）
  7 收盘 ≥ 52 周最高 × 0.75（接近高点）  8 6 个月动量在美股池里排前 30%（无横截面时退化为动量 > 0）
再加两个确认：ADX(14) ≥ adx_min 表示趋势有力度；收盘 ≤ SMA50 × (1 + max_ext) 表示还没过热。

阶段：
  主升   模板 ≥ 6 且 ADX ≥ 20，且不过热
  过热   模板 ≥ 6 但 收盘 > SMA50 × 1.30 或 RSI > 80
  启动   模板 4~5 且收盘 > SMA50 且 ADX 在上升，或最近 10 根 bar 内创 52 周新高
  整理   模板 ≥ 6 但 ADX < 20：趋势结构完好、暂时没有推进力度（主升前后的休整）
  蓄势   离 52 周高 15% 以内、最近 40 根 bar 振幅 ≤ 20%、模板 < 6
  衰竭   最近 20 根 bar 内曾 ≥ 6，现在收盘 < SMA50 且 SMA50 在下行
  其他   —
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from ..universe import is_us_listed
from .indicators import adx, momentum, rolling_max_prev, rsi, sma
from .swing import BASE_HELP, SwingStrategy

TEMPLATE_LABELS = ["收盘>SMA150/200", "SMA150>SMA200", "SMA200 上行", "SMA50>SMA150/200", "收盘>SMA50", "高于52周低25%", "距52周高25%内", "动量前30%"]


def trend_template(df: pd.DataFrame, rs_rank_pct: pd.Series | None = None, top_pct: float = 0.30) -> pd.DataFrame:
    """8 条模板逐 bar 的布尔值 + score（0~8）+ 辅助列。rs_rank_pct：横截面动量百分位（0 最强 → 1 最弱），可空。"""
    c, h, lo = df["close"], df["high"], df["low"]
    s50, s150, s200 = sma(c, 50), sma(c, 150), sma(c, 200)
    lo52 = lo.rolling(252, min_periods=120).min()
    hi52 = h.rolling(252, min_periods=120).max()
    mom6 = momentum(c, 126)
    cond = pd.DataFrame({
        "t1": (c > s150) & (c > s200),
        "t2": s150 > s200,
        "t3": s200 > s200.shift(21),
        "t4": (s50 > s150) & (s50 > s200),
        "t5": c > s50,
        "t6": c >= lo52 * 1.25,
        "t7": c >= hi52 * 0.75,
        "t8": (rs_rank_pct <= top_pct) if rs_rank_pct is not None else (mom6 > 0),
    }, index=df.index).fillna(False)
    cond["score"] = cond[[f"t{i}" for i in range(1, 9)]].sum(axis=1)
    a = adx(df, 14)
    cond["adx"] = a["adx"]
    cond["adx_up"] = a["adx"] > a["adx"].shift(5)
    cond["ext"] = c / s50 - 1  # 相对 SMA50 的乖离
    cond["rsi"] = rsi(c, 14)
    cond["mom6"] = mom6
    cond["hi52"] = hi52
    cond["s50"] = s50
    return cond


def stage_series(df: pd.DataFrame, tpl: pd.DataFrame, adx_min: float = 20.0, hot_ext: float = 0.30) -> pd.Series:
    c, h = df["close"], df["high"]
    score, a, ext, r = tpl["score"], tpl["adx"], tpl["ext"], tpl["rsi"]
    hot = (score >= 6) & ((ext > hot_ext) | (r > 80))
    main = (score >= 6) & (a >= adx_min) & ~hot
    new_high_recent = (c > rolling_max_prev(h, 252)).rolling(10).max().fillna(0).astype(bool)
    start = ((score.between(4, 5)) & (c > tpl["s50"]) & tpl["adx_up"]) | (new_high_recent & (score < 6))
    base_range = (h.shift(1).rolling(40).max() - df["low"].shift(1).rolling(40).min()) / c
    basing = (c >= tpl["hi52"] * 0.85) & (base_range <= 0.20) & (score < 6)
    was_strong = (score >= 6).rolling(20).max().fillna(0).astype(bool)
    fading = was_strong & (c < tpl["s50"]) & (tpl["s50"] < tpl["s50"].shift(5)) & (score < 6)
    resting = (score >= 6) & (a < adx_min) & ~hot  # 趋势结构完好，但当下没有推进力度：趋势中的整理
    out = pd.Series("—", index=df.index)
    out[basing] = "蓄势"
    out[start] = "启动"
    out[fading] = "衰竭"
    out[resting] = "整理"
    out[main] = "主升"
    out[hot] = "过热"
    return out


@dataclass
class MainWave(SwingStrategy):
    """主升浪：趋势模板 ≥ min_score、ADX ≥ adx_min、未过热时持有；跌破 SMA50 两天或模板 ≤ exit_score 退出。

    状态型信号：只要还满足就一直“在列”，显示上区分 ▲ 新进 / ● 在列。排序分数 = 6 个月动量。
    """

    min_score: int = 6
    exit_score: int = 3
    adx_min: float = 20.0
    max_ext: float = 0.20
    top_pct: float = 0.30
    atr_stop_mult: float = 3.0
    name: str = "main_wave"
    _cs: dict = field(default_factory=dict, repr=False, compare=False)

    PARAM_HELP: ClassVar[dict[str, str]] = {
        **BASE_HELP,
        "min_score": "趋势模板至少满足几条（共 8）才算主升", "exit_score": "模板掉到几条以下退出",
        "adx_min": "ADX(14) 至少多少才算趋势有力度", "max_ext": "收盘高于 SMA50 超过这个比例算过热，不新开仓（0.2 = 20%）",
        "top_pct": "6 个月动量在美股池里排前多少算强（0.3 = 前 30%）",
    }
    signal_kind: ClassVar[str] = "state"

    def prepare(self, data: dict[str, pd.DataFrame]) -> None:
        pool = {s: df for s, df in data.items() if is_us_listed(s)}
        self._cs = {}
        if pool:
            moms = pd.DataFrame({s: momentum(df["close"], 126) for s, df in pool.items()})
            pct = moms.rank(axis=1, ascending=False, pct=True)  # 0 最强 → 1 最弱
        for s, df in data.items():
            rank_pct = pct[s].reindex(df.index) if s in pool else None
            tpl = trend_template(df, rank_pct, self.top_pct)
            self._cs[s] = self._signals(df, tpl) if s in pool else pd.DataFrame({"entry": False, "exit": False, "score": np.nan}, index=df.index)

    def _signals(self, df: pd.DataFrame, tpl: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        entry = (tpl["score"] >= self.min_score) & (tpl["adx"] >= self.adx_min) & (tpl["ext"] <= self.max_ext)
        below50 = c < tpl["s50"]
        exit_ = (below50 & below50.shift(1).fillna(False)) | (tpl["score"] <= self.exit_score)
        return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False), "score": tpl["mom6"]}, index=df.index)

    def compute(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        cached = self._cs.get(symbol)
        if cached is not None and len(cached) == len(df) and cached.index.equals(df.index):
            return cached
        return self._signals(df, trend_template(df, None, self.top_pct))

    def warmup_bars(self) -> int:
        return 260


def checklist(df: pd.DataFrame, rank_pct: float | None = None) -> dict:
    """给详情弹窗用：最后一根 bar 的 8 条模板逐项 ✓/✗、得分、ADX、乖离、阶段。"""
    tpl = trend_template(df, None)
    last = tpl.iloc[-1].copy()
    if rank_pct is not None:
        last["t8"] = bool(rank_pct <= 0.30)
        last["score"] = int(sum(bool(last[f"t{i}"]) for i in range(1, 9)))
    st = stage_series(df, tpl)
    return {
        "items": [(TEMPLATE_LABELS[i - 1], bool(last[f"t{i}"])) for i in range(1, 9)],
        "score": int(last["score"]), "adx": float(last["adx"]) if pd.notna(last["adx"]) else float("nan"), "adx_up": bool(last["adx_up"]),
        "ext_pct": float(last["ext"] * 100) if pd.notna(last["ext"]) else float("nan"), "rsi": float(last["rsi"]) if pd.notna(last["rsi"]) else float("nan"),
        "stage": str(st.iloc[-1]),
    }
