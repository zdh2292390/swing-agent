"""把策略的目标权重变成账户要执行的目标权重。

entry_mode:
  - "fresh"：账户里没有的票，只在最后一根 bar 刚入场（前一根权重 0、这一根 > 0）时才买；
             策略早就持有的老仓位不追。账户里已有的票完整跟随策略（含出场）。适合从空仓起步。
  - "align"：直接对齐策略当前权重，中途上车。
"""
from __future__ import annotations

import pandas as pd

ENTRY_MODES = ("fresh", "align")


def select_targets(last: pd.Series, prev: pd.Series, held: set[str], mode: str = "fresh") -> tuple[dict[str, float], list[str]]:
    """返回 (目标权重, 说明)。last/prev 是策略最后两根 bar 的权重。"""
    if mode not in ENTRY_MODES:
        raise ValueError(f"entry_mode={mode!r} 不支持，可选 {ENTRY_MODES}")
    targets: dict[str, float] = {}
    notes: list[str] = []
    for sym in last.index:
        w, w0 = float(last[sym]), float(prev.get(sym, 0.0))
        if mode == "align" or sym in held:
            targets[sym] = w
            continue
        if w > 0 and w0 == 0:
            targets[sym] = w  # 新入场
        elif w > 0:
            targets[sym] = 0.0  # 策略的老仓位，账户没有，不追
            notes.append(f"{sym}: 策略已持有 {w:.0%}，但属于老仓位，只跟新入场，不追")
        else:
            targets[sym] = 0.0
    return targets, notes
