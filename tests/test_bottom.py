import numpy as np
import pandas as pd

from tradebot.bottom import BottomParams, analyze_bottom, build_bottom_table, swing_lows

NY = "America/New_York"


def _df(closes, vol=None):
    c = np.asarray(closes, dtype=float)
    idx = pd.bdate_range("2026-01-02", periods=len(c), tz=NY)
    o = np.r_[c[0], c[:-1]]
    v = np.asarray(vol, dtype=float) if vol is not None else np.full(len(c), 1e6)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.005, "low": np.minimum(o, c) * 0.995, "close": c, "volume": v}, index=idx)


def test_swing_lows_finds_local_minima():
    low = pd.Series([5, 4, 3, 4, 5, 6, 5, 4, 3.5, 4, 5, 6, 7, 8, 9, 10, 11])
    assert swing_lows(low, 2) == [2, 8]


def test_w_bottom_confirmed():
    # 下跌 140->100，反弹到 112，回踩 103（低点抬高），放量突破颈线 112，站上 SMA20
    down = np.linspace(140, 100, 40)
    bounce = np.linspace(100, 112, 8)
    pull = np.linspace(112, 103, 8)
    up = np.linspace(103, 120, 14)
    closes = np.r_[down, bounce, pull, up]
    vol = np.full(len(closes), 1e6)
    first_break = len(down) + len(bounce) + len(pull) + int(np.argmax(up > 112.6))
    vol[first_break] = 2e6
    res = analyze_bottom(_df(closes, vol), BottomParams(lookback=120, swing_k=3))
    assert res["阶段"] == "确认", res
    assert res["低点抬高"] == "✓" and res["突破颈线"] == "✓" and res["收复SMA20"] == "✓" and res["量能"] == "✓"
    assert res["得分"] >= 5


def test_downtrend_not_bottomed_and_no_decline():
    res = analyze_bottom(_df(np.linspace(150, 100, 80)), BottomParams(swing_k=3))
    assert res["阶段"] in ("未见底", "跌破低点")
    flat = 100 + np.sin(np.arange(80) / 3.0)  # 无明显下跌
    assert analyze_bottom(_df(flat), BottomParams(swing_k=3))["阶段"] == "无明显下跌"


def test_table_orders_confirmed_first():
    down = np.linspace(140, 100, 40); bounce = np.linspace(100, 112, 8); pull = np.linspace(112, 103, 8); up = np.linspace(103, 120, 14)
    good = _df(np.r_[down, bounce, pull, up])
    bad = _df(np.linspace(150, 100, 70))
    tbl = build_bottom_table(["GOOD", "BAD"], {"GOOD": good, "BAD": bad}, BottomParams(swing_k=3), groups={"g": ["GOOD", "BAD"]})
    assert list(tbl["标的"]) == ["GOOD", "BAD"] and tbl.iloc[0]["板块"] == "g"
