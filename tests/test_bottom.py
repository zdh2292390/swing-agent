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


def test_low_support_lines_picks_two_separate_troughs():
    import numpy as np
    from tradebot.bottom import low_support_lines
    # 两段独立探底：第 30 根 80（最低），第 90 根 85（第二低），中间反弹到 110
    c = np.r_[np.linspace(100, 80, 31), np.linspace(80, 110, 30), np.linspace(110, 85, 30), np.linspace(85, 105, 29)]
    df = _df(c)
    lines = low_support_lines(df, months=12, gap_bars=10)
    assert [l["标签"] for l in lines] == ["12个月最低", "第二低"]
    assert lines[0]["价格"] < lines[1]["价格"]
    assert abs((lines[0]["日期"] - df.index[30]).days) <= 3
    assert abs((lines[1]["日期"] - df.index[90]).days) <= 3


def test_low_support_lines_second_low_is_not_the_neighbouring_bar():
    import numpy as np
    from tradebot.bottom import low_support_lines
    df = _df(np.r_[np.linspace(120, 60, 60), np.linspace(60, 130, 60)])   # 单个 V 底
    lines = low_support_lines(df, months=12, gap_bars=10)
    assert len(lines) == 2
    assert abs(lines[0]["距今bar"] - lines[1]["距今bar"]) > 10   # 第二低必须跳开最低那一段


def test_low_support_lines_handles_short_data():
    import numpy as np
    from tradebot.bottom import low_support_lines
    assert low_support_lines(_df(np.array([100.0])), months=6) == []


def test_price_position_counts_days_below():
    import numpy as np
    from tradebot.bottom import price_position
    df = _df(np.arange(1, 101, dtype=float))          # 单调上升，最后一天最高
    p = price_position(df, months=12)
    assert p["交易日"] == 100 and p["低于天数"] == 99
    assert abs(p["低于天数%"] - 99.0) < 1e-9
    assert p["区间位置%"] > 99                          # 也在区间顶部
    assert p["昨天低于天数%"] is not None and abs(p["昨天低于天数%"] - 98 / 99 * 100) < 1e-9


def test_price_position_middle_and_custom_price():
    import numpy as np
    from tradebot.bottom import price_position
    df = _df(np.r_[np.arange(1, 51, dtype=float), np.arange(50, 0, -1, dtype=float)])   # 上去又下来，收在 1
    p = price_position(df, months=12)
    assert p["低于天数%"] == 0.0                        # 收在最低，没有哪天比它更低
    p2 = price_position(df, months=12, price=25.0)      # 换个参考价：上下各有约一半的天数低于 25
    assert 45 < p2["低于天数%"] < 55 and p2["参考价"] == 25.0


def test_price_position_needs_enough_bars():
    import numpy as np
    from tradebot.bottom import price_position
    assert price_position(_df(np.arange(1, 4, dtype=float)), months=6) == {}
