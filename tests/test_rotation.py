import numpy as np
import pandas as pd

from tradebot.rotation import basket_index, group_breadth, monthly_returns, quadrant, relative_strength_table, rrg_series, rrg_tail


def _closes(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-02", periods=n)
    bench = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    strong = bench * np.cumprod(1 + np.full(n, 0.0015))          # 持续跑赢
    weak = bench * np.cumprod(1 + np.full(n, -0.0015))           # 持续跑输
    return pd.DataFrame({"SPY": bench, "STRONG": strong, "WEAK": weak}, index=idx)


def test_rrg_quadrants_for_persistent_out_and_underperformer():
    closes = _closes()
    ratio, mom = rrg_series(closes, "SPY")
    assert "SPY" not in ratio.columns
    assert ratio["STRONG"].iloc[-1] > 100 and mom["STRONG"].iloc[-1] > 100 and quadrant(ratio["STRONG"].iloc[-1], mom["STRONG"].iloc[-1]) == "领先"
    assert ratio["WEAK"].iloc[-1] < 100 and quadrant(ratio["WEAK"].iloc[-1], mom["WEAK"].iloc[-1]) == "落后"
    assert quadrant(float("nan"), 100.0) == ""


def test_relative_strength_table_and_tail():
    closes = _closes()
    tbl = relative_strength_table(closes, "SPY", {"STRONG": "强", "WEAK": "弱"}, {"STRONG": "t", "WEAK": "t"})
    assert list(tbl["代码"]) == ["STRONG", "WEAK"]
    assert tbl.iloc[0]["3月超额"] > 0 > tbl.iloc[1]["3月超额"]
    assert tbl.iloc[0]["3月超额排名"] == 1 and tbl.iloc[0]["象限"] == "领先"
    tails = rrg_tail(closes, "SPY", ["STRONG", "WEAK"], tail_points=6, step=5)
    assert len(tails["STRONG"]) == 6 and list(tails["STRONG"].columns) == ["ratio", "mom"]


def test_basket_index_and_monthly_and_breadth():
    closes = _closes()
    idx = closes.index
    a = pd.DataFrame({"close": closes["STRONG"]}, index=idx.tz_localize("America/New_York"))
    b = pd.DataFrame({"close": closes["WEAK"].iloc[100:]}, index=idx[100:].tz_localize("America/New_York"))  # 晚上市
    bi = basket_index({"A": a, "B": b}, idx)
    assert abs(bi.iloc[0] - 100) < 1e-9 and bi.notna().all() and len(bi) == len(idx)
    mr = monthly_returns(closes, ["STRONG", "WEAK"], months=6)
    assert mr.shape == (2, 6) and (mr.loc["STRONG"] > mr.loc["WEAK"]).all()
    br = group_breadth({"g": {"A": a, "B": b}})
    assert br.iloc[0]["成员"] == 2 and 0 <= br.iloc[0][">SMA50%"] <= 100
