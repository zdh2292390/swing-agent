import datetime as dt

import numpy as np
import pandas as pd

from tradebot.leaps import bs_call_delta, bs_call_price, bs_price_path, implied_vol, implied_vol_series, iv_stats, build_leap_table, horizon_stats, payoff_curve, pick_expiry, pick_strike_by_delta, prob_above


def test_black_scholes_sanity():
    S, T, r, q, sig = 100.0, 0.5, 0.04, 0.0, 0.3
    d_atm = bs_call_delta(S, 100, T, r, q, sig)
    assert 0.5 < d_atm < 0.65
    assert bs_call_delta(S, 60, T, r, q, sig) > 0.95 and bs_call_delta(S, 200, T, r, q, sig) < 0.05
    assert abs(bs_call_price(S, 100, T, r, q, sig) - 9.4) < 0.5  # ATM 半年 30% vol 约 9.4
    assert 0.4 < prob_above(S, 100, T, r, q, sig) < 0.5     # 风险中性下略低于 50%（对数正态漂移项）
    assert prob_above(S, 60, T, r, q, sig) > 0.95


def test_pick_expiry_and_strike():
    today = dt.date(2026, 9, 5)
    exps = [dt.date(2026, 9, 18), dt.date(2026, 12, 18), dt.date(2027, 1, 15), dt.date(2027, 3, 19), dt.date(2027, 6, 17), dt.date(2028, 1, 21)]
    assert pick_expiry(exps, today, 6) == dt.date(2027, 3, 19)
    assert pick_expiry(exps, today, 10) == dt.date(2027, 6, 17)   # 7 月没有，6 月 17 在 45 天容差内
    assert pick_expiry(exps, today, 8) == dt.date(2027, 6, 17) or pick_expiry(exps, today, 8) == dt.date(2027, 3, 19)
    assert pick_expiry([dt.date(2026, 9, 18)], today, 6) is None
    calls = pd.DataFrame({"strike": [60.0, 80.0, 100.0, 120.0], "bid": [40, 24, 10, 3], "ask": [42, 25, 11, 3.5], "lastPrice": [41, 24.5, 10.5, 3.2],
                          "impliedVolatility": [0.35, 0.32, 0.30, 0.31], "openInterest": [10, 20, 30, 40], "volume": [1, 2, 3, 4]})
    row, delta = pick_strike_by_delta(calls, 100.0, 0.5, 0.04, 0.0, 0.85)
    assert row["strike"] == 80.0 and 0.8 < delta < 0.95
    row2, delta2 = pick_strike_by_delta(calls, 100.0, 0.5, 0.04, 0.0, 0.6)
    assert row2["strike"] == 100.0 and 0.5 < delta2 < 0.65


def test_horizon_stats_and_table():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2016-01-04", periods=2600)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.015, len(idx))), index=idx)
    hs = horizon_stats(close, 182, 10.0)
    assert hs["n"] > 2000 and 0 <= hs["p_hist_be"] <= 100 and hs["p_hist_up"] >= hs["p_hist_be"] and hs["rv"] > 0
    c = {"symbol": "X", "months": 6, "expiry": "2027-03-19", "dte": 195, "T": 195 / 365, "spot": 100.0, "strike": 80.0, "premium": 26.0,
         "bid": 25.5, "ask": 26.5, "spread_pct": 3.8, "iv": 30.0, "delta": 0.78, "leverage": 3.0, "intrinsic": 20.0, "extrinsic": 6.0,
         "extrinsic_pct": 23.0, "annual_cost_pct": 11.2, "breakeven": 106.0, "be_move_pct": 6.0, "p_itm_implied": 80.0, "p_be_implied": 45.0,
         "oi": 100, "volume": 5, "r": 0.04, "q": 0.0}
    tbl = build_leap_table({"X": [c], "Y": []}, {"X": pd.DataFrame({"close": close})}, groups={"g": ["X"]})
    assert len(tbl) == 1 and tbl.iloc[0]["板块"] == "g" and tbl.iloc[0]["历史胜率"] is not None and tbl.iloc[0]["IV/RV"] is not None
    pay = payoff_curve(c, np.array([-0.5, 0.0, 0.06, 0.5]))
    assert pay[0] == -100.0 and abs(pay[2]) < 1e-9 and pay[3] > 100  # 跌穿行权全亏；盈亏平衡处 0；大涨杠杆放大


def test_bs_price_path_tracks_underlying_and_converges_to_intrinsic():
    idx = pd.bdate_range("2026-01-05", periods=60, tz="America/New_York")
    S = pd.Series(np.linspace(90, 130, 60), index=idx)
    expiry = dt.date(2026, 3, 27)  # 落在序列内
    path = bs_price_path(S, 100.0, expiry, 0.04, 0.0, 0.35)
    assert len(path) == 60 and path.is_monotonic_increasing
    after = path[S.index.date >= expiry] if hasattr(S.index, "date") else path
    last_after = path.iloc[-1]
    assert abs(last_after - (130 - 100)) < 1e-9       # 到期后 = 内在价值
    assert path.iloc[0] < path.iloc[10] < path.iloc[30]


def test_implied_vol_round_trip_and_series():
    S, K, T, r, q = 230.0, 210.0, 0.4, 0.04, 0.0
    for sig in (0.25, 0.45, 0.8):
        px = bs_call_price(S, K, T, r, q, sig)
        assert abs(implied_vol(px, S, K, T, r, q) - sig) < 1e-3
    assert np.isnan(implied_vol(S - K - 1, S, K, T, r, q))        # 低于内在价值：无解
    idx = pd.bdate_range("2026-03-02", periods=40, tz="America/New_York")
    under = pd.Series(np.linspace(200, 240, 40), index=idx)
    expiry = dt.date(2027, 1, 15)
    true_iv = 0.4
    prices = [bs_call_price(float(s_), K, (expiry - d.date()).days / 365, r, q, true_iv) for s_, d in zip(under.values, idx)]
    oh = pd.DataFrame({"close": prices, "volume": [10] * 39 + [0]}, index=idx)
    iv = implied_vol_series(oh, under, K, expiry, r, q)
    assert len(iv) == 40 and np.isnan(iv.iloc[-1]) and abs(iv.dropna().mean() - 40.0) < 0.2
    st_ = iv_stats(iv, 45.0)
    assert st_["n"] == 39 and st_["iv_pct"] > 95 and abs(st_["iv_median"] - 40.0) < 0.2
