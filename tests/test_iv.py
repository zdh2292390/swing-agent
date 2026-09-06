import numpy as np
import pandas as pd

from tradebot.iv import _atm_iv, hv30, iv_rank_percentile, record_snapshots, load_history


def test_atm_iv_averages_nearest_strikes():
    calls = pd.DataFrame({"strike": [90, 100, 110], "impliedVolatility": [0.40, 0.30, 0.35], "bid": [1, 1, 1]})
    puts = pd.DataFrame({"strike": [90, 100, 110], "impliedVolatility": [0.42, 0.32, 0.33], "bid": [1, 1, 1]})
    iv = _atm_iv(calls, puts, 101.0)
    assert abs(iv - np.mean([0.30, 0.35, 0.32, 0.33])) < 1e-9


def test_hv30_and_rank():
    idx = pd.bdate_range("2026-01-05", periods=60)
    close = pd.Series(100 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.02, 60)), index=idx)
    hv = hv30(close)
    assert 15 < hv < 60
    hist = pd.Series([20, 25, 30, 35, 40])
    r = iv_rank_percentile(hist, 30.0)
    assert abs(r["iv_rank"] - 50) < 1e-9 and abs(r["iv_pct"] - 40) < 1e-9 and r["n"] == 5
    assert np.isnan(iv_rank_percentile(pd.Series([20.0]), 30.0)["iv_rank"])


def test_record_snapshots_offline(tmp_path, monkeypatch):
    import tradebot.iv as ivmod
    monkeypatch.setattr(ivmod, "iv30_snapshot", lambda sym, today=None: {"symbol": sym, "spot": 100.0, "iv30": 33.3} if sym != "BAD" else None)
    idx = pd.bdate_range("2026-01-05", periods=60)
    closes = {"AAA": pd.Series(np.linspace(90, 110, 60), index=idx)}
    import datetime as dt
    h1 = record_snapshots(["AAA", "BAD"], tmp_path, closes, today=dt.date(2026, 9, 5))
    assert list(h1["symbol"]) == ["AAA"] and abs(h1.iloc[0]["iv30"] - 33.3) < 1e-9
    h2 = record_snapshots(["AAA"], tmp_path, closes, today=dt.date(2026, 9, 5))  # 同一天覆盖
    assert len(h2) == 1
    h3 = record_snapshots(["AAA"], tmp_path, closes, today=dt.date(2026, 9, 8))
    assert len(h3) == 2 and len(load_history(tmp_path)) == 2
