import numpy as np
import pandas as pd

from tradebot.strategies import MainWave, get_strategy
from tradebot.strategies.composite import checklist, stage_series, trend_template
from tradebot.strategies.indicators import adx

NY = "America/New_York"


def _df(closes, seed=0):
    c = np.asarray(closes, dtype=float)
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=len(c), tz=NY)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + abs(rng.normal(0, 0.004, len(c)))),
                         "low": np.minimum(o, c) * (1 - abs(rng.normal(0, 0.004, len(c)))), "close": c, "volume": 1e6}, index=idx)


def test_adx_is_high_in_a_steady_trend_and_low_in_chop():
    trend = _df(100 * np.cumprod(1 + np.full(300, 0.004)))
    chop = _df(100 + np.sin(np.arange(300) / 3.0) * 2, seed=1)
    assert adx(trend, 14)["adx"].iloc[-1] > 40
    assert adx(chop, 14)["adx"].iloc[-1] < 30


def test_trend_template_scores_uptrend_high_and_downtrend_low():
    up = _df(100 * np.cumprod(1 + np.full(400, 0.0025)))
    down = _df(400 * np.cumprod(1 - np.full(400, 0.0025)))
    tu, td = trend_template(up), trend_template(down)
    assert tu["score"].iloc[-1] >= 7 and td["score"].iloc[-1] <= 1
    assert stage_series(up, tu).iloc[-1] in ("主升", "过热")
    ck = checklist(up)
    assert ck["score"] >= 7 and len(ck["items"]) == 8 and ck["stage"] in ("主升", "过热")


def test_main_wave_strategy_holds_uptrend_and_excludes_foreign():
    n = 420
    data = {"UP": _df(100 * np.cumprod(1 + np.full(n, 0.0025))), "DN": _df(300 * np.cumprod(1 - np.full(n, 0.002)), seed=2),
            "005930.KS": _df(100 * np.cumprod(1 + np.full(n, 0.005)), seed=3)}
    strat = MainWave(regime_filter=False)
    W = strat.target_weights(data)
    assert W["UP"].iloc[-1] > 0 and W["DN"].iloc[-1] == 0 and (W["005930.KS"] == 0).all()
    assert MainWave.signal_kind == "state" and isinstance(get_strategy("main_wave"), MainWave)
    assert not strat.compute("UP", data["UP"])["entry"].iloc[:150].any()  # 预热期（SMA200 未形成）不给信号
