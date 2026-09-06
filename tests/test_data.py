import numpy as np
import pandas as pd

from tradebot.data import _normalize, drop_incomplete_last_bar

NY = "America/New_York"


def test_normalize_daily_naive_dates_localize_to_new_york():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1, 2], "Volume": [10, 20]}, index=idx)
    out = _normalize(df)
    assert str(out.index.tz) == NY
    assert out.index[0] == pd.Timestamp("2024-01-02 00:00", tz=NY)  # 不能被当成 UTC 后偏到前一天
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_normalize_hourly_utc_converts_to_new_york():
    idx = pd.to_datetime(["2024-01-02 14:30", "2024-01-02 15:30"], utc=True)
    df = pd.DataFrame({"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1, 2], "Volume": [1, 1]}, index=idx)
    out = _normalize(df)
    assert out.index[0] == pd.Timestamp("2024-01-02 09:30", tz=NY)


def _bars(times):
    idx = pd.DatetimeIndex(pd.to_datetime(times)).tz_localize(NY)
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)


def test_drop_incomplete_hourly_bar():
    df = _bars(["2024-01-02 09:30", "2024-01-02 10:30"])
    at_1031 = pd.Timestamp("2024-01-02 10:31", tz=NY)
    at_1130 = pd.Timestamp("2024-01-02 11:30", tz=NY)
    assert len(drop_incomplete_last_bar(df, "1h", at_1031)) == 1
    assert len(drop_incomplete_last_bar(df, "1h", at_1130)) == 2


def test_last_half_hour_bar_completes_at_close():
    df = _bars(["2024-01-02 14:30", "2024-01-02 15:30"])
    assert len(drop_incomplete_last_bar(df, "1h", pd.Timestamp("2024-01-02 15:59", tz=NY))) == 1
    assert len(drop_incomplete_last_bar(df, "1h", pd.Timestamp("2024-01-02 16:00", tz=NY))) == 2


def test_drop_incomplete_daily_bar():
    df = _bars(["2024-01-02", "2024-01-03"])
    assert len(drop_incomplete_last_bar(df, "1d", pd.Timestamp("2024-01-03 09:35", tz=NY))) == 1
    assert len(drop_incomplete_last_bar(df, "1d", pd.Timestamp("2024-01-03 16:00", tz=NY))) == 2
    assert len(drop_incomplete_last_bar(df, "1d", pd.Timestamp("2024-01-04 09:35", tz=NY))) == 2
