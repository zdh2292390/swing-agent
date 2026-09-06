import datetime as dt

import pandas as pd

from tradebot.econ_calendar import (
    _business_day, _nth_weekday, bls_events, market_session_open, nyse_early_closes, nyse_holidays, parse_bea, parse_census_retail, parse_fomc,
    rule_based_events, sync_official_events,
)

FOMC_SNIPPET = """
2026 FOMC Meetings
January
27-28
March
17-18*
December
8-9*
2027 FOMC Meetings
January
26-27
January/February
31-1
"""


def test_parse_fomc_decision_minutes_beige_book():
    ev = parse_fomc(FOMC_SNIPPET, 2026)
    by = {(e["日期"], e["事件"]) for e in ev}
    assert ("2026-01-28", "FOMC 利率决议") in by
    assert ("2026-03-18", "FOMC 利率决议（含经济预测 SEP）") in by
    assert ("2026-02-18", "FOMC 会议纪要") in by      # 决议后 3 周
    assert ("2026-01-14", "美联储褐皮书") in by        # 决议前 2 周
    assert all(e["日期"].startswith("2026") for e in ev) and len(ev) == 9
    ev27 = parse_fomc(FOMC_SNIPPET, 2027)
    assert ("2027-02-01", "FOMC 利率决议") in {(e["日期"], e["事件"]) for e in ev27}  # 跨月会议取结束日


def test_parse_bea_and_census():
    bea = "News\nPersonal Income and Outlays, August 2026\nOctober 6\n8:30 AM\nNews\nGDP (Advance Estimate), 3rd Quarter 2026\nOctober 29\n8:30 AM\nNews\nGDP (Third Estimate), 4th Quarter 2026\nMarch 25\n8:30 AM\n"
    ev = parse_bea(bea)
    got = {(e["日期"], e["事件"]) for e in ev}
    assert ("2026-10-06", "PCE 物价 / 个人收支") in got and ("2026-10-29", "GDP 初值") in got
    assert ("2027-03-25", "GDP 终值") in got  # 4 季度数据次年 3 月发布
    census = "Advance Monthly Retail Trade Report\nData Month\nRelease Date at 8:30 am\nAugust 2026\nSeptember 16, 2026\nSeptember 2026\nOctober 15, 2026\nMonthly Retail Trade Report\nSeptember 16, 2026\n"
    ev = parse_census_retail(census)
    assert [(e["日期"], e["事件"]) for e in ev] == [("2026-09-16", "零售销售"), ("2026-10-15", "零售销售")]
    assert "August 2026" in ev[0]["备注"]


def test_rules_and_bls_builtin():
    assert _nth_weekday(2026, 9, 4, 3) == dt.date(2026, 9, 18)  # 2026-09 第三个周五
    assert _business_day(2026, 9, 1) == dt.date(2026, 9, 1) and _business_day(2026, 9, 3) == dt.date(2026, 9, 3)
    assert _business_day(2026, 1, 1) == dt.date(2026, 1, 2)  # 元旦休市
    ev = rule_based_events([2026])
    names = {e["事件"] for e in ev}
    assert "四巫日（股指期货 / 期权到期）" in names and "美股休市（劳动节）" in names and "ISM 制造业 PMI" in names
    bls, errors = bls_events([2026], try_fetch=False)
    assert ("2026-09-11", "CPI 通胀") in {(e["日期"], e["事件"]) for e in bls}
    assert ("2026-09-04", "非农就业报告") in {(e["日期"], e["事件"]) for e in bls}
    assert errors == []


def test_sync_merges_without_duplicates(tmp_path):
    path = tmp_path / "ev.json"
    pd.DataFrame([{"日期": "2026-09-11", "事件": "CPI 通胀", "影响": "高", "备注": "我手写的"}]).pipe(
        lambda df: __import__("tradebot.macro", fromlist=["save_manual_events"]).save_manual_events(df, path))
    r1 = sync_official_events(path, years=[2026], fetch=False)
    r2 = sync_official_events(path, years=[2026], fetch=False)
    assert r1["added"] > 50 and r2["added"] == 0 and r1["total"] == r2["total"]
    df = pd.read_json(path)
    row = df[(df["日期"] == "2026-09-11") & (df["事件"] == "CPI 通胀")]
    assert len(row) == 1 and row.iloc[0]["备注"] == "我手写的"  # 用户的行保留


def test_market_session_open_handles_holidays_and_early_close():
    NY = "America/New_York"
    assert dt.date(2026, 9, 7) in nyse_holidays(2026)                     # 劳动节
    assert not market_session_open(pd.Timestamp("2026-09-07 10:00", tz=NY))   # 休市日
    assert market_session_open(pd.Timestamp("2026-09-08 09:35", tz=NY))       # 正常交易日
    assert not market_session_open(pd.Timestamp("2026-09-08 16:00", tz=NY))   # 收盘后
    assert not market_session_open(pd.Timestamp("2026-09-05 11:00", tz=NY))   # 周六
    if dt.date(2026, 11, 27) in nyse_early_closes(2026):
        assert market_session_open(pd.Timestamp("2026-11-27 12:30", tz=NY))
        assert not market_session_open(pd.Timestamp("2026-11-27 13:30", tz=NY))  # 提前收盘后
