"""官方经济日程：美联储 FOMC、BLS（CPI / PPI / 非农 / JOLTS）、BEA（PCE / GDP）、普查局零售，
加上按规则推算的 ISM、密歇根信心、期权到期、四巫日、美股休市、财政部再融资。

sync_official_events() 把这些合并进 macro_events.json（按 日期+事件 去重，不动用户手写的行）。
BLS 网站拦截脚本，2026 年日程内置在 BLS_2026 里；其余来源实时抓取，失败时跳过并报告。
"""
from __future__ import annotations

import calendar
import datetime as dt
import html
import re
from pathlib import Path

import pandas as pd
import requests

from .macro import EVENT_COLUMNS, MANUAL_EVENTS_PATH, load_manual_events, save_manual_events

UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
MONTHS = {name: i for i, name in enumerate(calendar.month_name) if name}
MONTH_RE = "January|February|March|April|May|June|July|August|September|October|November|December"

FED_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BEA_URL = "https://www.bea.gov/news/schedule"
CENSUS_URL = "https://www.census.gov/retail/release_schedule.html"
BLS_URLS = {
    "cpi": "https://www.bls.gov/schedule/news_release/cpi.htm",
    "ppi": "https://www.bls.gov/schedule/news_release/ppi.htm",
    "empsit": "https://www.bls.gov/schedule/news_release/empsit.htm",
    "jolts": "https://www.bls.gov/schedule/news_release/jolts.htm",
}

# ---- BLS 2026 官方发布日（来自 bls.gov/schedule，2026-09-04 抓取；网站拦截脚本所以内置）----
BLS_2026 = {
    "cpi": [("2026-01-13", "12月"), ("2026-02-13", "1月"), ("2026-03-11", "2月"), ("2026-04-10", "3月"), ("2026-05-12", "4月"),
            ("2026-06-10", "5月"), ("2026-07-14", "6月"), ("2026-08-12", "7月"), ("2026-09-11", "8月"), ("2026-10-14", "9月"),
            ("2026-11-10", "10月"), ("2026-12-10", "11月")],
    "empsit": [("2026-01-09", "12月"), ("2026-02-11", "1月"), ("2026-03-06", "2月"), ("2026-04-03", "3月"), ("2026-05-08", "4月"),
               ("2026-06-05", "5月"), ("2026-07-02", "6月"), ("2026-08-07", "7月"), ("2026-09-04", "8月"), ("2026-10-02", "9月"),
               ("2026-11-06", "10月"), ("2026-12-04", "11月")],
    "ppi": [("2026-01-14", "11月"), ("2026-01-30", "12月"), ("2026-02-27", "1月"), ("2026-03-18", "2月"), ("2026-04-14", "3月"),
            ("2026-05-13", "4月"), ("2026-06-11", "5月"), ("2026-07-15", "6月"), ("2026-08-13", "7月"), ("2026-09-10", "8月"),
            ("2026-10-15", "9月"), ("2026-11-13", "10月"), ("2026-12-15", "11月")],
    "jolts": [("2026-01-07", "11月"), ("2026-02-05", "12月"), ("2026-03-13", "1月"), ("2026-03-31", "2月"), ("2026-05-05", "3月"),
              ("2026-06-02", "4月"), ("2026-06-30", "5月"), ("2026-08-04", "6月"), ("2026-09-01", "7月"), ("2026-09-29", "8月"),
              ("2026-11-03", "9月"), ("2026-12-01", "10月")],
}
BLS_LABEL = {
    "cpi": ("CPI 通胀", "高", "BLS · {ref}数据 · 8:30 ET"),
    "empsit": ("非农就业报告", "高", "BLS · {ref}数据 · 8:30 ET"),
    "ppi": ("PPI 生产者价格", "中", "BLS · {ref}数据 · 8:30 ET"),
    "jolts": ("JOLTS 职位空缺", "中", "BLS · {ref}数据 · 10:00 ET"),
}

# ---- NYSE 休市 / 提前收盘（2027 请以交易所公告为准）----
NYSE_HOLIDAYS = {
    2026: [("01-01", "元旦"), ("01-19", "马丁·路德·金日"), ("02-16", "总统日"), ("04-03", "耶稣受难日"), ("05-25", "阵亡将士纪念日"),
           ("06-19", "六月节"), ("07-03", "独立日（补休）"), ("09-07", "劳动节"), ("11-26", "感恩节"), ("12-25", "圣诞节")],
    2027: [("01-01", "元旦"), ("01-18", "马丁·路德·金日"), ("02-15", "总统日"), ("03-26", "耶稣受难日"), ("05-31", "阵亡将士纪念日"),
           ("06-18", "六月节（补休）"), ("07-05", "独立日（补休）"), ("09-06", "劳动节"), ("11-25", "感恩节"), ("12-24", "圣诞节（补休）")],
}
NYSE_EARLY_CLOSE = {2026: [("11-27", "感恩节次日"), ("12-24", "平安夜")], 2027: [("11-26", "感恩节次日")]}


def _ev(date: dt.date | str, event: str, impact: str, note: str) -> dict:
    d = date if isinstance(date, str) else date.isoformat()
    return {"日期": d, "事件": event, "影响": impact, "备注": note}


def _get(url: str, timeout: int = 25) -> str:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.text


def _text(page: str) -> str:
    page = re.sub(r"<script.*?</script>|<style.*?</style>", " ", page, flags=re.S)
    page = re.sub(r"<[^>]+>", "\n", page)
    return re.sub(r"[ \t\r]+", " ", html.unescape(page))


# ================= 美联储 =================
def parse_fomc(page_text: str, year: int) -> list[dict]:
    i = page_text.find(f"{year} FOMC Meetings")
    if i < 0:
        return []
    block = page_text[i + 10:]
    nxt = re.search(r"20\d\d FOMC Meetings", block)
    block = block[: nxt.start()] if nxt else block[:6000]
    pat = rf"({MONTH_RE})(?:/({MONTH_RE}))?\s*\n\s*([0-9]{{1,2}})(?:-([0-9]{{1,2}}))?(\*?)"
    events = []
    for m1, m2, d1, d2, star in re.findall(pat, block):
        start_m, end_m = MONTHS[m1], MONTHS[m2 or m1]
        first, last = int(d1), int(d2 or d1)
        decision = dt.date(year, end_m, last)
        span = f"{start_m}/{first}-{end_m}/{last}" if m2 else f"{start_m}/{first}-{last}"
        label = "FOMC 利率决议" + ("（含经济预测 SEP）" if star else "")
        events.append(_ev(decision, label, "高", f"美联储 · 会议 {span} · 14:00 ET 声明，14:30 记者会"))
        events.append(_ev(decision + dt.timedelta(days=21), "FOMC 会议纪要", "中", f"美联储 · {end_m}月会议纪要 · 14:00 ET"))
        events.append(_ev(decision - dt.timedelta(days=14), "美联储褐皮书", "低", f"美联储 · {end_m}月会议前 · 14:00 ET"))
    return events


def fetch_fomc(years: list[int]) -> list[dict]:
    text = _text(_get(FED_URL))
    out = []
    for y in years:
        out += parse_fomc(text, y)
    return out


# ================= BEA：PCE / GDP =================
def _infer_release_year(title: str, release_month: int) -> int | None:
    m = re.search(r"(20\d\d)", title)
    if not m:
        return None
    y = int(m.group(1))
    late_period = re.search(r"October|November|December|4th Quarter|Fourth Quarter", title, re.I)
    return y + 1 if (release_month <= 3 and late_period) else y


def parse_bea(page_text: str) -> list[dict]:
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    events = []
    for i, line in enumerate(lines):
        m = re.fullmatch(rf"({MONTH_RE}) (\d{{1,2}})", line)
        if not m or i == 0:
            continue
        title = lines[i - 1]
        month, day = MONTHS[m.group(1)], int(m.group(2))
        year = _infer_release_year(title, month)
        if year is None:
            continue
        date = dt.date(year, month, day)
        ref = re.search(r", ([A-Za-z]+ 20\d\d|\d(?:st|nd|rd|th) Quarter 20\d\d)", title)
        ref_txt = ref.group(1) if ref else ""
        if "Personal Income and Outlays" in title:
            events.append(_ev(date, "PCE 物价 / 个人收支", "高", f"BEA · {ref_txt} · 8:30 ET"))
        elif "GDP (Advance Estimate)" in title:
            events.append(_ev(date, "GDP 初值", "高", f"BEA · {ref_txt} · 8:30 ET"))
        elif "GDP (Second Estimate)" in title:
            events.append(_ev(date, "GDP 修正值", "中", f"BEA · {ref_txt} · 8:30 ET"))
        elif "GDP (Third Estimate)" in title:
            events.append(_ev(date, "GDP 终值", "低", f"BEA · {ref_txt} · 8:30 ET"))
    return events


def fetch_bea() -> list[dict]:
    return parse_bea(_text(_get(BEA_URL)))


# ================= 普查局：零售销售 =================
def parse_census_retail(page_text: str) -> list[dict]:
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    start = next((i for i, l in enumerate(lines) if "Advance Monthly" in l), 0)
    end = next((i for i, l in enumerate(lines) if i > start + 3 and re.search(r"Monthly Retail Trade|Quarterly", l)), len(lines))
    events, seen = [], set()
    for i in range(start, end):
        m = re.fullmatch(rf"({MONTH_RE}) (\d{{1,2}}), (20\d\d)", lines[i])
        if not m:
            continue
        date = dt.date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
        if date in seen:
            continue
        seen.add(date)
        # 表格是 "Data Month | Release Date"，数据月份在日期前一行；兼容反过来的情况
        ref = ""
        for j in (i - 1, i + 1):
            if 0 <= j < len(lines) and re.fullmatch(rf"({MONTH_RE}) 20\d\d", lines[j]):
                ref = lines[j]
                break
        events.append(_ev(date, "零售销售", "中", f"普查局 · {ref} 数据 · 8:30 ET".replace("  ", " ")))
    return events


def fetch_census_retail() -> list[dict]:
    return parse_census_retail(_text(_get(CENSUS_URL)))


# ================= BLS：实时抓取（通常 403）+ 内置 2026 =================
def parse_bls(page_text: str, kind: str) -> list[dict]:
    label, impact, note = BLS_LABEL[kind]
    events = []
    for m in re.finditer(rf"({MONTH_RE}) (\d{{1,2}}), (20\d\d)", page_text):
        date = dt.date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
        events.append(_ev(date, label, impact, note.format(ref="")))
    return events


def bls_events(years: list[int], try_fetch: bool = True) -> tuple[list[dict], list[str]]:
    events, errors = [], []
    for kind, rows in BLS_2026.items():
        label, impact, note = BLS_LABEL[kind]
        for date, ref in rows:
            if int(date[:4]) in years:
                events.append(_ev(date, label, impact, note.format(ref=ref)))
    if try_fetch:
        for kind, url in BLS_URLS.items():
            try:
                fetched = [e for e in parse_bls(_text(_get(url, timeout=15)), kind) if int(e["日期"][:4]) in years and int(e["日期"][:4]) != 2026]
                events += fetched
            except Exception as e:
                errors.append(f"BLS {kind}: {type(e).__name__}（网站拦截脚本，已用内置 2026 日程）")
    return events, errors


# ================= 规则推算 =================
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def nyse_holidays(year: int) -> set[dt.date]:
    return {dt.date.fromisoformat(f"{year}-{md}") for md, _ in NYSE_HOLIDAYS.get(year, [])}


def nyse_early_closes(year: int) -> set[dt.date]:
    """13:00 提前收盘的日子（感恩节次日、平安夜等）。"""
    return {dt.date.fromisoformat(f"{year}-{md}") for md, _ in NYSE_EARLY_CLOSE.get(year, [])}


def market_session_open(now) -> bool:
    """美股常规交易时段是否开着：周一到周五 9:30 起，正常 16:00 收，提前收盘日 13:00 收，休市日全天关。now 需带纽约时区。"""
    d = now.date()
    if now.weekday() >= 5 or d in nyse_holidays(d.year):
        return False
    close_minutes = 13 * 60 if d in nyse_early_closes(d.year) else 16 * 60
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= minutes < close_minutes


def _business_day(year: int, month: int, n: int) -> dt.date:
    d, hol, count = dt.date(year, month, 1), nyse_holidays(year), 0
    while True:
        if d.weekday() < 5 and d not in hol:
            count += 1
            if count == n:
                return d
        d += dt.timedelta(days=1)


def rule_based_events(years: list[int]) -> list[dict]:
    events = []
    for y in years:
        for md, name in NYSE_HOLIDAYS.get(y, []):
            events.append(_ev(f"{y}-{md}", f"美股休市（{name}）", "中", "NYSE" + ("" if y == 2026 else " · 请以交易所公告核对")))
        for md, name in NYSE_EARLY_CLOSE.get(y, []):
            events.append(_ev(f"{y}-{md}", f"美股提前收盘 13:00（{name}）", "低", "NYSE"))
        for m in range(1, 13):
            opex = _nth_weekday(y, m, 4, 3)  # 第三个周五
            if m in (3, 6, 9, 12):
                events.append(_ev(opex, "四巫日（股指期货 / 期权到期）", "中", "成交量放大，收盘前波动"))
            else:
                events.append(_ev(opex, "月度期权到期", "低", "第三个周五"))
            events.append(_ev(_business_day(y, m, 1), "ISM 制造业 PMI", "中", "ISM · 10:00 ET · 首个交易日（推算）"))
            events.append(_ev(_business_day(y, m, 3), "ISM 服务业 PMI", "中", "ISM · 10:00 ET · 第三个交易日（推算）"))
            events.append(_ev(_nth_weekday(y, m, 4, 2), "密歇根消费者信心（初值）", "低", "密歇根大学 · 10:00 ET · 第二个周五（推算）"))
        for m in (2, 5, 8, 11):
            events.append(_ev(_nth_weekday(y, m, 2, 1), "财政部季度再融资公告", "低", "美国财政部 · 8:30 ET · 首个周三（推算）"))
    return events


# ================= 合并 =================
def build_official_events(years: list[int], fetch: bool = True) -> tuple[list[dict], list[str], list[str]]:
    events, errors, sources = [], [], []
    if fetch:
        for name, fn in (("美联储 FOMC", lambda: fetch_fomc(years)), ("BEA PCE/GDP", fetch_bea), ("普查局零售", fetch_census_retail)):
            try:
                got = fn()
                events += got
                sources.append(f"{name} {len(got)} 条")
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}: {e}")
    bls, bls_err = bls_events(years, try_fetch=fetch)
    events += bls
    sources.append(f"BLS {len(bls)} 条")
    errors += bls_err
    rules = rule_based_events(years)
    events += rules
    sources.append(f"规则推算 {len(rules)} 条")
    events = [e for e in events if int(e["日期"][:4]) in years]
    return events, errors, sources


def sync_official_events(path: Path = MANUAL_EVENTS_PATH, years: list[int] | None = None, fetch: bool = True) -> dict:
    today = dt.date.today()
    years = years or [today.year, today.year + 1]
    events, errors, sources = build_official_events(years, fetch=fetch)
    existing = load_manual_events(path)
    have = set(zip(existing["日期"], existing["事件"]))
    new_rows = [e for e in events if (e["日期"], e["事件"]) not in have]
    merged = pd.concat([existing, pd.DataFrame(new_rows, columns=EVENT_COLUMNS)], ignore_index=True) if new_rows else existing
    merged = save_manual_events(merged, path)
    return {"added": len(new_rows), "total": len(merged), "errors": errors, "sources": sources, "years": years}
