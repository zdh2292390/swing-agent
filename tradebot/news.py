"""个股资讯：yfinance 新闻列表 + 基本面快照，JSON 文件缓存。

新闻只是原始标题和摘要；"利好 / 利空"的判断在 agent/research.py 里由 Claude 完成（可选）。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .universe import to_yahoo

log = logging.getLogger("tradebot.news")
NEWS_TTL_HOURS = 3.0
INFO_TTL_HOURS = 24.0
INFO_FIELDS = [
    "longName", "sector", "industry", "marketCap", "beta",
    "trailingPE", "forwardPE", "priceToSalesTrailing12Months", "profitMargins",
    "revenueGrowth", "earningsGrowth", "dividendYield", "shortPercentOfFloat",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "targetMeanPrice", "numberOfAnalystOpinions",
    "recommendationKey", "longBusinessSummary",
]


def _cache_path(cache_dir: Path | None, kind: str, symbol: str) -> Path | None:
    if cache_dir is None:
        return None
    p = Path(cache_dir) / kind
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{symbol.replace('.', '_')}.json"


def _read_cache(path: Path | None, ttl_hours: float):
    if path and path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _write_cache(path: Path | None, obj) -> None:
    if path:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def fetch_news(symbol: str, count: int = 15, cache_dir: Path | None = None) -> list[dict]:
    """返回 [{title, summary, time, source, url}]，按时间倒序。失败返回空列表。"""
    symbol = symbol.upper()
    path = _cache_path(cache_dir, "news", symbol)
    cached = _read_cache(path, NEWS_TTL_HOURS)
    if cached is not None:
        return cached
    items: list[dict] = []
    try:
        import yfinance as yf

        for it in yf.Ticker(to_yahoo(symbol)).get_news(count=count) or []:
            c = it.get("content", it) or {}
            items.append({
                "title": c.get("title") or "",
                "summary": c.get("summary") or c.get("description") or "",
                "time": c.get("pubDate") or c.get("displayTime") or "",
                "source": (c.get("provider") or {}).get("displayName", ""),
                "url": ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url") or ""),
            })
        items = [x for x in items if x["title"]]
        items.sort(key=lambda x: x["time"], reverse=True)
    except Exception as e:
        log.warning("新闻获取失败 %s: %s", symbol, e)
    _write_cache(path, items)
    return items


def fetch_info(symbol: str, cache_dir: Path | None = None) -> dict:
    """基本面快照（yfinance Ticker.info 的部分字段）。失败返回空 dict。"""
    symbol = symbol.upper()
    path = _cache_path(cache_dir, "info", symbol)
    cached = _read_cache(path, INFO_TTL_HOURS)
    if cached is not None:
        return cached
    out: dict = {}
    try:
        import yfinance as yf

        info = yf.Ticker(to_yahoo(symbol)).info or {}
        out = {k: info.get(k) for k in INFO_FIELDS if info.get(k) is not None}
    except Exception as e:
        log.warning("基本面获取失败 %s: %s", symbol, e)
    _write_cache(path, out)
    return out


def format_info(info: dict) -> list[tuple[str, str]]:
    """把基本面字段整理成可展示的 (标签, 值) 列表。"""
    def num(x, fmt):
        try:
            return fmt.format(x)
        except Exception:
            return "n/a"

    cap = info.get("marketCap")
    cap_txt = f"{cap / 1e12:.2f} 万亿" if cap and cap >= 1e12 else (f"{cap / 1e9:.0f} 亿" if cap else "n/a")
    rows = [
        ("行业", f"{info.get('sector', '')} / {info.get('industry', '')}".strip(" /")),
        ("市值 ($)", cap_txt),
        ("PE（TTM / 前瞻）", f"{num(info.get('trailingPE'), '{:.1f}')} / {num(info.get('forwardPE'), '{:.1f}')}"),
        ("市销率", num(info.get("priceToSalesTrailing12Months"), "{:.1f}")),
        ("净利率", num(info.get("profitMargins", float('nan')) * 100 if info.get("profitMargins") is not None else None, "{:.1f}%")),
        ("营收增速（同比）", num(info.get("revenueGrowth", float('nan')) * 100 if info.get("revenueGrowth") is not None else None, "{:+.1f}%")),
        ("盈利增速（同比）", num(info.get("earningsGrowth", float('nan')) * 100 if info.get("earningsGrowth") is not None else None, "{:+.1f}%")),
        ("52 周区间", f"{num(info.get('fiftyTwoWeekLow'), '{:.2f}')} ~ {num(info.get('fiftyTwoWeekHigh'), '{:.2f}')}"),
        ("分析师目标价（均值 / 人数）", f"{num(info.get('targetMeanPrice'), '{:.2f}')} / {info.get('numberOfAnalystOpinions', 'n/a')}"),
        ("评级", str(info.get("recommendationKey", "n/a")).replace("_", " ")),
        ("空头占流通比", num(info.get("shortPercentOfFloat", float('nan')) * 100 if info.get("shortPercentOfFloat") is not None else None, "{:.1f}%")),
        ("Beta", num(info.get("beta"), "{:.2f}")),
    ]
    return [(k, v) for k, v in rows if v and v not in {"n/a", "n/a / n/a", "n/a ~ n/a"}]
