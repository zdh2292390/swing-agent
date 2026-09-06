"""Claude 研究助手（可选，需要 Anthropic 凭证）：

- catalysts(): 把个股最近的新闻和基本面整理成 利好 / 利空 催化剂
- macro_brief(): 联网搜索近期大盘与科技板块的宏观大事件，结构化输出

两个功能都只是研究摘要，不产生任何交易动作。
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

DEFAULT_MODEL = "claude-opus-5"


# ---------------- 结构 ----------------
class Catalyst(BaseModel):
    headline: str = Field(description="一句话概括这条催化剂")
    why: str = Field(description="为什么对股价重要，一两句")
    importance: Literal["高", "中", "低"]
    date: str = Field(default="", description="新闻日期，YYYY-MM-DD，未知留空")
    source: str = Field(default="", description="来源媒体")


class CatalystReport(BaseModel):
    symbol: str
    one_liner: str = Field(description="一句话总结当前多空格局")
    bullish: list[Catalyst] = Field(description="近期利好催化剂，按重要性排序")
    bearish: list[Catalyst] = Field(description="近期利空催化剂，按重要性排序")
    watch: list[str] = Field(default_factory=list, description="接下来值得盯的事件或数据点")


class MacroEvent(BaseModel):
    date: str = Field(description="YYYY-MM-DD 或日期范围")
    event: str
    why_it_matters: str = Field(description="对大盘或科技/半导体板块的含义")
    importance: Literal["高", "中", "低"]


class MacroNews(BaseModel):
    title: str
    impact: Literal["利好", "利空", "中性"]
    affected: list[str] = Field(description="受影响的板块或标的，例如 半导体、七巨头、NVDA")
    why: str
    date: str = ""
    source: str = ""


class MacroBrief(BaseModel):
    as_of: str
    regime: str = Field(description="两三句话：当前大盘与科技股所处的宏观环境")
    upcoming_events: list[MacroEvent] = Field(description="未来两周左右的重要事件：FOMC、CPI、非农、重要财报、发布会等")
    recent_news: list[MacroNews] = Field(description="过去一周对大盘/科技/半导体影响最大的新闻")


# ---------------- 凭证 ----------------
def has_credentials() -> bool:
    """有 API key、auth token，或 `ant auth login` 留下的 profile 即可。"""
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return True
    cfg = Path.home() / ".config" / "anthropic"
    return cfg.exists() and any(cfg.iterdir())


# ---------------- 研究助手 ----------------
CATALYST_SYSTEM = """You are an equity research assistant. You receive recent news headlines/summaries and a fundamentals
snapshot for one US stock. Sort the news into bullish and bearish catalysts for the stock price over the next few weeks.
Use ONLY the material provided; do not invent events. Drop items that are irrelevant to this company. Rate importance
by how much it could move the stock. Write every text field in Chinese, keep sentences short. Never give buy/sell advice."""

MACRO_RESEARCH_SYSTEM = """You are a markets research assistant. Use web search to find (1) the most important scheduled
macro and market events in the next ~2 weeks (FOMC, CPI/PCE, jobs report, Treasury auctions, major earnings, big tech
product events, policy/tariff deadlines), and (2) the past week's news that most affected the US stock market, the
Magnificent 7, semiconductors, and memory/storage stocks. Prefer primary or major financial sources, include dates.
Write a concise research note in Chinese with clearly separated sections: 环境判断 / 未来事件（含日期）/ 近期要闻（含利好利空判断和影响板块）.
Never give buy/sell advice."""


class Researcher:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.client = anthropic.Anthropic()
        self.model = model

    def catalysts(self, symbol: str, news: list[dict], info: dict, price_note: str = "") -> CatalystReport:
        lines = [f"# {symbol} {info.get('longName', '')}", price_note, "", "## Fundamentals"]
        for k in ["sector", "industry", "marketCap", "trailingPE", "forwardPE", "revenueGrowth", "earningsGrowth",
                  "targetMeanPrice", "recommendationKey", "shortPercentOfFloat"]:
            if info.get(k) is not None:
                lines.append(f"- {k}: {info[k]}")
        lines += ["", "## Recent news (newest first)"]
        for n in news[:20]:
            lines.append(f"- [{(n.get('time') or '')[:10]}] {n.get('title', '')} ({n.get('source', '')})")
            if n.get("summary"):
                lines.append(f"    {n['summary'][:300]}")
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=CATALYST_SYSTEM,
            messages=[{"role": "user", "content": "\n".join(lines)}],
            output_format=CatalystReport,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("模型拒绝了本次请求")
        return response.parsed_output

    def macro_brief(self, focus: list[str], max_search: int = 6) -> MacroBrief:
        today = datetime.now().strftime("%Y-%m-%d")
        user = (f"Today is {today}. Focus groups: {', '.join(focus)}. "
                "Research the upcoming two weeks of scheduled events and the past week's most market-moving news, then write the note.")
        messages: list[dict] = [{"role": "user", "content": user}]
        tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_search}]
        response = None
        for _ in range(4):  # 服务端搜索循环可能以 pause_turn 暂停，原样续上即可
            response = self.client.messages.create(
                model=self.model, max_tokens=16000, system=MACRO_RESEARCH_SYSTEM, tools=tools, messages=messages,
            )
            if response.stop_reason != "pause_turn":
                break
            messages.append({"role": "assistant", "content": response.content})
        assert response is not None
        if response.stop_reason == "refusal":
            raise RuntimeError("模型拒绝了本次请求")
        note = "\n".join(b.text for b in response.content if b.type == "text").strip()
        if not note:
            raise RuntimeError("搜索没有返回可用文本")

        structured = self.client.messages.parse(
            model=self.model,
            max_tokens=16000,
            system="Convert the research note into the requested structure faithfully. Keep Chinese text. Do not add facts.",
            messages=[{"role": "user", "content": f"as_of: {today}\n\n{note}"}],
            output_format=MacroBrief,
        )
        if structured.stop_reason == "refusal":
            raise RuntimeError("模型拒绝了结构化请求")
        return structured.parsed_output
