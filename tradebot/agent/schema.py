from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SymbolView(BaseModel):
    symbol: str
    stance: Literal["bullish", "neutral", "bearish"]
    confidence: float = Field(ge=0.0, le=1.0, description="0 到 1")
    reasoning: str = Field(description="两三句话，说明依据的是哪些给定数据")
    risk_flags: list[str] = Field(default_factory=list, description="例如 波动骤增 / 成交量异常 / 信号刚翻转")


class AdvisorReport(BaseModel):
    market_note: str = Field(description="对整体环境的一句话判断")
    views: list[SymbolView]
