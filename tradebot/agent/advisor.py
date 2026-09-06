"""Claude 决策层：读最近的小时线摘要和策略信号，输出结构化观点。

原则：LLM 只能否决或降低仓位，不能加仓，也不能直接下单。
"""
from __future__ import annotations

import anthropic
import pandas as pd

from .schema import AdvisorReport

SYSTEM = """You are a risk-aware research assistant embedded in an hourly US-equity trading system.
You receive, for each symbol, recent hourly OHLCV statistics and the current signal from a rules-based strategy.
Your job is to sanity-check each signal using ONLY the data provided: flag regime shifts, volatility spikes,
volume anomalies, or signals that just flipped and may whipsaw. You never place orders and you cannot increase
position size; a "bearish" view with confidence >= 0.6 will cause the system to skip or reduce that position.
Be conservative and specific. Write `reasoning` and `market_note` in Chinese."""


def _summarize_symbol(sym: str, df: pd.DataFrame, signal: float, bars: int = 30) -> str:
    recent = df.tail(bars)
    close = recent["close"]
    ret_1h = close.pct_change().dropna()
    last = recent.iloc[-1]
    day_ago = df["close"].iloc[-8] if len(df) >= 8 else df["close"].iloc[0]
    week_ago = df["close"].iloc[-35] if len(df) >= 35 else df["close"].iloc[0]
    vol_recent = ret_1h.tail(10).std()
    vol_base = ret_1h.std()
    avg_volume = recent["volume"].mean()
    lines = [
        f"## {sym}",
        f"last_close={last['close']:.2f} bar_time={recent.index[-1].isoformat()}",
        f"ret_since_1_day_ago={last['close'] / day_ago - 1:+.2%} ret_since_1_week_ago={last['close'] / week_ago - 1:+.2%}",
        f"hourly_vol_last10={vol_recent:.4f} hourly_vol_last{bars}={vol_base:.4f} vol_ratio={vol_recent / vol_base if vol_base else float('nan'):.2f}",
        f"last_bar_volume_vs_avg={last['volume'] / avg_volume if avg_volume else float('nan'):.2f}x",
        f"strategy_target_weight={signal:.3f} (组合目标权重, 0=空仓)",
        "last 8 hourly closes: " + ", ".join(f"{v:.2f}" for v in close.tail(8)),
    ]
    return "\n".join(lines)


class ClaudeAdvisor:
    def __init__(self, model: str = "claude-opus-5", veto_confidence: float = 0.6):
        self.client = anthropic.Anthropic()
        self.model = model
        self.veto_confidence = veto_confidence

    def build_prompt(self, data: dict[str, pd.DataFrame], signals: dict[str, float]) -> str:
        parts = [
            "Review the following hourly signals. For every symbol return a view.",
            *[_summarize_symbol(s, data[s], signals[s]) for s in signals],
        ]
        return "\n\n".join(parts)

    def review(self, data: dict[str, pd.DataFrame], signals: dict[str, float]) -> AdvisorReport:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": self.build_prompt(data, signals)}],
            output_format=AdvisorReport,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("模型拒绝了本次请求，保持策略原始信号")
        return response.parsed_output

    def apply(self, report: AdvisorReport, targets: dict[str, float]) -> tuple[dict[str, float], list[str]]:
        """bearish 且置信度足够 -> 该标的权重置零；其他情况保持策略权重不变。"""
        adjusted = dict(targets)
        notes: list[str] = []
        for view in report.views:
            sym = view.symbol.upper()
            if sym not in adjusted:
                continue
            if view.stance == "bearish" and view.confidence >= self.veto_confidence and adjusted[sym] > 0:
                notes.append(f"{sym}: Claude 否决（{view.confidence:.2f}）{view.reasoning}")
                adjusted[sym] = 0.0
        return adjusted, notes
