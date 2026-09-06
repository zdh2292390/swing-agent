"""风控层：纯规则，任何来源的目标仓位（策略或 LLM）都必须经过这里。

规则顺序：
  1. 最大回撤停机：净值从历史高点回撤超过阈值 -> 全部清仓，halted=True
  2. 当日亏损熔断：当日净值跌幅超过阈值 -> 全部清仓，当日不再开仓
  3. 单标的仓位上限
  4. 总仓位上限（超出按比例缩减）
  5. 只做多：负权重一律置零
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PortfolioState:
    equity: float
    cash: float
    positions: dict[str, float]  # symbol -> 市值
    day_start_equity: float
    peak_equity: float


@dataclass
class RiskLimits:
    max_position_pct: float = 0.20
    max_gross_exposure: float = 1.0
    daily_loss_limit_pct: float = 0.02
    max_drawdown_pct: float = 0.10


@dataclass
class RiskDecision:
    approved: dict[str, float]
    halted: bool = False
    reasons: list[str] = field(default_factory=list)


class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def approve(self, targets: dict[str, float], state: PortfolioState) -> RiskDecision:
        lim = self.limits
        reasons: list[str] = []

        drawdown = state.equity / state.peak_equity - 1 if state.peak_equity > 0 else 0.0
        if drawdown <= -lim.max_drawdown_pct:
            reasons.append(
                f"KILL_SWITCH 回撤 {drawdown:.2%} 超过上限 {lim.max_drawdown_pct:.0%}，全部清仓并停机"
            )
            return RiskDecision({s: 0.0 for s in targets}, halted=True, reasons=reasons)

        day_pnl = state.equity / state.day_start_equity - 1 if state.day_start_equity > 0 else 0.0
        if day_pnl <= -lim.daily_loss_limit_pct:
            reasons.append(
                f"DAILY_STOP 当日亏损 {day_pnl:.2%} 超过上限 {lim.daily_loss_limit_pct:.0%}，清仓且当日不再开仓"
            )
            return RiskDecision({s: 0.0 for s in targets}, halted=True, reasons=reasons)

        approved: dict[str, float] = {}
        for sym, w in targets.items():
            w = float(w)
            if w < 0:
                reasons.append(f"{sym}: 只做多，负权重 {w:.3f} 置零")
                w = 0.0
            if w > lim.max_position_pct:
                reasons.append(f"{sym}: 权重 {w:.3f} 超过单标的上限 {lim.max_position_pct:.2f}，截断")
                w = lim.max_position_pct
            approved[sym] = w

        gross = sum(approved.values())
        if gross > lim.max_gross_exposure and gross > 0:
            scale = lim.max_gross_exposure / gross
            reasons.append(f"总仓位 {gross:.3f} 超过上限 {lim.max_gross_exposure:.2f}，整体缩放 x{scale:.3f}")
            approved = {s: w * scale for s, w in approved.items()}

        return RiskDecision(approved, halted=False, reasons=reasons)
