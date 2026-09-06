from __future__ import annotations

from .base import EqualWeightAdapter, PortfolioStrategy, Strategy, as_portfolio
from .buy_and_hold import BuyAndHold
from .ma_cross import MACross
from .composite import MainWave
from .momentum import High52Breakout, RelativeMomentum
from .reversal import BottomRecovery, OversoldBounce
from .swing import Breakout, PostEarningsMomentum, SwingStrategy, TrendPullback

STRATEGIES: dict[str, type] = {
    "trend_pullback": TrendPullback,
    "breakout": Breakout,
    "post_earnings": PostEarningsMomentum,
    "oversold_bounce": OversoldBounce,
    "bottom_recovery": BottomRecovery,
    "relative_momentum": RelativeMomentum,
    "high52_breakout": High52Breakout,
    "main_wave": MainWave,
    "ma_cross": MACross,
    "buy_and_hold": BuyAndHold,
}


def get_strategy(name: str, **params):
    try:
        cls = STRATEGIES[name]
    except KeyError as e:
        raise KeyError(f"未知策略 {name!r}，可选: {list(STRATEGIES)}") from e
    return cls(**params)


__all__ = [
    "Strategy", "PortfolioStrategy", "EqualWeightAdapter", "as_portfolio", "SwingStrategy",
    "TrendPullback", "Breakout", "PostEarningsMomentum", "OversoldBounce", "BottomRecovery", "RelativeMomentum", "High52Breakout", "MainWave", "MACross", "BuyAndHold", "STRATEGIES", "get_strategy",
]
