from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Strategy


@dataclass
class MACross(Strategy):
    """演示用基线：快均线上穿慢均线持有，下穿空仓。

    这只是让整条管道跑通的占位策略，不构成任何投资建议，
    小时线上的均线交叉在成本后通常没有优势，请替换成你自己的策略。
    """

    fast: int = 20
    slow: int = 50
    name: str = "ma_cross"

    def target_weights(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        fast = close.rolling(self.fast).mean()
        slow = close.rolling(self.slow).mean()
        w = (fast > slow).astype(float)
        w[slow.isna()] = 0.0
        return w.rename("weight")

    def warmup_bars(self) -> int:
        return self.slow + 5

    def describe(self) -> str:
        return f"{self.name}(fast={self.fast}, slow={self.slow})"
