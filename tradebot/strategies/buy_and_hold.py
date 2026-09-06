from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Strategy


@dataclass
class BuyAndHold(Strategy):
    """满仓持有，作为回测基准。"""

    name: str = "buy_and_hold"

    def target_weights(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index, name="weight")
