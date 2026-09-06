from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


def union_index(data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    idx: pd.DatetimeIndex | None = None
    for df in data.values():
        idx = df.index if idx is None else idx.union(df.index)
    if idx is None:
        raise ValueError("没有数据")
    return idx


class Strategy(ABC):
    """单标的策略接口（简单场景用）。

    target_weights 输入单个标的的 OHLCV，输出与 df 同索引、取值 [0, 1] 的 Series，
    表示"在该 bar 收盘时希望持有该标的分配额度的多少"。只能用截至当前 bar 的数据。
    回测和实盘会自动把它包成 EqualWeightAdapter：按当时已上市标的数等分资金。
    """

    name: str = "base"

    @abstractmethod
    def target_weights(self, df: pd.DataFrame) -> pd.Series: ...

    def warmup_bars(self) -> int:
        return 0

    def describe(self) -> str:
        return self.name


class PortfolioStrategy(ABC):
    """组合级策略接口（swing 策略用）。

    target_weights 一次看到全部标的，输出 DataFrame：索引是全部标的时间戳的并集，
    列是标的，值是组合权重 [0, 1]，每行之和不超过 1。同样只能用截至当前 bar 的数据，
    引擎会把权重后移一根 bar 再成交，这里不要自己 shift。
    """

    name: str = "portfolio"

    @abstractmethod
    def target_weights(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame: ...

    def warmup_bars(self) -> int:
        return 0

    def describe(self) -> str:
        return self.name


class EqualWeightAdapter(PortfolioStrategy):
    """把单标的策略包装成组合策略：每个标的的 [0,1] 信号 × 1/当时已上市标的数。"""

    def __init__(self, inner: Strategy):
        self.inner = inner
        self.name = inner.name

    def target_weights(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        union = union_index(data)
        sigs, avail = {}, {}
        for sym, df in data.items():
            sig = self.inner.target_weights(df).clip(lower=0.0, upper=1.0).fillna(0.0)
            sigs[sym] = sig.reindex(union).ffill().fillna(0.0)  # 缺 bar 持仓不变；上市前为 0
            avail[sym] = pd.Series(union >= df.index[0], index=union)
        n_avail = pd.DataFrame(avail).sum(axis=1).replace(0, 1)
        return pd.DataFrame(sigs).div(n_avail, axis=0).fillna(0.0)

    def warmup_bars(self) -> int:
        return self.inner.warmup_bars()

    def describe(self) -> str:
        return self.inner.describe()


def as_portfolio(strategy: Strategy | PortfolioStrategy) -> PortfolioStrategy:
    if isinstance(strategy, PortfolioStrategy):
        return strategy
    if isinstance(strategy, Strategy):
        return EqualWeightAdapter(strategy)
    raise TypeError(f"不是策略对象: {type(strategy)}")
