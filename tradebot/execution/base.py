from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Account:
    equity: float
    cash: float


@dataclass
class Position:
    symbol: str
    qty: float
    market_value: float
    avg_price: float


@dataclass
class OrderResult:
    symbol: str
    side: str  # buy | sell
    qty: float
    est_price: float
    order_id: str
    status: str


class Broker(ABC):
    """券商适配接口。本地模拟盘和 Alpaca 实现同一接口，run.py 不感知差异。"""

    name: str = "broker"

    @abstractmethod
    def account(self) -> Account: ...

    @abstractmethod
    def positions(self) -> dict[str, Position]: ...

    @abstractmethod
    def last_price(self, symbol: str) -> float: ...

    @abstractmethod
    def submit_market_order(self, symbol: str, qty: float, side: str) -> OrderResult: ...

    @abstractmethod
    def is_market_open(self) -> bool: ...

    def set_prices(self, prices: dict[str, float]) -> None:
        """本地模拟盘需要外部喂价；真实券商忽略即可。"""
        return None
