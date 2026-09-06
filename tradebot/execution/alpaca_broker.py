"""Alpaca 适配。paper=True 走模拟盘 endpoint；paper=False 才是实盘。

只用市价单、当日有效、整数股。alpaca-py 0.44 接口已核对。
"""
from __future__ import annotations

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from .base import Account, Broker, OrderResult, Position


class AlpacaBroker(Broker):
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        if not api_key or not secret_key:
            raise ValueError("缺少 ALPACA_API_KEY / ALPACA_SECRET_KEY")
        self.paper = paper
        self.name = "alpaca_paper" if paper else "alpaca_live"
        self.trading = TradingClient(api_key, secret_key, paper=paper)
        self.data = StockHistoricalDataClient(api_key, secret_key)

    def account(self) -> Account:
        a = self.trading.get_account()
        return Account(equity=float(a.equity), cash=float(a.cash))

    def positions(self) -> dict[str, Position]:
        out = {}
        for p in self.trading.get_all_positions():
            out[p.symbol] = Position(p.symbol, float(p.qty), float(p.market_value), float(p.avg_entry_price))
        return out

    def last_price(self, symbol: str) -> float:
        res = self.data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
        return float(res[symbol].price)

    def is_market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)

    def submit_market_order(self, symbol: str, qty: float, side: str) -> OrderResult:
        qty_int = int(qty)
        if qty_int <= 0:
            raise ValueError("整数股数量必须 >= 1")
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty_int,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading.submit_order(req)
        return OrderResult(symbol, side, qty_int, self.last_price(symbol), str(order.id), str(order.status))
