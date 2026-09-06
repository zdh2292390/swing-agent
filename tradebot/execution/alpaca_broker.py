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

    # ---------- 面板用的详细信息 ----------
    def account_details(self) -> dict:
        a = self.trading.get_account()
        f = lambda x: float(x) if x is not None else None
        return {
            "status": str(getattr(a, "status", "")).split(".")[-1], "currency": getattr(a, "currency", "USD"),
            "equity": f(a.equity), "cash": f(a.cash), "buying_power": f(a.buying_power), "portfolio_value": f(a.portfolio_value),
            "last_equity": f(a.last_equity), "daytrade_count": int(a.daytrade_count or 0), "pattern_day_trader": bool(a.pattern_day_trader),
            "multiplier": f(a.multiplier), "paper": self.paper,
        }

    def positions_detail(self) -> list[dict]:
        out = []
        for p in self.trading.get_all_positions():
            out.append({
                "symbol": p.symbol, "qty": float(p.qty), "side": str(p.side).split(".")[-1].lower(), "avg_price": float(p.avg_entry_price),
                "current_price": float(p.current_price) if p.current_price is not None else None, "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl is not None else None,
                "unrealized_plpc": float(p.unrealized_plpc) * 100 if p.unrealized_plpc is not None else None,
                "change_today": float(p.change_today) * 100 if p.change_today is not None else None,
            })
        return out

    def orders(self, status: str = "open", limit: int = 50, days: int = 30) -> list[dict]:
        from datetime import datetime, timedelta, timezone

        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        st = {"open": QueryOrderStatus.OPEN, "closed": QueryOrderStatus.CLOSED, "all": QueryOrderStatus.ALL}[status]
        req = GetOrdersRequest(status=st, limit=limit, after=datetime.now(timezone.utc) - timedelta(days=days) if status != "open" else None)
        out = []
        for o in self.trading.get_orders(req):
            out.append({
                "symbol": o.symbol, "side": str(o.side).split(".")[-1].lower(), "type": str(o.type).split(".")[-1].lower(),
                "qty": float(o.qty) if o.qty is not None else None, "filled_qty": float(o.filled_qty) if o.filled_qty is not None else 0.0,
                "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price is not None else None,
                "limit_price": float(o.limit_price) if o.limit_price is not None else None,
                "status": str(o.status).split(".")[-1].lower(), "submitted_at": o.submitted_at, "filled_at": o.filled_at,
            })
        return out

    def portfolio_history(self, period: str = "1M", timeframe: str = "1D"):
        import pandas as pd
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        h = self.trading.get_portfolio_history(GetPortfolioHistoryRequest(period=period, timeframe=timeframe))
        idx = pd.to_datetime(h.timestamp, unit="s", utc=True).tz_convert("America/New_York")
        return pd.DataFrame({"equity": h.equity, "profit_loss": h.profit_loss, "profit_loss_pct": [x * 100 if x is not None else None for x in h.profit_loss_pct]}, index=idx)
