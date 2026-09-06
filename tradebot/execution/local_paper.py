"""本地模拟撮合：不需要任何账户，状态存在 logs/paper_state.json。

成交价 = 最新 bar 收盘价 ± 滑点。只做多，不支持融资。
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .base import Account, Broker, OrderResult, Position

NY = ZoneInfo("America/New_York")


class LocalPaperBroker(Broker):
    name = "local_paper"

    def __init__(self, state_path: Path, initial_cash: float = 100_000.0, slippage_bps: float = 2.0):
        self.state_path = Path(state_path)
        self.slippage = slippage_bps / 1e4
        self._prices: dict[str, float] = {}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {"cash": float(initial_cash), "positions": {}, "fills": []}
            self._save()

    # ---- 状态 ----
    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False))

    def set_prices(self, prices: dict[str, float]) -> None:
        self._prices.update({k.upper(): float(v) for k, v in prices.items()})

    def last_price(self, symbol: str) -> float:
        symbol = symbol.upper()
        if symbol not in self._prices:
            raise KeyError(f"本地模拟盘没有 {symbol} 的价格，请先 set_prices")
        return self._prices[symbol]

    # ---- 查询 ----
    def positions(self) -> dict[str, Position]:
        out = {}
        for sym, p in self.state["positions"].items():
            qty = float(p["qty"])
            if qty == 0:
                continue
            price = self._prices.get(sym, float(p["avg_price"]))
            out[sym] = Position(sym, qty, qty * price, float(p["avg_price"]))
        return out

    def account(self) -> Account:
        mv = sum(p.market_value for p in self.positions().values())
        cash = float(self.state["cash"])
        return Account(equity=cash + mv, cash=cash)

    def is_market_open(self) -> bool:
        from ..econ_calendar import market_session_open  # 含 NYSE 休市日与 13:00 提前收盘

        return market_session_open(datetime.now(NY))

    # ---- 下单 ----
    def submit_market_order(self, symbol: str, qty: float, side: str) -> OrderResult:
        symbol = symbol.upper()
        qty = float(qty)
        if qty <= 0:
            raise ValueError("qty 必须为正")
        px = self.last_price(symbol)
        fill_px = px * (1 + self.slippage) if side == "buy" else px * (1 - self.slippage)
        pos = self.state["positions"].setdefault(symbol, {"qty": 0.0, "avg_price": 0.0})

        if side == "buy":
            cost = fill_px * qty
            if cost > self.state["cash"] + 1e-6:
                raise ValueError(f"现金不足：需要 {cost:.2f}，可用 {self.state['cash']:.2f}")
            new_qty = pos["qty"] + qty
            pos["avg_price"] = (pos["avg_price"] * pos["qty"] + fill_px * qty) / new_qty
            pos["qty"] = new_qty
            self.state["cash"] -= cost
        elif side == "sell":
            if qty > pos["qty"] + 1e-9:
                raise ValueError(f"持仓不足：想卖 {qty}，持有 {pos['qty']}")
            pos["qty"] -= qty
            self.state["cash"] += fill_px * qty
            if pos["qty"] <= 1e-9:
                self.state["positions"].pop(symbol, None)
        else:
            raise ValueError(f"未知方向 {side!r}")

        oid = uuid.uuid4().hex[:12]
        self.state["fills"].append(
            {"time": datetime.now(NY).isoformat(), "symbol": symbol, "side": side, "qty": qty, "price": fill_px, "id": oid}
        )
        self._save()
        return OrderResult(symbol, side, qty, fill_px, oid, "filled")
