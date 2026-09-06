from __future__ import annotations

from pathlib import Path

from ..config import Settings
from .base import Account, Broker, OrderResult, Position
from .local_paper import LocalPaperBroker


def make_broker(settings: Settings) -> Broker:
    if settings.mode == "local":
        return LocalPaperBroker(
            state_path=settings.log_dir / "paper_state.json",
            initial_cash=settings.initial_cash,
            slippage_bps=settings.slippage_bps,
        )
    if settings.mode == "alpaca":
        from .alpaca_broker import AlpacaBroker

        return AlpacaBroker(settings.alpaca_api_key, settings.alpaca_secret_key, paper=settings.alpaca_paper)
    raise ValueError(f"未知 MODE={settings.mode!r}，可选 local / alpaca")


__all__ = ["Account", "Broker", "OrderResult", "Position", "LocalPaperBroker", "make_broker"]
