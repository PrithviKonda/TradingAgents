"""What every venue looks like to the trading session: a book, and a way to order.

The paper broker and real brokers both implement ``Broker``, so a session that
runs against paper today runs against a live account by swapping one object.
Each broker reports its book as a ``PortfolioContext``, the same shape a
``--portfolio`` file has, so the decision agents size against what is actually
held rather than against a file the caller has to keep in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tradingagents.portfolio import PortfolioContext


class BrokerError(RuntimeError):
    """A venue refused or failed an order, or could not report the book."""


@dataclass(frozen=True)
class Order:
    ticker: str
    side: str          # "buy" or "sell"
    quantity: float    # always positive; the side carries the direction

    def __post_init__(self):
        if self.side not in ("buy", "sell"):
            raise ValueError(f"order side must be buy or sell, got {self.side!r}")
        if self.quantity <= 0:
            raise ValueError(f"order quantity must be positive, got {self.quantity}")

    def signed(self) -> float:
        return self.quantity if self.side == "buy" else -self.quantity


@dataclass(frozen=True)
class Fill:
    """What happened to an order: filled at a price, or accepted and still working."""

    ticker: str
    side: str
    quantity: float
    price: float | None
    commission: float = 0.0
    date: str | None = None
    status: str = "filled"     # "filled", or the venue's status for a working order
    order_id: str | None = None


class Broker(Protocol):
    name: str

    def portfolio(self) -> PortfolioContext:
        """Cash and positions as the venue reports them now."""
        ...

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus positions marked at ``prices`` (the venue's own mark if it has one)."""
        ...

    def submit(self, order: Order, price: float, date: str) -> Fill:
        """Place ``order``. ``price`` is the reference price it was sized at."""
        ...
