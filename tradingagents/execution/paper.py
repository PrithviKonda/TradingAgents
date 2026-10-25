"""A paper account: a cash ledger and positions, filled at a reference price.

Used two ways. The simulator drives one in memory over historical closes, and a
paper trading session keeps one on disk between runs, so a paper account
accumulates fills day after day the way a real one would.

Fills are deliberately pessimistic: a buy pays ``slippage_bps`` over the
reference price and a sell receives that much under it, and each fill pays a
fixed plus a proportional commission. A buy the cash cannot cover is cut to what
it can; the account is long-only unless ``allow_short`` is set.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tradingagents.execution.broker import BrokerError, Fill, Order
from tradingagents.portfolio import PortfolioContext, Position


@dataclass
class CostModel:
    slippage_bps: float = 5.0
    commission_per_order: float = 0.0
    commission_pct: float = 0.0

    @classmethod
    def from_config(cls, config: dict) -> CostModel:
        execution = config.get("execution", {})
        return cls(
            slippage_bps=float(execution.get("slippage_bps", 5.0)),
            commission_per_order=float(execution.get("commission_per_order", 0.0)),
            commission_pct=float(execution.get("commission_pct", 0.0)),
        )

    def fill_price(self, side: str, price: float) -> float:
        drift = price * self.slippage_bps / 10_000
        return price + drift if side == "buy" else price - drift

    def commission(self, notional: float) -> float:
        return self.commission_per_order + abs(notional) * self.commission_pct


@dataclass
class _Holding:
    quantity: float
    average_price: float


@dataclass
class PaperBroker:
    cash: float
    costs: CostModel = field(default_factory=CostModel)
    currency: str = "USD"
    allow_short: bool = False
    allow_fractional: bool = False
    path: Path | None = None
    starting_cash: float | None = None
    holdings: dict[str, _Holding] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    name: str = "paper"

    def __post_init__(self):
        if self.starting_cash is None:
            self.starting_cash = self.cash

    # -- persistence ---------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path, starting_cash: float, **kwargs) -> PaperBroker:
        """The account saved at ``path``, or a new one funded with ``starting_cash``."""
        path = Path(path)
        if not path.exists():
            broker = cls(cash=starting_cash, path=path, **kwargs)
            broker.save()
            return broker
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            broker = cls(cash=float(data["cash"]), path=path,
                         starting_cash=float(data.get("starting_cash", data["cash"])),
                         currency=data.get("currency", "USD"), **kwargs)
            broker.holdings = {t: _Holding(float(h["quantity"]), float(h["average_price"]))
                               for t, h in data.get("positions", {}).items()}
            broker.fills = [Fill(**f) for f in data.get("fills", [])]
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise BrokerError(f"paper account {path} is not readable: {exc}") from exc
        return broker

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "cash": self.cash,
            "starting_cash": self.starting_cash,
            "currency": self.currency,
            "positions": {t: asdict(h) for t, h in self.holdings.items()},
            "fills": [asdict(f) for f in self.fills],
        }
        # Written beside the target and renamed over it, so an interrupted
        # write leaves the previous account rather than half of a new one.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    # -- the Broker interface ------------------------------------------------

    def quantity(self, ticker: str) -> float:
        held = self.holdings.get(ticker.upper())
        return held.quantity if held else 0.0

    def portfolio(self) -> PortfolioContext:
        return PortfolioContext(
            cash=round(self.cash, 2),
            currency=self.currency,
            positions=[Position(ticker=t, quantity=h.quantity, average_price=round(h.average_price, 4))
                       for t, h in sorted(self.holdings.items())],
        )

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus positions at ``prices``; a position with no price is marked at cost."""
        value = self.cash
        for ticker, held in self.holdings.items():
            value += held.quantity * prices.get(ticker, held.average_price)
        return value

    def submit(self, order: Order, price: float, date: str) -> Fill:
        if price <= 0 or not math.isfinite(price):
            raise BrokerError(f"no usable price for {order.ticker}: {price}")
        ticker = order.ticker.upper()
        quantity = order.quantity
        held = self.quantity(ticker)
        fill_price = self.costs.fill_price(order.side, price)

        if order.side == "sell" and not self.allow_short:
            quantity = min(quantity, max(held, 0.0))
        if order.side == "buy":
            quantity = min(quantity, self._affordable(fill_price))
        if not self.allow_fractional:
            quantity = math.floor(quantity + 1e-9)
        if quantity <= 0:
            return Fill(ticker, order.side, 0.0, None, date=date, status="rejected")

        notional = quantity * fill_price
        commission = self.costs.commission(notional)
        if order.side == "buy":
            self.cash -= notional + commission
        else:
            self.cash += notional - commission
        self._apply(ticker, quantity if order.side == "buy" else -quantity, fill_price)

        fill = Fill(ticker, order.side, quantity, round(fill_price, 6),
                    commission=round(commission, 6), date=date)
        self.fills.append(fill)
        self.save()
        return fill

    # -- internals -------------------------------------------------------------

    def _affordable(self, fill_price: float) -> float:
        """Most units the cash covers at ``fill_price``, commission included."""
        budget = self.cash - self.costs.commission_per_order
        if budget <= 0:
            return 0.0
        return budget / (fill_price * (1 + self.costs.commission_pct))

    def _apply(self, ticker: str, signed: float, price: float) -> None:
        held = self.holdings.get(ticker)
        if held is None:
            self.holdings[ticker] = _Holding(signed, price)
            return
        new_quantity = held.quantity + signed
        if abs(new_quantity) < 1e-9:
            del self.holdings[ticker]
            return
        # Adding in the position's direction moves the average; reducing it
        # realizes against the existing average and leaves it as it was.
        if held.quantity * signed > 0:
            held.average_price = (held.quantity * held.average_price + signed * price) / new_quantity
        elif held.quantity * new_quantity < 0:
            held.average_price = price      # crossed through zero into the other side
        held.quantity = new_quantity
