"""A paper account: a cash ledger and positions, filled at a reference price.

Used two ways. The simulator drives one in memory over historical closes, and a
paper trading session keeps one on disk between runs, so a paper account
accumulates fills day after day the way a real one would.

Fills are deliberately pessimistic: a buy pays ``slippage_bps`` over the
reference price and a sell receives that much under it, and each fill pays a
fixed plus a proportional commission. A buy the cash cannot cover is cut to what
it can; the account is long-only unless ``allow_short`` is set.

A paper trading session (``fill_next_close``) does not fill at the close its
analysis already read: that would trade on news the close had priced in. Its
orders wait as pending and fill at the first close after the analysis date,
the rule the simulator and settlement use (#1518), when ``settle`` is called
on a later run.

Every price source here is split-adjusted, so the ledger has to be too: opened
with a ``splits`` source, the account applies each split since a holding's last
fill to its units and average price, and records the adjustment, so a 10:1
split reads as ten times the units at a tenth of the cost, not a 90% loss.
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from tradingagents.dataflows.files import locked
from tradingagents.execution.broker import BrokerError, Fill, Order
from tradingagents.portfolio import PortfolioContext, Position

logger = logging.getLogger(__name__)

# closes(ticker, start, end) -> daily closes from start up to, not including, end.
CloseSource = Callable[[str, str, str], pd.Series]

# splits(ticker) -> new-shares-per-old ratios keyed by ex-date, e.g. 10.0 for 10:1.
SplitSource = Callable[[str], pd.Series]


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
    # Splits applied to the holdings: {"ticker", "date", "ratio"}, oldest first.
    adjustments: list[dict] = field(default_factory=list)
    # Queue orders for the next close instead of filling them at ``price``.
    fill_next_close: bool = False
    # Orders waiting for that close: {"ticker", "side", "quantity", "date"}.
    pending: list[dict] = field(default_factory=list)
    name: str = "paper"

    def __post_init__(self):
        if self.starting_cash is None:
            self.starting_cash = self.cash

    # -- persistence ---------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path, starting_cash: float, splits: SplitSource | None = None,
             **kwargs) -> PaperBroker:
        """The account saved at ``path``, or a new one funded with ``starting_cash``.

        With ``splits``, any split since a holding's last fill is applied first.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        broker = cls(cash=starting_cash, path=path, **kwargs)
        with locked(path):
            if not path.exists():
                broker._write()
                return broker
            broker._reload()
            if splits is not None and broker.apply_splits(splits):
                broker._write()
        return broker

    def save(self) -> None:
        if self.path is None:
            return
        with locked(self.path):
            self._write()

    def _state(self) -> dict:
        return {
            "cash": self.cash,
            "starting_cash": self.starting_cash,
            "currency": self.currency,
            "positions": {t: asdict(h) for t, h in self.holdings.items()},
            "fills": [asdict(f) for f in self.fills],
            "adjustments": [dict(a) for a in self.adjustments],
            "pending": [dict(p) for p in self.pending],
        }

    def _write(self) -> None:
        """Write the account; the caller holds the lock on ``path``."""
        data = json.dumps(self._state(), indent=2)
        # Through a uniquely named temp file renamed over the target, so an
        # interrupted write leaves the previous account rather than half of a
        # new one, and two writers never share (and remove) one temp file.
        temp = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp.write_text(data, encoding="utf-8")
            os.replace(temp, self.path)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise

    def _reload(self) -> None:
        """Replace the in-memory account with the one on disk."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.cash = float(data["cash"])
            self.starting_cash = float(data.get("starting_cash", data["cash"]))
            self.currency = data.get("currency", "USD")
            self.holdings = {t: _Holding(float(h["quantity"]), float(h["average_price"]))
                             for t, h in data.get("positions", {}).items()}
            self.fills = [Fill(**f) for f in data.get("fills", [])]
            self.adjustments = list(data.get("adjustments", []))
            self.pending = list(data.get("pending", []))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise BrokerError(f"paper account {self.path} is not readable: {exc}") from exc

    def apply_splits(self, splits: SplitSource) -> list[dict]:
        """Apply each split since a holding's last fill; returns the adjustments made."""
        made = []
        done = {(a["ticker"], a["date"]) for a in self.adjustments}
        for ticker, held in self.holdings.items():
            # A fill on the ex-date was already at the post-split price.
            since = max((f.date or "" for f in self.fills if f.ticker == ticker), default="")
            try:
                history = splits(ticker)
            except Exception as exc:  # no split data must not lock the account
                logger.warning("Could not read splits for %s: %s", ticker, exc)
                continue
            for day, ratio in history.sort_index().items():
                date = pd.Timestamp(day).strftime("%Y-%m-%d")
                ratio = float(ratio)
                if date <= since or (ticker, date) in done or not math.isfinite(ratio) or ratio <= 0 or ratio == 1:
                    continue
                held.quantity *= ratio
                held.average_price /= ratio
                made.append({"ticker": ticker, "date": date, "ratio": ratio})
        self.adjustments.extend(sorted(made, key=lambda a: (a["date"], a["ticker"])))
        return made

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

    def open_orders(self) -> list[Order]:
        return [Order(p["ticker"], p["side"], p["quantity"]) for p in self.pending]

    def submit(self, order: Order, price: float, date: str) -> Fill:
        if price <= 0 or not math.isfinite(price):
            raise BrokerError(f"no usable price for {order.ticker}: {price}")
        if self.fill_next_close:
            return self._transact(lambda: self._queue(order, date))
        return self._transact(lambda: self._fill(order, price, date))

    def settle(self, closes: CloseSource, through: str | None = None) -> list[Fill]:
        """Fill each pending order at the first close after its date, up to ``through``.

        ``through`` defaults to the last session that has closed, so an
        in-progress bar is never taken for a close. An order with no such close
        yet stays pending.
        """
        through = through or _last_closed_session()
        return self._transact(lambda: self._settle(closes, through))

    def _transact(self, change: Callable[[], object]):
        """Run ``change`` on the account as it is on disk, and save it, under the lock."""
        if self.path is None:
            return change()
        # Another session may have traded this account since it was opened:
        # re-read it under the lock, so its fills are built on, not overwritten.
        with locked(self.path):
            self._reload()
            before = self._state()
            result = change()
            try:
                self._write()
            except BaseException:
                self._restore(before)      # not saved, so not applied
                raise
        return result

    def _queue(self, order: Order, date: str) -> Fill:
        ticker = order.ticker.upper()
        self.pending.append({"ticker": ticker, "side": order.side, "quantity": order.quantity, "date": date})
        return Fill(ticker, order.side, order.quantity, None, date=date, status="pending")

    def _settle(self, closes: CloseSource, through: str) -> list[Fill]:
        filled, waiting = [], []
        end = (datetime.strptime(through, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        # Oldest first, and sells before buys, so a sell's cash can fund a buy.
        for p in sorted(self.pending, key=lambda p: (p["date"], p["side"] != "sell")):
            start = (datetime.strptime(p["date"], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            try:
                series = _daily(closes(p["ticker"], start, end)) if start < end else pd.Series(dtype=float)
            except Exception as exc:  # an outage leaves the order for the next run
                logger.warning("No close to settle %s %s: %s", p["side"], p["ticker"], exc)
                series = pd.Series(dtype=float)
            if series.empty:
                waiting.append(p)
                continue
            day = series.index[0].strftime("%Y-%m-%d")
            fill = self._fill(Order(p["ticker"], p["side"], float(p["quantity"])), float(series.iloc[0]), day)
            filled.append(fill)
        self.pending = waiting
        return filled

    def _fill(self, order: Order, price: float, date: str) -> Fill:
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
        return fill

    def _restore(self, state: dict) -> None:
        self.cash = state["cash"]
        self.holdings = {t: _Holding(**h) for t, h in state["positions"].items()}
        self.fills = [Fill(**f) for f in state["fills"]]
        self.adjustments = state["adjustments"]
        self.pending = state["pending"]

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


def _daily(closes: pd.Series) -> pd.Series:
    """Closes keyed by calendar day, missing and non-positive closes dropped."""
    closes = closes.dropna()
    closes = closes[closes > 0]
    index = pd.DatetimeIndex(closes.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    closes.index = index.normalize()
    return closes.sort_index()


def _last_closed_session(now: datetime | None = None) -> str:
    """Today once the US close has printed, else yesterday."""
    now = now or datetime.now(ZoneInfo("America/New_York"))
    day = now.date() if now.time() >= time(16, 30) else now.date() - timedelta(days=1)
    return day.isoformat()
