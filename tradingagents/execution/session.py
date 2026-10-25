"""One trading session: analyze tickers, size the ratings, and place the orders.

The broker decides where the orders go (a paper ledger on disk, Alpaca paper,
Alpaca live); everything else is the same. A session runs in three phases so a
human can look at the whole plan before anything is sent:

1. Analyze every ticker, each run seeing the broker's book as its portfolio
   context, so the agents know what is held. Each ticker runs as its own asset
   type, so a list mixing stocks and coins gives each the right analysts: one
   graph is built per distinct analyst set and shared by the tickers that use it.
2. Plan an order per decision from one equity figure, priced at the latest close.
   The plan already holds every limit an order is held to: a buy is cut to the
   per-order cap (``max_order_pct`` of equity, or the broker's
   ``max_order_notional``) and to the cash on hand plus what the session's sells free, so what the human approves is what gets sent. A sell
   that only reduces a position is never capped, so a grown position can exit.
3. Ask ``confirm(plan)`` once, then send sells before buys, so the cash a sell
   frees can fund a buy in the same session. A sell that is refused frees
   nothing, so the buys are cut again to the cash actually left. At a broker
   that only accepts an order (Alpaca), the session waits a bounded time for
   the sells to fill and counts only what filled; the paper ledger fills its
   queued sells before its buys, so a sell it accepted counts in full.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.market import get_closes
from tradingagents.execution.broker import Broker, Fill, Order
from tradingagents.execution.sizing import SizingRules, order_cap, plan_order, within_cap

logger = logging.getLogger(__name__)


def latest_close(ticker: str, date: str) -> float:
    """The last close on or before ``date``: the reference an order is sized at."""
    day = datetime.strptime(date, "%Y-%m-%d")
    closes = get_closes(ticker, (day - timedelta(days=10)).strftime("%Y-%m-%d"),
                        (day + timedelta(days=1)).strftime("%Y-%m-%d")).dropna()
    closes = closes[closes > 0]
    if closes.empty:
        raise ValueError(f"no recent close for {ticker} on or before {date}")
    return float(closes.iloc[-1])


@dataclass
class SessionStep:
    ticker: str
    rating: str | None = None
    price: float | None = None
    order: Order | None = None
    fill: Fill | None = None
    error: str | None = None
    note: str | None = None     # why the order differs from what the rating asked for

    def describe(self) -> str:
        if self.error:
            return f"{self.ticker}: failed - {self.error}"
        head = f"{self.ticker}: {self.rating}"
        note = f" ({self.note})" if self.note else ""
        if self.order is None:
            return f"{head}, no trade{note}"
        plan = f"{head}, {self.order.side} {self.order.quantity:g} @ ~{self.price:,.2f}{note}"
        if self.fill is None:
            return plan
        if self.fill.status == "filled" and self.fill.price is not None:
            return f"{plan} -> filled {self.fill.quantity:g} @ {self.fill.price:,.2f}"
        return f"{plan} -> {self.fill.status}"


def run_session(
    tickers: list[str],
    broker: Broker,
    config: dict,
    trade_date: str | None = None,
    asset_type: str | Mapping[str, str] = "stock",
    selected_analysts: Sequence[str] | Mapping[str, Sequence[str]] = ("market", "social", "news", "fundamentals"),
    price: Callable[[str, str], float] = latest_close,
    confirm: Callable[[list[SessionStep]], bool] | None = None,
    submit: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
    graph=None,
) -> list[SessionStep]:
    """Run the session; with ``submit=False`` it stops after planning (a dry run).

    ``asset_type`` is one type for every ticker, or a type per ticker (any
    spelling; a ticker it leaves out is a stock). ``selected_analysts`` is one
    analyst list, or a list per asset type. ``graph``, when given, runs every
    ticker whatever its type.
    """
    date = trade_date or get_current_date()
    rules = SizingRules.from_config(config)
    book = broker.portfolio()
    # Canonical symbols, so BTCUSD and BTC-USD are one instrument and one
    # position, not two that each get bought.
    steps = [SessionStep(t) for t in dict.fromkeys(normalize_symbol(t) for t in tickers if t.strip())]
    holding: dict[str, float] = {}
    for held in book.positions:
        key = normalize_symbol(held.ticker)
        holding[key] = holding.get(key, 0.0) + held.quantity

    if isinstance(asset_type, Mapping):
        types = {normalize_symbol(t): k for t, k in asset_type.items()}
        kinds = {s.ticker: types.get(s.ticker, "stock") for s in steps}
    else:
        kinds = {s.ticker: asset_type for s in steps}
    # Built before any analysis, so a setup error (a missing key) stops the whole session.
    graphs = (dict.fromkeys(kinds.values(), graph) if graph
              else _graphs(set(kinds.values()), selected_analysts, config))

    for index, step in enumerate(steps, 1):
        if progress:
            progress(index, len(steps), step.ticker)
        try:
            kind = kinds[step.ticker]
            _, step.rating = graphs[kind].propagate(step.ticker, date, kind, portfolio=book)
            step.price = price(step.ticker, date)
        except Exception as exc:  # one ticker's outage must not cancel the others' trades
            logger.warning("Session step %s failed: %s", step.ticker, exc)
            step.error = str(exc)

    marks = {s.ticker: s.price for s in steps if s.price}
    # A venue that values its own book (Alpaca) needs no marks for the rest of it.
    for ticker in holding if not getattr(broker, "marks_itself", False) else ():
        if ticker not in marks:
            # An unpriced holding is marked at cost by the paper broker.
            with contextlib.suppress(Exception):
                marks[ticker] = price(ticker, date)
    equity = broker.equity(marks)
    cap = order_cap(rules, equity, getattr(broker, "max_order_notional", None))
    # Orders still working at the venue (a buy queued overnight) already move
    # the position; a re-run must count them or it orders the same trade again.
    pending: dict[str, float] = {}
    committed = 0.0
    for order in broker.open_orders() if hasattr(broker, "open_orders") else []:
        ticker = normalize_symbol(order.ticker)
        pending[ticker] = pending.get(ticker, 0.0) + order.signed()
        if order.side == "buy":
            if ticker not in marks:
                with contextlib.suppress(Exception):
                    marks[ticker] = price(ticker, date)
            committed += order.quantity * marks.get(ticker, 0.0)
    for step in steps:
        if step.error or step.price is None:
            continue
        current = holding.get(step.ticker, 0.0) + pending.get(step.ticker, 0.0)
        step.order = plan_order(step.ticker, step.rating, equity, step.price, current, rules)
        if step.order is not None:
            step.order, step.note = within_cap(step.order, step.price, current, cap, rules)
        if pending.get(step.ticker):
            opened = f"{pending[step.ticker]:+g} already on order"
            step.note = f"{opened}; {step.note}" if step.note else opened

    cash = book.cash - committed
    sells = [s for s in steps if s.order is not None and s.order.side == "sell"]
    buys = [s for s in steps if s.order is not None and s.order.side == "buy"]
    _fund(buys, cash + sum(s.order.quantity * s.price for s in sells), rules)

    planned = sells + [s for s in buys if s.order is not None]
    if not submit or not planned or (confirm is not None and not confirm(steps)):
        return steps
    for step in sells:
        _send(broker, step, date)
    _fund(buys, cash + _freed(broker, sells, wait=any(s.order is not None for s in buys)), rules)
    for step in buys:
        if step.order is not None:
            _send(broker, step, date)
    return steps


def _graphs(kinds: set[str], selected_analysts, config: dict) -> dict[str, object]:
    """A graph per asset type; types that share an analyst list share one graph."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    built: dict[tuple[str, ...], object] = {}
    graphs = {}
    for kind in sorted(kinds):
        chosen = tuple(selected_analysts[kind] if isinstance(selected_analysts, Mapping) else selected_analysts)
        if chosen not in built:
            built[chosen] = TradingAgentsGraph(list(chosen), config=config)
        graphs[kind] = built[chosen]
    return graphs


def _send(broker: Broker, step: SessionStep, date: str) -> None:
    try:
        step.fill = broker.submit(step.order, step.price, date)
    except Exception as exc:  # a refused order is reported, and the rest still go
        logger.warning("Order for %s failed: %s", step.ticker, exc)
        step.error = str(exc)


def _freed(broker: Broker, sells: list[SessionStep], wait: bool) -> float:
    """The cash the session's sells freed for its buys."""
    sent = [s for s in sells if s.fill is not None and s.fill.status != "rejected"]
    wait_for_fills = getattr(broker, "wait_for_fills", None)
    if wait_for_fills is None:
        return sum(s.fill.quantity * s.price for s in sent)
    if wait and sent:
        try:
            for step, fill in zip(sent, wait_for_fills([s.fill for s in sent]), strict=True):
                step.fill = fill
        except Exception as exc:  # unknown fills free nothing; the buys are cut instead
            logger.warning("Could not confirm the sells' fills: %s", exc)
            return 0.0
    # An accepted order has freed nothing yet; only what filled is cash.
    return sum(s.fill.quantity * s.price for s in sent if s.fill.status in ("filled", "partially_filled"))


def _units(amount: float, price: float, rules: SizingRules) -> float:
    units = max(amount, 0.0) / price
    return units if rules.allow_fractional else math.floor(units + 1e-9)


def _fund(buys: list[SessionStep], cash: float, rules: SizingRules) -> None:
    """Cut buys, in order, to ``cash``: a session never borrows what the plan did not show."""
    for step in buys:
        if step.order is None or step.error:
            continue
        cost = step.order.quantity * step.price
        if cost <= cash + 1e-6:
            cash -= cost
            continue
        units = _units(cash, step.price, rules)
        reason = f"cut from {step.order.quantity:g} to the {max(cash, 0.0):,.2f} cash available"
        step.note = f"{step.note}; {reason}" if step.note else reason
        step.order = Order(step.ticker, "buy", units) if units > 0 else None
        cash -= units * step.price
