"""One trading session: analyze tickers, size the ratings, and place the orders.

The broker decides where the orders go (a paper ledger on disk, Alpaca paper,
Alpaca live); everything else is the same. A session runs in three phases so a
human can look at the whole plan before anything is sent:

1. Analyze every ticker, each run seeing the broker's book as its portfolio
   context, so the agents know what is held.
2. Plan an order per decision from one equity figure, priced at the latest close.
3. Ask ``confirm(plan)`` once, then send sells before buys, so the cash a sell
   frees can fund a buy in the same session.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.vendors.yahoo.market import get_closes
from tradingagents.execution.broker import Broker, Fill, Order
from tradingagents.execution.sizing import SizingRules, plan_order

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

    def describe(self) -> str:
        if self.error:
            return f"{self.ticker}: failed - {self.error}"
        head = f"{self.ticker}: {self.rating}"
        if self.order is None:
            return f"{head}, no trade"
        plan = f"{head}, {self.order.side} {self.order.quantity:g} @ ~{self.price:,.2f}"
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
    asset_type: str = "stock",
    selected_analysts=("market", "social", "news", "fundamentals"),
    price: Callable[[str, str], float] = latest_close,
    confirm: Callable[[list[SessionStep]], bool] | None = None,
    submit: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
    graph=None,
) -> list[SessionStep]:
    """Run the session; with ``submit=False`` it stops after planning (a dry run)."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    date = trade_date or get_current_date()
    graph = graph or TradingAgentsGraph(list(selected_analysts), config=config)
    rules = SizingRules.from_config(config)
    book = broker.portfolio()
    steps = [SessionStep(t) for t in dict.fromkeys(t.strip().upper() for t in tickers if t.strip())]

    for index, step in enumerate(steps, 1):
        if progress:
            progress(index, len(steps), step.ticker)
        try:
            _, step.rating = graph.propagate(step.ticker, date, asset_type, portfolio=book)
            step.price = price(step.ticker, date)
        except Exception as exc:  # one ticker's outage must not cancel the others' trades
            logger.warning("Session step %s failed: %s", step.ticker, exc)
            step.error = str(exc)

    marks = {s.ticker: s.price for s in steps if s.price}
    for held in book.positions:
        if held.ticker.upper() not in marks:
            # An unpriced holding is marked at cost by the paper broker.
            with contextlib.suppress(Exception):
                marks[held.ticker.upper()] = price(held.ticker, date)
    equity = broker.equity(marks)
    for step in steps:
        if step.error or step.price is None:
            continue
        held = book.position_in(step.ticker)
        step.order = plan_order(step.ticker, step.rating, equity, step.price,
                                held.quantity if held else 0.0, rules)

    planned = [s for s in steps if s.order is not None]
    if not submit or not planned or (confirm is not None and not confirm(steps)):
        return steps
    for step in sorted(planned, key=lambda s: s.order.side != "sell"):
        try:
            step.fill = broker.submit(step.order, step.price, date)
        except Exception as exc:  # a refused order is reported, and the rest still go
            logger.warning("Order for %s failed: %s", step.ticker, exc)
            step.error = str(exc)
    return steps
