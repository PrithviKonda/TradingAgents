"""Alpaca, as a ``Broker``: its paper account by default, a live one only when unlocked.

Paper and live share one REST API, so a session proven on Alpaca paper runs
unchanged against live money; only the base URL and the keys differ. Keys come
from ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY``, the names Alpaca's own
tools read.

Live trading needs two separate yeses: ``paper=False`` from the caller, and
``TRADINGAGENTS_LIVE_TRADING=1`` in the environment. Neither alone is enough, so
no flag typed by habit, and no config file copied between machines, can place a
real order. A session counts the orders still open at Alpaca as part of the
position it plans from, so re-running it (the next morning, after an order was
queued overnight for the open) does not send the same trade twice. Each order's
client order id carries the exchange date, ticker and side, plus a suffix that
makes every attempt unique, so a retry after a rejected order is not refused.

Tickers stay in the Yahoo form the rest of the package uses; only the requests
to Alpaca carry its own symbols. A class share is ``BRK-B`` here and ``BRK.B``
there, and a crypto pair is ``BTC-USD`` here and ``BTC/USD`` there (Alpaca
reports a crypto position as ``BTCUSD``). A crypto order is good till
cancelled, as Alpaca requires; an equity order lasts the day.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from zoneinfo import ZoneInfo

import requests

from tradingagents.dataflows.symbols import crypto_base
from tradingagents.execution.broker import BrokerError, Fill, Order
from tradingagents.portfolio import PortfolioContext, Position

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
LIVE_UNLOCK_ENV = "TRADINGAGENTS_LIVE_TRADING"


def live_trading_unlocked() -> bool:
    return os.getenv(LIVE_UNLOCK_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def format_quantity(quantity: float) -> str:
    """``quantity`` in fixed notation to Alpaca's 9 decimals, rounded down.

    Rounding down means a full exit never asks to sell more than is held.
    """
    exact = Decimal(repr(quantity)).quantize(Decimal("1e-9"), rounding=ROUND_DOWN)
    return f"{exact.normalize():f}"


def to_alpaca_symbol(ticker: str) -> str:
    """``ticker`` as Alpaca spells it: ``BRK-B`` -> ``BRK.B``, ``BTC-USD`` -> ``BTC/USD``."""
    base = crypto_base(ticker)
    return f"{base}/USD" if base else ticker.strip().upper().replace("-", ".")


def from_alpaca_symbol(symbol: str) -> str:
    """An Alpaca symbol in the package's Yahoo form; the inverse of ``to_alpaca_symbol``."""
    base = crypto_base(symbol.replace("/", ""))
    return f"{base}-USD" if base else symbol.strip().upper().replace(".", "-")


def client_order_id(order: Order) -> str:
    """``tradingagents-<US exchange date>-<TICKER>-<side>-<unique suffix>``."""
    day = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    return f"tradingagents-{day}-{order.ticker.upper()}-{order.side}-{uuid.uuid4().hex[:8]}"


class AlpacaBroker:
    def __init__(self, paper: bool = True, key_id: str | None = None, secret_key: str | None = None,
                 max_order_notional: float | None = None, max_order_pct: float | None = None,
                 timeout: float = 15.0,
                 session: requests.Session | None = None):
        if not paper and not live_trading_unlocked():
            raise BrokerError(
                f"live trading is locked: set {LIVE_UNLOCK_ENV}=1 in the environment to "
                "trade a live Alpaca account, or trade its paper account instead"
            )
        key_id = key_id or os.getenv("APCA_API_KEY_ID")
        secret_key = secret_key or os.getenv("APCA_API_SECRET_KEY")
        if not key_id or not secret_key:
            raise BrokerError("Alpaca keys are missing: set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
        self.paper = paper
        self.name = "alpaca-paper" if paper else "alpaca-live"
        self.base_url = PAPER_URL if paper else LIVE_URL
        self.max_order_notional = max_order_notional
        self.max_order_pct = max_order_pct
        self._equity: float | None = None      # as last reported, for max_order_pct
        self.timeout = timeout
        self._session = session or requests.Session()
        self._held: dict[str, float] = {}      # positions as last reported, for the cap
        self._session.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key})

    def _request(self, method: str, path: str, **kwargs):
        try:
            response = self._session.request(method, self.base_url + path, timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            # The keys travel in headers, which requests never quotes in its errors.
            raise BrokerError(f"Alpaca {method} {path} failed: {exc}") from None
        if response.status_code >= 400:
            try:
                detail = response.json().get("message", response.text)
            except ValueError:
                detail = response.text
            raise BrokerError(f"Alpaca {method} {path} returned {response.status_code}: {detail}")
        return response.json()

    def account(self) -> dict:
        account = self._request("GET", "/v2/account")
        if account.get("trading_blocked") or account.get("account_blocked"):
            raise BrokerError("the Alpaca account is blocked from trading")
        return account

    def portfolio(self) -> PortfolioContext:
        account = self.account()
        positions = self._request("GET", "/v2/positions")
        self._held = {from_alpaca_symbol(p["symbol"]): float(p["qty"]) for p in positions}
        self._equity = float(account.get("equity") or account["cash"])
        return PortfolioContext(
            cash=float(account["cash"]),
            currency=account.get("currency", "USD"),
            positions=[Position(ticker=from_alpaca_symbol(p["symbol"]), quantity=float(p["qty"]),
                                average_price=float(p["avg_entry_price"])) for p in positions],
        )

    def open_orders(self) -> list[Order]:
        """The unfilled part of every order still working at Alpaca."""
        working = []
        for placed in self._request("GET", "/v2/orders", params={"status": "open", "limit": 500}):
            # A notional (dollar-amount) order has no quantity to net; this
            # package never places one.
            if placed.get("qty") is None:
                continue
            remaining = float(placed["qty"]) - float(placed.get("filled_qty") or 0)
            if remaining > 0:
                working.append(Order(from_alpaca_symbol(placed["symbol"]), placed["side"], remaining))
        return working

    def equity(self, prices: dict[str, float]) -> float:
        # Alpaca marks the book itself, with fresher prices than the caller has.
        return float(self.account()["equity"])

    def _order_limit(self) -> float | None:
        limits = [self.max_order_notional] if self.max_order_notional is not None else []
        if self.max_order_pct is not None and self._equity is not None:
            limits.append(self.max_order_pct * max(self._equity, 0.0))
        return min(limits) if limits else None

    def submit(self, order: Order, price: float, date: str) -> Fill:
        # The session already cut the plan to the cap; this is the backstop. A
        # sell that only reduces a reported position is exempt, so it can exit.
        held = max(self._held.get(order.ticker.upper(), 0.0), 0.0)
        capped = order.quantity - min(order.quantity, held) if order.side == "sell" else order.quantity
        notional = order.quantity * price
        limit = self._order_limit()
        if limit is not None and capped * price > limit + 1e-6:
            raise BrokerError(
                f"{order.side} {order.quantity:g} {order.ticker} is about {notional:,.2f}, over the "
                f"per-order cap of {limit:,.2f}; not sent"
            )
        placed = self._request("POST", "/v2/orders", json={
            "symbol": to_alpaca_symbol(order.ticker),
            "qty": format_quantity(order.quantity),
            "side": order.side,
            "type": "market",
            # Alpaca takes crypto orders only as gtc or ioc.
            "time_in_force": "gtc" if crypto_base(order.ticker) else "day",
            "client_order_id": client_order_id(order),
        })
        filled_price = placed.get("filled_avg_price")
        return Fill(
            ticker=order.ticker.upper(),
            side=order.side,
            quantity=float(placed.get("filled_qty") or 0) or order.quantity,
            price=float(filled_price) if filled_price else None,
            date=date,
            status=placed.get("status", "accepted"),
            order_id=placed.get("id"),
        )
