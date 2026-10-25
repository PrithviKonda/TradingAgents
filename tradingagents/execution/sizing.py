"""Turning a rating into an order: how much of the book the rating asks for.

The decision agents say which way to lean, not how many units to hold. This is
the one place that turns the 5-tier rating into a target position, so the
simulator, the paper broker and a real broker all trade the same rule.

Each rating maps to a fraction of the per-position cap (``max_position_pct`` of
equity). Hold maps to ``None``: keep whatever is held, rather than rebalancing to
some invented weight. REVIEW is no decision at all and never trades (#1170).

No single order may exceed the per-order cap: ``max_order_pct`` of equity, and
a broker's ``max_order_notional`` in dollars where one is set, whichever is
lower. A larger buy is cut to it, so a position is built over several sessions;
a sell that only reduces a position is exempt, so it can always exit.

The book is long-only unless ``allow_short`` is set: a negative target weight
(the only way a rating asks for a short) is refused when the rules are built,
and no plan ever sells more than is held.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tradingagents.agents.rating import RATINGS_5_TIER, is_review
from tradingagents.execution.broker import Order

# Fraction of the per-position cap each rating targets. Hold is absent on
# purpose: it keeps the current position.
DEFAULT_TARGET_WEIGHTS: dict[str, float] = {
    "Buy": 1.0,
    "Overweight": 0.75,
    "Underweight": 0.25,
    "Sell": 0.0,
}


@dataclass(frozen=True)
class SizingRules:
    max_position_pct: float = 0.10
    target_weights: dict[str, float] | None = None
    allow_fractional: bool = False
    # A trade smaller than this share of equity is noise, not a rebalance.
    min_trade_pct: float = 0.005
    # The largest single order, as a share of equity; None for no such cap.
    max_order_pct: float | None = None
    allow_short: bool = False

    def __post_init__(self):
        if self.allow_short:
            return
        short = sorted(r for r, w in (self.target_weights or {}).items() if float(w) < 0)
        if short:
            raise ValueError(
                f"target_weights for {', '.join(short)} are negative, which asks for a short; "
                "set execution.allow_short to trade short, or use weights of 0 or more"
            )

    @classmethod
    def from_config(cls, config: dict) -> SizingRules:
        execution = config.get("execution", {})
        return cls(
            max_position_pct=float(execution.get("max_position_pct", 0.10)),
            target_weights=execution.get("target_weights"),
            allow_fractional=bool(execution.get("allow_fractional", False)),
            min_trade_pct=float(execution.get("min_trade_pct", 0.005)),
            max_order_pct=(float(execution["max_order_pct"])
                           if execution.get("max_order_pct") is not None else None),
            allow_short=bool(execution.get("allow_short", False)),
        )

    def weights(self) -> dict[str, float]:
        return {**DEFAULT_TARGET_WEIGHTS, **(self.target_weights or {})}


def target_quantity(rating: str, equity: float, price: float, current: float,
                    rules: SizingRules) -> float | None:
    """Units the rating asks to hold, or ``None`` to leave the position alone."""
    if is_review(rating) or rating == "Hold":
        return None
    if rating not in RATINGS_5_TIER:
        raise ValueError(f"unknown rating {rating!r}")
    if price <= 0 or equity <= 0:
        return None
    weight = rules.weights()[rating]
    units = equity * rules.max_position_pct * weight / price
    if not rules.allow_fractional:
        units = math.floor(units)
    # Bearish ratings never grow a position: an Underweight on a book already
    # below a quarter of the cap is satisfied, not a signal to buy up to it.
    if rating in ("Underweight", "Sell"):
        units = min(units, max(current, 0.0))
    return units


def plan_order(ticker: str, rating: str, equity: float, price: float, current: float,
               rules: SizingRules) -> Order | None:
    """The order that moves ``current`` units to the rating's target, if any."""
    target = target_quantity(rating, equity, price, current, rules)
    if target is None:
        return None
    if not rules.allow_short:
        # A long-only book sells at most what it holds, and never sells into a short.
        target = max(target, min(current, 0.0))
    if target == current:
        return None
    delta = target - current
    # A full exit always goes through; any other trade below the floor is noise.
    if target != 0 and abs(delta) * price < rules.min_trade_pct * equity:
        return None
    return Order(ticker=ticker, side="buy" if delta > 0 else "sell", quantity=abs(delta))


def order_cap(rules: SizingRules, equity: float,
              max_order_notional: float | None = None) -> tuple[float, str] | None:
    """The per-order cap in notional and the setting it comes from, or ``None``."""
    caps = []
    if rules.max_order_pct is not None:
        caps.append((rules.max_order_pct * max(equity, 0.0), f"max_order_pct cap of {rules.max_order_pct:.0%}"))
    if max_order_notional is not None:
        caps.append((float(max_order_notional), "max_order_notional cap"))
    return min(caps, key=lambda c: c[0]) if caps else None


def within_cap(order: Order, price: float, current: float, cap: tuple[float, str] | None,
               rules: SizingRules) -> tuple[Order | None, str | None]:
    """``order`` cut to ``cap``, with a note saying so; a sell that only reduces is exempt."""
    if cap is None:
        return order, None
    limit, name = cap
    # Only the part of a sell beyond the position (a short) is held to the cap.
    reducing = min(order.quantity, max(current, 0.0)) if order.side == "sell" else 0.0
    if (order.quantity - reducing) * price <= limit + 1e-6:
        return order, None
    units = max(limit, 0.0) / price
    units = reducing + (units if rules.allow_fractional else math.floor(units + 1e-9))
    if units <= 0:
        return None, f"one unit is over the {name} ({limit:,.2f})"
    return (Order(order.ticker, order.side, units),
            f"cut from {order.quantity:g} by the {name} ({limit:,.2f})")
