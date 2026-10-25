"""Turning a rating into an order: how much of the book the rating asks for.

The decision agents say which way to lean, not how many units to hold. This is
the one place that turns the 5-tier rating into a target position, so the
simulator, the paper broker and a real broker all trade the same rule.

Each rating maps to a fraction of the per-position cap (``max_position_pct`` of
equity). Hold maps to ``None``: keep whatever is held, rather than rebalancing to
some invented weight. REVIEW is no decision at all and never trades (#1170).
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

    @classmethod
    def from_config(cls, config: dict) -> SizingRules:
        execution = config.get("execution", {})
        return cls(
            max_position_pct=float(execution.get("max_position_pct", 0.10)),
            target_weights=execution.get("target_weights"),
            allow_fractional=bool(execution.get("allow_fractional", False)),
            min_trade_pct=float(execution.get("min_trade_pct", 0.005)),
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
    if target is None or target == current:
        return None
    delta = target - current
    # A full exit always goes through; any other trade below the floor is noise.
    if target != 0 and abs(delta) * price < rules.min_trade_pct * equity:
        return None
    return Order(ticker=ticker, side="buy" if delta > 0 else "sell", quantity=abs(delta))
