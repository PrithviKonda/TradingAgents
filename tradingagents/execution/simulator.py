"""Replay dated decisions through a paper account over historical closes.

``tradingagents.backtest`` scores each decision on its own; this answers the
other question: what would an account that traded those decisions have done.
It takes (date, ticker, rating) triples, normally the memory log a backtest
sweep wrote, so a sweep's LLM cost is paid once and its decisions can be
replayed under any sizing and cost assumptions for free.

Timing follows settlement (#1518): a decision dated D read news through D's
close, so it executes at the first close after D. A position is then carried,
marked to market every session, until a later decision on the same ticker
changes it. Hold leaves a position as it is; REVIEW never trades.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from tradingagents.agents.rating import RATING_REVIEW
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.market import get_closes
from tradingagents.execution.broker import Fill
from tradingagents.execution.paper import CostModel, PaperBroker
from tradingagents.execution.sizing import SizingRules, plan_order
from tradingagents.memory import TradingMemoryLog

TRADING_DAYS = 252

# closes(ticker, start, end) -> daily closes from start up to, not including, end.
CloseSource = Callable[[str, str, str], pd.Series]


@dataclass(frozen=True)
class Decision:
    date: str
    ticker: str
    rating: str


def decisions_from_log(path: str | Path) -> list[Decision]:
    """Every rated decision in a memory log, oldest first; REVIEW entries never trade."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no memory log at {path}")
    entries = TradingMemoryLog({"memory_log_path": str(path)}).load_entries()
    found = [Decision(e["date"], e["ticker"], e["rating"]) for e in entries
             if e["rating"] != RATING_REVIEW]
    return sorted(found, key=lambda d: (d.date, d.ticker))


def _daily(closes: pd.Series) -> pd.Series:
    """Closes keyed by calendar day, missing and non-positive closes dropped."""
    closes = closes.dropna()
    closes = closes[closes > 0]
    index = pd.DatetimeIndex(closes.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    closes.index = index.normalize()
    return closes[~closes.index.duplicated(keep="last")].sort_index()


@dataclass
class SimulationResult:
    equity: pd.Series
    fills: list[Fill]
    metrics: dict
    benchmark: pd.Series | None = None
    unpriced: list[Decision] = field(default_factory=list)

    def render(self) -> str:
        m = self.metrics
        lines = [
            f"Period: {m['start']} to {m['end']} ({m['sessions']} sessions)",
            f"Equity: {m['starting_equity']:,.2f} -> {m['ending_equity']:,.2f} "
            f"({m['total_return']:+.2%}, CAGR {m['cagr']:+.2%})",
            f"Volatility {m['volatility']:.2%} · Sharpe {m['sharpe']:.2f} · "
            f"max drawdown {m['max_drawdown']:.2%}",
            f"Trades: {m['trades']} · commissions {m['commissions']:,.2f} · "
            f"average exposure {m['average_exposure']:.0%}",
        ]
        if m.get("benchmark"):
            lines.append(f"Benchmark {m['benchmark']} buy-and-hold: {m['benchmark_return']:+.2%} "
                         f"(excess {m['total_return'] - m['benchmark_return']:+.2%})")
        if self.unpriced:
            lines.append(f"{len(self.unpriced)} decision(s) had no close after their date and did not trade.")
        lines.append("Fills are at the first close after each decision date, with the configured "
                     "slippage and commission; one model sampling per decision, so indicative only.")
        return "\n".join(lines)

    def write(self, directory: str | Path) -> Path:
        """Save the equity curve, fills and metrics beside each other."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({"equity": self.equity})
        if self.benchmark is not None:
            frame["benchmark"] = self.benchmark
        frame.index.name = "date"
        frame.to_csv(directory / "equity.csv")
        pd.DataFrame([asdict(f) for f in self.fills]).to_csv(directory / "fills.csv", index=False)
        (directory / "metrics.json").write_text(json.dumps(self.metrics, indent=2), encoding="utf-8")
        return directory


def simulate(
    decisions: list[Decision],
    config: dict,
    starting_cash: float | None = None,
    end_date: str | None = None,
    benchmark: str | None = "SPY",
    closes: CloseSource | None = None,
) -> SimulationResult:
    """Trade ``decisions`` through a fresh paper account and mark it daily to ``end_date``."""
    closes = closes or get_closes
    if not decisions:
        raise ValueError("there are no rated decisions to simulate")
    # Keyed as the paper account keys its holdings, so "nvda" in a log is the
    # NVDA position it bought and is marked to market, not valued at cost.
    decisions = [replace(d, ticker=normalize_symbol(d.ticker)) for d in decisions]
    execution = config.get("execution", {})
    rules = SizingRules.from_config(config)
    broker = PaperBroker(
        cash=float(starting_cash if starting_cash is not None else execution.get("starting_cash", 100_000)),
        costs=CostModel.from_config(config),
        allow_short=bool(execution.get("allow_short", False)),
        allow_fractional=rules.allow_fractional,
    )

    start = min(d.date for d in decisions)
    end = end_date or get_current_date()
    # get_closes excludes its end date, so ask for the day after.
    fetch_end = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    tickers = sorted({d.ticker for d in decisions})
    prices = {t: _daily(closes(t, start, fetch_end)) for t in tickers}

    # Each decision trades at the first session after its date in its own market.
    # Two decisions landing on one session (a weekend grid) collapse to the later one.
    by_session: dict[pd.Timestamp, dict[str, Decision]] = {}
    unpriced = []
    for decision in decisions:
        series = prices[decision.ticker]
        after = series.index[series.index > pd.Timestamp(decision.date)]
        if after.empty:
            unpriced.append(decision)
            continue
        by_session.setdefault(after[0], {})[decision.ticker] = decision

    calendar = sorted(set().union(*(s.index for s in prices.values())))
    calendar = [day for day in calendar if day > pd.Timestamp(start)]
    if not calendar:
        raise ValueError(f"no closes after {start} for {', '.join(tickers)}")

    curve, exposure = {}, []
    last: dict[str, float] = {}
    for day in calendar:
        for ticker, series in prices.items():
            if day in series.index:
                last[ticker] = float(series.loc[day])
        todays = by_session.get(day, {})
        if todays:
            equity = broker.equity(last)
            orders = [plan_order(t, d.rating, equity, last[t], broker.quantity(t), rules)
                      for t, d in todays.items()]
            # Sells first, so the cash they free can fund the same session's buys.
            for order in sorted(filter(None, orders), key=lambda o: o.side != "sell"):
                broker.submit(order, last[order.ticker], day.strftime("%Y-%m-%d"))
        value = broker.equity(last)
        curve[day] = value
        invested = value - broker.cash
        exposure.append(invested / value if value > 0 else 0.0)

    equity_curve = pd.Series(curve, name="equity")
    bench_curve = None
    if benchmark:
        bench = _daily(closes(benchmark, start, fetch_end))
        bench = bench[bench.index > pd.Timestamp(start)].reindex(equity_curve.index).ffill().dropna()
        if not bench.empty:
            bench_curve = bench / bench.iloc[0] * broker.starting_cash

    metrics = _metrics(equity_curve, broker, exposure, benchmark, bench_curve)
    return SimulationResult(equity_curve, broker.fills, metrics, bench_curve, unpriced)


def _metrics(curve: pd.Series, broker: PaperBroker, exposure: list[float],
             benchmark: str | None, bench_curve: pd.Series | None) -> dict:
    start_value = broker.starting_cash
    end_value = float(curve.iloc[-1])
    returns = curve.pct_change().dropna()
    # Day zero's return is against the cash the account started with.
    first = pd.Series([float(curve.iloc[0]) / start_value - 1], index=curve.index[:1])
    returns = pd.concat([first, returns])
    days = max((curve.index[-1] - curve.index[0]).days, 1)
    total = end_value / start_value - 1
    std = float(returns.std(ddof=0)) if len(returns) > 1 else 0.0
    peak = curve.cummax().clip(lower=start_value)
    filled = [f for f in broker.fills if f.status == "filled"]
    metrics = {
        "start": curve.index[0].strftime("%Y-%m-%d"),
        "end": curve.index[-1].strftime("%Y-%m-%d"),
        "sessions": len(curve),
        "starting_equity": start_value,
        "ending_equity": round(end_value, 2),
        "total_return": total,
        "cagr": (1 + total) ** (365.25 / days) - 1 if total > -1 else -1.0,
        "volatility": std * math.sqrt(TRADING_DAYS),
        "sharpe": float(returns.mean()) / std * math.sqrt(TRADING_DAYS) if std > 0 else 0.0,
        "max_drawdown": float((curve / peak - 1).min()),
        "trades": len(filled),
        "commissions": round(sum(f.commission for f in filled), 2),
        "average_exposure": sum(exposure) / len(exposure) if exposure else 0.0,
    }
    if benchmark and bench_curve is not None:
        metrics["benchmark"] = benchmark
        metrics["benchmark_return"] = float(bench_curve.iloc[-1] / start_value - 1)
    return metrics
