"""What the dashboard reads: every artifact a run, sweep or account leaves behind.

Everything is found under ``results_dir``:

- ``paper/<account>.json``: paper accounts (``tradingagents paper``)
- any directory holding ``metrics.json`` and ``equity.csv``: saved simulations
- ``backtest/<run>/trading_memory.md``: a sweep's decisions, plus the live memory log
- ``reports/<TICKER>_<stamp>/``: the analysts' reports of a single run

Prices come through the data layer like every other read, so a chart shows the
same bars an analysis saw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.vendors.yahoo.market import get_closes
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv
from tradingagents.execution.paper import PaperBroker
from tradingagents.memory import TradingMemoryLog

_FILL_COLUMNS = ["ticker", "side", "quantity", "price", "commission", "date", "status", "order_id"]


@dataclass(frozen=True)
class Account:
    """An equity curve and the fills that made it, from a simulation or a paper account."""

    name: str
    equity: pd.Series
    fills: pd.DataFrame
    benchmark: pd.Series | None = None
    starting_cash: float | None = None


@dataclass(frozen=True)
class Sources:
    results_dir: Path
    memory_log_path: Path | None = None

    @classmethod
    def from_config(cls, config: dict, results_dir: str | Path | None = None) -> Sources:
        log = config.get("memory_log_path")
        return cls(Path(results_dir or config["results_dir"]), Path(log) if log else None)

    def paper_accounts(self) -> dict[str, Path]:
        folder = self.results_dir / "paper"
        return {p.stem: p for p in sorted(folder.glob("*.json"))} if folder.is_dir() else {}

    def simulations(self) -> dict[str, Path]:
        if not self.results_dir.is_dir():
            return {}
        found = {}
        for metrics in sorted(self.results_dir.rglob("metrics.json")):
            folder = metrics.parent
            if (folder / "equity.csv").is_file():
                found[str(folder.relative_to(self.results_dir))] = folder
        return found

    def decision_logs(self) -> dict[str, Path]:
        logs = {}
        if self.memory_log_path and self.memory_log_path.is_file():
            logs["live memory log"] = self.memory_log_path
        folder = self.results_dir / "backtest"
        if folder.is_dir():
            for log in sorted(folder.glob("*/trading_memory.md")):
                logs[f"backtest {log.parent.name}"] = log
        return logs

    def reports(self) -> dict[str, Path]:
        folder = self.results_dir / "reports"
        if not folder.is_dir():
            return {}
        return {p.name: p for p in sorted(folder.iterdir(), reverse=True) if p.is_dir()}


# -- loading ---------------------------------------------------------------------------

def _fills_frame(rows: list[dict]) -> pd.DataFrame:
    fills = pd.DataFrame(rows, columns=_FILL_COLUMNS)
    if fills.empty:
        return fills
    fills["date"] = pd.to_datetime(fills["date"])
    fills = fills[fills["status"].fillna("filled") == "filled"]
    return fills.sort_values("date", kind="stable").reset_index(drop=True)


def load_simulation(folder: str | Path) -> Account:
    folder = Path(folder)
    frame = pd.read_csv(folder / "equity.csv", index_col="date", parse_dates=["date"])
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    rows = _read_fills(folder / "fills.csv")
    benchmark = frame["benchmark"].dropna() if "benchmark" in frame else None
    return Account(name=folder.name, equity=frame["equity"], fills=_fills_frame(rows),
                   benchmark=benchmark, starting_cash=metrics.get("starting_equity"))


def _read_fills(path: Path) -> list[dict]:
    """A saved fills.csv as rows; a run that never traded may have left one with no header."""
    try:
        return pd.read_csv(path).to_dict("records") if path.is_file() else []
    except pd.errors.EmptyDataError:
        return []


def load_paper(path: str | Path, closes=None, end_date: str | None = None) -> Account:
    """A paper account, with an equity curve rebuilt by marking its fills to market daily."""
    broker = PaperBroker.open(path, starting_cash=0.0)
    fills = _split_adjusted(_fills_frame([{k: getattr(f, k) for k in _FILL_COLUMNS} for f in broker.fills]),
                            broker.adjustments)
    equity = replay_equity(fills, broker.starting_cash, closes or daily_closes, end_date)
    return Account(name=Path(path).stem, equity=equity, fills=fills, starting_cash=broker.starting_cash)


def _split_adjusted(fills: pd.DataFrame, adjustments: list[dict]) -> pd.DataFrame:
    """Fills before each recorded split restated in post-split units, as the closes are."""
    if fills.empty or not adjustments:
        return fills
    fills = fills.copy()
    for a in adjustments:
        before = (fills["ticker"] == a["ticker"]) & (fills["date"] < pd.Timestamp(a["date"]))
        fills.loc[before, "quantity"] *= a["ratio"]
        fills.loc[before, "price"] /= a["ratio"]
    return fills


def replay_equity(fills: pd.DataFrame, starting_cash: float, closes, end_date: str | None = None) -> pd.Series:
    """Cash plus positions at each session's close, from the first fill to ``end_date``."""
    if fills.empty:
        return pd.Series(dtype=float, name="equity")
    start = fills["date"].min().strftime("%Y-%m-%d")
    end = end_date or get_current_date()
    prices = {t: closes(t, start, end) for t in fills["ticker"].unique()}
    calendar = sorted(set().union(*(s.index for s in prices.values())) | set(fills["date"]))
    calendar = [d for d in calendar if d <= pd.Timestamp(end)]

    cash, held, curve, cursor = starting_cash, {}, {}, 0
    rows = fills.to_dict("records")
    for day in calendar:
        while cursor < len(rows) and rows[cursor]["date"] <= day:
            f = rows[cursor]
            signed = f["quantity"] if f["side"] == "buy" else -f["quantity"]
            cash -= signed * f["price"] + (f["commission"] or 0.0)
            held[f["ticker"]] = held.get(f["ticker"], 0.0) + signed
            cursor += 1
        value = cash
        for ticker, quantity in held.items():
            series = prices[ticker]
            known = series[series.index <= day]
            mark = float(known.iloc[-1]) if not known.empty else _last_fill_price(rows[:cursor], ticker)
            value += quantity * mark
        curve[day] = value
    return pd.Series(curve, name="equity")


def _last_fill_price(rows: list[dict], ticker: str) -> float:
    return next((r["price"] for r in reversed(rows) if r["ticker"] == ticker), 0.0)


def load_decisions(path: str | Path) -> pd.DataFrame:
    """A memory log as a table, one row per decision, oldest first."""
    entries = TradingMemoryLog({"memory_log_path": str(path)}).load_entries()
    columns = ["date", "ticker", "rating", "pending", "raw", "alpha", "holding", "resolved",
               "decision", "reflection"]
    frame = pd.DataFrame(entries, columns=columns)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("raw", "alpha"):
        frame[column] = pd.to_numeric(frame[column].astype(str).str.rstrip("%"), errors="coerce") / 100
    return frame.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def load_report(folder: str | Path) -> list[tuple[str, str]]:
    """A report tree's markdown sections as (section/file, text), the complete report first."""
    folder = Path(folder)
    parts = []
    complete = folder / "complete_report.md"
    if complete.is_file():
        parts.append(("complete_report.md", complete.read_text(encoding="utf-8")))
    for md in sorted(folder.rglob("*.md")):
        if md != complete:
            parts.append((str(md.relative_to(folder)), md.read_text(encoding="utf-8")))
    return parts


# -- prices ----------------------------------------------------------------------------

def daily_closes(ticker: str, start: str, end: str) -> pd.Series:
    """Daily closes from ``start`` through ``end`` inclusive, keyed by calendar day."""
    stop = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    closes = get_closes(ticker, start, stop).dropna()
    closes = closes[closes > 0]
    index = pd.DatetimeIndex(closes.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    closes.index = index.normalize()
    return closes[~closes.index.duplicated(keep="last")].sort_index()


def ohlcv(ticker: str, end: str | None = None) -> pd.DataFrame:
    """Daily OHLCV bars through ``end`` (about five years), indexed by date, as reported."""
    data = load_ohlcv(ticker, end or get_current_date(), fill_gaps=False)
    return data.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].sort_index()
