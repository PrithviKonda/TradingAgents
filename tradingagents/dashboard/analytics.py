"""The numbers behind the dashboard: performance, trades and signal quality.

Pure pandas, no UI, so every figure on screen has a test. Conventions:

- Returns are simple daily returns of the equity curve; annualization uses 252
  sessions and a zero risk-free rate. The headline figures come from
  ``tradingagents.execution.metrics``, as the simulator's do.
- A decision is entered at the first close after its date (#1518), as the
  simulator and settlement do, so forward returns here are the returns a
  trade on that decision could actually have had.
- Ratings are scored on one axis, Buy +2 to Sell -2, for rank statistics such
  as the information coefficient.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

from tradingagents.execution.metrics import TRADING_DAYS, curve_drawdown, curve_returns, curve_stats
from tradingagents.memory.settlement import benchmark_return

RATING_SCORE = {"Buy": 2, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -2}
RATING_ORDER = list(RATING_SCORE)
HORIZONS = (1, 5, 10, 20)


def direction(rating: str) -> int:
    return int(np.sign(RATING_SCORE.get(rating, 0)))


# -- performance -----------------------------------------------------------------------

def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.astype(float).pct_change().dropna()


def drawdown(equity: pd.Series, starting_cash: float | None = None) -> pd.Series:
    """Fractional distance below the running peak, 0 at a new high."""
    return curve_drawdown(equity, starting_cash)


def longest_drawdown(equity: pd.Series, starting_cash: float | None = None) -> int:
    """Most consecutive sessions spent below a previous peak."""
    longest = run = 0
    for below in (drawdown(equity, starting_cash) < 0):
        run = run + 1 if below else 0
        longest = max(longest, run)
    return longest


def performance(equity: pd.Series, benchmark: pd.Series | None = None,
                starting_cash: float | None = None) -> dict:
    """Headline statistics of an equity curve, and against ``benchmark`` when given.

    With ``starting_cash``, returns are measured from the cash the account
    started with, so the first session's costs count, and the headline figures
    match what ``tradingagents simulate`` reports for the same run.
    """
    equity = equity.dropna().astype(float)
    if len(equity) < 2:
        return {}
    r = curve_returns(equity, starting_cash)
    headline = curve_stats(equity, starting_cash)
    downside = r[r < 0]
    down_vol = math.sqrt((downside ** 2).sum() / len(r)) * math.sqrt(TRADING_DAYS) if len(r) else 0.0
    cagr, max_dd = headline["cagr"], headline["max_drawdown"]
    stats = {
        **{k: headline[k] for k in ("total_return", "cagr", "volatility", "sharpe")},
        "sortino": float(r.mean() * TRADING_DAYS / down_vol) if down_vol > 0 else 0.0,
        "max_drawdown": max_dd,
        "calmar": float(cagr / -max_dd) if max_dd < 0 else 0.0,
        "longest_drawdown_sessions": longest_drawdown(equity, starting_cash),
        "positive_sessions": float((r > 0).mean()) if len(r) else 0.0,
        "best_session": float(r.max()) if len(r) else 0.0,
        "worst_session": float(r.min()) if len(r) else 0.0,
        "skew": float(r.skew()) if len(r) > 2 else 0.0,
        "kurtosis": float(r.kurt()) if len(r) > 3 else 0.0,
        "sessions": len(equity),
    }
    if benchmark is not None and len(benchmark.dropna()) > 2:
        b = daily_returns(benchmark.reindex(equity.index).ffill().dropna())
        joined = pd.concat([r, b], axis=1, join="inner").dropna()
        if len(joined) > 2 and joined.iloc[:, 1].var() > 0:
            beta = joined.iloc[:, 0].cov(joined.iloc[:, 1]) / joined.iloc[:, 1].var()
            alpha = (joined.iloc[:, 0].mean() - beta * joined.iloc[:, 1].mean()) * TRADING_DAYS
            active = joined.iloc[:, 0] - joined.iloc[:, 1]
            stats.update({
                "beta": float(beta),
                "alpha": float(alpha),
                "correlation": float(joined.iloc[:, 0].corr(joined.iloc[:, 1])),
                "tracking_error": float(active.std(ddof=1) * math.sqrt(TRADING_DAYS)),
                "information_ratio": float(active.mean() / active.std(ddof=1) * math.sqrt(TRADING_DAYS))
                if active.std(ddof=1) > 0 else 0.0,
                "benchmark_return": float(benchmark.dropna().iloc[-1] / benchmark.dropna().iloc[0] - 1),
            })
    return stats


def rolling_sharpe(equity: pd.Series, window: int = 63) -> pd.Series:
    r = daily_returns(equity)
    return r.rolling(window).mean() / r.rolling(window).std(ddof=1) * math.sqrt(TRADING_DAYS)


def rolling_volatility(equity: pd.Series, window: int = 21) -> pd.Series:
    return daily_returns(equity).rolling(window).std(ddof=1) * math.sqrt(TRADING_DAYS)


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Calendar-month returns, years by months, with a full-year column."""
    month_end = equity.resample("ME").last().dropna()
    # The first month is measured from the curve's first value, not from nothing.
    base = pd.concat([pd.Series([equity.iloc[0]], index=[equity.index[0] - pd.Timedelta(days=1)]), month_end])
    monthly = base.pct_change().dropna()
    table = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "r": monthly.values})
    grid = table.pivot(index="year", columns="month", values="r").reindex(columns=range(1, 13))
    grid.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    grid["Year"] = table.groupby("year")["r"].apply(lambda s: float(np.prod(1 + s) - 1))
    return grid


def cash_series(fills: pd.DataFrame, starting_cash: float, index: pd.DatetimeIndex) -> pd.Series:
    """The account's cash at each session in ``index``, from its fills."""
    if fills.empty:
        return pd.Series(float(starting_cash), index=index, name="cash")
    signed = np.where(fills["side"] == "buy", fills["quantity"], -fills["quantity"])
    # Plain arrays: a Series handed a new index would be realigned to it, not relabelled.
    values = -(signed * fills["price"].to_numpy()) - fills["commission"].fillna(0.0).to_numpy()
    flows = pd.Series(values, index=pd.DatetimeIndex(fills["date"])).groupby(level=0).sum()
    cash = starting_cash + flows.cumsum()
    return cash.reindex(index.union(cash.index)).ffill().fillna(starting_cash).reindex(index).rename("cash")


def exposure(equity: pd.Series, cash: pd.Series) -> pd.Series:
    """Share of equity held in positions rather than cash."""
    return ((equity - cash) / equity).rename("exposure")


# -- trades ----------------------------------------------------------------------------

def _match(fills: pd.DataFrame) -> tuple[list[dict], dict[str, deque]]:
    """FIFO matching: the closed trades, and the lots still open per ticker."""
    trips, lots = [], {}
    for f in fills.sort_values("date", kind="stable").itertuples(index=False):
        if not f.quantity:
            continue
        per_unit_cost = (f.commission or 0.0) / f.quantity
        sign = 1 if f.side == "buy" else -1
        book = lots.setdefault(f.ticker, deque())
        remaining = float(f.quantity)
        # A fill against the open side closes its oldest lots first.
        while remaining > 1e-9 and book and book[0][0] * sign < 0:
            lot_sign, qty, price, date, cost = book[0]
            take = min(qty, remaining)
            pnl = lot_sign * take * (f.price - price) - take * (cost + per_unit_cost)
            trips.append({
                "ticker": f.ticker, "side": "long" if lot_sign > 0 else "short",
                "entry_date": date, "exit_date": f.date, "quantity": take,
                "entry_price": price, "exit_price": f.price, "pnl": pnl,
                "return": pnl / (take * price) if price else 0.0,
                "holding_days": (f.date - date).days,
            })
            remaining -= take
            if qty - take > 1e-9:
                book[0] = (lot_sign, qty - take, price, date, cost)
            else:
                book.popleft()
        if remaining > 1e-9:
            book.append((sign, remaining, float(f.price), f.date, per_unit_cost))
    return trips, lots


def round_trips(fills: pd.DataFrame) -> pd.DataFrame:
    """Fills matched first-in, first-out into closed trades, one row per matched lot.

    A sell closes the oldest open long units first; a sell with nothing long
    opens a short, closed in turn by later buys. Commission is spread over the
    units of the fill that paid it.
    """
    columns = ["ticker", "side", "entry_date", "exit_date", "quantity", "entry_price",
               "exit_price", "pnl", "return", "holding_days"]
    return pd.DataFrame(_match(fills)[0], columns=columns)


def open_lots(fills: pd.DataFrame) -> pd.DataFrame:
    """Units still open after FIFO matching, per ticker, with their average entry and first entry."""
    rows = []
    for ticker, book in _match(fills)[1].items():
        if not book:
            continue
        units = sum(lot[0] * lot[1] for lot in book)
        cost = sum(lot[1] * lot[2] for lot in book)
        rows.append({"ticker": ticker, "quantity": float(units),
                     "average_entry": cost / sum(lot[1] for lot in book), "opened": book[0][3]})
    return pd.DataFrame(rows, columns=["ticker", "quantity", "average_entry", "opened"])


def trade_stats(trips: pd.DataFrame) -> dict:
    if trips.empty:
        return {"trades": 0}
    wins, losses = trips[trips["pnl"] > 0], trips[trips["pnl"] <= 0]
    gross_loss = -losses["pnl"].sum()
    return {
        "trades": len(trips),
        "win_rate": len(wins) / len(trips),
        "average_win": float(wins["pnl"].mean()) if len(wins) else 0.0,
        "average_loss": float(losses["pnl"].mean()) if len(losses) else 0.0,
        "profit_factor": float(wins["pnl"].sum() / gross_loss) if gross_loss > 0 else math.inf,
        "expectancy": float(trips["pnl"].mean()),
        "average_return": float(trips["return"].mean()),
        "average_holding_days": float(trips["holding_days"].mean()),
        "total_pnl": float(trips["pnl"].sum()),
    }


def add_excursions(trips: pd.DataFrame, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Maximum adverse and favourable excursion of each trade, from daily highs and lows.

    MAE is the worst price move against the position while it was open, MFE the
    best, both as a fraction of the entry price; the entry and exit sessions count.
    """
    trips = trips.copy()
    mae, mfe = [], []
    for t in trips.itertuples(index=False):
        frame = bars.get(t.ticker)
        if frame is None or frame.empty:
            mae.append(np.nan)
            mfe.append(np.nan)
            continue
        window = frame[(frame.index >= t.entry_date) & (frame.index <= t.exit_date)]
        if window.empty:
            mae.append(np.nan)
            mfe.append(np.nan)
            continue
        low, high = window["Low"].min() / t.entry_price - 1, window["High"].max() / t.entry_price - 1
        if t.side == "long":
            mae.append(min(low, 0.0))
            mfe.append(max(high, 0.0))
        else:
            mae.append(min(-high, 0.0))
            mfe.append(max(-low, 0.0))
    trips["mae"], trips["mfe"] = mae, mfe
    return trips


# -- signal quality --------------------------------------------------------------------

def forward_returns(decisions: pd.DataFrame, closes: dict[str, pd.Series],
                    benchmark: pd.Series | None = None, horizons=HORIZONS) -> pd.DataFrame:
    """Each decision with its forward return over each horizon, raw and in excess of ``benchmark``.

    Entry is the first close after the decision date; ``fwd_h`` is the return from
    there to ``h`` sessions later. A horizon that has not fully traded is NaN.
    The benchmark leg is measured as settlement measures alpha, from its last
    close by the entry to its last close by the exit, so a weekend entry in a
    seven-day market is scored against the same dates the memory log scores it.
    """
    if benchmark is not None:
        benchmark = benchmark.dropna()
        benchmark = benchmark[benchmark > 0].sort_index()
    rows = []
    for d in decisions.itertuples(index=False):
        series = closes.get(d.ticker)
        row = {"date": d.date, "ticker": d.ticker, "rating": d.rating,
               "score": RATING_SCORE.get(d.rating, np.nan), "direction": direction(d.rating)}
        entry_index = None
        if series is not None and not series.empty:
            after = np.flatnonzero(series.index > d.date)
            entry_index = after[0] if after.size else None
        row["entry_date"] = series.index[entry_index] if entry_index is not None else pd.NaT
        for h in horizons:
            fwd = excess = np.nan
            if entry_index is not None and entry_index + h < len(series):
                start, stop = series.index[entry_index], series.index[entry_index + h]
                fwd = series.iloc[entry_index + h] / series.iloc[entry_index] - 1
                bench_ret = benchmark_return(benchmark, start, stop) if benchmark is not None else None
                if bench_ret is not None:
                    excess = fwd - bench_ret
            row[f"fwd_{h}"], row[f"excess_{h}"] = fwd, excess
        rows.append(row)
    return pd.DataFrame(rows)


def _target(fr: pd.DataFrame, horizon: int, excess: bool) -> pd.Series:
    column = f"excess_{horizon}" if excess else f"fwd_{horizon}"
    return fr[column] if column in fr else pd.Series(np.nan, index=fr.index)


def by_rating(fr: pd.DataFrame, horizon: int = 5, excess: bool = True) -> pd.DataFrame:
    """Per rating: count, mean forward return, its t-statistic, and the hit rate.

    The hit rate counts a directional call right when the forward return has its
    sign; Hold claims no direction and has none.
    """
    target = _target(fr, horizon, excess)
    rows = []
    for rating in [r for r in RATING_ORDER if r in set(fr["rating"])]:
        values = target[fr["rating"] == rating].dropna()
        if values.empty:
            continue
        sign = direction(rating)
        std = values.std(ddof=1) if len(values) > 1 else np.nan
        rows.append({
            "rating": rating, "count": len(values), "mean": float(values.mean()),
            "median": float(values.median()),
            "t_stat": float(values.mean() / (std / math.sqrt(len(values)))) if std and std > 0 else np.nan,
            "hit_rate": float((values * sign > 0).mean()) if sign else np.nan,
        })
    return pd.DataFrame(rows, columns=["rating", "count", "mean", "median", "t_stat", "hit_rate"])


def _spearman(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 3 or joined.iloc[:, 0].nunique() < 2 or joined.iloc[:, 1].nunique() < 2:
        return np.nan
    ranked = joined.rank()
    return float(ranked.iloc[:, 0].corr(ranked.iloc[:, 1]))


def information_coefficient(fr: pd.DataFrame, horizon: int = 5, excess: bool = True) -> float:
    """Rank correlation of rating score against forward return, pooled over all decisions."""
    return _spearman(fr["score"], _target(fr, horizon, excess))


def rolling_ic(fr: pd.DataFrame, horizon: int = 5, window: int = 20, excess: bool = True) -> pd.Series:
    """The IC over each trailing ``window`` decisions, in date order."""
    ordered = fr.sort_values("date", kind="stable").reset_index(drop=True)
    target = _target(ordered, horizon, excess)
    values = [(_spearman(ordered["score"].iloc[i - window:i], target.iloc[i - window:i])
               if i >= window else np.nan) for i in range(1, len(ordered) + 1)]
    return pd.Series(values, index=ordered["date"], name="ic").dropna()


def signal_decay(fr: pd.DataFrame, horizons=HORIZONS, excess: bool = True) -> pd.DataFrame:
    """How the edge evolves with holding horizon: IC and direction-signed mean return."""
    rows = []
    for h in horizons:
        target = _target(fr, h, excess)
        signed = (fr["direction"] * target)[fr["direction"] != 0].dropna()
        rows.append({"horizon": h, "ic": information_coefficient(fr, h, excess),
                     "signed_return": float(signed.mean()) if len(signed) else np.nan,
                     "hit_rate": float((signed > 0).mean()) if len(signed) else np.nan,
                     "n": int(target.notna().sum())})
    return pd.DataFrame(rows)


def confusion(fr: pd.DataFrame, horizon: int = 5, excess: bool = True) -> pd.DataFrame:
    """Called direction (rows) against realized direction (columns), as counts."""
    target = _target(fr, horizon, excess)
    known = fr.assign(realized=np.where(target > 0, "up", "down"))[target.notna()]
    called = known["direction"].map({1: "bullish", 0: "neutral", -1: "bearish"})
    table = pd.crosstab(called, known["realized"])
    return table.reindex(index=["bullish", "neutral", "bearish"], columns=["up", "down"], fill_value=0)


def transitions(decisions: pd.DataFrame) -> pd.DataFrame:
    """How each ticker's rating changed from one decision to its next, as counts."""
    pairs = []
    for _, group in decisions.sort_values("date", kind="stable").groupby("ticker"):
        ratings = group["rating"].tolist()
        pairs += list(zip(ratings, ratings[1:], strict=False))
    order = [r for r in RATING_ORDER + ["REVIEW"] if r in set(decisions["rating"])]
    table = pd.DataFrame(0, index=order, columns=order)
    for before, after in pairs:
        if before in table.index and after in table.columns:
            table.loc[before, after] += 1
    table.index.name, table.columns.name = "from", "to"
    return table
