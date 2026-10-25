"""The TradingAgents quant desk: trades, signals, research and neural models in one place.

Run with ``tradingagents dashboard``, which starts this script under Streamlit.
Every number shown comes from ``tradingagents.dashboard.analytics`` or
``models``; this file only lays them out.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tradingagents.dashboard import analytics as A, models as M, sources as S
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.execution import Decision, simulate

RATING_COLORS = {"Buy": "#0b8a5f", "Overweight": "#5cc49a", "Hold": "#8a939b",
                 "Underweight": "#e8964a", "Sell": "#c8423a", "REVIEW": "#6c5ce7"}
RATING_SYMBOLS = {"Buy": "triangle-up", "Overweight": "triangle-up", "Hold": "circle",
                  "Underweight": "triangle-down", "Sell": "triangle-down", "REVIEW": "x"}
TODAY = get_current_date()


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None)
    return parser.parse_known_args(sys.argv[1:])[0]


def pct(value, digits: int = 2) -> str:
    return "—" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:+.{digits}%}"


def num(value, digits: int = 2) -> str:
    return "—" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:,.{digits}f}"


def _mtime(path: Path) -> float:
    return max((p.stat().st_mtime for p in [path, *path.glob("*")] if p.exists()), default=0.0)


# -- cached reads ----------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def closes(ticker: str, start: str, end: str) -> pd.Series:
    return S.daily_closes(ticker, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def bars(ticker: str, end: str) -> pd.DataFrame:
    return S.ohlcv(ticker, end)


def _sim_closes(ticker: str, start: str, end_exclusive: str) -> pd.Series:
    # The simulator asks for closes up to, not including, its end date.
    end = (datetime.strptime(end_exclusive, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    return closes(ticker, start, end)


# A cached read takes everything its answer depends on: ``stamp`` is the
# artifact's modification time and ``today`` the end of its price window.
# st.cache_data leaves parameters with a leading underscore out of the key.

@st.cache_data(show_spinner=False)
def load_simulation(path: str, stamp: float) -> S.Account:
    return S.load_simulation(path)


@st.cache_data(ttl=600, show_spinner=False)
def load_paper(path: str, stamp: float, today: str) -> S.Account:
    return S.load_paper(path, closes=closes, end_date=today)


@st.cache_data(show_spinner=False)
def load_decisions(path: str, stamp: float) -> pd.DataFrame:
    return S.load_decisions(path)


@st.cache_data(show_spinner="Simulating the decisions…")
def simulate_log(path: str, stamp: float, today: str, cash: float, max_position: float, slippage: float,
                 commission: float, benchmark: str) -> S.Account:
    frame = S.load_decisions(path)
    decisions = [Decision(d.date.strftime("%Y-%m-%d"), d.ticker, d.rating) for d in frame.itertuples()
                 if d.rating != "REVIEW"]
    config = {**DEFAULT_CONFIG, "execution": {**DEFAULT_CONFIG.get("execution", {}),
                                              "max_position_pct": max_position, "slippage_bps": slippage,
                                              "commission_pct": commission}}
    result = simulate(decisions, config, starting_cash=cash, end_date=today,
                      benchmark=benchmark or None, closes=_sim_closes)
    fills = pd.DataFrame([vars(f) for f in result.fills])
    if not fills.empty:
        fills["date"] = pd.to_datetime(fills["date"])
    return S.Account("simulation", result.equity, fills, result.benchmark, cash)


def forward_table(decisions: pd.DataFrame, benchmark: str, today: str) -> pd.DataFrame:
    # Not cached itself: the closes are, and a failed fetch is retried on the
    # next rerun rather than kept as an empty series.
    start = (decisions["date"].min() - timedelta(days=7)).strftime("%Y-%m-%d")
    series = {}
    for ticker in decisions["ticker"].unique():
        try:
            series[ticker] = closes(ticker, start, today)
        except Exception:  # an unpriced ticker scores nothing rather than stopping the page
            series[ticker] = pd.Series(dtype=float)
    bench = None
    if benchmark:
        try:
            bench = closes(benchmark, start, today)
        except Exception:  # without the benchmark, raw forward returns still score
            bench = None
    return A.forward_returns(decisions, series, bench)


@st.cache_data(show_spinner="Training the network walk-forward…")
def walk_forward(ticker: str, horizon: int, kind: str, min_train: int, step: int, today: str) -> M.WalkForward:
    return M.walk_forward(bars(ticker, today), horizon=horizon, kind=kind, min_train=min_train, step=step)


@st.cache_resource(show_spinner="Loading Chronos-Bolt…")
def chronos_pipeline():
    return M._pipeline(M.CHRONOS_MODEL)


@st.cache_data(show_spinner="Forecasting with Chronos…")
def chronos_forecast(ticker: str, horizon: int, context: int, today: str) -> M.Forecast:
    return M.chronos_forecast(bars(ticker, today)["Close"], horizon, context, pipeline=chronos_pipeline())


@st.cache_data(show_spinner="Backtesting Chronos over past origins…")
def chronos_backtest(ticker: str, horizon: int, origins: int, spacing: int, today: str) -> pd.DataFrame:
    return M.chronos_backtest(bars(ticker, today)["Close"], horizon, origins, spacing,
                              pipeline=chronos_pipeline())


# -- charts ----------------------------------------------------------------------------

def _layout(fig: go.Figure, height: int = 360, **kwargs) -> go.Figure:
    fig.update_layout(height=height, margin={"l": 10, "r": 10, "t": 40, "b": 10}, hovermode="x unified",
                      legend={"orientation": "h", "y": 1.08, "x": 0}, **kwargs)
    return fig


def equity_chart(account: S.Account, log_scale: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=account.equity.index, y=account.equity, name="Strategy", line={"width": 2}))
    if account.benchmark is not None and not account.benchmark.empty:
        rebased = account.benchmark / account.benchmark.iloc[0] * account.equity.iloc[0]
        fig.add_trace(go.Scatter(x=rebased.index, y=rebased, name="Benchmark (rebased)",
                                 line={"width": 1.5, "dash": "dot"}))
    if not account.fills.empty:
        marks = account.equity.reindex(pd.DatetimeIndex(account.fills["date"]), method="ffill")
        fig.add_trace(go.Scatter(x=account.fills["date"], y=marks.values, mode="markers", name="Fills",
                                 marker={"size": 6, "symbol": "line-ns-open", "color": "#888"},
                                 text=account.fills["side"] + " " + account.fills["ticker"],
                                 hovertemplate="%{text}<extra></extra>"))
    fig.update_yaxes(type="log" if log_scale else "linear", title="Equity")
    return _layout(fig, 420, title="Equity curve")


def drawdown_chart(equity: pd.Series) -> go.Figure:
    dd = A.drawdown(equity)
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line={"color": "#c8423a"}, name="Drawdown"))
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, 260, title="Underwater (drawdown from peak)")


def heatmap(table: pd.DataFrame, title: str, fmt: str = ".1%", center: float | None = 0.0,
            scale: str = "RdYlGn") -> go.Figure:
    values = table.astype(float)
    fig = px.imshow(values, text_auto=fmt, aspect="auto", color_continuous_scale=scale,
                    color_continuous_midpoint=center)
    return _layout(fig, max(220, 60 + 38 * len(table)), title=title)


def price_chart(ticker: str, frame: pd.DataFrame, decisions: pd.DataFrame, fills: pd.DataFrame,
                fr: pd.DataFrame | None, overlays: dict) -> go.Figure:
    rows = 3 if overlays.get("probability") is not None else 2
    heights = [0.6, 0.15, 0.25] if rows == 3 else [0.78, 0.22]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)
    fig.add_trace(go.Candlestick(x=frame.index, open=frame["Open"], high=frame["High"], low=frame["Low"],
                                 close=frame["Close"], name=ticker, showlegend=False), row=1, col=1)
    for window, color in ((50, "#3b82f6"), (200, "#a855f7")):
        if overlays.get(f"sma{window}"):
            fig.add_trace(go.Scatter(x=frame.index, y=frame["Close"].rolling(window).mean(), name=f"SMA {window}",
                                     line={"width": 1, "color": color}), row=1, col=1)
    forecast = overlays.get("forecast")
    if forecast is not None:
        b = forecast.bands
        fig.add_trace(go.Scatter(x=list(b.index) + list(b.index[::-1]), y=list(b["q90"]) + list(b["q10"][::-1]),
                                 fill="toself", fillcolor="rgba(20,194,144,0.18)", line={"width": 0},
                                 name="Chronos 10–90%", hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=b.index, y=b["q50"], name="Chronos median",
                                 line={"color": "#14c290", "dash": "dash"}), row=1, col=1)
    own = decisions[decisions["ticker"] == ticker]
    if not own.empty:
        close = frame["Close"]
        at = close.reindex(own["date"], method="ffill").values
        info = own[["rating"]].copy()
        if fr is not None and not fr.empty:
            joined = fr[fr["ticker"] == ticker].set_index("date")
            info["fwd5"] = joined["fwd_5"].reindex(own["date"]).map(pct).values
            info["ex5"] = joined["excess_5"].reindex(own["date"]).map(pct).values
        else:
            info["fwd5"] = info["ex5"] = "—"
        for rating in [r for r in RATING_COLORS if r in set(own["rating"])]:
            mask = (own["rating"] == rating).values
            fig.add_trace(go.Scatter(
                x=own["date"][mask], y=at[mask], mode="markers", name=rating,
                marker={"size": 13, "color": RATING_COLORS[rating], "symbol": RATING_SYMBOLS[rating],
                            "line": {"width": 1, "color": "white"}},
                customdata=np.stack([info["fwd5"][mask], info["ex5"][mask]], axis=-1),
                hovertemplate=f"{rating}<br>%{{x|%Y-%m-%d}}<br>5d fwd %{{customdata[0]}} · "
                              "excess %{customdata[1]}<extra></extra>"), row=1, col=1)
    own_fills = fills[fills["ticker"] == ticker] if not fills.empty else fills
    if not own_fills.empty:
        fig.add_trace(go.Scatter(x=own_fills["date"], y=own_fills["price"], mode="markers", name="Fills",
                                 marker={"size": 9, "symbol": "x-thin", "line": {"width": 2, "color": "#111"}},
                                 text=own_fills["side"] + " " + own_fills["quantity"].map("{:g}".format),
                                 hovertemplate="%{text} @ %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=frame.index, y=frame["Volume"], name="Volume", marker_color="#9aa6af",
                         showlegend=False), row=2, col=1)
    if rows == 3:
        p = overlays["probability"]
        fig.add_trace(go.Scatter(x=p.index, y=p, name="MLP P(up)", line={"color": "#e8964a"}), row=3, col=1)
        fig.add_hline(y=0.5, line_dash="dot", line_color="#888", row=3, col=1)
        fig.update_yaxes(range=[0, 1], title="P(up)", row=3, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _layout(fig, 720 if rows == 3 else 620, title=f"{ticker}: price, decisions and fills")


# -- pages -----------------------------------------------------------------------------

def portfolio_page(account: S.Account | None):
    if account is None or account.equity.empty:
        st.info("Pick an account in the sidebar: a saved simulation, a paper account, or simulate a decision log.")
        return
    stats = A.performance(account.equity, account.benchmark)
    tiles = [("Total return", pct(stats.get("total_return"))), ("CAGR", pct(stats.get("cagr"))),
             ("Sharpe", num(stats.get("sharpe"))), ("Sortino", num(stats.get("sortino"))),
             ("Max drawdown", pct(stats.get("max_drawdown"))), ("Calmar", num(stats.get("calmar"))),
             ("Volatility", pct(stats.get("volatility")).lstrip("+"))]
    if "beta" in stats:
        tiles += [("Beta", num(stats["beta"])), ("Alpha (ann.)", pct(stats["alpha"])),
                  ("vs benchmark", pct(stats["total_return"] - stats["benchmark_return"]))]
    for column, (label, value) in zip(st.columns(len(tiles)), tiles, strict=True):
        column.metric(label, value)

    st.plotly_chart(equity_chart(account, st.toggle("Log scale", value=False)), width="stretch")
    st.plotly_chart(drawdown_chart(account.equity), width="stretch")

    left, right = st.columns(2)
    window = left.slider("Rolling window (sessions)", 21, 252, 63, step=21)
    rolling = pd.DataFrame({"Rolling Sharpe": A.rolling_sharpe(account.equity, window),
                            "Rolling volatility": A.rolling_volatility(account.equity, window)})
    left.plotly_chart(_layout(px.line(rolling, y="Rolling Sharpe"), 280, title=f"Rolling Sharpe ({window})"),
                      width="stretch")
    right.plotly_chart(_layout(px.line(rolling, y="Rolling volatility"), 280,
                               title=f"Rolling volatility ({window}), annualized"), width="stretch")

    if account.starting_cash:
        cash = A.cash_series(account.fills, account.starting_cash, account.equity.index)
        exp = A.exposure(account.equity, cash)
        st.plotly_chart(_layout(px.area(exp.to_frame(), y="exposure"), 220, title="Gross exposure (share of equity invested)"),
                        width="stretch")
    st.plotly_chart(heatmap(A.monthly_returns(account.equity), "Monthly returns"), width="stretch")
    with st.expander("All statistics"):
        st.dataframe(pd.DataFrame({"value": stats}), width="stretch")


def trades_page(account: S.Account | None):
    if account is None or account.fills.empty:
        st.info("This account has no fills yet.")
        return
    fills = account.fills
    trips = A.round_trips(fills)
    stats = A.trade_stats(trips)
    tiles = [("Closed trades", f"{stats['trades']}")]
    if stats["trades"]:
        tiles += [("Win rate", f"{stats['win_rate']:.0%}"), ("Profit factor", num(stats["profit_factor"])),
                  ("Expectancy", num(stats["expectancy"])), ("Avg win", num(stats["average_win"])),
                  ("Avg loss", num(stats["average_loss"])), ("Avg hold (days)", num(stats["average_holding_days"], 1)),
                  ("Realized P&L", num(stats["total_pnl"]))]
    for column, (label, value) in zip(st.columns(len(tiles)), tiles, strict=True):
        column.metric(label, value)

    lots = A.open_lots(fills)
    if not lots.empty:
        st.subheader("Open positions")
        marks = {}
        for ticker in lots["ticker"]:
            try:
                marks[ticker] = float(closes(ticker, (pd.Timestamp(TODAY) - timedelta(days=10)).strftime("%Y-%m-%d"),
                                             TODAY).iloc[-1])
            except Exception:
                marks[ticker] = np.nan
        lots["last"] = lots["ticker"].map(marks)
        lots["unrealized"] = (lots["last"] - lots["average_entry"]) * lots["quantity"]
        lots["unrealized_pct"] = lots["last"] / lots["average_entry"] - 1
        st.dataframe(lots, width="stretch", hide_index=True)

    if not trips.empty:
        frames = {}
        for ticker in trips["ticker"].unique():
            # A ticker with no bars just shows no excursions.
            with contextlib.suppress(Exception):
                frames[ticker] = bars(ticker, TODAY)
        trips = A.add_excursions(trips, frames)
        st.subheader("Closed trades (FIFO)")
        st.dataframe(trips, width="stretch", hide_index=True)
        a, b = st.columns(2)
        a.plotly_chart(_layout(px.histogram(trips, x="return", nbins=30, color="ticker"), 300,
                               title="Trade return distribution"), width="stretch")
        b.plotly_chart(_layout(px.scatter(trips, x="holding_days", y="return", color="ticker", size="quantity"),
                               300, title="Return vs holding period"), width="stretch")
        c, d = st.columns(2)
        by_ticker = trips.groupby("ticker", as_index=False)["pnl"].sum().sort_values("pnl")
        c.plotly_chart(_layout(px.bar(by_ticker, x="ticker", y="pnl", color="pnl",
                                      color_continuous_scale="RdYlGn"), 300, title="Realized P&L by ticker"),
                       width="stretch")
        if trips["mae"].notna().any():
            d.plotly_chart(_layout(px.scatter(trips, x="mae", y="return", color="ticker",
                                              hover_data=["entry_date", "exit_date"]), 300,
                                   title="Return vs maximum adverse excursion"), width="stretch")
    st.subheader("Blotter")
    st.dataframe(fills, width="stretch", hide_index=True)
    st.download_button("Download fills (CSV)", fills.to_csv(index=False), "fills.csv", "text/csv")


def signals_page(decisions: pd.DataFrame, account: S.Account | None, benchmark: str):
    fills = account.fills if account is not None else pd.DataFrame()
    tickers = sorted(set(decisions["ticker"]) | (set(fills["ticker"]) if not fills.empty else set()))
    typed = st.text_input("Ticker (or pick below)", "")
    ticker = typed.strip().upper() or (st.selectbox("Ticker", tickers) if tickers else None)
    if not ticker:
        st.info("No decisions or fills yet; type a ticker to chart it.")
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    lookback = c1.selectbox("Window", ["6M", "1Y", "2Y", "5Y"], index=1)
    overlays = {"sma50": c2.checkbox("SMA 50", True), "sma200": c3.checkbox("SMA 200", True)}
    want_mlp = c4.checkbox("MLP P(up)", False)
    want_chronos = c5.checkbox("Chronos forecast", False, disabled=not M.chronos_available())
    try:
        frame = bars(ticker, TODAY)
    except Exception as exc:
        st.error(f"No prices for {ticker}: {exc}")
        return
    months = {"6M": 6, "1Y": 12, "2Y": 24, "5Y": 60}[lookback]
    shown = frame[frame.index >= frame.index[-1] - pd.DateOffset(months=months)]
    if want_mlp:
        try:
            wf = walk_forward(ticker, 5, "mlp", 252, 21, TODAY)
            overlays["probability"] = wf.probability[wf.probability.index >= shown.index[0]]
        except ValueError as exc:
            st.warning(str(exc))
    if want_chronos:
        overlays["forecast"] = chronos_forecast(ticker, 20, 512, TODAY)
    fr = forward_table(decisions, benchmark, TODAY) if not decisions.empty else None
    st.plotly_chart(price_chart(ticker, shown, decisions, fills, fr, overlays), width="stretch")
    if fr is not None:
        own = fr[fr["ticker"] == ticker]
        if not own.empty:
            st.subheader(f"{ticker} decisions and what followed")
            st.dataframe(own, width="stretch", hide_index=True)


def quality_page(decisions: pd.DataFrame, benchmark: str):
    scored = decisions[decisions["rating"] != "REVIEW"]
    if scored.empty:
        st.info("Pick a decision log with rated decisions in the sidebar.")
        return
    c1, c2, c3 = st.columns(3)
    horizon = c1.selectbox("Horizon (sessions)", list(A.HORIZONS), index=1)
    excess = c2.toggle(f"Excess over {benchmark or 'benchmark'}", value=bool(benchmark), disabled=not benchmark)
    window = c3.slider("Rolling IC window (decisions)", 5, 60, 20)
    fr = forward_table(scored, benchmark, TODAY)
    target = f"excess_{horizon}" if excess else f"fwd_{horizon}"
    known = fr[fr[target].notna()]
    directional = known[known["direction"] != 0]
    signed = directional["direction"] * directional[target]
    ic = A.information_coefficient(fr, horizon, excess)
    tiles = [("Decisions scored", f"{len(known)} / {len(fr)}"), ("IC (Spearman)", num(ic, 3)),
             ("Directional hit rate", f"{(signed > 0).mean():.0%}" if len(signed) else "—"),
             ("Mean signed return", pct(signed.mean()) if len(signed) else "—"),
             ("t-stat", num(signed.mean() / (signed.std(ddof=1) / np.sqrt(len(signed))))
              if len(signed) > 2 and signed.std(ddof=1) > 0 else "—")]
    for column, (label, value) in zip(st.columns(len(tiles)), tiles, strict=True):
        column.metric(label, value)

    table = A.by_rating(fr, horizon, excess)
    a, b = st.columns([3, 2])
    if not table.empty:
        a.plotly_chart(_layout(px.bar(table, x="rating", y="mean", color="rating", color_discrete_map=RATING_COLORS,
                                      text=table["hit_rate"].map(lambda h: "" if pd.isna(h) else f"hit {h:.0%}"),
                                      category_orders={"rating": A.RATING_ORDER}), 320,
                               title=f"Mean {horizon}-session {'excess ' if excess else ''}return by rating"),
                       width="stretch")
    b.dataframe(table, width="stretch", hide_index=True)

    c, d = st.columns(2)
    decay = A.signal_decay(fr, excess=excess)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=decay["horizon"], y=decay["ic"], name="IC", mode="lines+markers"))
    fig.add_trace(go.Bar(x=decay["horizon"], y=decay["signed_return"], name="Signed return", opacity=0.5),
                  secondary_y=True)
    c.plotly_chart(_layout(fig, 320, title="Signal decay by horizon"), width="stretch")
    rolling = A.rolling_ic(fr, horizon, window, excess)
    d.plotly_chart(_layout(px.line(rolling.to_frame(), y="ic"), 320, title=f"Rolling IC ({window} decisions)")
                   .add_hline(y=0, line_dash="dot"), width="stretch")

    e, f = st.columns(2)
    e.plotly_chart(heatmap(A.confusion(fr, horizon, excess), "Called vs realized direction", fmt="d",
                           center=None, scale="Blues"), width="stretch")
    f.plotly_chart(heatmap(A.transitions(scored), "Rating transitions (per ticker)", fmt="d", center=None,
                           scale="Blues"), width="stretch")

    g, h = st.columns(2)
    g.plotly_chart(_layout(px.box(known, x="rating", y=target, color="rating", color_discrete_map=RATING_COLORS,
                                  points="all", category_orders={"rating": A.RATING_ORDER}), 340,
                           title="Forward return distribution by rating"), width="stretch")
    curve = directional.assign(signed=signed).sort_values("date")
    curve["cumulative"] = curve["signed"].cumsum()
    h.plotly_chart(_layout(px.line(curve, x="date", y="cumulative"), 340,
                           title="Cumulative direction-signed return (equal weight per call)"), width="stretch")

    with st.expander("Does the neural network agree with the agents?"):
        st.caption("Trains the walk-forward MLP on each ticker at this horizon, then splits the agents' directional "
                   "calls by whether the network's out-of-sample probability pointed the same way.")
        if st.button("Compare with the MLP"):
            probabilities = {}
            for ticker in fr["ticker"].unique():
                try:
                    probabilities[ticker] = walk_forward(ticker, horizon, "mlp", 252, 21, TODAY).probability
                except Exception as exc:
                    st.warning(f"{ticker}: {exc}")
            detail, summary = M.agreement(fr, probabilities, horizon, excess)
            st.dataframe(summary, width="stretch", hide_index=True)
            st.dataframe(detail, width="stretch", hide_index=True)
    st.download_button("Download decisions with forward returns (CSV)", fr.to_csv(index=False),
                       "forward_returns.csv", "text/csv")


def research_page(decisions: pd.DataFrame, sources: S.Sources):
    if not decisions.empty:
        c1, c2 = st.columns(2)
        tickers = c1.multiselect("Tickers", sorted(decisions["ticker"].unique()))
        ratings = c2.multiselect("Ratings", [r for r in RATING_COLORS if r in set(decisions["rating"])])
        view = decisions
        if tickers:
            view = view[view["ticker"].isin(tickers)]
        if ratings:
            view = view[view["rating"].isin(ratings)]
        monthly = view.assign(month=view["date"].dt.to_period("M").dt.to_timestamp())
        counts = monthly.groupby(["month", "rating"]).size().reset_index(name="count")
        st.plotly_chart(_layout(px.bar(counts, x="month", y="count", color="rating", color_discrete_map=RATING_COLORS,
                                       category_orders={"rating": list(RATING_COLORS)}), 260,
                                title="Ratings issued per month"), width="stretch")
        st.dataframe(view.drop(columns=["decision", "reflection"]), width="stretch", hide_index=True)
        labels = [f"{d.date:%Y-%m-%d} · {d.ticker} · {d.rating}" for d in view.itertuples()]
        if labels:
            choice = st.selectbox("Read a decision", range(len(labels)), format_func=labels.__getitem__)
            row = view.iloc[choice]
            left, right = st.columns(2)
            left.markdown("**Decision**")
            left.markdown(row["decision"] or "_(empty)_")
            right.markdown("**Reflection after the outcome**")
            right.markdown(row["reflection"] or "_Not settled yet._")
    else:
        st.info("No decision log selected.")

    st.divider()
    st.subheader("Analyst reports")
    reports = sources.reports()
    if not reports:
        st.caption(f"No saved reports under {sources.results_dir / 'reports'} yet; save one at the end of a run.")
        return
    name = st.selectbox("Report", list(reports))
    folder = reports[name]
    html = folder / "complete_report.html"
    if html.is_file():
        st.download_button("Download the complete report (HTML)", html.read_bytes(), html.name, "text/html")
    for section, text in S.load_report(folder):
        with st.expander(section, expanded=section == "complete_report.md"):
            st.markdown(text)


def models_page(decisions: pd.DataFrame):
    st.markdown("Two neural signals, each scored out of sample before you trust it: a **walk-forward MLP** "
                "trained here on technical features, and **Chronos-Bolt**, a pretrained time-series "
                "transformer that forecasts zero-shot.")
    default = decisions["ticker"].iloc[-1] if not decisions.empty else "SPY"
    c1, c2, c3, c4 = st.columns(4)
    ticker = c1.text_input("Ticker", default).strip().upper()
    horizon = c2.selectbox("Forecast horizon (sessions)", [1, 5, 10, 20], index=1)
    min_train = c3.number_input("Initial training sessions", 126, 1000, 252, step=21)
    step = c4.number_input("Refit every (sessions)", 5, 126, 21, step=1)

    st.subheader("Walk-forward MLP vs logistic baseline")
    if st.button("Train and evaluate", type="primary"):
        st.session_state["mlp_run"] = (ticker, horizon, int(min_train), int(step))
    run = st.session_state.get("mlp_run")
    if run and run[0] == ticker:
        try:
            mlp = walk_forward(*run[:2], "mlp", *run[2:], TODAY)
            logit = walk_forward(*run[:2], "logistic", *run[2:], TODAY)
        except Exception as exc:
            st.error(str(exc))
        else:
            when, p = mlp.latest
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"MLP P(up over {run[1]}d) as of {when:%Y-%m-%d}", f"{p:.0%}")
            m2.metric("MLP out-of-sample AUC", num(mlp.metrics.get("auc"), 3),
                      delta=num(mlp.metrics.get("auc", np.nan) - logit.metrics.get("auc", np.nan), 3) + " vs logistic")
            m3.metric("MLP Brier", num(mlp.metrics.get("brier"), 4),
                      delta=num(mlp.metrics.get("brier", np.nan) - mlp.metrics.get("brier_baseline", np.nan), 4)
                      + " vs base rate", delta_color="inverse")
            m4.metric("MLP IC vs forward return", num(mlp.metrics.get("ic"), 3))
            st.dataframe(pd.DataFrame({"MLP": mlp.metrics, "Logistic": logit.metrics}), width="stretch")
            frame = bars(ticker, TODAY)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.04)
            fig.add_trace(go.Scatter(x=frame.index, y=frame["Close"], name="Close"), row=1, col=1)
            fig.add_trace(go.Scatter(x=mlp.probability.index, y=mlp.probability, name="MLP"), row=2, col=1)
            fig.add_trace(go.Scatter(x=logit.probability.index, y=logit.probability, name="Logistic",
                                     line={"dash": "dot"}), row=2, col=1)
            fig.add_hline(y=0.5, line_dash="dot", row=2, col=1)
            st.plotly_chart(_layout(fig, 520, title="Out-of-sample P(up)"), width="stretch")
            a, b = st.columns(2)
            cal = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dot"}, name="Perfect"),
                             go.Scatter(x=mlp.calibration["predicted"], y=mlp.calibration["observed"],
                                        mode="lines+markers", name="MLP", text=mlp.calibration["count"]),
                             go.Scatter(x=logit.calibration["predicted"], y=logit.calibration["observed"],
                                        mode="lines+markers", name="Logistic")])
            cal.update_xaxes(title="Predicted P(up)")
            cal.update_yaxes(title="Observed frequency")
            a.plotly_chart(_layout(cal, 340, title="Calibration"), width="stretch")
            if mlp.importance is not None and not mlp.importance.empty:
                b.plotly_chart(_layout(px.bar(mlp.importance.iloc[::-1], x="importance", y="feature",
                                              error_x="std", orientation="h"), 340,
                                       title="Permutation importance (AUC drop, latest 20%)"), width="stretch")
            out = pd.DataFrame({"p_up_mlp": mlp.probability, "p_up_logistic": logit.probability,
                                "label": mlp.label, "forward_return": mlp.forward})
            st.download_button("Download probabilities (CSV)", out.to_csv(), f"{ticker}_p_up.csv", "text/csv")

    st.subheader("Chronos-Bolt forecast")
    if not M.chronos_available():
        st.warning("Chronos is not installed in this environment: pip install chronos-forecasting")
        return
    d1, d2, d3 = st.columns(3)
    fc_horizon = d1.slider("Forecast sessions ahead", 5, 64, 20)
    origins = d2.slider("Backtest origins", 12, 120, 48)
    spacing = d3.slider("Sessions between origins", 1, 21, 5)
    if st.button("Forecast and backtest"):
        st.session_state["chronos_run"] = (ticker, fc_horizon, origins, spacing, horizon)
    run = st.session_state.get("chronos_run")
    if run and run[0] == ticker:
        try:
            forecast = chronos_forecast(ticker, run[1], 512, TODAY)
            record = chronos_backtest(ticker, run[4], run[2], run[3], TODAY)
        except Exception as exc:
            st.error(str(exc))
            return
        history = bars(ticker, TODAY)["Close"].iloc[-180:]
        b = forecast.bands
        fig = go.Figure([
            go.Scatter(x=history.index, y=history, name="Close"),
            go.Scatter(x=list(b.index) + list(b.index[::-1]), y=list(b["q90"]) + list(b["q10"][::-1]),
                       fill="toself", fillcolor="rgba(20,194,144,0.18)", line={"width": 0}, name="10–90%"),
            go.Scatter(x=list(b.index) + list(b.index[::-1]), y=list(b["q70"]) + list(b["q30"][::-1]),
                       fill="toself", fillcolor="rgba(20,194,144,0.30)", line={"width": 0}, name="30–70%"),
            go.Scatter(x=b.index, y=b["q50"], name="Median", line={"color": "#14c290", "dash": "dash"}),
        ])
        st.plotly_chart(_layout(fig, 420, title=f"{ticker}: Chronos-Bolt {run[1]}-session forecast"), width="stretch")
        p_metrics = M.classification_metrics(record["prob_up"], record["label"], record["actual_return"])
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(f"P(up in {run[1]}d)", f"{forecast.prob_up:.0%}")
        m2.metric("Median expected return", pct(forecast.expected_return))
        m3.metric(f"Backtest hit rate ({run[4]}d)", f"{p_metrics.get('accuracy', np.nan):.0%}")
        m4.metric("Backtest AUC", num(p_metrics.get("auc"), 3))
        m5.metric("80% band coverage", f"{record['inside_80'].mean():.0%}")
        st.plotly_chart(_layout(px.scatter(record, x="predicted_return", y="actual_return", color="prob_up",
                                           color_continuous_scale="RdYlGn", hover_data=["origin"]), 340,
                                title="Chronos backtest: predicted median vs realized return"), width="stretch")
        with st.expander("Backtest record"):
            st.dataframe(record, width="stretch", hide_index=True)


# -- main ------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="TradingAgents Quant Desk", page_icon="📈", layout="wide")
    sources = S.Sources.from_config(DEFAULT_CONFIG, _args().results_dir)
    st.sidebar.title("Quant Desk")
    st.sidebar.caption(f"Reading {sources.results_dir}")

    logs = sources.decision_logs()
    log_name = st.sidebar.selectbox("Decision log", ["(none)", *logs])
    log_path = logs.get(log_name)
    decisions = load_decisions(str(log_path), _mtime(log_path)) if log_path else pd.DataFrame(
        columns=["date", "ticker", "rating", "pending", "raw", "alpha", "holding", "resolved", "decision",
                 "reflection"])

    sims, papers = sources.simulations(), sources.paper_accounts()
    options = ["(none)"] + [f"simulation: {k}" for k in sims] + [f"paper: {k}" for k in papers]
    if log_path:
        options.insert(1, "simulate the selected log")
    choice = st.sidebar.selectbox("Account", options, index=1 if len(options) > 1 else 0)
    benchmark = st.sidebar.text_input("Benchmark", "SPY").strip().upper()

    account = None
    try:
        if choice == "simulate the selected log":
            with st.sidebar.expander("Simulation settings", expanded=True):
                cash = st.number_input("Starting cash", 1_000.0, 1e9, 100_000.0, step=10_000.0)
                max_position = st.slider("Max position (% of equity)", 1, 100, 10) / 100
                slippage = st.slider("Slippage (bps)", 0, 50, 5)
                commission = st.slider("Commission (bps of notional)", 0, 50, 0) / 10_000
            account = simulate_log(str(log_path), _mtime(log_path), TODAY, cash, max_position, float(slippage),
                                   commission, benchmark)
        elif choice.startswith("simulation: "):
            path = sims[choice.removeprefix("simulation: ")]
            account = load_simulation(str(path), _mtime(path))
        elif choice.startswith("paper: "):
            path = papers[choice.removeprefix("paper: ")]
            account = load_paper(str(path), _mtime(path), TODAY)
    except Exception as exc:  # a broken artifact is reported, and the other tabs still work
        st.sidebar.error(f"Could not load the account: {exc}")

    tabs = st.tabs(["Portfolio", "Trades", "Signals", "Signal quality", "Research", "Models"])
    with tabs[0]:
        portfolio_page(account)
    with tabs[1]:
        trades_page(account)
    with tabs[2]:
        signals_page(decisions, account, benchmark)
    with tabs[3]:
        quality_page(decisions, benchmark)
    with tabs[4]:
        research_page(decisions, sources)
    with tabs[5]:
        models_page(decisions)


main()
