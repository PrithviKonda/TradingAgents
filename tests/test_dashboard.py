"""The quant dashboard: its statistics, its neural models and the artifacts it reads.

Every series here is synthetic and the Chronos pipeline is a stand-in, so the
suite neither downloads a model nor reaches a price vendor.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradingagents.dashboard import analytics as A, models as M, sources as S
from tradingagents.execution import CostModel, Order, PaperBroker
from tradingagents.memory import TradingMemoryLog


def _series(values, start="2026-01-05"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def _fills(rows):
    frame = pd.DataFrame(rows, columns=["ticker", "side", "quantity", "price", "commission", "date"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _bars(n=700, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    index = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"Open": close * (1 + rng.normal(0, 0.003, n)), "High": close * 1.01,
                         "Low": close * 0.99, "Close": close, "Volume": rng.integers(1e6, 2e6, n)}, index=index)


# -- performance -----------------------------------------------------------------------

@pytest.mark.unit
def test_drawdown_and_its_duration():
    equity = _series([100, 110, 99, 108.9, 121])
    assert A.drawdown(equity).min() == pytest.approx(99 / 110 - 1)
    assert A.longest_drawdown(equity) == 2


@pytest.mark.unit
def test_sharpe_and_sortino_follow_their_definitions():
    equity = _series([100, 101, 100.5, 102, 101, 103])
    r = equity.pct_change().dropna()
    stats = A.performance(equity)
    assert stats["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))
    downside = math.sqrt((r[r < 0] ** 2).sum() / len(r)) * math.sqrt(252)
    assert stats["sortino"] == pytest.approx(r.mean() * 252 / downside)
    assert stats["total_return"] == pytest.approx(0.03)


@pytest.mark.unit
def test_beta_against_a_benchmark_moved_half_as_much():
    bench_r = np.array([0.01, -0.02, 0.015, 0.005, -0.01, 0.02])
    bench = _series(100 * np.cumprod(np.r_[1, 1 + bench_r]))
    equity = _series(100 * np.cumprod(np.r_[1, 1 + 2 * bench_r]))
    stats = A.performance(equity, bench)
    assert stats["beta"] == pytest.approx(2.0)
    assert stats["alpha"] == pytest.approx(0.0, abs=1e-12)
    assert stats["correlation"] == pytest.approx(1.0)


@pytest.mark.unit
def test_monthly_returns_start_from_the_first_value():
    index = pd.to_datetime(["2026-01-02", "2026-01-30", "2026-02-27"])
    equity = pd.Series([100.0, 110.0, 99.0], index=index)
    grid = A.monthly_returns(equity)
    assert grid.loc[2026, "Jan"] == pytest.approx(0.10)
    assert grid.loc[2026, "Feb"] == pytest.approx(-0.10)
    assert grid.loc[2026, "Year"] == pytest.approx(-0.01)


@pytest.mark.unit
def test_cash_and_exposure_from_fills():
    fills = _fills([("AAPL", "buy", 10, 100.0, 1.0, "2026-01-06")])
    index = pd.bdate_range("2026-01-05", periods=3)
    cash = A.cash_series(fills, 2_000, index)
    assert list(cash) == [2_000, 999, 999]
    equity = pd.Series([2_000, 1_999, 2_099], index=index, dtype=float)
    assert A.exposure(equity, cash).iloc[-1] == pytest.approx(1_100 / 2_099)


# -- trades ----------------------------------------------------------------------------

@pytest.mark.unit
def test_round_trips_match_first_in_first_out():
    fills = _fills([("AAPL", "buy", 10, 100.0, 0.0, "2026-01-05"),
                    ("AAPL", "buy", 10, 110.0, 0.0, "2026-01-12"),
                    ("AAPL", "sell", 15, 120.0, 0.0, "2026-01-20")])
    trips = A.round_trips(fills)
    assert list(trips["quantity"]) == [10, 5]
    assert list(trips["pnl"]) == [200, 50]
    assert list(trips["holding_days"]) == [15, 8]
    lots = A.open_lots(fills)
    assert (lots.iloc[0]["quantity"], lots.iloc[0]["average_entry"]) == (5, 110)


@pytest.mark.unit
def test_a_sell_with_nothing_held_opens_a_short_and_commission_is_spread():
    fills = _fills([("TSLA", "sell", 5, 100.0, 5.0, "2026-01-05"),
                    ("TSLA", "buy", 5, 90.0, 5.0, "2026-01-09")])
    trip = A.round_trips(fills).iloc[0]
    assert trip["side"] == "short"
    assert trip["pnl"] == pytest.approx(50 - 10)


@pytest.mark.unit
def test_trade_stats_and_excursions():
    fills = _fills([("A", "buy", 1, 100.0, 0.0, "2026-01-05"), ("A", "sell", 1, 110.0, 0.0, "2026-01-07"),
                    ("A", "buy", 1, 100.0, 0.0, "2026-01-08"), ("A", "sell", 1, 95.0, 0.0, "2026-01-09")])
    trips = A.round_trips(fills)
    stats = A.trade_stats(trips)
    assert stats["win_rate"] == 0.5 and stats["profit_factor"] == pytest.approx(2.0)
    bars = pd.DataFrame({"High": [101, 112, 111, 104, 99], "Low": [99, 96, 108, 94, 94]},
                        index=pd.bdate_range("2026-01-05", periods=5))
    first = A.add_excursions(trips, {"A": bars}).iloc[0]
    assert first["mae"] == pytest.approx(-0.04) and first["mfe"] == pytest.approx(0.12)


# -- signal quality --------------------------------------------------------------------

def _decisions(rows):
    frame = pd.DataFrame(rows, columns=["date", "ticker", "rating"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


@pytest.mark.unit
def test_forward_returns_enter_at_the_first_close_after_the_decision():
    closes = _series([100, 102, 104, 108, 110, 121])        # Jan 5 .. Jan 12
    bench = _series([200, 200, 202, 202, 204, 204])
    fr = A.forward_returns(_decisions([("2026-01-05", "X", "Buy")]), {"X": closes}, bench, horizons=(1, 3, 5))
    row = fr.iloc[0]
    assert row["entry_date"] == pd.Timestamp("2026-01-06")
    assert row["fwd_1"] == pytest.approx(104 / 102 - 1)
    assert row["fwd_3"] == pytest.approx(110 / 102 - 1)
    assert row["excess_3"] == pytest.approx(110 / 102 - 1 - (204 / 200 - 1))
    assert np.isnan(row["fwd_5"])                           # not fully traded yet


@pytest.mark.unit
def test_rating_statistics_ic_and_confusion():
    fr = pd.DataFrame({
        "date": pd.bdate_range("2026-01-05", periods=6), "ticker": ["X"] * 6,
        "rating": ["Buy", "Overweight", "Hold", "Underweight", "Sell", "Buy"],
        "fwd_5": [0.05, 0.02, 0.0, -0.01, -0.04, 0.03], "excess_5": [0.05, 0.02, 0.0, -0.01, -0.04, -0.01],
    })
    fr["score"] = fr["rating"].map(A.RATING_SCORE)
    fr["direction"] = fr["rating"].map(A.direction)
    assert A.information_coefficient(fr, 5, excess=False) == pytest.approx(
        pd.Series(fr["score"]).rank().corr(fr["fwd_5"].rank()))
    table = A.by_rating(fr, 5, excess=True).set_index("rating")
    assert table.loc["Buy", "count"] == 2 and table.loc["Buy", "hit_rate"] == 0.5
    assert np.isnan(table.loc["Hold", "hit_rate"])
    grid = A.confusion(fr, 5, excess=True)
    assert grid.loc["bullish", "up"] == 2 and grid.loc["bearish", "down"] == 2
    assert grid.loc["bullish", "down"] == 1


@pytest.mark.unit
def test_rating_transitions_are_counted_per_ticker():
    decisions = _decisions([("2026-01-05", "A", "Buy"), ("2026-01-12", "A", "Hold"),
                            ("2026-01-05", "B", "Hold"), ("2026-01-12", "B", "Hold")])
    table = A.transitions(decisions)
    assert table.loc["Buy", "Hold"] == 1 and table.loc["Hold", "Hold"] == 1
    assert table.to_numpy().sum() == 2


# -- models ----------------------------------------------------------------------------

@pytest.mark.unit
def test_features_never_look_ahead():
    bars = _bars(400)
    later = bars.copy()
    later.iloc[300:, :4] *= 1.7
    a, b = M.technical_features(bars), M.technical_features(later)
    pd.testing.assert_frame_equal(a.iloc[:300], b.iloc[:300])


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["mlp", "logistic"])
def test_walk_forward_probabilities_are_out_of_sample(kind):
    bars = _bars(700)
    cut = bars.index[520]
    changed = bars.copy()
    changed.loc[changed.index > cut, ["Open", "High", "Low", "Close"]] *= 1.5
    a = M.walk_forward(bars, horizon=5, kind=kind, min_train=252, step=21, importance=False)
    b = M.walk_forward(changed, horizon=5, kind=kind, min_train=252, step=21, importance=False)
    # Every probability up to the cut was made from data through the cut at most.
    pd.testing.assert_series_equal(a.probability[a.probability.index <= cut],
                                   b.probability[b.probability.index <= cut])
    assert a.probability.between(0, 1).all()
    assert {"auc", "brier", "brier_baseline", "ic"} <= set(a.metrics)


@pytest.mark.unit
def test_walk_forward_reports_calibration_and_importance():
    result = M.walk_forward(_bars(700), horizon=5, kind="logistic", min_train=252, step=63)
    assert result.calibration["count"].sum() == result.label.notna().sum()
    assert set(result.importance["feature"]) == set(M.technical_features(_bars(50)).columns)
    when, p = result.latest
    assert when == result.probability.index[-1] and 0 <= p <= 1


@pytest.mark.unit
def test_walk_forward_refuses_too_little_history():
    with pytest.raises(ValueError, match="need at least"):
        M.walk_forward(_bars(260), horizon=5)


@pytest.mark.unit
def test_probability_above_interpolates_the_forecast_cdf():
    quantiles = {0.1: 90.0, 0.5: 100.0, 0.9: 110.0}
    assert M.probability_above(quantiles, 100.0) == pytest.approx(0.5)
    assert M.probability_above(quantiles, 105.0) == pytest.approx(0.3)
    assert M.probability_above(quantiles, 50.0) == pytest.approx(0.9)    # held at the outer band
    assert M.probability_above(quantiles, 500.0) == pytest.approx(0.1)


@pytest.mark.unit
def test_agreement_splits_calls_by_whether_the_model_concurred():
    fr = pd.DataFrame({"date": pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-19"]),
                       "ticker": ["X"] * 3, "rating": ["Buy", "Sell", "Hold"],
                       "excess_5": [0.02, 0.03, 0.0]})
    p = pd.Series([0.7, 0.8], index=pd.to_datetime(["2026-01-02", "2026-01-09"]))
    detail, summary = M.agreement(fr, {"X": p}, 5)
    assert list(detail["agrees"]) == [True, False]           # Hold claims no direction
    summary = summary.set_index("group")
    assert summary.loc["model agrees", "hit_rate"] == 1.0
    assert summary.loc["model disagrees", "hit_rate"] == 0.0


class _FakeChronos:
    """Forecasts a flat path at the last value, with symmetric bands; records its inputs."""

    def __init__(self):
        self.inputs = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        import torch

        batch = inputs if isinstance(inputs, list) else [inputs]
        self.inputs += [x.numpy() for x in batch]
        out = []
        for x in batch:
            last = float(x[-1])
            out.append([[last * (1 + (q - 0.5) / 5) for q in quantile_levels]] * prediction_length)
        q = torch.tensor(out)
        return q, q[..., len(quantile_levels) // 2]


@pytest.mark.unit
def test_chronos_forecast_shapes_and_probability():
    closes = _series(np.linspace(100, 120, 300))
    forecast = M.chronos_forecast(closes, horizon=10, pipeline=_FakeChronos())
    assert len(forecast.bands) == 10 and {"q10", "q50", "q90", "mean"} <= set(forecast.bands)
    assert forecast.bands.index[0] > closes.index[-1]
    assert forecast.prob_up == pytest.approx(0.5)
    assert forecast.expected_return == pytest.approx(0.0)


@pytest.mark.unit
def test_chronos_backtest_sees_only_history_up_to_each_origin():
    closes = _series(np.linspace(100, 160, 300))
    fake = _FakeChronos()
    record = M.chronos_backtest(closes, horizon=5, origins=6, spacing=10, pipeline=fake)
    assert len(record) == 6
    for row, seen in zip(record.itertuples(), fake.inputs, strict=True):
        assert seen[-1] == pytest.approx(closes.loc[row.origin])
    assert (record["actual_return"] > 0).all()
    assert record["inside_80"].dtype == bool


# -- sources ---------------------------------------------------------------------------

@pytest.mark.unit
def test_sources_find_every_artifact(tmp_path):
    PaperBroker.open(tmp_path / "paper" / "main.json", starting_cash=1_000)
    sim = tmp_path / "backtest" / "run1" / "simulation"
    sim.mkdir(parents=True)
    pd.DataFrame({"date": ["2026-01-05", "2026-01-06"], "equity": [100.0, 101.0],
                  "benchmark": [100.0, 100.5]}).to_csv(sim / "equity.csv", index=False)
    (sim / "fills.csv").write_text("ticker,side,quantity,price,commission,date,status,order_id\n"
                                   "AAPL,buy,1,100.0,0.0,2026-01-05,filled,\n", encoding="utf-8")
    (sim / "metrics.json").write_text('{"starting_equity": 100.0}', encoding="utf-8")
    log = tmp_path / "backtest" / "run1" / "trading_memory.md"
    TradingMemoryLog({"memory_log_path": str(log)}).store_decision("AAPL", "2026-01-05", "Rating: Buy\n\nup")
    report = tmp_path / "reports" / "AAPL_20260105_120000"
    (report / "1_analysts").mkdir(parents=True)
    (report / "complete_report.md").write_text("# AAPL", encoding="utf-8")
    (report / "1_analysts" / "market.md").write_text("market view", encoding="utf-8")

    sources = S.Sources(tmp_path)
    assert list(sources.paper_accounts()) == ["main"]
    assert list(sources.simulations()) == ["backtest/run1/simulation"]
    assert list(sources.decision_logs()) == ["backtest run1"]
    assert [name for name, _ in S.load_report(sources.reports()["AAPL_20260105_120000"])] == [
        "complete_report.md", "1_analysts/market.md"]

    account = S.load_simulation(sim)
    assert account.starting_cash == 100.0 and len(account.fills) == 1 and account.benchmark.iloc[-1] == 100.5
    decisions = S.load_decisions(log)
    assert decisions.iloc[0]["rating"] == "Buy" and bool(decisions.iloc[0]["pending"])


@pytest.mark.unit
def test_a_paper_account_is_marked_to_market_from_its_fills(tmp_path):
    path = tmp_path / "paper.json"
    broker = PaperBroker.open(path, starting_cash=1_000, costs=CostModel(slippage_bps=0))
    broker.submit(Order("AAPL", "buy", 5), 100.0, "2026-01-06")
    prices = {"AAPL": _series([99, 100, 104, 110])}          # Jan 5 .. Jan 8

    def closes(ticker, start, end):
        series = prices[ticker]
        return series[(series.index >= start) & (series.index <= end)]

    account = S.load_paper(path, closes=closes, end_date="2026-01-08")
    assert list(account.equity) == [1_000, 1_020, 1_050]
    assert account.starting_cash == 1_000


@pytest.mark.unit
def test_a_paper_account_replays_across_a_split_in_the_split_adjusted_basis(tmp_path):
    path = tmp_path / "paper.json"
    broker = PaperBroker.open(path, starting_cash=10_000, costs=CostModel(slippage_bps=0))
    broker.submit(Order("NVDA", "buy", 5), 1_000.0, "2026-01-05")
    PaperBroker.open(path, starting_cash=0, splits=lambda t: pd.Series({pd.Timestamp("2026-01-07"): 10.0}))
    # Split-adjusted closes: the pre-split 1,000 reads as 100.
    prices = {"NVDA": _series([100, 100, 100, 100])}          # Jan 5 .. Jan 8

    def closes(ticker, start, end):
        series = prices[ticker]
        return series[(series.index >= start) & (series.index <= end)]

    account = S.load_paper(path, closes=closes, end_date="2026-01-08")
    assert list(account.equity) == [10_000] * 4
    assert account.fills.iloc[0][["quantity", "price"]].tolist() == [50, 100]


# -- the app ---------------------------------------------------------------------------

@pytest.mark.unit
def test_the_app_renders_every_tab_from_saved_artifacts(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    sim = tmp_path / "simulation"
    sim.mkdir()
    pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=40).strftime("%Y-%m-%d"),
                  "equity": np.linspace(100_000, 104_000, 40)}).to_csv(sim / "equity.csv", index=False)
    (sim / "fills.csv").write_text("ticker,side,quantity,price,commission,date,status,order_id\n"
                                   "AAPL,buy,10,100.0,0.0,2026-01-06,filled,\n"
                                   "AAPL,sell,10,104.0,0.0,2026-02-02,filled,\n", encoding="utf-8")
    (sim / "metrics.json").write_text('{"starting_equity": 100000.0}', encoding="utf-8")

    app = Path(M.__file__).with_name("app.py")
    monkeypatch.setattr(sys, "argv", [str(app), "--results-dir", str(tmp_path)])
    at = AppTest.from_file(str(app), default_timeout=60)
    at.run()
    assert not at.exception
    at.sidebar.selectbox[1].select("simulation: simulation").run()
    assert not at.exception
    assert any(m.label == "Sharpe" for m in at.metric)
    assert any(m.label == "Closed trades" and m.value == "1" for m in at.metric)


def _app(tmp_path, monkeypatch):
    """The app under AppTest, reading ``tmp_path`` with synthetic prices and no model downloads."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    # st.cache_data is process-wide, so one test's prices never reach another's.
    st.cache_data.clear()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    def ohlcv(ticker, end=None):
        frame = _bars(700)
        frame.index = pd.bdate_range(end=end, periods=700)
        return frame

    def daily_closes(ticker, start, end):
        close = ohlcv(ticker, end)["Close"]
        return close[close.index >= start]

    monkeypatch.setattr(S, "ohlcv", ohlcv)
    monkeypatch.setattr(S, "daily_closes", daily_closes)
    app = Path(M.__file__).with_name("app.py")
    monkeypatch.setattr(sys, "argv", [str(app), "--results-dir", str(tmp_path)])
    return AppTest.from_file(str(app), default_timeout=120)


@pytest.mark.unit
def test_the_app_rereads_a_decision_log_that_changed(tmp_path, monkeypatch):
    log = tmp_path / "backtest" / "run1" / "trading_memory.md"
    memory = TradingMemoryLog({"memory_log_path": str(log)})
    memory.store_decision("AAPL", "2026-01-05", "Rating: Buy\n\nup")
    at = _app(tmp_path, monkeypatch)
    at.run()
    at.sidebar.selectbox[0].select("backtest run1").run()
    assert not at.exception
    read = next(s for s in at.selectbox if s.label == "Read a decision")
    assert len(read.options) == 1

    memory.store_decision("MSFT", "2026-01-06", "Rating: Sell\n\ndown")
    at.run()
    read = next(s for s in at.selectbox if s.label == "Read a decision")
    assert len(read.options) == 2


@pytest.mark.unit
def test_the_app_recomputes_a_model_when_the_date_moves(tmp_path, monkeypatch):
    from tradingagents.dataflows import date_window

    monkeypatch.setattr(date_window, "get_current_date", lambda: "2026-03-02")
    at = _app(tmp_path, monkeypatch)
    at.run()
    next(b for b in at.button if b.label == "Train and evaluate").click().run()
    assert not at.exception
    assert any(m.label.endswith("as of 2026-03-02") for m in at.metric)

    monkeypatch.setattr(date_window, "get_current_date", lambda: "2026-03-09")
    at.run()
    assert any(m.label.endswith("as of 2026-03-09") for m in at.metric)
