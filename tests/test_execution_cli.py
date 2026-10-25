"""The execution commands: one instrument per symbol, and errors as one red line.

Like tests/test_execution.py, every price is synthetic and every broker local or
faked: nothing here reaches a venue or the network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

import cli.main as m
import tradingagents.execution.simulator as sim
from tests.test_execution import DAYS, FREE, _closes, _config
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.execution import Decision, PaperBroker, run_session, simulate
from tradingagents.llm_clients.chatgpt_auth import ChatGPTAuthError
from tradingagents.memory import TradingMemoryLog
from tradingagents.portfolio import PortfolioContext, Position


class _OneRating:
    """A graph that rates every ticker the same, and records the names it was asked."""

    def __init__(self, rating):
        self.rating, self.seen = rating, []

    def propagate(self, ticker, date, asset_type="stock", portfolio=None):
        self.seen.append(ticker)
        return {}, self.rating


# -- one instrument, one position --------------------------------------------------

@pytest.mark.unit
def test_two_spellings_of_one_coin_are_one_holding(tmp_path):
    path = tmp_path / "main.json"
    for spelling in ("BTCUSD", "BTC-USD"):
        broker = PaperBroker.open(path, starting_cash=10_000, costs=FREE)
        run_session([spelling], broker, _config(), trade_date="2026-01-05",
                    price=lambda t, d: 100.0, graph=_OneRating("Buy"))
    broker = PaperBroker.open(path, starting_cash=10_000, costs=FREE)
    # The second run sees the first run's 50 and buys nothing more.
    assert list(broker.holdings) == ["BTC-USD"] and broker.quantity("BTCUSD") == 50
    assert len(broker.fills) == 1


@pytest.mark.unit
def test_an_account_saved_with_two_spellings_reads_as_one_position(tmp_path):
    path = tmp_path / "main.json"
    path.write_text(json.dumps({
        "cash": 1_000, "starting_cash": 5_000,
        "positions": {"700.HK": {"quantity": 10, "average_price": 300},
                      "0700.HK": {"quantity": 30, "average_price": 400}},
        "fills": [{"ticker": "700.HK", "side": "buy", "quantity": 10, "price": 300, "date": "2026-01-02"}],
    }))
    broker = PaperBroker.open(path, starting_cash=1, costs=FREE)
    held = broker.portfolio().position_in("0700.HK")
    assert [p.ticker for p in broker.portfolio().positions] == ["0700.HK"]
    assert (held.quantity, held.average_price) == (40, 375)
    assert broker.fills[0].ticker == "0700.HK"


@pytest.mark.unit
def test_paper_hands_the_session_canonical_tickers_and_detects_a_coin(tmp_path, monkeypatch):
    seen = {}

    def fake_run_session(names, broker, config, **kwargs):
        seen.update(names=names, asset_type=kwargs["asset_type"])
        return []

    monkeypatch.setitem(m.DEFAULT_CONFIG, "results_dir", str(tmp_path))
    monkeypatch.setattr(m, "run_session", fake_run_session)
    monkeypatch.setattr(m, "get_splits", lambda ticker: pd.Series(dtype=float))
    result = CliRunner().invoke(m.app, ["paper", "btcusd,BTC-USD,ethusdt", "--account", "demo"])
    assert result.exit_code == 0, result.output
    assert seen == {"names": ["BTC-USD", "ETH-USD"], "asset_type": {"BTC-USD": "crypto", "ETH-USD": "crypto"}}


def _session_args(tmp_path, monkeypatch, *args):
    seen = {}

    def fake_run_session(names, broker, config, **kwargs):
        seen.update(asset_type=kwargs["asset_type"], analysts=kwargs["selected_analysts"])
        return []

    monkeypatch.setitem(m.DEFAULT_CONFIG, "results_dir", str(tmp_path))
    monkeypatch.setattr(m, "run_session", fake_run_session)
    monkeypatch.setattr(m, "get_splits", lambda ticker: pd.Series(dtype=float))
    return CliRunner().invoke(m.app, ["paper", *args, "--account", "demo"]), seen


@pytest.mark.unit
def test_a_mixed_list_runs_each_ticker_as_its_own_asset_type(tmp_path, monkeypatch):
    result, seen = _session_args(tmp_path, monkeypatch, "NVDA,btcusd")
    assert result.exit_code == 0, result.output
    assert seen["asset_type"] == {"NVDA": "stock", "BTC-USD": "crypto"}
    # The coin runs without the fundamentals analyst; the stock keeps it.
    assert seen["analysts"] == {"stock": ["market", "social", "news", "fundamentals"],
                                "crypto": ["market", "social", "news"]}


@pytest.mark.unit
def test_asset_type_still_sets_one_type_for_every_ticker(tmp_path, monkeypatch):
    result, seen = _session_args(tmp_path, monkeypatch, "NVDA,BTC-USD", "--asset-type", "stock",
                                 "--analysts", "market,fundamentals")
    assert result.exit_code == 0, result.output
    assert seen == {"asset_type": {"NVDA": "stock", "BTC-USD": "stock"},
                    "analysts": {"stock": ["market", "fundamentals"]}}


@pytest.mark.unit
def test_analysts_no_coin_can_run_are_refused_in_a_mixed_list(tmp_path, monkeypatch):
    result, seen = _session_args(tmp_path, monkeypatch, "NVDA,BTC-USD", "--analysts", "fundamentals")
    assert result.exit_code == 1 and "not available for crypto" in result.output
    assert seen == {}


@pytest.mark.unit
def test_a_session_builds_one_graph_per_analyst_set(monkeypatch):
    import tradingagents.graph.trading_graph as tg

    built = []

    class _Graph:
        def __init__(self, analysts, config):
            self.analysts, self.runs = analysts, []
            built.append(self)

        def propagate(self, ticker, date, asset_type="stock", portfolio=None):
            self.runs.append((ticker, asset_type))
            return {}, "Hold"

    monkeypatch.setattr(tg, "TradingAgentsGraph", _Graph)
    broker = PaperBroker(cash=10_000, costs=FREE)
    run_session(["NVDA", "btcusd", "AAPL"], broker, _config(), trade_date="2026-01-05",
                asset_type={"NVDA": "stock", "BTCUSD": "crypto", "AAPL": "stock"},
                selected_analysts={"stock": ["market", "fundamentals"], "crypto": ["market"]},
                price=lambda t, d: 100.0)
    runs = {tuple(g.analysts): g.runs for g in built}
    assert runs == {("market", "fundamentals"): [("NVDA", "stock"), ("AAPL", "stock")],
                    ("market",): [("BTC-USD", "crypto")]}


# -- simulate's default output -----------------------------------------------------

def _one_buy_log(path):
    TradingMemoryLog({"memory_log_path": str(path)}).store_decision("AAPL", "2026-01-05", "Rating: Buy\n\nup")
    return path


@pytest.mark.unit
@pytest.mark.parametrize("source", ["run-id", "log"])
def test_simulate_saves_a_long_run_id_or_a_spaced_log_name_by_default(source, tmp_path, monkeypatch):
    run_id = "sweep-" + "x" * 26                     # 32 characters, as backtest allows
    results = tmp_path / "results"
    if source == "run-id":
        _one_buy_log(results / "backtest" / run_id / "trading_memory.md")
        args = ["--run-id", run_id]
    else:
        args = ["--log", str(_one_buy_log(tmp_path / "my trading log.md"))]
    monkeypatch.setattr(sim, "get_closes", _closes({"AAPL": dict(zip(DAYS, [100] * 5, strict=True))}))
    monkeypatch.setitem(m.DEFAULT_CONFIG, "results_dir", str(results))
    result = CliRunner().invoke(m.app, ["simulate", *args, "--end", "2026-01-09", "--benchmark", ""])
    assert result.exit_code == 0, result.output
    [saved] = (results / "simulations").glob("*/metrics.json")
    assert len(saved.parent.name) <= 32 and " " not in saved.parent.name


# -- errors are one red line -------------------------------------------------------

def _refused(result, message):
    """Exit 1 through typer, not an exception escaping, with ``message`` as the one line."""
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit), result.output
    assert result.output.strip().splitlines() == [result.output.strip()] and message in result.output


@pytest.mark.unit
def test_a_price_outage_during_a_replay_is_one_red_line(tmp_path, monkeypatch):
    def outage(ticker, start, end):
        raise VendorUnavailableError("yahoo is throttling")

    monkeypatch.setattr(sim, "get_closes", outage)
    log = _one_buy_log(tmp_path / "trading_memory.md")
    result = CliRunner().invoke(m.app, ["simulate", "--log", str(log), "--out", str(tmp_path / "out")])
    _refused(result, "yahoo is throttling")


@pytest.mark.unit
@pytest.mark.parametrize("flag, value, message", [
    ("--cash", "0", "--cash must be more than 0"),
    ("--max-position", "0", "--max-position"),
    ("--max-position", "1.5", "--max-position"),
])
def test_simulate_refuses_funding_no_account_can_trade_with(flag, value, message, tmp_path, monkeypatch):
    monkeypatch.setattr(sim, "get_closes", _closes({"AAPL": dict(zip(DAYS, [100] * 5, strict=True))}))
    log = _one_buy_log(tmp_path / "trading_memory.md")
    _refused(CliRunner().invoke(m.app, ["simulate", "--log", str(log), flag, value]), message)


@pytest.mark.unit
@pytest.mark.parametrize("args, message", [
    (["simulate", "--run-id", "my run"], "not allowed in a filesystem path"),
    (["paper", "--status", "--account", "my acct"], "not allowed in a filesystem path"),
    (["paper", "--status", "--cash", "-5"], "--cash must be more than 0"),
])
def test_a_bad_account_name_or_funding_is_one_red_line(args, message, tmp_path, monkeypatch):
    monkeypatch.setitem(m.DEFAULT_CONFIG, "results_dir", str(tmp_path))
    _refused(CliRunner().invoke(m.app, args), message)
    assert not (tmp_path / "paper").exists()       # no account was opened


@pytest.mark.unit
@pytest.mark.parametrize("error", [ChatGPTAuthError("sign in again: tradingagents login chatgpt"),
                                   ImportError("install the bedrock extra"), KeyError("OPENAI_API_KEY")])
def test_a_session_that_cannot_start_is_one_red_line(error, tmp_path, monkeypatch):
    def cannot_start(names, broker, config, **kwargs):
        raise error

    monkeypatch.setitem(m.DEFAULT_CONFIG, "results_dir", str(tmp_path))
    monkeypatch.setattr(m, "run_session", cannot_start)
    monkeypatch.setattr(m, "get_splits", lambda ticker: pd.Series(dtype=float))
    _refused(CliRunner().invoke(m.app, ["paper", "NVDA", "--account", "demo"]), str(error))


# -- backtest --portfolio --simulate -----------------------------------------------

@pytest.mark.unit
def test_a_simulation_can_open_holding_a_book():
    # 10 AAPL and 1,000 cash, worth 2,000 at the Jan 5 close; the Sell exits at Jan 6's 110.
    prices = _closes({"AAPL": {"2026-01-02": 90, **dict(zip(DAYS, [100, 110, 120, 130, 140], strict=True))}})
    book = PortfolioContext(cash=1_000, positions=[Position(ticker="aapl", quantity=10, average_price=80)])
    result = simulate([Decision("2026-01-05", "AAPL", "Sell")], _config(), end_date="2026-01-09",
                      benchmark=None, closes=prices, book=book)
    assert [(f.date, f.side, f.quantity, f.price) for f in result.fills] == [("2026-01-06", "sell", 10, 110)]
    assert result.metrics["starting_equity"] == 2_000
    assert result.metrics["total_return"] == pytest.approx(0.05)


@pytest.mark.unit
def test_a_held_position_no_decision_names_is_carried_and_marked():
    prices = _closes({"AAPL": dict(zip(DAYS, [100] * 5, strict=True)),
                      "MSFT": {"2026-01-02": 50, **dict(zip(DAYS, [50, 60, 60, 60, 70], strict=True))}})
    book = PortfolioContext(cash=0, positions=[Position(ticker="MSFT", quantity=10)])
    result = simulate([Decision("2026-01-05", "AAPL", "Hold")], _config(), end_date="2026-01-09",
                      benchmark=None, closes=prices, book=book)
    assert result.equity.loc["2026-01-09"] == 700 and result.metrics["starting_equity"] == 500


@pytest.mark.unit
def test_backtest_simulates_the_book_it_was_given(tmp_path, monkeypatch):
    portfolio = tmp_path / "book.json"
    portfolio.write_text(json.dumps({"cash": 500, "positions": [{"ticker": "NVDA", "quantity": 3}]}))
    log = _one_buy_log(tmp_path / "sweep" / "trading_memory.md")
    swept = SimpleNamespace(cells_run=1, skipped=0, log_path=log, run_id="r", failures=[], settlement_failures=[])
    monkeypatch.setattr(m, "run_backtest", lambda *a, **k: swept)
    monkeypatch.setattr(m, "summarize", lambda result: SimpleNamespace(render=lambda: ""))
    seen = {}

    def fake_simulate(decisions, config, **kwargs):
        seen["book"] = kwargs.get("book")
        raise ValueError("stop here")

    monkeypatch.setattr(m, "simulate_decisions", fake_simulate)
    CliRunner().invoke(m.app, ["backtest", "NVDA", "--start", "2026-01-05", "--end", "2026-01-05",
                               "--portfolio", str(portfolio), "--simulate"])
    assert seen["book"].cash == 500 and seen["book"].position_in("NVDA").quantity == 3


# -- the benchmark follows the config ----------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("tickers, configured, expected", [
    (["7203.T"], {}, "^N225"),                                 # one market: its benchmark_map entry
    (["7203.T", "AAPL"], {}, "SPY"),                           # several: the map's default
    (["7203.T"], {"benchmark_ticker": "qqq"}, "QQQ"),          # benchmark_ticker overrides the map
])
def test_a_simulation_benchmarks_what_settlement_would(tickers, configured, expected):
    flat = dict(zip(DAYS, [100] * 5, strict=True))
    prices = _closes(dict.fromkeys([*tickers, "SPY", "^N225", "QQQ"], flat))
    config = {**_config(), "benchmark_map": {".T": "^N225", "": "SPY"}, **configured}
    result = simulate([Decision("2026-01-05", t, "Buy") for t in tickers], config, starting_cash=1_000,
                      end_date="2026-01-09", closes=prices)
    assert result.metrics["benchmark"] == expected


@pytest.mark.unit
def test_simulate_without_a_benchmark_flag_uses_the_configured_one(tmp_path, monkeypatch):
    flat = dict(zip(DAYS, [100] * 5, strict=True))
    monkeypatch.setattr(sim, "get_closes", _closes({"AAPL": flat, "QQQ": flat}))
    monkeypatch.setitem(m.DEFAULT_CONFIG, "benchmark_ticker", "QQQ")
    log = _one_buy_log(tmp_path / "trading_memory.md")
    result = CliRunner().invoke(m.app, ["simulate", "--log", str(log), "--end", "2026-01-09",
                                        "--out", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert "Benchmark QQQ" in result.output


@pytest.mark.unit
@pytest.mark.parametrize("flag, passed", [([], ...), (["--benchmark", "QQQ"], "QQQ"), (["--benchmark", ""], None)])
def test_backtest_simulate_takes_a_benchmark(flag, passed, tmp_path, monkeypatch):
    log = _one_buy_log(tmp_path / "sweep" / "trading_memory.md")
    swept = SimpleNamespace(cells_run=1, skipped=0, log_path=log, run_id="r", failures=[], settlement_failures=[])
    monkeypatch.setattr(m, "run_backtest", lambda *a, **k: swept)
    monkeypatch.setattr(m, "summarize", lambda result: SimpleNamespace(render=lambda: ""))
    seen = {}

    def fake_simulate(decisions, config, **kwargs):
        seen["benchmark"] = kwargs["benchmark"]
        raise ValueError("stop here")

    monkeypatch.setattr(m, "simulate_decisions", fake_simulate)
    result = CliRunner().invoke(m.app, ["backtest", "NVDA", "--start", "2026-01-05", "--end", "2026-01-05",
                                        "--simulate", *flag])
    assert seen == {"benchmark": passed}, result.output
