"""The execution commands: one instrument per symbol, and errors as one red line.

Like tests/test_execution.py, every price is synthetic and every broker local or
faked: nothing here reaches a venue or the network.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

import cli.main as m
import tradingagents.execution.simulator as sim
from tests.test_execution import DAYS, FREE, _closes, _config
from tradingagents.execution import PaperBroker, run_session
from tradingagents.memory import TradingMemoryLog


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
    assert seen == {"names": ["BTC-USD", "ETH-USD"], "asset_type": "crypto"}


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
