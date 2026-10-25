"""Execution: ratings sized into orders, filled on paper, replayed, and sent to a broker.

Every price here is synthetic and every broker is local or faked: nothing in
this file reaches a venue or the network.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

import cli.main as m
from tradingagents.execution import (
    BrokerError,
    CostModel,
    Decision,
    Order,
    PaperBroker,
    SizingRules,
    alpaca,
    decisions_from_log,
    order_cap,
    plan_order,
    run_session,
    simulate,
    target_quantity,
)
from tradingagents.memory import TradingMemoryLog

FREE = CostModel(slippage_bps=0, commission_per_order=0, commission_pct=0)
RULES = SizingRules(max_position_pct=0.10)


# -- sizing ------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("rating, units", [("Buy", 100), ("Overweight", 75), ("Sell", 0)])
def test_each_rating_targets_its_share_of_the_position_cap(rating, units):
    # 10% of 100,000 at 100 a unit is 100 units at full weight.
    current = 40 if rating == "Sell" else 0
    assert target_quantity(rating, 100_000, 100.0, current, RULES) == units


@pytest.mark.unit
@pytest.mark.parametrize("rating", ["Hold", "REVIEW"])
def test_hold_and_review_leave_the_position_alone(rating):
    assert target_quantity(rating, 100_000, 100.0, 37, RULES) is None
    assert plan_order("AAPL", rating, 100_000, 100.0, 37, RULES) is None


@pytest.mark.unit
def test_a_bearish_rating_never_grows_a_position():
    # A quarter of the cap is 25 units; holding 10, Underweight must not buy 15.
    assert plan_order("AAPL", "Underweight", 100_000, 100.0, 10, RULES) is None
    assert plan_order("AAPL", "Underweight", 100_000, 100.0, 60, RULES) == Order("AAPL", "sell", 35)


@pytest.mark.unit
def test_a_trade_below_the_floor_is_skipped_but_a_full_exit_is_not():
    rules = SizingRules(max_position_pct=0.10, min_trade_pct=0.005)
    assert plan_order("AAPL", "Buy", 100_000, 100.0, 98, rules) is None      # 200 of 100,000
    assert plan_order("AAPL", "Sell", 100_000, 100.0, 2, rules) == Order("AAPL", "sell", 2)


@pytest.mark.unit
def test_target_weights_can_be_overridden_per_tier():
    rules = SizingRules(max_position_pct=0.10, target_weights={"Overweight": 0.5})
    assert target_quantity("Overweight", 100_000, 100.0, 0, rules) == 50


@pytest.mark.unit
def test_a_negative_target_weight_needs_allow_short():
    # A negative weight is the only way a rating asks for a short.
    with pytest.raises(ValueError, match="allow_short"):
        SizingRules.from_config({"execution": {"target_weights": {"Sell": -0.5}}})
    rules = SizingRules.from_config({"execution": {"target_weights": {"Sell": -0.5}, "allow_short": True}})
    assert plan_order("AAPL", "Sell", 100_000, 100.0, 0, rules) == Order("AAPL", "sell", 50)


@pytest.mark.unit
def test_a_long_only_plan_never_sells_more_than_is_held():
    # Rules built around the check still cannot plan a sale beyond the position.
    rules = SizingRules(max_position_pct=0.10)
    object.__setattr__(rules, "target_weights", {"Sell": -0.5})
    assert plan_order("AAPL", "Sell", 100_000, 100.0, 20, rules) == Order("AAPL", "sell", 20)
    assert plan_order("AAPL", "Sell", 100_000, 100.0, 0, rules) is None


# -- paper broker ------------------------------------------------------------------

@pytest.mark.unit
def test_paper_fills_pay_slippage_and_commission():
    broker = PaperBroker(cash=10_000, costs=CostModel(slippage_bps=10, commission_per_order=1.0))
    fill = broker.submit(Order("AAPL", "buy", 10), 100.0, "2026-01-05")
    assert fill.price == pytest.approx(100.1)
    assert broker.cash == pytest.approx(10_000 - 1001 - 1)
    sell = broker.submit(Order("AAPL", "sell", 10), 100.0, "2026-01-06")
    assert sell.price == pytest.approx(99.9)
    assert broker.quantity("AAPL") == 0


@pytest.mark.unit
def test_a_buy_is_cut_to_the_cash_and_a_sell_to_the_position():
    broker = PaperBroker(cash=1_000, costs=FREE)
    assert broker.submit(Order("AAPL", "buy", 50), 100.0, "2026-01-05").quantity == 10
    assert broker.submit(Order("AAPL", "sell", 50), 100.0, "2026-01-06").quantity == 10
    rejected = broker.submit(Order("AAPL", "sell", 5), 100.0, "2026-01-07")
    assert rejected.status == "rejected" and rejected.quantity == 0


@pytest.mark.unit
def test_average_price_moves_on_adds_and_holds_on_trims():
    broker = PaperBroker(cash=10_000, costs=FREE)
    broker.submit(Order("AAPL", "buy", 10), 100.0, "d1")
    broker.submit(Order("AAPL", "buy", 10), 200.0, "d2")
    assert broker.portfolio().position_in("AAPL").average_price == 150
    broker.submit(Order("AAPL", "sell", 5), 300.0, "d3")
    assert broker.portfolio().position_in("AAPL").average_price == 150
    assert broker.equity({"AAPL": 300.0}) == pytest.approx(broker.cash + 15 * 300)


@pytest.mark.unit
def test_a_paper_account_persists_between_sessions(tmp_path):
    path = tmp_path / "paper" / "main.json"
    first = PaperBroker.open(path, starting_cash=5_000, costs=FREE)
    first.submit(Order("MSFT", "buy", 3), 400.0, "2026-01-05")

    again = PaperBroker.open(path, starting_cash=999_999, costs=FREE)  # funding applies only to a new account
    assert again.cash == pytest.approx(3_800)
    assert again.quantity("MSFT") == 3
    assert again.starting_cash == 5_000
    assert len(again.fills) == 1


def _splits(table):
    return lambda ticker: pd.Series({pd.Timestamp(d): r for d, r in table.get(ticker, {}).items()}, dtype=float)


@pytest.mark.unit
def test_a_split_since_the_last_fill_is_applied_once_when_the_account_opens(tmp_path):
    path = tmp_path / "main.json"
    broker = PaperBroker.open(path, starting_cash=100_000, costs=FREE)
    broker.submit(Order("NVDA", "buy", 10), 1_000.0, "2026-01-05")
    # A 10:1 split after the fill, and one before it that the fill already reflects.
    splits = _splits({"NVDA": {"2025-06-02": 4.0, "2026-01-08": 10.0}})

    opened = PaperBroker.open(path, starting_cash=0, costs=FREE, splits=splits)
    held = opened.portfolio().position_in("NVDA")
    assert (held.quantity, held.average_price) == (100, 100)
    assert opened.equity({"NVDA": 100.0}) == pytest.approx(100_000)       # not 91,000
    assert opened.adjustments == [{"ticker": "NVDA", "date": "2026-01-08", "ratio": 10.0}]

    again = PaperBroker.open(path, starting_cash=0, costs=FREE, splits=splits)
    assert again.quantity("NVDA") == 100                                    # applied once, not twice
    # The Sell rating exits the 100 post-split units at no loss.
    assert again.submit(Order("NVDA", "sell", 100), 100.0, "2026-01-09").quantity == 100
    assert again.cash == pytest.approx(100_000)


@pytest.mark.unit
def test_split_data_that_cannot_be_read_does_not_lock_the_account(tmp_path):
    path = tmp_path / "main.json"
    PaperBroker.open(path, starting_cash=1_000, costs=FREE).submit(Order("NVDA", "buy", 1), 10.0, "2026-01-05")

    def down(ticker):
        raise RuntimeError("yahoo is down")

    assert PaperBroker.open(path, starting_cash=0, splits=down).quantity("NVDA") == 1


@pytest.mark.unit
def test_two_sessions_on_one_paper_account_both_keep_their_fills(tmp_path):
    path = tmp_path / "main.json"
    first = PaperBroker.open(path, starting_cash=100_000, costs=FREE)
    second = PaperBroker.open(path, starting_cash=100_000, costs=FREE)   # opened before first trades
    first.submit(Order("AAPL", "buy", 100), 100.0, "2026-01-05")
    second.submit(Order("MSFT", "buy", 100), 100.0, "2026-01-05")
    saved = PaperBroker.open(path, starting_cash=0, costs=FREE)
    assert saved.cash == pytest.approx(80_000)
    assert [f.ticker for f in saved.fills] == ["AAPL", "MSFT"]


@pytest.mark.unit
def test_concurrent_writers_never_lose_a_fill_or_trip_over_a_temp_file(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "main.json"
    PaperBroker.open(path, starting_cash=1_000_000, costs=FREE)

    def trade(worker):
        broker = PaperBroker.open(path, starting_cash=0, costs=FREE)
        for _ in range(25):
            broker.submit(Order(f"T{worker}", "buy", 1), 10.0, "2026-01-05")

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(trade, range(8)))          # re-raises any worker's error
    saved = PaperBroker.open(path, starting_cash=0, costs=FREE)
    assert len(saved.fills) == 200 and saved.cash == pytest.approx(1_000_000 - 2_000)


@pytest.mark.unit
def test_an_order_that_cannot_be_saved_is_not_applied(tmp_path, monkeypatch):
    path = tmp_path / "main.json"
    broker = PaperBroker.open(path, starting_cash=1_000, costs=FREE)
    monkeypatch.setattr(PaperBroker, "_write", lambda self: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        broker.submit(Order("AAPL", "buy", 1), 10.0, "2026-01-05")
    assert broker.cash == 1_000 and broker.fills == [] and broker.quantity("AAPL") == 0


@pytest.mark.unit
def test_an_unreadable_paper_account_is_reported(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(BrokerError, match="not readable"):
        PaperBroker.open(path, starting_cash=1)


# -- simulator ---------------------------------------------------------------------

def _closes(table: dict[str, dict[str, float]]):
    """A close source over a fixed table, honouring get_closes' exclusive end."""
    def closes(ticker, start, end):
        rows = table.get(ticker, {})
        series = pd.Series({pd.Timestamp(d): v for d, v in rows.items()}, dtype=float)
        if series.empty:
            return series
        return series[(series.index >= pd.Timestamp(start)) & (series.index < pd.Timestamp(end))]
    return closes


def _config(**execution):
    return {"execution": {"max_position_pct": 0.5, "slippage_bps": 0, "min_trade_pct": 0, **execution}}


DAYS = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]


@pytest.mark.unit
def test_a_decision_fills_at_the_first_close_after_its_date():
    prices = _closes({"AAPL": dict(zip(DAYS, [100, 110, 120, 130, 140], strict=True))})
    result = simulate([Decision("2026-01-05", "AAPL", "Buy")], _config(), starting_cash=10_000,
                      end_date="2026-01-09", benchmark=None, closes=prices)
    fill = result.fills[0]
    assert (fill.date, fill.price, fill.quantity) == ("2026-01-06", 110, 45)   # half of 10,000 at 110
    # Carried and marked each session: 45 units, 5,050 cash.
    assert result.equity.loc["2026-01-09"] == pytest.approx(5_050 + 45 * 140)
    assert result.metrics["trades"] == 1


@pytest.mark.unit
def test_a_simulation_is_held_to_the_same_per_order_cap():
    prices = _closes({"AAPL": dict(zip(DAYS, [100] * 5, strict=True))})
    decisions = [Decision(d, "AAPL", "Buy") for d in DAYS[:3]]
    result = simulate(decisions, _config(max_order_pct=0.1), starting_cash=10_000,
                      end_date="2026-01-09", benchmark=None, closes=prices)
    # Half the book is the target; 10% of equity a session builds it in steps.
    assert [f.quantity for f in result.fills] == [10, 10, 10]


@pytest.mark.unit
def test_hold_carries_and_sell_exits():
    prices = _closes({"AAPL": dict(zip(DAYS, [100, 100, 100, 100, 100], strict=True))})
    decisions = [Decision("2026-01-05", "AAPL", "Buy"), Decision("2026-01-06", "AAPL", "Hold"),
                 Decision("2026-01-07", "AAPL", "Sell")]
    result = simulate(decisions, _config(), starting_cash=10_000, end_date="2026-01-09",
                      benchmark=None, closes=prices)
    assert [(f.date, f.side, f.quantity) for f in result.fills] == [
        ("2026-01-06", "buy", 50), ("2026-01-08", "sell", 50)]


@pytest.mark.unit
def test_sells_fund_the_same_sessions_buys():
    # The account is fully in AAPL; a Sell of AAPL and a Buy of MSFT land on one session.
    table = {"AAPL": dict(zip(DAYS, [100] * 5, strict=True)), "MSFT": dict(zip(DAYS, [100] * 5, strict=True))}
    decisions = [Decision("2026-01-05", "AAPL", "Buy"), Decision("2026-01-06", "AAPL", "Sell"),
                 Decision("2026-01-06", "MSFT", "Buy")]
    result = simulate(decisions, _config(max_position_pct=1.0), starting_cash=1_000,
                      end_date="2026-01-09", benchmark=None, closes=_closes(table))
    day = [(f.ticker, f.side, f.quantity) for f in result.fills if f.date == "2026-01-07"]
    assert day == [("AAPL", "sell", 10), ("MSFT", "buy", 10)]


@pytest.mark.unit
def test_metrics_and_benchmark():
    table = {"AAPL": dict(zip(DAYS, [100, 100, 50, 100, 100], strict=True)),
             "SPY": dict(zip(DAYS, [400, 400, 400, 400, 440], strict=True))}
    result = simulate([Decision("2026-01-05", "AAPL", "Buy")], _config(max_position_pct=1.0),
                      starting_cash=1_000, end_date="2026-01-09", benchmark="SPY", closes=_closes(table))
    assert result.metrics["max_drawdown"] == pytest.approx(-0.5)
    assert result.metrics["total_return"] == pytest.approx(0.0)
    # Buy-and-hold from the first session the account existed (400) to 440.
    assert result.metrics["benchmark_return"] == pytest.approx(0.10)
    assert "Benchmark SPY" in result.render()


@pytest.mark.unit
def test_a_lowercase_ticker_is_marked_to_market_like_its_canonical_form():
    prices = _closes({"NVDA": dict(zip(DAYS, [100, 100, 110, 120, 130], strict=True))})
    run = lambda ticker: simulate([Decision("2026-01-05", ticker, "Buy")], _config(),  # noqa: E731
                                  starting_cash=10_000, end_date="2026-01-09", benchmark=None, closes=prices)
    lower, upper = run("nvda"), run("NVDA")
    assert lower.equity.tolist() == upper.equity.tolist()
    assert lower.metrics["total_return"] > 0


@pytest.mark.unit
def test_backtest_hands_the_sweep_canonical_tickers(monkeypatch):
    seen = {}

    def fake_run_backtest(names, dates, config, **kwargs):
        seen["names"] = names
        raise ValueError("stop here")

    monkeypatch.setattr(m, "run_backtest", fake_run_backtest)
    result = CliRunner().invoke(m.app, ["backtest", "nvda,btcusd", "--start", "2026-01-05",
                                        "--end", "2026-01-05", "--simulate"])
    assert seen["names"] == ["NVDA", "BTC-USD"], result.output


@pytest.mark.unit
def test_a_decision_with_no_later_close_is_reported_not_traded():
    prices = _closes({"AAPL": {"2026-01-05": 100, "2026-01-06": 101}})
    result = simulate([Decision("2026-01-05", "AAPL", "Buy"), Decision("2026-01-06", "AAPL", "Sell")],
                      _config(), starting_cash=1_000, end_date="2026-01-06", benchmark=None, closes=prices)
    assert result.unpriced == [Decision("2026-01-06", "AAPL", "Sell")]


@pytest.mark.unit
def test_results_are_written_beside_each_other(tmp_path):
    prices = _closes({"AAPL": dict(zip(DAYS, [100] * 5, strict=True))})
    result = simulate([Decision("2026-01-05", "AAPL", "Buy")], _config(), starting_cash=1_000,
                      end_date="2026-01-09", benchmark=None, closes=prices)
    out = result.write(tmp_path / "sim")
    assert {p.name for p in out.iterdir()} == {"equity.csv", "fills.csv", "metrics.json"}
    assert json.loads((out / "metrics.json").read_text())["trades"] == 1


@pytest.mark.unit
def test_decisions_are_read_from_a_memory_log_without_review(tmp_path):
    path = tmp_path / "trading_memory.md"
    log = TradingMemoryLog({"memory_log_path": str(path)})
    log.store_decision("NVDA", "2026-01-06", "Rating: Buy\n\nup")
    log.store_decision("AAPL", "2026-01-05", "no rating here")
    log.store_decision("AAPL", "2026-01-07", "Rating: Sell\n\ndown")
    assert decisions_from_log(path) == [Decision("2026-01-06", "NVDA", "Buy"),
                                        Decision("2026-01-07", "AAPL", "Sell")]


# -- Alpaca ------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._payload, self.text = status, payload, json.dumps(payload)

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, routes):
        self.routes, self.headers, self.sent = routes, {}, []

    def request(self, method, url, timeout=None, json=None, params=None):
        self.sent.append((method, url, json))
        path = url.split(".markets", 1)[1]
        route = self.routes[(method, path)]
        # A callable route answers per request, e.g. refusing only the sells.
        return _FakeResponse(*(route(json) if callable(route) else route))

    def orders(self):
        return [body for method, url, body in self.sent if method == "POST"]


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    monkeypatch.delenv(alpaca.LIVE_UNLOCK_ENV, raising=False)


@pytest.mark.unit
def test_live_trading_needs_the_environment_unlock(keys, monkeypatch):
    with pytest.raises(BrokerError, match="live trading is locked"):
        alpaca.AlpacaBroker(paper=False, session=_FakeSession({}))
    monkeypatch.setenv(alpaca.LIVE_UNLOCK_ENV, "1")
    assert alpaca.AlpacaBroker(paper=False, session=_FakeSession({})).base_url == alpaca.LIVE_URL


@pytest.mark.unit
def test_paper_is_the_default_and_keys_are_required(keys, monkeypatch):
    assert alpaca.AlpacaBroker(session=_FakeSession({})).base_url == alpaca.PAPER_URL
    monkeypatch.delenv("APCA_API_KEY_ID")
    with pytest.raises(BrokerError, match="keys are missing"):
        alpaca.AlpacaBroker(session=_FakeSession({}))


@pytest.mark.unit
def test_alpaca_reports_the_book_and_places_a_market_order(keys):
    session = _FakeSession({
        ("GET", "/v2/account"): (200, {"cash": "900.5", "equity": "1900.5", "currency": "USD"}),
        ("GET", "/v2/positions"): (200, [{"symbol": "AAPL", "qty": "5", "avg_entry_price": "200"}]),
        ("POST", "/v2/orders"): (200, {"id": "abc", "status": "accepted", "filled_qty": "0",
                                       "filled_avg_price": None}),
    })
    broker = alpaca.AlpacaBroker(session=session)
    book = broker.portfolio()
    assert (book.cash, book.position_in("AAPL").quantity) == (900.5, 5)
    assert broker.equity({}) == 1900.5

    fill = broker.submit(Order("aapl", "buy", 2), 210.0, "2026-01-05")
    assert (fill.status, fill.order_id, fill.quantity) == ("accepted", "abc", 2)
    body = dict(session.sent[-1][2])
    assert body.pop("client_order_id").startswith("tradingagents-")
    assert body == {"symbol": "AAPL", "qty": "2", "side": "buy", "type": "market", "time_in_force": "day"}


@pytest.mark.unit
@pytest.mark.parametrize("quantity, sent", [(12.3456789, "12.3456789"), (1_234_567, "1234567"),
                                            (2.0, "2"), (0.1234567891234, "0.123456789")])
def test_alpaca_quantities_are_sent_exactly_and_never_rounded_up(keys, quantity, sent):
    session = _FakeSession({("POST", "/v2/orders"): (200, {"id": "abc", "status": "accepted"})})
    alpaca.AlpacaBroker(session=session).submit(Order("AAPL", "sell", quantity), 1.0, "2026-01-05")
    assert session.orders()[0]["qty"] == sent


@pytest.mark.unit
@pytest.mark.parametrize("ticker, symbol, tif", [("BRK-B", "BRK.B", "day"), ("BTC-USD", "BTC/USD", "gtc"),
                                                 ("AAPL", "AAPL", "day")])
def test_alpaca_orders_use_its_symbols_and_crypto_is_good_till_cancelled(keys, ticker, symbol, tif):
    session = _FakeSession({("POST", "/v2/orders"): (200, {"id": "abc", "status": "accepted"})})
    fill = alpaca.AlpacaBroker(session=session).submit(Order(ticker, "buy", 1), 1.0, "2026-01-05")
    assert (session.orders()[0]["symbol"], session.orders()[0]["time_in_force"]) == (symbol, tif)
    assert fill.ticker == ticker


@pytest.mark.unit
def test_alpaca_positions_and_open_orders_come_back_as_yahoo_tickers(keys):
    session = _FakeSession({
        ("GET", "/v2/account"): (200, {"cash": "100", "equity": "100"}),
        ("GET", "/v2/positions"): (200, [{"symbol": "BRK.B", "qty": "2", "avg_entry_price": "400"},
                                         {"symbol": "BTCUSD", "qty": "0.5", "avg_entry_price": "60000",
                                          "asset_class": "crypto"}]),
        ("GET", "/v2/orders"): (200, [{"symbol": "ETH/USD", "side": "buy", "qty": "1", "filled_qty": "0"}]),
    })
    broker = alpaca.AlpacaBroker(session=session)
    book = broker.portfolio()
    assert book.position_in("BRK-B").quantity == 2 and book.position_in("BTC-USD").quantity == 0.5
    assert broker.open_orders() == [Order("ETH-USD", "buy", 1)]


@pytest.mark.unit
def test_every_order_attempt_gets_its_own_client_order_id(keys):
    # Alpaca refuses a reused id, so a retry after a rejection must not reuse one.
    session = _FakeSession({("POST", "/v2/orders"): (200, {"id": "abc", "status": "accepted"})})
    broker = alpaca.AlpacaBroker(session=session)
    for _ in range(2):
        broker.submit(Order("AAPL", "buy", 1), 1.0, "2026-01-05")
    first, second = (o["client_order_id"] for o in session.orders())
    assert first != second and first.rsplit("-", 1)[0].endswith("-AAPL-buy")


@pytest.mark.unit
def test_an_order_over_the_notional_cap_is_not_sent(keys):
    session = _FakeSession({})
    broker = alpaca.AlpacaBroker(session=session, max_order_notional=1_000)
    with pytest.raises(BrokerError, match="per-order cap of 1,000"):
        broker.submit(Order("AAPL", "buy", 10), 200.0, "2026-01-05")
    assert session.sent == []


@pytest.mark.unit
def test_the_broker_backstop_also_holds_orders_to_a_share_of_equity(keys):
    session = _FakeSession({
        ("GET", "/v2/account"): (200, {"cash": "100000", "equity": "100000", "currency": "USD"}),
        ("GET", "/v2/positions"): (200, []),
    })
    broker = alpaca.AlpacaBroker(session=session, max_order_pct=0.02)
    broker.portfolio()
    with pytest.raises(BrokerError, match="per-order cap of 2,000"):
        broker.submit(Order("AAPL", "buy", 30), 100.0, "2026-01-05")


@pytest.mark.unit
def test_an_alpaca_refusal_carries_its_message(keys):
    session = _FakeSession({("POST", "/v2/orders"): (422, {"message": "client_order_id must be unique"})})
    with pytest.raises(BrokerError, match="must be unique"):
        alpaca.AlpacaBroker(session=session).submit(Order("AAPL", "buy", 1), 1.0, "2026-01-05")


# -- session -----------------------------------------------------------------------

class _FakeGraph:
    def __init__(self, ratings):
        self.ratings, self.seen = ratings, []

    def propagate(self, ticker, date, asset_type="stock", portfolio=None):
        self.seen.append((ticker, portfolio))
        rating = self.ratings[ticker]
        if isinstance(rating, Exception):
            raise rating
        return {}, rating


def _session(broker, ratings, **kwargs):
    graph = _FakeGraph(ratings)
    steps = run_session(list(ratings), broker, _config(), trade_date="2026-01-05",
                        price=lambda t, d: 100.0, graph=graph, **kwargs)
    return steps, graph


@pytest.mark.unit
def test_a_session_shows_the_agents_the_book_and_trades_the_plan():
    broker = PaperBroker(cash=10_000, costs=FREE)
    broker.submit(Order("AAPL", "buy", 20), 100.0, "2026-01-02")
    steps, graph = _session(broker, {"AAPL": "Sell", "MSFT": "Buy", "NVDA": "Hold"})

    assert graph.seen[0][1].position_in("AAPL").quantity == 20
    by_ticker = {s.ticker: s for s in steps}
    assert by_ticker["NVDA"].order is None
    assert broker.quantity("AAPL") == 0
    assert broker.quantity("MSFT") == 50        # half of 10,000 equity at 100
    # The AAPL sell went first.
    assert [f.ticker for f in broker.fills[1:]] == ["AAPL", "MSFT"]


@pytest.mark.unit
def test_a_dry_run_or_a_declined_plan_sends_nothing():
    broker = PaperBroker(cash=10_000, costs=FREE)
    steps, _ = _session(broker, {"MSFT": "Buy"}, submit=False)
    assert steps[0].order == Order("MSFT", "buy", 50) and steps[0].fill is None

    steps, _ = _session(broker, {"MSFT": "Buy"}, confirm=lambda plan: False)
    assert broker.fills == []


@pytest.mark.unit
def test_one_failed_analysis_does_not_cancel_the_other_trades():
    broker = PaperBroker(cash=10_000, costs=FREE)
    steps, _ = _session(broker, {"AAPL": RuntimeError("vendor down"), "MSFT": "Buy"})
    assert steps[0].error == "vendor down"
    assert broker.quantity("MSFT") == 50


def _alpaca(cash, equity, positions=(), cap=10_000, post=(200, {"id": "x", "status": "accepted"}),
            open_orders=()):
    session = _FakeSession({
        ("GET", "/v2/orders"): (200, list(open_orders)),
        ("GET", "/v2/account"): (200, {"cash": str(cash), "equity": str(equity), "currency": "USD"}),
        ("GET", "/v2/positions"): (200, [{"symbol": t, "qty": str(q), "avg_entry_price": "50"}
                                         for t, q in positions]),
        ("POST", "/v2/orders"): post,
    })
    return alpaca.AlpacaBroker(session=session, max_order_notional=cap), session


def _alpaca_session(broker, ratings, confirm=None):
    config = {"execution": {"max_position_pct": 0.10, "min_trade_pct": 0}}
    return run_session(list(ratings), broker, config, trade_date="2026-01-05",
                       price=lambda t, d: 50.0, graph=_FakeGraph(ratings), confirm=confirm)


@pytest.mark.unit
def test_the_plan_shown_for_confirmation_is_already_cut_to_the_notional_cap(keys):
    # 10% of 100,500 at 50 is 201 units, 10,050: over the 10,000 cap.
    broker, session = _alpaca(cash=100_500, equity=100_500)
    shown = []
    steps = _alpaca_session(broker, {"NVDA": "Buy"}, confirm=lambda plan: shown.extend(
        s.describe() for s in plan) or True)
    assert steps[0].order == Order("NVDA", "buy", 200) and steps[0].error is None
    assert "max_order_notional" in shown[0] and "buy 200" in shown[0]
    assert [o["qty"] for o in session.orders()] == ["200"]


@pytest.mark.unit
def test_a_large_account_is_capped_at_a_share_of_its_equity_not_a_fixed_sum(keys):
    # At 20M a Buy targets 2M; the 2% cap lets 400k of it through per session.
    broker, session = _alpaca(cash=20_000_000, equity=20_000_000, cap=None)
    config = {"execution": {"max_position_pct": 0.10, "min_trade_pct": 0, "max_order_pct": 0.02}}
    steps = run_session(["NVDA"], broker, config, trade_date="2026-01-05",
                        price=lambda t, d: 100.0, graph=_FakeGraph({"NVDA": "Buy"}))
    assert steps[0].order == Order("NVDA", "buy", 4_000)
    assert "max_order_pct cap of 2%" in steps[0].note
    assert [o["qty"] for o in session.orders()] == ["4000"]


@pytest.mark.unit
def test_the_lower_of_the_two_per_order_caps_wins():
    rules = SizingRules(max_order_pct=0.02)
    assert order_cap(rules, 1_000_000, max_order_notional=5_000) == (5_000, "max_order_notional cap")
    assert order_cap(rules, 100_000, max_order_notional=5_000)[0] == 2_000
    assert order_cap(SizingRules(), 100_000) is None


@pytest.mark.unit
def test_a_position_grown_past_the_cap_can_still_exit(keys):
    # 300 units at 50 is 15,000, over the cap, but selling it only reduces risk.
    broker, session = _alpaca(cash=2_000, equity=17_000, positions=[("AAPL", 300)])
    steps = _alpaca_session(broker, {"AAPL": "Sell"})
    assert steps[0].error is None and steps[0].note is None
    assert [(o["side"], o["qty"]) for o in session.orders()] == [("sell", "300")]


@pytest.mark.unit
def test_a_refused_sell_leaves_its_buys_cut_to_the_cash_actually_there(keys):
    refuse_sells = lambda body: ((403, {"message": "insufficient qty"}) if body["side"] == "sell"  # noqa: E731
                                 else (200, {"id": "x", "status": "accepted"}))
    broker, session = _alpaca(cash=2_000, equity=100_000, positions=[("AAPL", 300)], post=refuse_sells)
    steps = {s.ticker: s for s in _alpaca_session(broker, {"AAPL": "Sell", "MSFT": "Buy"})}
    assert "insufficient qty" in steps["AAPL"].error
    # The plan counted on the sell's 15,000; with it refused only 2,000 is there.
    assert steps["MSFT"].order == Order("MSFT", "buy", 40)
    assert [(o["side"], o["qty"]) for o in session.orders()] == [("sell", "300"), ("buy", "40")]


@pytest.mark.unit
def test_a_rerun_counts_the_orders_still_open_and_does_not_double_them(keys):
    # Last night's run queued a buy of 200 NVDA for the open; this morning's run
    # (a new date) must see it rather than send the same buy again.
    queued = {"symbol": "NVDA", "side": "buy", "qty": "200", "filled_qty": "0"}
    broker, session = _alpaca(cash=100_000, equity=100_000, open_orders=[queued])
    steps = _alpaca_session(broker, {"NVDA": "Buy"})
    assert steps[0].order is None and "+200 already on order" in steps[0].describe()
    assert session.orders() == []


@pytest.mark.unit
def test_a_session_sends_no_short_sale_unless_allow_short(keys):
    broker, session = _alpaca(cash=10_000, equity=10_000)
    config = {"execution": {"max_position_pct": 0.10, "target_weights": {"Sell": -1.0}}}
    with pytest.raises(ValueError, match="allow_short"):
        run_session(["AAPL"], broker, config, trade_date="2026-01-05",
                    price=lambda t, d: 50.0, graph=_FakeGraph({"AAPL": "Sell"}))
    assert session.orders() == []


@pytest.mark.unit
def test_a_plan_never_buys_more_than_the_cash_on_hand(keys):
    broker, session = _alpaca(cash=1_000, equity=100_000)
    steps = _alpaca_session(broker, {"MSFT": "Buy"})
    assert steps[0].order == Order("MSFT", "buy", 20) and "cash available" in steps[0].note


@pytest.mark.unit
def test_a_paper_session_fills_at_the_close_after_its_analysis_not_the_one_it_read(tmp_path):
    # The analysis of Jan 5 read Jan 5's close (100); it trades at Jan 6's (110).
    prices = _closes({"MSFT": dict(zip(DAYS, [100, 110, 120, 130, 140], strict=True))})
    broker = PaperBroker.open(tmp_path / "main.json", starting_cash=10_000, costs=FREE, fill_next_close=True)
    steps, _ = _session(broker, {"MSFT": "Buy"})
    assert steps[0].fill.status == "pending" and broker.fills == []

    settled = broker.settle(prices, through="2026-01-09")
    assert [(f.date, f.price, f.quantity) for f in settled] == [("2026-01-06", 110, 50)]   # sized at 100, filled at 110
    assert broker.pending == [] and broker.quantity("MSFT") == 50


@pytest.mark.unit
def test_a_pending_order_waits_for_its_close_and_is_not_ordered_twice(tmp_path):
    prices = _closes({"MSFT": {"2026-01-05": 100}})
    broker = PaperBroker.open(tmp_path / "main.json", starting_cash=10_000, costs=FREE, fill_next_close=True)
    _session(broker, {"MSFT": "Buy"})
    assert broker.settle(prices, through="2026-01-05") == []          # no close after Jan 5 yet
    again = PaperBroker.open(tmp_path / "main.json", starting_cash=0, costs=FREE, fill_next_close=True)
    steps, _ = _session(again, {"MSFT": "Buy"})
    assert steps[0].order is None and len(again.pending) == 1


@pytest.mark.unit
@pytest.mark.parametrize("clock, session", [("15:59", "2026-01-05"), ("16:45", "2026-01-06")])
def test_an_in_progress_bar_is_never_taken_for_a_close(clock, session):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from tradingagents.execution.paper import _last_closed_session

    now = datetime.fromisoformat(f"2026-01-06T{clock}").replace(tzinfo=ZoneInfo("America/New_York"))
    assert _last_closed_session(now) == session


@pytest.mark.unit
def test_paper_settle_fills_what_is_pending(tmp_path, monkeypatch):
    monkeypatch.setitem(m.DEFAULT_CONFIG, "results_dir", str(tmp_path))
    path = tmp_path / "paper" / "demo.json"
    queued = PaperBroker.open(path, starting_cash=10_000, costs=FREE, fill_next_close=True)
    queued.submit(Order("MSFT", "buy", 10), 100.0, "2026-01-05")
    monkeypatch.setattr(m, "get_closes", _closes({"MSFT": dict(zip(DAYS, [100, 110, 120, 130, 140], strict=True))}))
    monkeypatch.setattr(m, "get_splits", lambda ticker: pd.Series(dtype=float))
    result = CliRunner().invoke(m.app, ["paper", "--settle", "--account", "demo"])
    assert result.exit_code == 0, result.output
    assert "Filled pending buy 10 MSFT" in result.output and "2026-01-06 close" in result.output
    assert "pending:" not in result.output


# -- CLI ---------------------------------------------------------------------------

@pytest.mark.unit
def test_simulate_needs_exactly_one_source():
    result = CliRunner().invoke(m.app, ["simulate"])
    assert result.exit_code == 1 and "exactly one" in result.output


@pytest.mark.unit
def test_simulate_replays_a_log(tmp_path, monkeypatch):
    path = tmp_path / "trading_memory.md"
    TradingMemoryLog({"memory_log_path": str(path)}).store_decision("AAPL", "2026-01-05", "Rating: Buy\n\nup")
    import tradingagents.execution.simulator as sim

    monkeypatch.setattr(sim, "get_closes", _closes({"AAPL": dict(zip(DAYS, [100] * 5, strict=True))}))
    result = CliRunner().invoke(m.app, ["simulate", "--log", str(path), "--end", "2026-01-09",
                                        "--benchmark", "", "--out", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert "Trades: 1" in result.output
    assert (tmp_path / "out" / "equity.csv").is_file()


@pytest.mark.unit
def test_paper_status_shows_a_new_account(tmp_path, monkeypatch):
    monkeypatch.setitem(m.DEFAULT_CONFIG, "results_dir", str(tmp_path))
    result = CliRunner().invoke(m.app, ["paper", "--status", "--account", "demo", "--cash", "2500"])
    assert result.exit_code == 0, result.output
    assert "Cash: 2,500.00" in result.output
    assert (tmp_path / "paper" / "demo.json").is_file()


@pytest.mark.unit
def test_trade_refuses_live_without_the_unlock(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    monkeypatch.delenv(alpaca.LIVE_UNLOCK_ENV, raising=False)
    result = CliRunner().invoke(m.app, ["trade", "AAPL", "--live", "--yes"])
    assert result.exit_code == 1 and "live trading is locked" in result.output


@pytest.mark.unit
def test_simulate_saves_under_results_dir_by_default(tmp_path, monkeypatch):
    # The dashboard lists simulations found under results_dir, so that is where they go.
    path = tmp_path / "trading_memory.md"
    TradingMemoryLog({"memory_log_path": str(path)}).store_decision("AAPL", "2026-01-05", "Rating: Buy\n\nup")
    import tradingagents.execution.simulator as sim

    monkeypatch.setattr(sim, "get_closes", _closes({"AAPL": dict(zip(DAYS, [100] * 5, strict=True))}))
    monkeypatch.setitem(m.DEFAULT_CONFIG, "results_dir", str(tmp_path / "results"))
    result = CliRunner().invoke(m.app, ["simulate", "--log", str(path), "--end", "2026-01-09", "--benchmark", ""])
    assert result.exit_code == 0, result.output
    saved = list((tmp_path / "results" / "simulations").glob("trading_memory_*/metrics.json"))
    assert len(saved) == 1
