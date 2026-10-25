# Trading guide: backtests, simulation, paper and live

The agents produce a **rating**, not an order. This guide covers how a rating becomes an
order, and the four places that order can go:

| Venue | Command | Money | What it answers |
|---|---|---|---|
| Backtest scoring | `tradingagents backtest` | none | Are the ratings right, call by call? |
| Simulation | `tradingagents simulate` / `backtest --simulate` | simulated | What would an account trading those ratings have earned? |
| Paper account | `tradingagents paper` | simulated, kept on disk | How does it trade day to day, going forward? |
| Broker | `tradingagents trade` | Alpaca paper, or live with an explicit unlock | The real thing |

All four use the same sizing rule, so a strategy tested in simulation sizes the same way
when it trades.

## Ratings

The portfolio manager ends every decision with one of five ratings:

| Rating | Meaning | Direction |
|---|---|---|
| **Buy** | Strong conviction the price rises | long, full size |
| **Overweight** | Lean long | long, reduced size |
| **Hold** | No change warranted | keep what you hold |
| **Underweight** | Lean against: trim | reduce, never add |
| **Sell** | Exit | flat |
| *REVIEW* | The decision had no readable rating | never trades; read the report |

## From rating to order

Every rating targets a share of one **per-position cap**: `max_position_pct` of current
equity, 10% by default.

```
target units = equity × max_position_pct × weight(rating) ÷ price      (rounded down unless fractional)
order        = target units − units held now
```

| Rating | weight |
|---|---|
| Buy | 1.00 |
| Overweight | 0.75 |
| Underweight | 0.25, and only ever trims |
| Sell | 0.00 |
| Hold, REVIEW | no order |

**Worked example.** Equity $100,000, cap 10%, AAPL at $250:

- **Buy** with no AAPL held: target = 100,000 × 0.10 × 1.00 ÷ 250 = 40 shares → **buy 40**.
- Later **Overweight**, holding 40: target = 30 → **sell 10**. Overweight is a smaller
  position than Buy.
- Later **Underweight**, holding 30: target = 10 → **sell 20**.
- **Underweight** while holding 5: the target of 10 is above what's held, and
  Underweight never adds, so **no order**.
- **Sell**: target 0 → **sell everything**.
- **Hold**: no order, whatever is held.

Two rules keep the account from churning:

- **Minimum trade.** A rebalance worth less than `min_trade_pct` of equity (0.5%) is
  skipped. A full exit always goes through.
- **Sells before buys.** When several tickers trade in the same session, sells go first,
  so the cash they free can fund that session's buys.

To change any of this, edit `execution` in `tradingagents/default_config.py`, or pass
`--max-position` / `--cash`. The [configuration reference](configuration.md#execution-turning-ratings-into-orders)
lists every key.

## 1. Backtest: score the calls

```bash
tradingagents backtest NVDA,AAPL --start 2026-03-01 --end 2026-09-01 --every 7
```

This runs the full agent pipeline for every (ticker, date) pair in the grid: here,
weekly. Each decision is written to that sweep's own log at
`results_dir/backtest/<run_id>/trading_memory.md`. Your live memory log is never touched.
A decision is scored once `holding_period_days` (5) sessions have traded after it.

The summary shows, per rating, the count, the **hit rate** (share of directional calls
whose alpha had the right sign) and the **mean alpha** against the benchmark.

- **Cost.** Every cell is a full LLM run. Two tickers weekly for six months is about 52
  runs. Start small, with `--analysts market,news` and a short range.
- **Resume.** An interrupted sweep continues with `--run-id <id>`; finished cells are
  skipped.
- **No look-ahead.** A run dated in the past reads only data published by that date:
  prices, filings as filed, news before the date.

## 2. Simulate: trade the calls through an account

```bash
tradingagents backtest NVDA,AAPL --start 2026-03-01 --end 2026-09-01 --every 7 --simulate
# or replay any finished sweep, as often as you like, without new LLM calls:
tradingagents simulate --run-id 20260901_101500
tradingagents simulate --run-id 20260901_101500 --cash 25000 --max-position 0.25
tradingagents simulate --log ~/.tradingagents/memory/trading_memory.md      # your live log
```

The simulator starts a paper account (all cash, or holding a backtest's `--portfolio` book,
valued at the last close before the first decision) and walks forward one session at a time:

1. **Entry timing.** A decision dated D executes at the **first close after D**. A run
   dated D reads news published after D's close, so trading at D's own close would use
   information that wasn't public yet.
2. **Fill price.** That close, plus slippage on a buy or minus slippage on a sell, plus
   commission.
3. **Carrying.** A position stays open, marked to market every session, until a later
   decision on that ticker changes it.
4. **Collisions.** Two decisions landing on the same session (a weekend date and Monday's)
   collapse to the later one.
5. **Caps.** Orders are held to the same per-position and per-order caps as a paper or
   broker session (see [the per-order cap](#the-per-order-cap)).

The benchmark defaults to the one settlement uses: `benchmark_ticker` if set, else the
`benchmark_map` index all the tickers share, else `SPY`. `--benchmark` overrides it.

It prints the return, CAGR, volatility, Sharpe, maximum drawdown, trade count,
commissions, average exposure, and the benchmark's buy-and-hold over the same sessions.
It also saves three files to `results_dir/simulations/<name>/`, where the dashboard finds
them (`--out` puts them elsewhere):

| File | Contents |
|---|---|
| `equity.csv` | `date, equity, benchmark`: one row per session, the benchmark rebased to starting cash |
| `fills.csv` | Every fill: ticker, side, quantity, price, commission, date, status |
| `metrics.json` | The summary figures above |

```python
from tradingagents.execution import decisions_from_log, simulate

sim = simulate(decisions_from_log("path/to/trading_memory.md"), config, starting_cash=50_000)
print(sim.render())
sim.equity          # pandas Series of account value by session
sim.fills           # list of Fill
sim.metrics         # dict
sim.write("out/")   # the three files
```

A simulation is indicative, not a promise. Each decision is one sample of a language
model, so another run could rate differently; and fills at the close with a fixed
slippage are optimistic for large or illiquid orders.

## 3. Paper account: trade forward without money

```bash
tradingagents paper NVDA,AAPL                     # analyze, size, queue for the next close
tradingagents paper NVDA,AAPL --dry-run           # show the plan, queue nothing
tradingagents paper --settle                      # fill pending orders whose close is in
tradingagents paper --status                      # cash, positions, pending orders, fill count
tradingagents paper MSFT --account swing --cash 25000   # a second, separate account
```

Each session does three things:

1. **Analyze** every ticker. The agents see the paper account's cash and positions as
   their portfolio context, so they know what is already held.
2. **Plan** an order per rating, sized at the latest close, held to the per-order cap
   and to the cash on hand plus what the session's sells free.
3. **Queue** the orders. Each fills, sells before buys, at the **first close after the
   analysis date**, with slippage and commission: the same entry rule as the simulator,
   so a paper fill never uses the close the analysis already read. A session dated in
   the past fills at once; today's orders fill on the next run after the close, or with
   `paper --settle`. A bar still in progress is never taken for a close.

Each account is one JSON file, `results_dir/paper/<account>.json`, holding its cash,
starting cash, positions with average prices, pending orders, split adjustments and every
fill. Opening an account applies any stock split since a holding's last fill, so the
ledger stays in the split-adjusted prices it is marked at. Tickers are canonical
(`BTCUSD` and `BTC-USD` are one position), and the file is locked while it is written,
so two sessions on one account both keep their fills. Run the same command daily, for
example from cron, and the account builds a real track record. `--cash` funds only a new
account; to start over, delete the file.

```bash
# every weekday at 16:30, after the US close
30 16 * * 1-5  cd ~/TradingAgents && .venv/bin/tradingagents paper NVDA,AAPL,MSFT >> paper.log 2>&1
```

## 4. Broker: Alpaca paper, then live

[Alpaca](https://alpaca.markets) gives every account a free **paper** account that uses
the same API as live. The paper account is the right place to prove a session end to
end.

### Setup

1. Create an Alpaca account. In the dashboard, switch to **Paper Trading** and generate
   API keys.
2. Put them in your `.env`:
   ```bash
   APCA_API_KEY_ID=PK...
   APCA_API_SECRET_KEY=...
   ```
3. Try a dry run, which analyzes and plans but sends nothing:
   ```bash
   tradingagents trade AAPL --dry-run
   ```
4. Trade the paper account. It shows the planned orders and asks before sending:
   ```bash
   tradingagents trade AAPL,MSFT
   ```

The agents see your Alpaca positions and cash as their portfolio context, just as with a
local paper account.

### Live trading

Live trading uses your live account's keys, which are different from the paper keys,
and needs **two** separate confirmations:

```bash
TRADINGAGENTS_LIVE_TRADING=1 tradingagents trade AAPL --live
```

| Safeguard | What it stops |
|---|---|
| `--live` **and** `TRADINGAGENTS_LIVE_TRADING=1` | A habitual flag or a copied config placing a real order alone |
| Confirmation prompt (unless `--yes`) | Sending a plan you haven't read |
| Per-order cap, applied while planning | An order larger than `max_order_pct` of equity (or `max_order_notional`); the plan you confirm already shows the cut |
| Cash check, applied while planning and again after the sells | Buying on margin the plan didn't show; a refused or unfilled sell frees no cash |
| Open orders counted while planning | A re-run (say, the morning after an order queued overnight) sending the same trade twice |
| Unique client order id per attempt | Alpaca refusing a retry after a rejected order |
| Blocked-account check | Trading an account Alpaca has restricted |

Orders are **market** orders (**day** for stocks, **gtc** for crypto) sized at the latest
close, with quantities sent exactly, never rounded up. They fill at the market's price,
which may differ, especially when sent outside market hours, where they queue for the
next open. `--yes` sends without asking, for scheduled jobs; use it only once the plan is
something you trust unattended.

### The per-order cap

No single order may exceed `max_order_pct` of equity (2% by default), or
`max_order_notional` dollars when that is set; the lower wins. A larger buy is cut to the
cap, so a position is built over several sessions: with a 10% position cap, a Buy fills in
about five, whatever the account's size. A sell that only reduces a position is never
capped, so you can always get out. Paper sessions and simulations apply the same cap, so
their results show what a broker session would actually send. Set `max_order_pct` to the
`max_position_pct` to fill a Buy in one order, keeping the cap only as a guard.

## Choosing between them

1. **Backtest** a few tickers over a few months to see if the ratings have any edge.
2. **Simulate** that sweep under different caps and costs; it's free.
3. Open the **dashboard** and study signal quality: hit rate by rating, information
   coefficient, decay. See [dashboard guide](dashboard.md).
4. **Paper trade** forward for weeks. Out-of-sample, forward-dated results are the only
   ones that can't have leaked.
5. Move to **Alpaca paper**, then **live**, with a small `max_order_notional` at first.

TradingAgents is a research framework. Nothing it produces is financial advice.
