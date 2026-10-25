# Dashboard guide

The dashboard is an optional extra: `pip install "tradingagents[dashboard]"` installs
Streamlit, Plotly, scikit-learn and Chronos (with PyTorch).

```bash
tradingagents dashboard                       # opens http://localhost:8501
tradingagents dashboard --port 8502 --no-browser
tradingagents dashboard --results-dir /path/to/other/results
tradingagents dashboard --address 0.0.0.0     # share on your network; it shows your accounts
```

The dashboard reads what is already saved under `results_dir`. It never runs the agents
or places orders. Prices come from Yahoo Finance and are cached.

## Sidebar

| Control | What it does |
|---|---|
| **Decision log** | The ratings to study: your live memory log, or any backtest sweep. Drives the Signals, Signal quality and Research tabs. |
| **Account** | Whose trades to show: a saved simulation, a paper account, or *simulate the selected log*, which replays the chosen log live. |
| **Simulation settings** | Shown with *simulate the selected log*: starting cash, max position (% of equity), slippage (bps), and commission (bps of notional). They change the replay immediately and are not saved. |
| **Benchmark** | Ticker for comparison; by default the configured `benchmark_ticker`, else the market index the log's tickers share, else `SPY`. |

Nothing shows up? Run a `backtest`, a `simulate`, or a `paper` session first; see
[trading guide](trading.md). Simulations saved with `--out` outside `results_dir` are not
listed.

## Conventions used everywhere

- **Returns** are simple daily returns of the equity curve. Annualizing uses 252 sessions
  and a 0% risk-free rate.
- **Entry** for any decision is the first close after its date, the same rule as the
  simulator and backtest scoring.
- **Rating score** puts ratings on one axis for rank statistics: Buy +2, Overweight +1,
  Hold 0, Underweight −1, Sell −2. **Direction** is the sign: long, none, or short.
- **Excess** return means the stock's return minus the benchmark's over the same sessions.

## Portfolio tab

| Metric | Definition | How to read it |
|---|---|---|
| Total return | last equity ÷ first − 1 | |
| CAGR | (1 + total)^(1 / years) − 1 | Annualized growth; noisy over short periods |
| Volatility | stdev(daily returns) × √252 | Annualized risk |
| Sharpe | mean / stdev of daily returns × √252 | Above 1 is good, above 2 rare; under ~6 months it is very noisy |
| Sortino | mean × 252 ÷ downside deviation | Like Sharpe, but only losses count as risk |
| Max drawdown | worst fall from a running peak | The pain you would have sat through |
| Calmar | CAGR ÷ \|max drawdown\| | Return earned per unit of worst-case pain |
| Longest drawdown | most consecutive sessions below a prior peak | How long recovery took |
| Beta | cov(account, benchmark) ÷ var(benchmark) | Market exposure; 0.5 moves half as much as the market |
| Alpha | (mean account − β × mean benchmark) × 252 | Annualized return not explained by market exposure |
| Correlation, tracking error, information ratio | of daily active returns (account − benchmark) | IR = annualized active return ÷ tracking error |
| Skew, kurtosis, best/worst session, % positive sessions | of daily returns | Shape of the return distribution |

Charts: equity against the rebased benchmark (with a log-scale toggle), the drawdown
("underwater") curve, 63-session rolling Sharpe, 21-session rolling volatility, gross
exposure (share of equity in positions rather than cash), and a monthly-returns heatmap
with a full-year column.

Low exposure explains most of the gap to a buy-and-hold benchmark. With a 10% cap and a
few tickers the account is mostly cash. Compare excess return per unit of exposure, or
raise `max_position_pct`.

## Trades tab

- **Open positions:** units, average entry, and unrealized P&L at the latest close.
- **Closed trades:** fills matched **first-in, first-out (FIFO)**, one row per matched
  lot. A sell closes the oldest open units first, and commission is spread over the units
  of the fill that paid it. Each row has entry, exit, P&L, return, and holding days, plus:
  - **MAE** (maximum adverse excursion): the worst move against the position while it was
    open, from daily lows (highs for shorts), as a fraction of entry. Large MAE on
    winners means the stop you'd have needed was wide.
  - **MFE** (maximum favorable excursion): the best move in your favor. MFE much larger
    than the final return means exits are giving gains back.
- **Stats:**
  - **Win rate.**
  - **Average win / loss.**
  - **Profit factor:** gross profit ÷ gross loss; above 1 is profitable.
  - **Expectancy:** mean P&L per trade.
  - **Average holding days.**
- **Blotter:** every fill, downloadable as CSV.

## Signals tab

One ticker: type it or pick one from the log. You get candlesticks with volume over 6M,
1Y, 2Y or 5Y, and optional 50- and 200-session moving averages. Each decision is a marker
colored by rating, and hovering shows its forward returns. Your fills appear beside the
decisions. Two optional overlays:

- **MLP P(up):** the walk-forward network's out-of-sample probability that the price is
  higher 5 sessions later.
- **Chronos forecast:** the next 20 sessions as a median with 10–90% bands.

## Signal quality tab

This tab answers "do the ratings carry information?" Pick a **horizon** (1, 5, 10 or 20
sessions) and whether to measure **raw** or **excess** returns. Excess is the honest
default: a Buy in a rising market is not skill.

| View | What it shows | What good looks like |
|---|---|---|
| Mean return by rating | Average forward return per rating, with hit rate | Monotone: Buy > Overweight > Hold > Underweight > Sell |
| Table: count, mean, median, **t-stat**, **hit rate** | t = mean ÷ (stdev ÷ √n); hit rate is the share of directional calls whose forward return had the called sign | \|t\| > 2 is the usual bar for "not noise"; with n < 30 treat everything as anecdote |
| **Information coefficient (IC)** | Spearman rank correlation between rating score and forward return, across all decisions | 0.02–0.05 is meaningful in practice, above 0.1 exceptional; negative means inverted |
| Rolling IC | IC over a sliding window of decisions | Stable sign over time; a one-off spike is luck |
| **Signal decay** | IC and direction-signed return at each horizon | Shows over what horizon the edge lives, and when it fades |
| Called vs realized direction | Confusion matrix of the called direction against the actual sign | Mass on the diagonal |
| Rating transitions | How often each rating follows each other, per ticker | Constant flipping means noise; heavy self-transitions mean conviction (or stickiness) |
| Return distribution by rating | Box plot of every decision's forward return | Separation, not just different means |
| Cumulative signed return | Running sum of direction × forward return, equal weight per call | A steady upward slope |
| **Does the network agree?** | Splits the agents' directional calls by whether the MLP's probability pointed the same way | A higher hit rate when it agrees suggests the two signals combine |

## Research tab

The decision log as a table: date, ticker, rating, outcome, and alpha once settled.
Picking a row shows the full decision and the reflection written once the outcome was
known. Below that, a browser for the saved analyst reports under `results_dir/reports`.

## Models tab: the neural signals

Two models give you a second opinion that doesn't come from the language models. Both are
scored **out of sample** before showing you their current call.

### Walk-forward MLP (trained here)

- **Model:** scikit-learn `MLPClassifier`, with hidden layers 32 and 16, strong L2
  regularization (`alpha=1.0`) and early stopping. Inputs are standardized.
- **Target:** whether the close is higher *N* sessions later. *N* is the
  **forecast horizon**: 1, 5, 10 or 20.
- **Features,** 15 in all, every one computed only from data up to that day:
  - returns over 1, 5, 10, 21 and 63 sessions
  - RSI(14)
  - MACD histogram ÷ price
  - Bollinger %B(20, 2σ)
  - ATR(14) ÷ price
  - 21-session realized volatility
  - volume z-score (20 sessions)
  - distance from the 50- and 200-session averages
- **Walk-forward:** first trained on **initial training sessions** (252, about a year),
  then retrained every **refit** sessions (21) on an expanding window. A row is used for
  training only once its outcome had fully traded, so no prediction ever sees the future.
- **Baseline:** a logistic regression on the same features. A network that can't beat it
  isn't earning its complexity.

| Metric | Meaning | Bar to clear |
|---|---|---|
| Accuracy | Share of P > 0.5 calls that were right | The base rate: the share of up periods. Stocks rise slightly more often than they fall, so 50% is not the bar |
| AUC | Ability to rank ups above downs | 0.5 is random; 0.52–0.55 is already interesting on daily data |
| Brier score | Mean squared error of the probability | Must be below `brier_baseline`, the score of always guessing the base rate |
| Log loss | Penalizes confident wrong calls | Lower is better |
| IC | Rank correlation of P(up) with the forward return | Same scale as the ratings' IC |
| Mean return when up / down | Average forward return when the model said up vs down | The "up" average should beat the "down" one |
| Calibration | Predicted probability against observed frequency, by bin | Points near the diagonal; a 70% call should come true about 70% of the time |
| Permutation importance | Drop in AUC when each feature is shuffled, on the latest 20% | Which inputs the model relies on; values near 0 are noise |

You can download the probabilities, labels and forward returns as CSV.

**Expect little.** On daily price features alone, AUC near 0.5 is normal: markets are
close to efficient at this frequency. The tab is built to tell you whether a model has an
edge *on this ticker* before you lean on it.

### Chronos-Bolt (pretrained)

- Amazon's `amazon/chronos-bolt-small`, a time-series transformer pretrained on a large
  corpus. It is not trained on your data ("zero-shot"); it reads the last 512 closes.
- The **forecast** gives the next 5–64 sessions as quantile bands (10th to 90th
  percentile). It also gives the **implied P(up)**, the chance the horizon close beats
  today's, read from the bands, and the median expected return.
- The **backtest** forecasts from past origins (**backtest origins**, spaced **sessions
  between origins** apart), each seeing only closes before it, and scores what happened:
  hit rate, AUC, and **80% band coverage**. Coverage is the share of outcomes inside the
  10–90% band; about 80% means the bands are honest, much less means overconfident.
- The weights (about 180 MB) download from Hugging Face on first use, then run locally
  on CPU.

## Using it with your own models

Anything with scikit-learn's `fit(X, y)` and `predict_proba(X)` can take the MLP's place:

```python
from tradingagents.dashboard import models, sources
from sklearn.ensemble import GradientBoostingClassifier

bars = sources.ohlcv("AAPL")
wf = models.walk_forward(bars, horizon=5, model_factory=lambda: GradientBoostingClassifier())
print(wf.metrics)            # same out-of-sample metrics as the tab
wf.probability.tail()        # P(up) per session
```

The analytics in `tradingagents/dashboard/analytics.py` work outside the app too:
`performance`, `round_trips`, `forward_returns`, `by_rating`, `information_coefficient`,
`signal_decay`.
