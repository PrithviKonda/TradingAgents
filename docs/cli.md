# CLI reference

Every command is `tradingagents <command> [options]`. `tradingagents <command> --help`
prints the same information.

## `tradingagents` — analyze one ticker

With no command, it runs an interactive analysis. Every flag answers its own question, so
a fully flagged run asks nothing.

| Option | Meaning |
|---|---|
| `--ticker NVDA` | Ticker; exchange suffixes work (`0700.HK`, `7203.T`, `BTC-USD`) |
| `--date YYYY-MM-DD` | Analysis date; past dates see only data published by then |
| `--analysts market,news` | Any of `market`, `sentiment`, `news`, `fundamentals` |
| `--portfolio book.json` | Your holdings and cash, so the agents size against them |
| `--save / --no-save` | Save the report under `results_dir/reports` |
| `--show / --no-show` | Print the full report at the end |
| `--html / --no-html` | Also write `complete_report.html` |
| `--checkpoint / --no-checkpoint` | Save state after each step so a crash resumes |
| `--clear-checkpoints` | Delete saved checkpoints first |

The provider, models, research depth and language come from prompts, or from
`TRADINGAGENTS_*` variables ([configuration](configuration.md)).

## `backtest` — run and score a grid

`tradingagents backtest NVDA,AAPL --start 2026-03-01 --end 2026-09-01`

| Option | Default | Meaning |
|---|---|---|
| `--start`, `--end` | required | Date range of analysis dates |
| `--every N` | 7 | Days between analysis dates |
| `--analysts` | all | As above |
| `--asset-type` | stock | `stock` or `crypto` |
| `--portfolio` | none | A standing book, the same for every cell |
| `--run-id ID` | new | Continue a sweep; finished cells are skipped |
| `--simulate` | off | Also replay the sweep through a simulated account |

## `simulate` — trade decisions through a simulated account

`tradingagents simulate --run-id 20260901_101500`

| Option | Default | Meaning |
|---|---|---|
| `--run-id ID` / `--log PATH` | one required | A backtest sweep, or any memory log file |
| `--cash` | `execution.starting_cash` | Starting cash |
| `--max-position` | `execution.max_position_pct` | Per-position cap as a fraction, e.g. `0.2` |
| `--end` | today | Mark the account through this date |
| `--benchmark` | SPY | Buy-and-hold comparison; `""` to skip |
| `--out DIR` | `results_dir/simulations/<name>` | Where to write `equity.csv`, `fills.csv`, `metrics.json` |

## `paper` — local paper account

`tradingagents paper NVDA,AAPL`

| Option | Default | Meaning |
|---|---|---|
| tickers | — | Comma-separated; omit with `--status` |
| `--account NAME` | default | Which account file, `results_dir/paper/NAME.json` |
| `--cash` | `execution.starting_cash` | Funding for a *new* account only |
| `--date` | today | Analysis date |
| `--analysts`, `--asset-type` | all, stock | As above |
| `--dry-run` | off | Analyze and plan, record no fills |
| `--status` | off | Show the account and exit |

## `trade` — broker

`tradingagents trade NVDA,AAPL`

| Option | Default | Meaning |
|---|---|---|
| `--broker` | alpaca | Only Alpaca is supported |
| `--live` | off | Live account; also needs `TRADINGAGENTS_LIVE_TRADING=1` |
| `--yes` | off | Send without the confirmation prompt |
| `--dry-run` | off | Plan only; send nothing |
| `--analysts`, `--asset-type` | all, stock | As above |

Needs `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`. See [trading guide](trading.md#4-broker-alpaca-paper-then-live).

## `dashboard`

`tradingagents dashboard`

| Option | Default | Meaning |
|---|---|---|
| `--port` | 8501 | Port |
| `--results-dir` | `results_dir` | Folder to read |
| `--no-browser` | off | Don't open a browser tab |
| `--address` | localhost | `0.0.0.0` exposes your accounts to the network |

## `login chatgpt` / `logout chatgpt`

Sign in (or out) with a ChatGPT Plus or Pro plan, for `llm_provider = "chatgpt"`.
