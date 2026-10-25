# Configuration reference

Every setting TradingAgents reads lives in one dictionary, `DEFAULT_CONFIG`, built in
[`tradingagents/default_config.py`](../tradingagents/default_config.py). This page lists
every key: what it means, its default, and how to change it.

## How to change a setting

There are four ways to change a setting. When more than one applies, later ones win.

| Where | Reaches | Use it for |
|---|---|---|
| **`tradingagents/default_config.py`** | Everything: the CLI, the dashboard and Python | Your own standing defaults, such as position cap, commissions or vendors |
| **Environment variables / `.env`** | Keys marked *env* below | Provider, models, paths and debate depth, without editing code |
| **CLI flags and prompts** | The run being started | One-off choices: ticker, date, analysts, `--cash`, `--max-position` |
| **Python** | The graph or call you pass it to | Scripts, notebooks, research |

### In Python

Copy the defaults, change what you need, and pass the copy in. Use `deepcopy`. The
config holds nested dictionaries (`execution`, `data_vendors`, `benchmark_map`), and with
a shallow `.copy()` an edit to a nested value would also change `DEFAULT_CONFIG` itself.

```python
import copy
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

config = copy.deepcopy(DEFAULT_CONFIG)
config["llm_provider"] = "anthropic"
config["deep_think_llm"] = "claude-opus-5-5"
config["quick_think_llm"] = "claude-sonnet-5-5"
config["max_debate_rounds"] = 2
config["execution"]["max_position_pct"] = 0.20      # nested keys edit in place on the copy

ta = TradingAgentsGraph(["market", "news", "fundamentals"], config=config)
```

### With environment variables

Copy `.env.example` to `.env` in the project folder and uncomment what you need. It is
read at start-up. Values are converted to the type of the default, so `true`, `2` and
`0.0` all work. A value that cannot be converted stops the program at start-up with the
variable's name, rather than being silently ignored.

```bash
cp .env.example .env
# then, in .env:
TRADINGAGENTS_LLM_PROVIDER=anthropic
TRADINGAGENTS_DEEP_THINK_LLM=claude-opus-5-5
TRADINGAGENTS_MAX_DEBATE_ROUNDS=2
```

### Editing the defaults file

The CLI commands (`backtest`, `simulate`, `paper`, `trade`, `dashboard`) all start from
`DEFAULT_CONFIG`. The `execution` block has no environment variables, so to change for
example the position cap or commission for every CLI run, edit it in
`tradingagents/default_config.py`. Several of these settings also have per-run flags
(`--cash`, `--max-position`) and dashboard sliders.

---

## Files and folders

| Key | Default | Env | Meaning |
|---|---|---|---|
| `results_dir` | `~/.tradingagents/logs` | `TRADINGAGENTS_RESULTS_DIR` | Root for everything a run writes: reports, backtests, paper accounts, simulations. The dashboard reads from here. |
| `data_cache_dir` | `~/.tradingagents/cache` | `TRADINGAGENTS_CACHE_DIR` | Cached market data and checkpoint databases. Safe to delete; it refills. |
| `memory_log_path` | `~/.tradingagents/memory/trading_memory.md` | `TRADINGAGENTS_MEMORY_LOG_PATH` | Your live decision log. Each run appends its decision; later runs settle it and learn from it. Backtests write their own logs, never this one. |
| `memory_log_max_entries` | `None` | none | Keep at most this many *settled* entries, pruning the oldest. Pending entries are never pruned. `None` keeps everything. |

What ends up under `results_dir`:

```
results_dir/
├── reports/<TICKER>_<timestamp>/      saved analyst reports, complete_report.html
├── backtest/<run_id>/trading_memory.md  each sweep's own decision log
│   └── simulation/                    written by `backtest --simulate`
├── simulations/<name>/                written by `simulate`: equity.csv, fills.csv, metrics.json
└── paper/<account>.json               paper accounts: cash, positions, every fill
```

## Language models

TradingAgents uses two model tiers. The **quick** tier serves the analysts, researchers,
risk debaters and trader, which make many calls. The **deep** tier serves the research
manager and portfolio manager, which make the final calls.

| Key | Default | Env | Meaning |
|---|---|---|---|
| `llm_provider` | `"openai"` | `TRADINGAGENTS_LLM_PROVIDER` | Provider for both tiers. One of `openai`, `anthropic`, `google`, `xai`, `deepseek`, `qwen`, `qwen-cn`, `glm`, `glm-cn`, `minimax`, `minimax-cn`, `openrouter`, `mistral`, `kimi`, `groq`, `nvidia`, `ollama`, `openai_compatible`, `azure`, `bedrock`, `chatgpt`. |
| `deep_think_llm` | `"gpt-6-sol"` | `TRADINGAGENTS_DEEP_THINK_LLM` | Model ID for the deep tier. Any ID your provider serves is accepted. |
| `quick_think_llm` | `"gpt-6-luna"` | `TRADINGAGENTS_QUICK_THINK_LLM` | Model ID for the quick tier. |
| `backend_url` | `None` | `TRADINGAGENTS_LLM_BACKEND_URL` | Custom endpoint, e.g. `http://localhost:8000/v1` for vLLM. `None` uses the provider's own. |
| `deep_think_provider` / `quick_think_provider` | `None` | `TRADINGAGENTS_DEEP_THINK_PROVIDER` / `…_QUICK_THINK_PROVIDER` | Run one tier on a different provider, e.g. managers on Anthropic, analysts on OpenAI. `None` uses `llm_provider`. |
| `deep_think_backend_url` / `quick_think_backend_url` | `None` | `TRADINGAGENTS_DEEP_THINK_BACKEND_URL` / `…_QUICK_THINK_BACKEND_URL` | Per-tier endpoint, for a local or relay server. |
| `openai_reasoning_effort` | `None` | `TRADINGAGENTS_OPENAI_REASONING_EFFORT` | `low` / `medium` / `high` for OpenAI and ChatGPT reasoning models. Setting it skips the CLI question. |
| `google_thinking_level` | `None` | `TRADINGAGENTS_GOOGLE_THINKING_LEVEL` | Gemini thinking level, e.g. `high`, `minimal`. |
| `anthropic_effort` | `None` | `TRADINGAGENTS_ANTHROPIC_EFFORT` | Claude effort: `low` / `medium` / `high`. |
| `temperature` | `None` | `TRADINGAGENTS_TEMPERATURE` | Sampling temperature for every provider. Lower means more repeatable on models that honor it; reasoning models largely ignore it. |
| `llm_max_retries` | `None` | `TRADINGAGENTS_LLM_MAX_RETRIES` | SDK retry budget. Raise it (e.g. `6`) on rate-limited keys that hit 429 errors. |
| `max_tokens` | `None` | `TRADINGAGENTS_MAX_TOKENS` | Cap on output tokens, for a model that runs long and trips a timeout. |
| `output_language` | `"English"` | `TRADINGAGENTS_OUTPUT_LANGUAGE` | Language of the reports and the final decision. The internal debate stays in English. |

API keys are separate environment variables, one per provider (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, …); the README's "Required APIs" section lists them all.

## How hard the agents think

| Key | Default | Env | Meaning |
|---|---|---|---|
| `max_debate_rounds` | `1` | `TRADINGAGENTS_MAX_DEBATE_ROUNDS` | Bull-vs-bear rounds before the research manager decides. More rounds cost more tokens. The CLI's "research depth" sets this. |
| `max_risk_discuss_rounds` | `1` | `TRADINGAGENTS_MAX_RISK_ROUNDS` | Rounds of the aggressive / neutral / conservative risk debate. |
| `max_tool_rounds` | `20` | `TRADINGAGENTS_MAX_TOOL_ROUNDS` | Data-tool calls an analyst may make before it must write its report. |
| `max_recur_limit` | `100` | none | LangGraph's step ceiling for one run; a safety stop, rarely needs changing. |
| `checkpoint_enabled` | `False` | `TRADINGAGENTS_CHECKPOINT_ENABLED` | Save state after every step so a crashed run resumes. Same as `--checkpoint`. |

## News and data

| Key | Default | Env | Meaning |
|---|---|---|---|
| `news_article_limit` | `20` | none | Most articles fetched per ticker. Lower it to save tokens. |
| `global_news_article_limit` | `10` | none | Most macro / global articles. |
| `global_news_lookback_days` | `7` | none | How far back macro news reaches. |
| `global_news_queries` | 5 macro queries | none | Search queries for macro headlines (Fed, earnings, geopolitics, central banks, commodities). Extend it to cover your sector or region. |
| `data_vendors` | see below | none | Which vendor serves each data category, as an ordered fallback chain. |
| `tool_vendors` | `{}` | none | Per-tool override of `data_vendors`, e.g. `{"get_stock_data": "alpha_vantage"}`. |

`data_vendors` defaults:

| Category | Default | Options |
|---|---|---|
| `core_stock_apis` | `yfinance` | `yfinance`, `alpha_vantage` |
| `technical_indicators` | `yfinance` | `yfinance`, `alpha_vantage` |
| `fundamental_data` | `sec_edgar,yfinance` | `sec_edgar` (US filings as filed), `yfinance`, `alpha_vantage` |
| `news_data` | `yfinance` | `yfinance`, `alpha_vantage` |
| `macro_data` | `fred` | `fred` (needs `FRED_API_KEY`) |
| `prediction_markets` | `polymarket` | `polymarket` (no key) |

A comma-separated value is tried in order, and only those vendors are used. `"default"`
uses every available vendor.

## Scoring decisions

These keys decide how a decision is judged once its outcome is known, both for the
memory log's reflections and for backtest scoring.

| Key | Default | Env | Meaning |
|---|---|---|---|
| `holding_period_days` | `5` | none | Trading days after entry over which a decision is scored. Entry is the first close after the decision date. |
| `benchmark_ticker` | `None` | `TRADINGAGENTS_BENCHMARK_TICKER` | One benchmark for every ticker. `None` picks a regional index from `benchmark_map`. |
| `benchmark_map` | suffix → index | none | Exchange suffix to benchmark, e.g. `.T` → `^N225`, `.HK` → `^HSI`, `.L` → `^FTSE`. Tickers with no suffix (US) use `SPY`. Add a row for a market that isn't listed. |

**Alpha** is the decision's return minus the benchmark's over the same window. Each is
measured in its own currency.

## Execution: turning ratings into orders

The `execution` block drives the simulator, paper accounts and broker trading.
[Trading guide](trading.md) explains how the pieces fit together.

| Key | Default | Meaning |
|---|---|---|
| `starting_cash` | `100000.0` | Cash for a new paper account or a simulation. `--cash` overrides it. |
| `max_position_pct` | `0.10` | The per-position cap: the most of equity one ticker may hold. Every rating sizes against this. `--max-position` overrides it. |
| `target_weights` | `None` | Override how much of the cap each rating takes, e.g. `{"Overweight": 0.5, "Underweight": 0.0}`. Unlisted ratings keep their defaults. |
| `min_trade_pct` | `0.005` | Skip a rebalance worth less than this share of equity, so a small drift doesn't trade. A full exit always goes through. |
| `allow_fractional` | `False` | Trade fractional units. Off rounds every order down to whole shares. |
| `allow_short` | `False` | Let a sell go below zero units. Off keeps paper and simulated accounts long-only. |
| `slippage_bps` | `5.0` | Basis points a fill pays: buys fill this much above the reference price, sells this much below. 1 bp = 0.01%. |
| `commission_per_order` | `0.0` | Fixed fee per fill, in account currency. |
| `commission_pct` | `0.0` | Fee as a fraction of the fill's value: `0.001` = 10 bps. |
| `max_order_notional` | `10000.0` | Real brokers refuse any single order worth more than this, before it is sent. `None` removes the cap; think twice. |

Default share of the cap per rating, and what changes it:

| Rating | Target | Override example |
|---|---|---|
| Buy | 100% of the cap | `{"Buy": 0.8}` |
| Overweight | 75% | `{"Overweight": 0.5}` |
| Hold | keep the current position | not configurable: Hold never trades |
| Underweight | 25%, and never adds to a position | `{"Underweight": 0.0}` makes it a full exit |
| Sell | 0%: exit | `{"Sell": 0.0}` |
| REVIEW | no trade (the decision had no readable rating) | not configurable |

## Broker and safety variables

These are environment variables, not config keys, so they cannot sit in a config file
that gets copied between machines.

| Variable | Meaning |
|---|---|
| `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` | Alpaca API keys. Paper and live accounts have different keys. |
| `TRADINGAGENTS_LIVE_TRADING` | Must be `1` (or `true`) for `trade --live` to place real orders. Without it, live is refused. |

## Other environment variables

| Variable | Meaning |
|---|---|
| `SEC_EDGAR_USER_AGENT` | Contact string SEC asks for, e.g. `"Your Name you@example.com"`. |
| `FRED_API_KEY` | Free key for FRED macro data. Without it, macro tools report that they are unavailable. |
| `ALPHA_VANTAGE_API_KEY` | Needed only if you route a category to `alpha_vantage`. |
| `TYPESAFE_API_KEY`, `TYPESAFE_DEFAULT_MODEL`, `TYPESAFE_BASE_URL` | Optional Jev screening of social posts for the sentiment analyst. |
| `OLLAMA_BASE_URL` | Remote Ollama server; default `http://localhost:11434/v1`. |
| `OPENAI_COMPATIBLE_API_KEY` | Key for an `openai_compatible` endpoint that requires one. |
| `AWS_DEFAULT_REGION`, `AWS_BEARER_TOKEN_BEDROCK`, `AWS_PROFILE` | AWS Bedrock credentials and region. |
| `TRADINGAGENTS_NO_UPDATE_CHECK` | `1` turns off the start-up note about new releases. |
| `TRADINGAGENTS_DATA_DIR` | Docker only: host folder for results, reports and the memory log. |
