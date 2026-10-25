# TradingAgents: guide for agents making changes

This is a fork of TauricResearch/TradingAgents, customized by a quant. A run is a LangGraph pipeline of LLM agents: analysts write reports, bull/bear researchers debate them, the research manager and trader propose a plan, three risk debaters argue it, and the portfolio manager emits a 5-tier rating (`agents/rating.py`).

## The core rule: Python computes, the LLM interprets

Any number a decision depends on (factor, signal, z-score, vol, position cap) is computed **deterministically in Python** and handed to the model as a tool result or prompt input. The model combines and explains; it never does the arithmetic. When asked to "make the logic smarter", reach for a tool or a hard rule before a longer prompt.

## Where each kind of change goes

| Change | Edit |
|---|---|
| Debate rounds, data vendors, benchmark, temperature | `tradingagents/default_config.py` keys, or `TRADINGAGENTS_*` in `.env` |
| An agent's reasoning or strategy | The system prompt inline in that agent's file under `tradingagents/agents/` (analysts/, researchers/, trader/, risk_mgmt/, managers/) |
| A new signal for an existing analyst | New `@tool` in `tradingagents/agents/tools.py` (fetch data via `dataflows/router.py`), add it to that analyst's `TOOLS` tuple (its tool node is built from the same tuple), then tell the prompt how to use it |
| A new analyst | See checklist below |
| Pipeline order, adding/skipping stages | `tradingagents/graph/setup.py` (nodes and edges), `graph/conditional_logic.py` (routing) |
| Rating scale or the final decision's shape | `agents/rating.py` and `agents/schemas.py` together |

### New analyst checklist

A report key is wired by name across the codebase, not discovered. Copy an existing analyst (e.g. `market_analyst.py` → `quant_analyst.py`), then:

1. Register it in `ANALYST_NODE_SPECS` (`graph/analyst_execution.py`) and `analyst_factories` (`graph/setup.py`).
2. Run `grep -rn fundamentals_report tradingagents cli` and mirror **every** hit for your new report key: state, initial state, logged state, the researchers and risk debaters that read it, the saved report, and the CLI display/selection.
3. Add the analyst choice to `cli/models.py`.

Done when the grep for your key hits the same set of files as `fundamentals_report`.

## Verifying a change

- `pytest -q` and `ruff check .`: both run in CI on every push. Three `tests/test_cache_writes.py::test_on_windows_*` tests fail on macOS on upstream too; any other failure is yours.
- Tests that drive the CLI patch `cli.selections` functions by name; renaming or removing one breaks them (`tests/test_cli_*.py`).
- To judge whether a logic change *helps*, use `tradingagents backtest` (`tradingagents/backtest.py`): run the same ticker/date grid before and after and compare `summarize()`. It scores decision quality against realized alpha; it is deliberately not a portfolio/P&L simulator, so keep sizing and fills out of it.

## Local environment

- LLM: local Ollama, `qwen3:latest` (8B) for both tiers, set in the gitignored `.env`. The machine has 8 GB RAM, so models of 20B and up will not run. Expect noisy output: make anything that must be reproducible a Python tool.
- Python: `.venv` built with `uv pip install -e .`.

## Git

- Push and open PRs only to the fork `PrithviKonda/TradingAgents` (remote `fork`; `origin` is upstream and its push URL is redirected to the fork). Use `gh pr create --repo PrithviKonda/TradingAgents`.
- The welcome screen credits "Built by Prithvi Kothakonda" and the upstream "What's new" update check was removed on purpose; keep both that way.
