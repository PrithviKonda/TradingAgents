import sys
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from cli.display import console
from cli.models import AnalystType, AssetType
from cli.prompts import filter_analysts_for_asset_type, parse_analysts
from cli.run import run_analysis
from tradingagents.backtest import iter_grid, run_backtest, summarize
from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.execution import (
    BrokerError,
    CostModel,
    PaperBroker,
    decisions_from_log,
    run_session,
    simulate as simulate_decisions,
)
from tradingagents.execution.alpaca import LIVE_UNLOCK_ENV, AlpacaBroker
from tradingagents.llm_clients.openai_client import SIGNED_IN_PROVIDERS
from tradingagents.portfolio import load_portfolio

# prompt_toolkit's win32 output module is importable only on Windows (it asserts
# the platform at import time), so gate on the platform rather than catching the
# failure — that way a genuinely broken prompt_toolkit on Windows still surfaces
# instead of silently disabling the handler below. Off Windows this stays an
# empty tuple, which `except` accepts and never matches (#1138).
if sys.platform == "win32":  # pragma: no cover - platform dependent
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError

    _NO_CONSOLE_ERRORS: tuple[type[BaseException], ...] = (NoConsoleScreenBufferError,)
else:
    _NO_CONSOLE_ERRORS = ()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


@app.callback(invoke_without_command=True)
def analyze(
    ctx: typer.Context,
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
    portfolio: str = typer.Option(
        None,
        "--portfolio",
        help="JSON file with current holdings and cash, so the trader, risk and "
        "portfolio agents size against your actual position.",
    ),
    ticker: str = typer.Option(None, "--ticker", help="Ticker to analyze, e.g. NVDA or 0700.HK; skips the prompt"),
    date: str = typer.Option(None, "--date", help="Analysis date, YYYY-MM-DD; skips the prompt"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts, e.g. market,news; skips the prompt"
    ),
    save: bool | None = typer.Option(
        None, "--save/--no-save", help="Save the report under results_dir without asking"
    ),
    show: bool | None = typer.Option(
        None, "--show/--no-show", help="Show the full report at the end without asking"
    ),
    html: bool | None = typer.Option(
        None, "--html/--no-html",
        help="Save the report as one HTML page too, complete_report.html, without asking (default: yes)",
    ),
):
    """Run an analysis. This is what a bare `tradingagents` does.

    Flags answer their questions; with provider, models, depth and language also
    set through TRADINGAGENTS_* variables, the run asks nothing.
    """
    if ctx.invoked_subcommand is not None:
        return
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    portfolio_context = None
    if portfolio:
        try:
            portfolio_context = load_portfolio(portfolio)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None

    try:
        flags = {"ticker": ticker, "date": date, "analysts": analysts, "save": save, "show": show, "html": html}
        run_analysis(checkpoint=checkpoint, portfolio=portfolio_context, flags=flags)
    except _NO_CONSOLE_ERRORS:
        # A terminal with no console buffer cannot host the interactive prompts.
        # Emit one actionable line on stderr instead of a prompt_toolkit
        # traceback; plain text, since rich may not render here either (#1138).
        typer.echo(
            "Error: no Windows console available. The interactive CLI needs a real "
            "console buffer — run it from Windows Terminal, PowerShell, or cmd.exe "
            "rather than a piped or embedded terminal.",
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command()
def backtest(
    tickers: str = typer.Argument(..., help="Comma-separated tickers, e.g. NVDA,AAPL"),
    start: str = typer.Option(..., "--start", help="First analysis date, YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="Last analysis date, YYYY-MM-DD"),
    every: int = typer.Option(7, "--every", help="Days between analysis dates"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts to run: market, sentiment, news, fundamentals; omit for all the asset type allows"
    ),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock or crypto"),
    portfolio: str = typer.Option(
        None, "--portfolio", help="JSON file with holdings and cash, held constant across the grid"
    ),
    run_id: str = typer.Option(
        None, "--run-id", help="Continue an earlier sweep: its cells are skipped and its log reused"
    ),
    simulate_book: bool = typer.Option(
        False, "--simulate", help="Also trade the sweep's decisions through a simulated account"
    ),
):
    """Score past decisions over a grid of tickers and dates."""

    try:
        dates = iter_grid(start, end, every)
        book = load_portfolio(portfolio) if portfolio else None
        kind = AssetType(asset_type.strip().lower())
        # The analysts are named and checked as for an analysis; without a
        # choice, every analyst the asset type allows runs.
        chosen = (parse_analysts(analysts, kind) if analysts
                  else filter_analysts_for_asset_type(list(AnalystType), kind))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    names = [t.strip() for t in tickers.split(",") if t.strip()]
    if not names:
        console.print("[red]No ticker to analyze; pass them comma-separated, e.g. NVDA,AAPL[/red]")
        raise typer.Exit(code=1)

    def show_progress(done, total, ticker, date):
        console.print(f"[dim][{done}/{total}] {ticker} {date}[/dim]")

    kwargs = {"asset_type": kind.value, "portfolio": book, "run_id": run_id, "progress": show_progress,
              "selected_analysts": [a.value for a in chosen]}

    try:
        result = run_backtest(names, dates, DEFAULT_CONFIG, **kwargs)
    except Exception as exc:  # a missing key or an unknown analyst is a setup error
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(summarize(result).render())
    console.print(f"\nRan {result.cells_run} cells, skipped {result.skipped}. Log: {result.log_path}")
    console.print(f"Continue or settle this sweep: --run-id {result.run_id}")
    for ticker, date, reason in result.failures:
        console.print(f"[yellow]failed:[/yellow] {ticker} {date}: {reason}")
    for ticker, reason in result.settlement_failures:
        console.print(f"[yellow]unsettled:[/yellow] {ticker}: {reason}")
    if simulate_book and result.log_path.is_file():
        console.print("")
        _print_simulation(result.log_path, cash=None, max_position=None, end=None,
                          benchmark="SPY", out=result.log_path.parent / "simulation")


def _print_simulation(log_path: Path, cash, max_position, end, benchmark, out) -> None:
    config = _with_execution(DEFAULT_CONFIG, max_position_pct=max_position)
    try:
        decisions = decisions_from_log(log_path)
        result = simulate_decisions(decisions, config, starting_cash=cash, end_date=end,
                                    benchmark=benchmark or None)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(escape(result.render()))
    if out:
        console.print(f"Equity curve, fills and metrics: {result.write(out)}")


def _with_execution(config: dict, **overrides) -> dict:
    """``config`` with the given execution settings replaced, the rest kept."""
    execution = {**config.get("execution", {}), **{k: v for k, v in overrides.items() if v is not None}}
    return {**config, "execution": execution}


@app.command()
def simulate(
    run_id: str = typer.Option(None, "--run-id", help="Replay the decisions of this backtest sweep"),
    log: str = typer.Option(None, "--log", help="Or replay the decisions in this memory log file"),
    cash: float = typer.Option(None, "--cash", help="Starting cash (default: execution.starting_cash)"),
    max_position: float = typer.Option(None, "--max-position", help="Largest position as a share of equity, e.g. 0.1"),
    end: str = typer.Option(None, "--end", help="Mark the account to this date, YYYY-MM-DD (default: today)"),
    benchmark: str = typer.Option("SPY", "--benchmark", help="Buy-and-hold comparison; empty to skip"),
    out: str = typer.Option(
        None, "--out",
        help="Directory for equity.csv, fills.csv and metrics.json (default: results_dir/simulations/<name>)",
    ),
):
    """Trade past decisions through a simulated account: equity curve, drawdown, Sharpe."""
    if bool(run_id) == bool(log):
        console.print("[red]Pass exactly one of --run-id or --log[/red]")
        raise typer.Exit(code=1)
    if run_id:
        path = Path(DEFAULT_CONFIG["results_dir"]) / "backtest" / safe_ticker_component(run_id) / "trading_memory.md"
    else:
        path = Path(log)
    if not out:
        # Saved under results_dir by default, where the dashboard looks for simulations.
        name = f"{run_id or path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
        out = Path(DEFAULT_CONFIG["results_dir"]) / "simulations" / safe_ticker_component(name)
    _print_simulation(path, cash, max_position, end, benchmark, out)


def _session_analysts(analysts: str | None, asset_type: str):
    kind = AssetType(asset_type.strip().lower())
    chosen = (parse_analysts(analysts, kind) if analysts
              else filter_analysts_for_asset_type(list(AnalystType), kind))
    return kind, [a.value for a in chosen]


def _run_and_report(names, broker, kind, chosen, date, dry_run, confirm=None):
    def show_progress(done, total, ticker):
        console.print(f"[dim][{done}/{total}] analyzing {ticker}[/dim]")

    try:
        steps = run_session(names, broker, DEFAULT_CONFIG, trade_date=date, asset_type=kind.value,
                            selected_analysts=chosen, confirm=confirm, submit=not dry_run,
                            progress=show_progress)
    except (BrokerError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print("")
    for step in steps:
        console.print(escape(step.describe()))
    if dry_run:
        console.print("[dim]Dry run: nothing was sent.[/dim]")
    return steps


@app.command()
def paper(
    tickers: str = typer.Argument(None, help="Comma-separated tickers to analyze and trade"),
    account: str = typer.Option("default", "--account", help="Name of the paper account"),
    cash: float = typer.Option(None, "--cash", help="Funding for a new account (default: execution.starting_cash)"),
    date: str = typer.Option(None, "--date", help="Analysis date, YYYY-MM-DD (default: today)"),
    analysts: str = typer.Option(None, "--analysts", help="Comma-separated analysts; omit for all"),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock or crypto"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Analyze and plan, but record no fills"),
    status: bool = typer.Option(False, "--status", help="Show the account and exit"),
):
    """Paper trade: analyze, size the ratings, and fill them in a local paper account."""
    path = Path(DEFAULT_CONFIG["results_dir"]) / "paper" / f"{safe_ticker_component(account)}.json"
    execution = DEFAULT_CONFIG.get("execution", {})
    try:
        broker = PaperBroker.open(
            path, starting_cash=cash if cash is not None else float(execution.get("starting_cash", 100_000)),
            costs=CostModel.from_config(DEFAULT_CONFIG),
            allow_short=bool(execution.get("allow_short", False)),
            allow_fractional=bool(execution.get("allow_fractional", False)),
        )
    except BrokerError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if not status:
        names = [t.strip() for t in (tickers or "").split(",") if t.strip()]
        if not names:
            console.print("[red]No ticker to trade; pass them comma-separated, e.g. NVDA,AAPL[/red]")
            raise typer.Exit(code=1)
        try:
            kind, chosen = _session_analysts(analysts, asset_type)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None
        _run_and_report(names, broker, kind, chosen, date, dry_run)
        console.print("")

    book = broker.portfolio()
    console.print(f"Paper account '{account}' ({path})")
    console.print(f"Cash: {book.cash:,.2f} {book.currency} · funded with {broker.starting_cash:,.2f}")
    for held in book.positions:
        console.print(f"- {held.ticker}: {held.quantity:g} @ {held.average_price:,.2f} average")
    console.print(f"Fills recorded: {len(broker.fills)}")


@app.command()
def trade(
    tickers: str = typer.Argument(..., help="Comma-separated tickers to analyze and trade"),
    broker_name: str = typer.Option("alpaca", "--broker", help="Brokerage to trade at: alpaca"),
    live: bool = typer.Option(
        False, "--live", help=f"Trade the live account (real money); also needs {LIVE_UNLOCK_ENV}=1"
    ),
    yes: bool = typer.Option(False, "--yes", help="Send the planned orders without asking"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Analyze and plan, but send nothing"),
    analysts: str = typer.Option(None, "--analysts", help="Comma-separated analysts; omit for all"),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock or crypto"),
):
    """Trade at a brokerage: its paper account by default, the live one with --live."""
    if broker_name.strip().lower() != "alpaca":
        console.print(f"[red]Unknown broker {broker_name!r}; supported: alpaca[/red]")
        raise typer.Exit(code=1)
    names = [t.strip() for t in tickers.split(",") if t.strip()]
    try:
        kind, chosen = _session_analysts(analysts, asset_type)
        broker = AlpacaBroker(paper=not live,
                              max_order_notional=DEFAULT_CONFIG.get("execution", {}).get("max_order_notional"))
    except (BrokerError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    if live:
        console.print("[bold red]LIVE account: orders placed here spend real money.[/bold red]")

    def confirm(steps) -> bool:
        console.print("\nPlanned orders:")
        for step in steps:
            console.print(f"  {escape(step.describe())}")
        if yes:
            return True
        return typer.confirm(f"Send these orders to {broker.name}?", default=False)

    _run_and_report(names, broker, kind, chosen, None, dry_run, confirm=confirm)


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port", help="Port to serve the dashboard on"),
    results_dir: str = typer.Option(None, "--results-dir", help="Read runs and accounts from here (default: results_dir)"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Serve without opening a browser"),
    address: str = typer.Option(
        "localhost", "--address", help="Interface to listen on; 0.0.0.0 shares your accounts with the network"
    ),
):
    """Open the quant dashboard: trades, signals, research and neural models."""
    import subprocess
    from importlib.resources import files

    app_path = str(files("tradingagents.dashboard").joinpath("app.py"))
    # Streamlit listens on every interface by default; the dashboard shows your
    # accounts, so it stays on this machine unless asked otherwise.
    command = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(port),
               "--server.address", address,
               "--server.headless", "true" if no_browser else "false", "--browser.gatherUsageStats", "false"]
    if results_dir:
        command += ["--", "--results-dir", results_dir]
    try:
        raise typer.Exit(code=subprocess.call(command))
    except KeyboardInterrupt:
        raise typer.Exit(code=0) from None


# The plans `login` and `logout` take: the providers signed in to, not keyed.
Plan = StrEnum("Plan", {name: name for name in SIGNED_IN_PROVIDERS})


@app.command()
def login(plan: Annotated[Plan, typer.Argument(help="The plan to sign in with")]):
    """Sign in with your ChatGPT Plus or Pro plan, for the chatgpt provider."""
    from tradingagents.llm_clients import chatgpt_auth

    try:
        # One unwrapped line, so the sign-in address can be copied whole.
        email = chatgpt_auth.sign_in(
            say=lambda line: console.print(line, soft_wrap=True, markup=False, highlight=False))
    except chatgpt_auth.ChatGPTAuthError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[green]Signed in to ChatGPT as {escape(email)}.[/green]")


@app.command()
def logout(plan: Annotated[Plan, typer.Argument(help="The plan to sign out of")]):
    """Sign out of your ChatGPT plan: revoke this machine's access and forget it."""
    from tradingagents.llm_clients import chatgpt_auth

    outcome = chatgpt_auth.sign_out()
    if outcome == "revoked":
        console.print("Signed out of ChatGPT.")
    elif outcome == "forgotten":
        console.print("[yellow]Signed out here, but OpenAI could not be reached to revoke the sign-in; "
                      "it stays valid at OpenAI until it expires.[/yellow]")
    else:
        console.print("Not signed in to ChatGPT.")


if __name__ == "__main__":
    app()
