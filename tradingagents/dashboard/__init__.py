"""The quant dashboard: sources and analytics for trades, signals and research, and neural models.

``sources`` finds and loads what runs, sweeps and accounts saved; ``analytics``
computes performance, trade and signal-quality statistics; ``models`` holds the
walk-forward MLP and the Chronos forecaster. ``app`` is the Streamlit view over
them, started by ``tradingagents dashboard``.
"""
