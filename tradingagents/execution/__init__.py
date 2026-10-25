"""Execution: turning ratings into orders, on paper, in simulation, or at a broker.

Kept apart from ``tradingagents.backtest``, which scores decisions and holds no
execution model. Everything that needs a quantity, a fill price or a cash
ledger lives here.
"""

from tradingagents.execution.broker import Broker, BrokerError, Fill, Order
from tradingagents.execution.paper import CostModel, PaperBroker
from tradingagents.execution.session import SessionStep, run_session
from tradingagents.execution.simulator import (
    Decision,
    SimulationResult,
    decisions_from_log,
    simulate,
)
from tradingagents.execution.sizing import (
    DEFAULT_TARGET_WEIGHTS,
    SizingRules,
    plan_order,
    target_quantity,
)
