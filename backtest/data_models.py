from dataclasses import dataclass
from typing import Dict, List, Any

import pandas as pd


@dataclass
class Trade:
    """Represents one executed rebalance transaction."""

    date: pd.Timestamp
    shares: Dict[str, float]
    price: Dict[str, float]
    commission: float
    note: str = ""


@dataclass
class TradingState:
    """Snapshot of trading state passed into strategy predict each tick."""

    position_history: pd.DataFrame
    nav_history: pd.Series
    current_date: pd.Timestamp
    current_shares: pd.Series
    current_nav: float
    trade_history: List[Trade]
    signal_history: List[Any]
