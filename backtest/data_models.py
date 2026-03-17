from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class Trade:
    """Represents one executed rebalance transaction."""

    date: pd.Timestamp
    position: pd.Series
    price: pd.Series
    commission: float
    slippage: float


@dataclass
class TradingState:
    """Snapshot of trading state passed into strategy predict each tick."""

    position_history: pd.DataFrame
    nav_history: pd.Series
    current_date: pd.Timestamp
    current_position: pd.Series
    current_nav: float
    trade_history: List[Trade]
    signal_history: List[Optional[pd.Series]]
