from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class Trade:
    """Represents one executed rebalance transaction."""

    date: pd.Timestamp
    shares: Dict[str, float]
    price: Dict[str, float]
    commission: float
