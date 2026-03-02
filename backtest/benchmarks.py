from typing import Callable, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

from backtest.strategy import Strategy
from backtest.data_models import TradingState
from backtest.exceptions import StrategyError, StrategyNotImplemented

class BuyAndHold(Strategy):
    """
    Simple buy-and-hold strategy that invests all cash into the first tradable asset at the start of the backtest and holds it until the end.
    """
    def __init__(self, notional_amount=1000000):
        super().__init__()
        self.notional_amount = notional_amount

    def predict(self, tradable_assets, history, trading_state: TradingState):
        if not self._initialized:
            self.initialize()

        if len(tradable_assets) == 0:
            raise StrategyError("No tradable assets available for BuyAndHold strategy.")

        # Hold existing positions (no changes)
        return 0, pd.Series([self.notional_amount / len(tradable_assets)] * len(tradable_assets), index=tradable_assets)