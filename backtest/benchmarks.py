import pandas as pd

from backtest.strategy import Strategy
from backtest.data_models import TradingState
from backtest.exceptions import StrategyError

class BuyAndHold(Strategy):
    """
    Simple buy-and-hold strategy that invests all cash into the first tradable asset at the start of the backtest and holds it until the end.
    """
    def __init__(self, notional_amount=1000000):
        super().__init__()
        self.notional_amount = notional_amount
        self.initial_shares = None

    def predict(self, tradable_assets, history, trading_state: TradingState):
        if not self._initialized:
            self.initialize()

        if len(tradable_assets) == 0:
            raise StrategyError("No tradable assets available for BuyAndHold strategy.")

        # Determine price format depending on whether history columns are MultiIndexed
        if isinstance(history.columns, pd.MultiIndex):
            if 'adj_close' in history.columns.levels[0]:
                prices = history['adj_close'].iloc[-1]
            else:
                prices = history.iloc[-1]
        else:
            prices = history.iloc[-1]
            
        prices = prices.reindex(tradable_assets)

        if self.initial_shares is None:
            # On day 1: find how many nominal shares we would get for an equal split
            target_dollars = self.notional_amount / len(tradable_assets)
            self.initial_shares = target_dollars / prices
            return pd.Series(target_dollars, index=tradable_assets)

        # To truly 'Buy and Hold' without triggering endless daily rebalancing under the hood,
        # we must track the valuation of the initial shares and ask the backtester 
        # to target exactly that drifted dollar amount.
        drifting_target = self.initial_shares.reindex(tradable_assets, fill_value=0.0) * prices
        return drifting_target