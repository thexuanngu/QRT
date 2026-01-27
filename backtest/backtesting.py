# (Trimmed snippet — use the full code run above for direct copy)

from typing import Callable, Optional, List, Dict, Any, Union, Tuple, Sequence

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import seaborn as sns
import inspect

sns.set_style("whitegrid")

@dataclass
class StrategyMeta:

    name: str
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


class StrategyError(Exception):
    pass

class StrategyNotImplemented(Exception):
    pass


class Strategy(ABC):
    """
    Abstract base Strategy class.

    Lifecycle:
      - initialize() : called once at the start of backtest
      - predict(history) : called every tick with historical data up to now -> returns target
      - finalize() : called once at end (reporting/cleanup)
      - reset() : optional to reuse the object in repeated tests

    The class stores internal state in self._state and metadata in self.meta.
    """


    def __init__(self):
        #self.meta = meta or StrategyMeta(name=self.__class__.__name__)
        self._state: Dict[str, Any] = {}
        self._initialized = False


    # lifecycle hooks
    def initialize(self) -> None:
        """Override for one-off initialization (seeding RNGs, precomputing)."""
        self._initialized = True


    @abstractmethod
    def predict(self, tradable_assets, history, position_history):
        """
        Core API that the backtester will call at each tick. Raises StrategyNotImplemented if not implemented yet. 

        Parameters
        ----------
        history : pd.DataFrame
            Time-indexed DataFrame containing all data up to and including the current tick.

        Returns
        -------
        float or pd.Series or dict
            The target portfolio. For single-asset strategies return a float (-1..1).
            For multi-asset strategies return a Series or dict mapping asset->weight.
        """
        raise StrategyNotImplemented


    def finalize(self) -> None:
        """Called once after backtest finishes."""
        pass


    def reset(self) -> None:
        """Reset internal state so strategy can be reused in repeated backtests."""
        self._state = {}
        self._initialized = False

    # def to_dict(self) -> dict:
    #     """Serialize metadata and state (non-callable)."""
    #     return {"meta": asdict(self.meta), "state": dict(self._state)}

    # def __repr__(self) -> str:
    #     return f"<Strategy {self.meta.name} params={self.meta.params}>"

    # small helper for validation
    @staticmethod
    def _validate_target(target) -> None:
        """
        Ensure target is a supported return type and numeric values are finite.
        """
        valid_types = (float, int, pd.Series, dict, type(None))
        if not isinstance(target, valid_types):
            raise StrategyError(f"Unsupported target type: {type(target)}. "
                                "Must be float, pd.Series, or dict.")
        
        if isinstance(target, (float, int)):
            if not np.isfinite(float(target)):
                raise StrategyError("Numeric return must be finite.")
            
        if isinstance(target, pd.Series):
            if not np.all(np.isfinite(target.values.astype(float))):
                raise StrategyError("Series contains non-finite values.")
            
        if isinstance(target, dict):
            for k, v in target.items():
                if not np.isfinite(float(v)):
                    raise StrategyError(f"Non-finite weight for asset {k}.")


class FunctionStrategy(Strategy):

    """
    Adapter that wraps a user function into a Strategy.

    Two supported user function signatures:
     - vectorized: fn(history: pd.DataFrame) -> pd.Series/float/dict
         (faster: user produces entire series given history; engine will call predict per tick
         but under hood the function can be vectorized)
     - per-tick (event): fn(current_index, history_up_to_index: pd.DataFrame, state: dict) -> float/Series/dict
         (state is provided to let user keep counters/cooldown etc.)
    
    Create with `mode='vector'` or `mode='event'`.
    """

    def __init__(self,
                 user_fn: Callable, # MUST be a function 
                 tradable_assets: Tuple[str], 
                 required_params: Tuple[str] = None,
                 optional_params: Tuple[str] = None,
                 mode: str = "event",
                 params: Optional[Dict[str, Any]] = None):
       
        if mode not in ("event", "vector"):
            raise ValueError("mode must be 'event' or 'vector'")

        #meta = meta or StrategyMeta(name=getattr(user_fn, "__name__", "function_strategy"),
        #                            params=params or {})
        # Force the strategy to be a function. 
    

        super().__init__()
        self.user_fn = user_fn
        self.required_params = required_params
        self.optional_params = optional_params
        self.tradable_assets = tradable_assets

        self.mode = mode 

        # keep a cache for vector mode to avoid recomputing on every tick
        self._vector_cache: Optional[pd.Series] = None


    def initialize(self) -> None:
        super().initialize()
        # clear caches/state
        self._state = {}
        self._vector_cache = None


    # Update - predict now includes an argument called tradable_assets. 
    # NB : will remove the history argument once I figure out how to generalise
    def predict(self, tradable_assets, history, position_history):
        """
        Return the target portfolio for the *current* tick. Backtester should call this
        once per tick passing history up to and including the current tick.

        In 'vector' mode, we lazily compute the whole series on first call and then
        return the value for the current index.

        Parameters
        ----------
        tradable_assets: an iterable (either list, or array) of tradable assets. This is so that the algorithm only spits out portfolio consisting of assets that we ACTUALLY want to trade.


        """

        if not isinstance(history, pd.DataFrame):
            raise StrategyError("history must be a pandas DataFrame")
        

        if history.empty:
            # no data yet -> neutral
            return 0.0

        current_index = history.index[-1]

        target_cash = None
        target_portfolio = None
        if self.mode == "event": # event mode 
            # user_fn(index, history_up_to_index, state) -> target

            try:
                target_cash, target_portfolio = self.user_fn(tradable_assets, history.copy(), position_history.copy(), self._state)
            except TypeError:
                # some users may ignore index and state; try fallback
                target_cash, target_portfolio = self.user_fn(tradable_assets, history.copy(), position_history.copy())
            except AttributeError:
                target_cash, target_portfolio = self.user_fn(tradable_assets, history.copy(), position_history.copy())

        else:  # vector mode
            # compute vectorized outputs only once (or if history length changed significantly)
            if self._vector_cache is None or len(self._vector_cache) < len(history):
                # user_fn should accept a DataFrame and return a Series aligned with history index or a scalar/dict
                vector_out = self.user_fn(history.copy())
                if isinstance(vector_out, (pd.Series, pd.DataFrame)):
                    vec = pd.Series(vector_out.squeeze()).reindex(history.index)
                    # store
                    self._vector_cache = vec.astype(float)
                elif isinstance(vector_out, (float, int)):
                    # scalar constant signal
                    self._vector_cache = pd.Series([float(vector_out)] * len(history), index=history.index)
                elif isinstance(vector_out, dict):
                    # Multi-asset vectorized mode returning dict of arrays/series
                    # For per-tick use, extract current dict
                    self._vector_cache = pd.Series([vector_out], index=history.index)  # fallback
                else:
                    raise StrategyError("vector mode: user function returned unsupported type")
            # fetch current target


            # If _vector_cache is a Series of scalars: return scalar for current index
            target_portfolio = self._vector_cache.loc[current_index]

        # validate returned target
        self._validate_target(target_portfolio)
        if not isinstance(target_cash, (float, type(None))):
            raise StrategyError("Cash must be returned as a float.")

        
        # Gabriel's notes: I'm thinking on whether not I should include a separate class that my Strategy class can read off and know exactly what trades it needs to make.

        return target_cash, target_portfolio
    
    def finalize(self) -> None:
        # user may want to inspect state or free heavy cached objects
        # do not clear _state by default so backtester can inspect it after finalize
        pass


@dataclass
class Trade:
    date: pd.Timestamp
    shares: float
    price: float
    commission: float
    note: Optional[str] = None



class Backtester:
    """
    
    The core Backtesting Engine. 
    
    Parameters
    ----------
    tradable_assets : List[str]
        The assets that the strategy will make trades on. 
        
    strategy_fn : Strategy
        The function that holds the core logic of the strategy. Must be wrapped into a Strategy object before it works. 
    
    data : pd.DataFrame
        The dataset that the strategy reads to make predictions. Can include more than price data from tradable_assets. 
        The dataset inputted will have the following restrictions:
        1. The columns that consists of the price that the strategy trades on MUST be named the ticker of the asset. E.g. if trading on Adj_Close price of NVDA, change Adj_Close column name to NVDA
        2. The index of the dataset must be in the form of a datetime index. 

    allow_short : bool, optional
        Straightforward.
        
    cash_start : float, optional
        The amount of capital that the strategy starts with. 
        
    commission : float, optional
        The commission obtained by the "broker" when trades are made. (default=0.0)
        
    slippage : float, optional
        
        
    execute_on_next_tick : bool, optional
        If True, portfolio is submitted
        
    periods_per_year : int, optional
        Will be depreciated soon. 
    
    riskfree_per_period : float, optional
        Will be depreciated soon. 
    
        
    Attributes
    ----------
    tradable_assets : List[str]
        The assets that the strategy will make trades on. 

    data : pd.DataFrame
        The dataset that the strategy reads to make predictions. 

    dates : pd.DatetimeIndex
        The date index of the dataset.
    
    cash_start : float
        The cash at the start of the 
        
    current_nav : float
        
    position : pd.Series
        
    current_shares : pd.Series
        
    cash : float
        
    commission : float
        
    slippage : float
        
    allow_short : bool
        Straightforward.

    execute_on_next_tick : bool
        If True, a requested portfolio is only executed on the prices of the next tick in the dataset.
    
    periods_per_year : int
        Staged for depreciation. 

    riskfree : float
        Staged for depreciation
        
    nav_history : pd.Series
        The history of Net Asset Value of the dataset. 
    
    position_history : pd.DataFrame
        The history of weights in each asset during the history of the backtest. 
        
    cash_history : pd.Series
        The amount of cash held by the strategy throughout the history of the backtest. 
        
    trades : List[Trade]
        The list of all trades sent to the 
        
    signal_history : List[float]
        The list of all signals generated by the strategy. 
        
    strategy_object : Strategy
        The core strategy object that is tested in the engine. 
        
    strategy_name : str
        The name of the strategy that can be used in plots later. 
        
    backtest_complete : bool
        
    portfolio_returns : pd.Series
        The returns series of the strategy. 
        
    """
    def __init__(self,
                tradable_assets: List[str], 
                strategy_fn: Strategy, # The strategy function I want to test on 
                strategy_name: str, # The name of the strategy for plotting
                data: pd.DataFrame, # the data that the strategy operates on
                allow_short=True,
                cash_start: float = 100_000.0,
                commission: float = 0.0,  
                slippage: float = 0.0, 
                execute_on_next_tick: bool = True,
                periods_per_year: int = 252,  # used for annualization
                riskfree_per_period: float = 0.0  # per-period risk-free rate (e.g., daily)   
                ):
        
        

        assert isinstance(data, pd.DataFrame), "Input data must be given as a pd.DataFrame indexed by date."
        

        # Check the data that we need for signal generation
        self.tradable_assets = tradable_assets 
        self.data = data

        ## need to force that the data index is in the form of dates
        self.dates = data.index
        
        # Current portfolio state
        self.cash_start = cash_start
        self.current_nav = cash_start
        self.position : pd.Series = pd.Series({asset:0 for asset in self.tradable_assets})
        self.current_shares: pd.Series = pd.Series({asset:0 for asset in self.tradable_assets})
        self.cash = cash_start

        # execution and model parameters
        self.commission = commission
        self.slippage = slippage
        self.allow_short = allow_short
        self.execute_on_next_tick = execute_on_next_tick
        self.periods_per_year = periods_per_year
        self.riskfree = riskfree_per_period

        # history
        self.nav_history: pd.Series = pd.Series(np.zeros(len(self.data)), index=self.dates)
        self.position_history: pd.DataFrame = pd.DataFrame(np.zeros((len(self.data), len(self.tradable_assets))),  columns= self.tradable_assets, index= self.dates)                            
        self.cash_history: pd.Series = pd.Series(np.zeros(len(self.data)), index=self.dates)
        # fraction of NAV invested in asset (dollars_in_asset / nav)
        self.trades: List[Trade] = []
        self.signal_history: List[float] = []     # raw signals returned by strategy

        # Make the engine flexible enough that it 
        self.strategy_object = strategy_fn
        self.strategy_name = strategy_name

        # For plotting later 
        self.backtest_complete = False

        # For visualising results
        self.portfolio_returns = None


    # Debugging Functions 
    def _current_state(self):
        # Return a snapshot of current portfolio state (useful for debugging).
        current_prices = self.data.loc[:, self.tradable_assets]
        return {
            "cash": self.cash,
            # To dict method converts pandas Series to dictionary for easier view 
            "shares": self.position.to_dict(),
            "nav": self.cash + self.position * current_prices
        }

    # Another debugging function
    def _reset(self):
        # Reset only histories, not prices/strategy/config — useful if you want to re-run with same object.
        self.cash = float(self.cash_start)
        self.position = 0.0
        self.nav_history.clear()
        self.position_history.clear()
        self.trades.clear()
        self.signal_history.clear()


    def run(self, verbose=False):
        """
        Runs the backtesting engine.
        
        Parameters
        ----------
        verbose : bool, optional
        If True, returns output log each time a trade occurs, with the size of the trade and the amount traded for.
            
        Returns
        -------
        None
        """
        self.strategy_object.initialize()

        pending_target = None
        T = len(self.data) 

        pending_cash_frac = None 
        pending_asset_frac = None

        # Starting next tick execution 
        for t in range(T):
            date = self.dates[t]
            current_prices = self.data.loc[date, self.tradable_assets]

            # 1) record NAV and position BEFORE generating new signal or executing new trades.
            self.current_nav = self.cash + (current_prices * self.current_shares).sum()
            self.nav_history.loc[date] = self.current_nav
            
            self.cash_history[date] = self.cash
            self.current_position = (current_prices * self.current_shares / self.current_nav) if self.current_nav != 0 else 0.0 
            self.position_history.loc[date, :] = self.current_position
    

            # 2) Perform next tick execution. 
            if self.execute_on_next_tick and (pending_cash_frac is not None) and (pending_asset_frac is not None):
                if verbose:
                    print(f"{date.date()}: executing pending target\n{pending_asset_frac}")

                    print(f"Current prices executed at: \n{current_prices}\n")
                self._execute_target_portfolio(pending_cash_frac, pending_asset_frac, current_prices, date)
                pending_asset_frac = None
                pending_cash_frac= None

            # 3) Generate strategy signal using price history up to t

            # The try except blocks are here for debugging - if the code breaks here we know that it broke here.
            try:
                # The strategy is returned as a class of portfolio weights. I think it only makes sense to return cash as a fraction of NAV as well.
                pending_cash_frac, pending_asset_frac = self.strategy_object.predict(self.tradable_assets, self.data.iloc[:t+1, :], self.position_history.iloc[:t + 1, :])

            except Exception as e:
                raise RuntimeError(f"Error in strategy.predict at index {t} date {date}: {e}")
            
            # 4) Execution: Code for executing immediately if specified in the model
            if not self.execute_on_next_tick:
                if verbose:
                    print(f"{date.date()}: executing pending target \n{pending_asset_frac}")
                    print(f"\nCurrent prices executed at: \n{current_prices}")
                self._execute_target_portfolio(pending_cash_frac, pending_asset_frac, current_prices, date)
                pending_target = None
        

        # Calculate backtest returns after backtest is complete.
        self.portfolio_returns = ( (self.nav_history) / (self.nav_history.shift(1)) - 1).dropna()
        
        # To call other plots later, must first be sure that the backtest is done.
        self.backtest_complete=True 
        return 


    def _execute_target_portfolio(self, target_cash: float, target_asset_frac: pd.Series, exec_price: pd.Series, date: pd.Timestamp):

        # """
        # Convert a target fraction into a trade and update cash/shares.
        # target_asset_frac is clamped to [-1,1] unless allow_short is False (then clamp to [0,1]).
        # The logic:
        #     - compute target dollar exposure = target_asset_frac * NAV
        #     - compute delta dollars = target_dollars - current_dollars
        #     - compute trade price with slippage
        #     - compute trade shares = delta_dollars / trade_price
        #     - update shares and cash and charge commission
        # """
        assert len(target_asset_frac) == len(exec_price), "For some reason your execution price is not the same as the tradable assets. Try checking the backtesting code."

        # clamp
        if not self.allow_short:
            target_asset_frac = target_asset_frac.clip(lower=0, upper=1)
        else:
            target_asset_frac = target_asset_frac.clip(lower=-1, upper=1)


        # if nav is zero, we can't sensibly trade
        # """
        # There needs to be error handling in this section. 
        # """
        if self.current_nav <= 0:
            print("Net Asset Value is now 0. No trades can occur.")
            return 

        # target_dollars = target_asset_frac * nav + target_cash * nav
        # current_dollars = current_shares * exec_price  + target_cash * nav
        # delta_dollars = target_dollars - current_dollars
        
        # UPDATE: 4/12/25
        # Modelling decision - we allow cash to go negative 

        # """
        # Logic of this section:
        # 1. An order is received, based on the relative weights of the portfolio wrt to net asset value
        # 2. From the weights relative to NAV, calculate the number of shares we need to purchase to achieve the desired weights of the portfolio. 
        # """

        # Calculating the number of shares I want to buy of each asset (not accounting for transaction costs yet). This is done by calculating the absolute amount of cash holding in each asset desired, then dividing by the current trade price to obtain the number of shares

        target_asset_value = target_asset_frac * self.current_nav
        target_cash_value = target_cash * self.current_nav
        current_asset_value = self.current_position * self.current_nav
        current_cash_value = self.cash 

        # I calculate the total amount I want to spend on each asset here, by subtracting my target off my current asset value
        delta_asset_value = target_asset_value - current_asset_value
        delta_cash_value = target_cash_value - current_cash_value

        shares_wanted = delta_asset_value / exec_price 

        # if tiny change, skip (reduce noise): 
        if abs(delta_asset_value.sum()) + abs(delta_cash_value) < 1e-8 * self.current_nav:
            print("Trade not executed because position rebalance was too small.")
            return

        # Important to change the dynamics here - each item that is rebalanced in the portfolio should be charged for. 

        # slippage: worse price depending on direction (buy => higher price, sell => lower price)
        trade_price = exec_price * (1.0 + self.slippage * np.sign(delta_asset_value))
        
        # Obtain the net effect of the trade - the portfolio might sell and buy shares, and this just calculates the excess profit to subtract off cash 
        trade_value_excess = (shares_wanted * trade_price).sum()
        
        # commission handling: fraction if <1 else flat fee
        if 0 <= self.commission < 1:
            commission_cost = abs(trade_value_excess) * self.commission
        else:
            commission_cost = float(self.commission)
        
        trade_value_excess += commission_cost

        # update portfolio
        #if self.cash - trade_value_excess < 0:
            #print(f"Not enough to pay the commission at time {date.date()}- trade was not executed.")
            #return
        
        self.current_shares += shares_wanted
        self.cash -= trade_value_excess

        # record trade for auditing
        self.trades.append(Trade(date=date, shares=dict(shares_wanted), price=dict(trade_price), commission=float(commission_cost)))

        return
