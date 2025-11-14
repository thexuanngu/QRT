# (Trimmed snippet — use the full code run above for direct copy)

from typing import Callable, Optional, List, Dict, Any, Union, Sequence

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import statsmodels.api as sm 


@dataclass
class StrategyMeta:

    name: str
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


class StrategyError(Exception):
    pass

class NotImplementedError(Exception):
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


    def __init__(self, name):
        #self.meta = meta or StrategyMeta(name=self.__class__.__name__)
        self.name = name
        self._state: Dict[str, Any] = {}
        self._initialized = False


    # lifecycle hooks
    def initialize(self) -> None:
        """Override for one-off initialization (seeding RNGs, precomputing)."""
        self._initialized = True


    @abstractmethod
    def predict(self, history: pd.DataFrame) -> Union[float, pd.Series, Dict[str, float]]:
        """
        Core API that the backtester will call at each tick.

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
        raise NotImplementedError


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
        valid_types = (float, int, pd.Series, dict)
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
                 name: str,
                 user_fn: Callable, # MUST be a class event 
                 mode: str = "event",
                 params: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        user_fn : callable
            The user-supplied function.
        mode : {'event', 'vector'}
            'event' calls the function once per tick with (index, history_up_to_index, state).
            'vector' calls the function once with the full history and expects vector output.
        """
        if mode not in ("event", "vector"):
            raise ValueError("mode must be 'event' or 'vector'")

        #meta = meta or StrategyMeta(name=getattr(user_fn, "__name__", "function_strategy"),
        #                            params=params or {})
        # Force the strategy to be a function. 
        assert (isinstance(user_fn, Strategy)), "Please only use a Strategy class as the function"

        super().__init__(name=name)
        self.user_fn = user_fn
        self.mode = mode
        # keep a cache for vector mode to avoid recomputing on every tick
        self._vector_cache: Optional[pd.Series] = None

    def initialize(self) -> None:
        super().initialize()
        # clear caches/state
        self._state = {}
        self._vector_cache = None

    def predict(self, history: pd.DataFrame) -> Union[float, pd.Series, Dict[str, float]]:
        """
        Return the target portfolio for the *current* tick. Backtester should call this
        once per tick passing history up to and including the current tick.

        In 'vector' mode, we lazily compute the whole series on first call and then
        return the value for the current index.
        """

        if not isinstance(history, pd.DataFrame):
            raise StrategyError("history must be a pandas DataFrame")

        if history.empty:
            # no data yet -> neutral
            return 0.0

        current_index = history.index[-1]

        if self.mode == "event":
            # user_fn(index, history_up_to_index, state) -> target
            try:
                target_portfolio = self.user_fn(current_index, history.copy(), self._state)
            except TypeError:
                # some users may ignore index and state; try fallback
                target_portfolio = self.user_fn(history.copy())

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
        return target_portfolio

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
    def __init__(self,
                tradable_assets: List[str], 
                data: pd.DataFrame, # the data that the strategy operates on
                name: str, # The name of the strategy
                strategy_fn: Callable, # type is undecided
                benchmark: Optional[pd.Series], 
                allow_short=False,
                cash_start: float = 100_000.0,
                commission: float = 0.0,  
                slippage: float = 0.0, 
                execute_on_next_tick: bool = True,
                periods_per_year: int = 252,  # used for annualization
                riskfree_per_period: float = 0.0  # per-period risk-free rate (e.g., daily)   
                 ):
        """
        
        
        
        
        
        
        """
        
        assert isinstance(data, pd.DataFrame), "Input data must be given as a pd.DataFrame indexed by date."

        if benchmark is not None:
            assert len(benchmark) == len(data)
        

        # Check the data that we need for signal generation
        self.tradable_assets = tradable_assets # Check if I can optimise it here 
        self.data = data
        self.benchmark = benchmark.copy().astype(float) if benchmark is not None else None
        self.dates = data.index
        
        # Question for me in the future: Should 


        # Current portfolio state
        self.start_cash = cash_start
        self.position : pd.Series = pd.Series({asset:0 for asset in self.tradable_assets})
        self.cash = self.start_cash

        # execution and model parameters
        self.commission = commission
        self.slippage = slippage
        self.allow_short = allow_short
        self.execute_on_next_tick = execute_on_next_tick
        self.periods_per_year = periods_per_year
        self.riskfree = riskfree_per_period

        # history
        self.nav_history: pd.Series = pd.Series(np.zeros(len(self.data)), index=self.dates)
        self.position_history: pd.DataFrame = pd.DataFrame(np.zeros(len(self.data), len(self.tradable_assets)), 
                                                           columns=self.tradable_assets, 
                                                           index=self.dates
                                                           )   # fraction of NAV invested in asset (dollars_in_asset / nav)
        self.trades: List[Trade] = []
        self.signal_history: List[float] = []     # raw signals returned by strategy

        # Change the strategy function to be compatible with Strategy Class.
        self.strategy_fn = FunctionStrategy(name, strategy_fn, mode="event")

        # For plotting later 
        self.backtest_complete = False

        # For visualising results
        self.portfolio_returns = None


    def run(self, verbose=False):
        self.strategy_fn.initialize()

        pending_target = None
        T = len(self.data) 

        # Starting next tick execution 
        for t in range(T):
            date = self.dates[t]
            current_prices = self.data[t, self.tradable_assets]

            # 1) Start with next tick execution. 
            if self.execute_on_next_tick and pending_target:
                if verbose:
                    print(f"{date.date()}: executing pending target {pending_target:.3f} at price {px:.4f}")
                self._execute_target_portfolio(pending_target, current_prices, date)
                pending_target = None

            # 2) record NAV and position BEFORE generating new signal or executing new trades.
            nav = self.cash + current_prices * self.position
            self.nav_history.append(nav)

            pos_frac = (current_prices * self.position / nav) if nav != 0 else 0.0 
            self.position_history.append(pos_frac) 

            # 3) Generate strategy signal using price history up to t
            try:
                # The strategy is returned as a class of portfolio weights.
                target_frac = float(self.strategy_fn.predict(self.data.iloc[:t+1, :], ))

            except Exception as e:
                raise RuntimeError(f"Error in strategy.predict at index {t} date {date}: {e}")

            # 4) Execution: Either execute trade now or execute on next tick
            if self.execute_on_next_tick:
                pending_target = target_frac
            else:
                if verbose:
                    print(f"{date.date()}: executing target immediately: {pending_target:.3f} at price {px:.4f}")
                self._execute_target_portfolio(target_frac, current_prices, date)
                pending_target = None
        
        # To call other plots later, must first be sure that the backtest is done.
        self.backtest_complete=True 
        return 

    def _execute_target_portfolio(self, target_frac: pd.Series, exec_price: pd.Series, date: pd.Timestamp):

        """
        Convert a target fraction into a trade and update cash/shares.
        target_frac is clamped to [-1,1] unless allow_short is False (then clamp to [0,1]).
        The logic:
            - compute target dollar exposure = target_frac * NAV
            - compute delta dollars = target_dollars - current_dollars
            - compute trade price with slippage
            - compute trade shares = delta_dollars / trade_price
            - update shares and cash and charge commission
        """

        # clamp
        if not self.allow_short:
            target_frac = max(0.0, target_frac)
        else:
            target_frac = max(-1.0, min(1.0, target_frac))


        nav = self.cash + self.position * exec_price
        # if nav is zero, we can't sensibly trade
        if nav <= 0:
            return 

        # The amount of extra cash we need to perform the trade 
        target_dollars = target_frac * nav
        current_dollars = self.position * exec_price
        delta_dollars = target_dollars - current_dollars 

        # Include logic to stop execution if delta dollars is greater than current cash



        # if tiny change, skip (reduce noise)
        if abs(delta_dollars.sum()) < 1e-8 * nav:
            return

        # slippage: worse price depending on direction (buy => higher price, sell => lower price)
        trade_price = exec_price * (1.0 + self.slippage * np.sign(delta_dollars))

        # trade shares (signed)
        trade_shares = delta_dollars / trade_price

        trade_value = (trade_shares * trade_price).sum()  # signed value

        # commission handling: fraction if <1 else flat fee
        if 0 <= self.commission < 1:
            commission_cost = abs(trade_value) * self.commission
        else:
            commission_cost = float(self.commission)
        
        # update portfolio
        self.position += trade_shares
        self.cash -= trade_value + commission_cost

        # record trade for auditing
        self.trades.append(Trade(date=date, shares=dict(trade_shares), price=dict(trade_price), commission=float(commission_cost)))

        return



    def current_state(self):
        """Return a snapshot of current portfolio state (useful for debugging)."""
        current_prices = self.data.loc[:, self.tradable_assets]
        return {
            "cash": self.cash,
            # To dict method converts pandas Series to dictionary for easier view 
            "shares": self.position.to_dict(),
            "nav": self.cash + self.position * current_prices
        }


    def reset(self):
        """Reset only histories, not prices/strategy/config — useful if you want to re-run with same object."""
        self.cash = float(self.start_cash)
        self.position = 0.0
        self.nav_history.clear()
        self.position_history.clear()
        self.trades.clear()
        self.signal_history.clear()
    
    def calculate_results(self):
        assert self.backtest_complete, ".run() method has not been called."
        
        # Calculate returns of the portfolio
        self.portfolio_returns = (self.nav_history / self.nav_history.shift(1)) - 1 

        return
    
    def visualise_pnl(self, target_risk):
        assert self.backtest_complete, ".run() method has not been called."

        # Risk-Weighting Portfolio. 
        risk_adjusted_returns = target_risk * self.portfolio_returns / self.portfolio_returns.std()

        return risk_adjusted_returns
    
    def calculate_alpha(self):
        # 
        portfolio_excess = self.portfolio_returns - self.riskfree
        market_excess = self.benchmark - self.riskfree

        # Prepares the x variable (market risk-adjusted returns) for regression
        portfolio_excess_new = sm.add_constant(market_excess)

        # Fit the OLS regression
        model = sm.OLS(portfolio_excess_new, market_excess)
        model.fit()

        # Summarise the findings for alpha (also need to make it intepretable)
        print("NB: The alpha value is given in the constant term.")
        model.summary()
        return
        