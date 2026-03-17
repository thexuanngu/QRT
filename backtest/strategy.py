from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

from backtest.exceptions import StrategyError, StrategyNotImplemented
from backtest.data_models import TradingState


class Strategy(ABC):
    """
    Abstract base Strategy class.

    Lifecycle:
      - initialize() : called once at the start of backtest
      - predict(history) : called every tick with historical data up to now -> returns target
      - finalize() : called once at end (reporting/cleanup)
      - reset() : optional to reuse the object in repeated tests

    The class stores internal state in self._state.
    """

    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Override for one-off initialization (seeding RNGs, precomputing)."""
        self._initialized = True

    @abstractmethod
    def predict(self, tradable_assets, history, trading_state: TradingState):
        """
        Core API that the backtester will call at each tick.
        Raises StrategyNotImplemented if not implemented yet.
        """
        raise StrategyNotImplemented

    def finalize(self) -> None:
        """Called once after backtest finishes."""
        pass

    def reset(self) -> None:
        """Reset internal state so strategy can be reused in repeated backtests."""
        self._state = {}
        self._initialized = False

    @staticmethod
    def _validate_target(target) -> None:
        """
        Ensure target is a supported return type and numeric values are finite.
        """
        valid_types = (float, int, pd.Series, dict, type(None))
        if not isinstance(target, valid_types):
            raise StrategyError(
                f"Unsupported target type: {type(target)}. Must be float, pd.Series, or dict."
            )

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
     - per-tick (event): fn(current_index, history_up_to_index: pd.DataFrame, state: dict) -> float/Series/dict

    Create with mode='vector' or mode='event'.
    """

    def __init__(
        self,
        user_fn: Callable,
        tradable_assets: Tuple[str],
        required_params: Tuple[str] = None,
        optional_params: Tuple[str] = None,
        mode: str = "event",
        params: Optional[Dict[str, Any]] = None,
    ):
        if mode not in ("event", "vector"):
            raise ValueError("mode must be 'event' or 'vector'")

        super().__init__()
        self.user_fn = user_fn
        self.required_params = required_params
        self.optional_params = optional_params
        self.tradable_assets = tradable_assets
        self.mode = mode

        self._vector_cache: Optional[pd.Series] = None

    def initialize(self) -> None:
        super().initialize()
        self._state = {}
        self._vector_cache = None

    def predict(self, tradable_assets, history, trading_state: TradingState):
        """
        Return the target portfolio for the current tick.
        """
        if not isinstance(history, pd.DataFrame):
            raise StrategyError("history must be a pandas DataFrame")

        if history.empty:
            return 0.0

        current_index = history.index[-1]
        history_view = history.copy(deep=False)

        target_portfolio = None
        if self.mode == "event":
            try:
                target_portfolio = self.user_fn(
                    tradable_assets,
                    history_view,
                    trading_state,
                    self._state,
                )
            except TypeError:
                try:
                    target_portfolio = self.user_fn(
                        tradable_assets,
                        history_view,
                        trading_state,
                    )
                except TypeError:
                    target_portfolio = self.user_fn(
                        tradable_assets,
                        history_view,
                        trading_state.position_history.copy(deep=False),
                    )
            except AttributeError:
                target_portfolio = self.user_fn(
                    tradable_assets,
                    history_view,
                    trading_state,
                )
        else:
            if self._vector_cache is None or len(self._vector_cache) < len(history):
                vector_out = self.user_fn(history_view)
                if isinstance(vector_out, (pd.Series, pd.DataFrame)):
                    vec = pd.Series(vector_out.squeeze()).reindex(history.index)
                    self._vector_cache = vec.astype(float)
                elif isinstance(vector_out, (float, int)):
                    self._vector_cache = pd.Series(
                        [float(vector_out)] * len(history), index=history.index
                    )
                elif isinstance(vector_out, dict):
                    self._vector_cache = pd.Series([vector_out], index=history.index)
                else:
                    raise StrategyError("vector mode: user function returned unsupported type")

            target_portfolio = self._vector_cache.loc[current_index]

        self._validate_target(target_portfolio)
        return target_portfolio

    def finalize(self) -> None:
        pass
