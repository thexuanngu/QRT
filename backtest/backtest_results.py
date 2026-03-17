from backtest.backtesting import Backtester
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

class VisualiseBacktestResults:
    """
    Docstring for VisualiseBacktestResults

    """
    def __init__(self, 
                 backtester: Backtester,
                 benchmark: Backtester, 
                 riskfree_rate: float = 0.0):
        
        assert backtester.backtest_complete, "This backtest has not been run."
        assert benchmark.backtest_complete, "This benchmark backtest has not been run."
        
        self.backtester = backtester
        self.benchmark = benchmark
        self.riskfree_rate = riskfree_rate

        # performance metrics 
        self._performance_metrics = ["net_nav", "cagr", "sharpe", "sortino", "calmar", "max_drawdown", "max_drawdown_duration", "average_recovery_duration", "average_drawdown"]

        # benchmark performance metrics 
        self._strategy_metrics = dict((metric, None) for metric in self._performance_metrics)
        self._benchmark_metrics = dict((metric, None) for metric in self._performance_metrics)
        self._drawdown = pd.Series(dtype=float)
        
    def _calculate_pnl(self):
        # basic performance metrics - cagr and pnl 

        nav = self.backtester.nav_history 
        start_nav, final_nav = nav.iloc[0], nav.iloc[-1]
        net_nav= final_nav - start_nav

        num_years = (nav.index[-1] - nav.index[0]).days / 365.25
        cagr = (final_nav / start_nav) ** (1 / num_years) - 1 
        self._strategy_metrics["net_nav"] = net_nav
        self._strategy_metrics["cagr"] = cagr
        return dict(
            net_nav=self._strategy_metrics["net_nav"],
            cagr=self._strategy_metrics["cagr"],
        )


    def _calculate_drawdown(self):

        nav = self.backtester.nav_history
        rolling_max = nav.cummax()
        drawdown = (nav - rolling_max) / rolling_max
        self._drawdown = drawdown
        max_drawdown = drawdown.min()
        self._strategy_metrics["max_drawdown"] = max_drawdown

        # Max Drawdown Duration
        end_of_drawdown = drawdown[drawdown == 0].index
        max_duration = pd.Timedelta(0)
        for i in range(1, len(end_of_drawdown)):
            start = end_of_drawdown[i - 1]
            end = end_of_drawdown[i]
            duration = end - start
            if duration > max_duration:
                max_duration = duration
        self._strategy_metrics["max_drawdown_duration"] = max_duration

        # Average Drawdown
        self._strategy_metrics["average_drawdown"] = drawdown[drawdown < 0].mean()

        return dict(
            max_drawdown=self._strategy_metrics["max_drawdown"],
            max_drawdown_duration=self._strategy_metrics["max_drawdown_duration"],
            average_drawdown=self._strategy_metrics["average_drawdown"],
        )
    

    def _calculate_performance_metrics(self):
        returns = self.backtester.portfolio_returns
        self._strategy_metrics["sharpe"] = returns.mean() / returns.std() * np.sqrt(252)  # Annualized Sharpe Ratio

        # Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std()
        self._strategy_metrics["sortino"] = returns.mean() / downside_deviation * np.sqrt(252)

        # Calmar ratio 
        self._strategy_metrics["calmar"] = returns.mean() / abs(self._strategy_metrics["max_drawdown"])
        return 

    def performance_dashboard(self):    

        self._calculate_pnl()
        self._calculate_drawdown()
        self._calculate_performance_metrics()

        fig = plt.figure(figsize=(14, 9), constrained_layout=True)

        # Create a grid: 6 rows x 6 columns
        gs = fig.add_gridspec(6, 6)

        # Top: equity curve (rows 0–2, all columns)
        ax_pnl = fig.add_subplot(gs[0:3, 0:6])
        ax_pnl.plot(self.backtester.nav_history.index, self.backtester.nav_history - self.backtester.nav_history.iloc[0], color="blue", label=self.backtester.strategy_name)
        ax_pnl.plot(self.benchmark.nav_history.index, self.benchmark.nav_history - self.benchmark.nav_history.iloc[0], color="orange", linestyle="--", label=self.benchmark.strategy_name)
        ax_pnl.legend()
        ax_pnl.set_title("Total PnL Over Time")

        # Middle: drawdown (row 3, all columns)
        ax_drawdown = fig.add_subplot(gs[3:4, 0:6])
        ax_drawdown.plot(self._drawdown.index, self._drawdown, color="red")
        ax_drawdown.fill_between(self._drawdown.index, 0, self._drawdown, color="salmon", alpha=0.5)
        ax_drawdown.set_title("Drawdown Over Time")
        ax_drawdown.axhline(y=self._strategy_metrics["max_drawdown"], color='black', linestyle='--', label='Max Drawdown')

        # Bottom-right: metrics card (rows 4–5, cols 0–2)
        ax_metrics = fig.add_subplot(gs[4:6, 0:2])
        metrics_text = f'''CAGR: {self._strategy_metrics["cagr"]:.2%}
        \nSharpe: {self._strategy_metrics["sharpe"]: .4f}
        \nSortino: {self._strategy_metrics["sortino"]: .4f}
        \nCalmar: {self._strategy_metrics["calmar"]: .4f}
        \nMax Drawdown: {self._strategy_metrics["max_drawdown"]: .4f}
        \nMax Drawdown Duration: {self._strategy_metrics["max_drawdown_duration"]}
        \nAverage Drawdown: {self._strategy_metrics["average_drawdown"]: .4f}'''

        ax_metrics.text(0.00, 0.98, metrics_text,
        transform=ax_metrics.transAxes,
        va='top',
        family='monospace', 
        linespacing=0.50)
        ax_metrics.axis('off')
        ax_metrics.set_title("Performance Metrics", loc='left')
        
        return 

    def calculate_alpha_versus_benchmark(self):

        # Basic error handling - exceptions should be thrown if not completed.
        assert len(self.benchmark.portfolio_returns) == len(self.backtester.portfolio_returns), "Benchmark returns length does not match portfolio returns length."
        
        # Straightforward - calculating the portfolio returns 
        portfolio_excess = self.backtester.portfolio_returns - self.riskfree_rate
        market_excess = self.benchmark.portfolio_returns - self.riskfree_rate

        # Prepares the x variable (market risk-adjusted returns) for regression
        market_excess_regready = sm.add_constant(market_excess)

        # Fit the OLS regression
        model = sm.OLS(portfolio_excess, market_excess_regready).fit()

        # Summarise the findings for alpha (also need to make it intepretable)
        print(f"Constant term of the regression: {model.params['const']} \n")
        print("NB: The alpha value is given in the constant term. \n")
        print(model.summary())

        return
    
    def ex_ante_risk_calculator(self, prices: pd.DataFrame, lookback: int = 60) -> pd.Series:
        """
        Compute ex-ante risk (annualised std dev of hypothetical daily P&L)
        following the method in the screenshot:
        - For each date t, form daily returns for the previous `lookback` days
            and compute the P&L series that WOULD have occurred using today's
            dollar positions (positions at t).
        - For days where price is NaN for an asset (asset not tradable then),
            that asset contributes 0 P&L for that day.
        - Risk at t = std(daily_pnl_series) * sqrt(252)  (annualised volatility).
        
        Returns
        -------
        pd.Series
            Annualised risk (same index as position_history).
        """
        # Basic checks / assertions
        assert hasattr(self.backtester, "position_history"), "Backtester must have position_history."
        if self.backtester.portfolio_returns is None:
            raise AssertionError("Backtester.portfolio_returns is None — run backtest before calling this method.")
        assert isinstance(self.backtester.portfolio_returns.index, pd.DatetimeIndex), (
            "backtester.portfolio_returns must be indexed by pd.DatetimeIndex."
        )

        position_history = self.backtester.position_history
        dates = position_history.index
        N = len(position_history)

        # Validate prices shape / index alignment assumptions
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise AssertionError("`prices` must have a pd.DatetimeIndex.")

        # Prepare output series
        risk = pd.Series(index=dates, dtype=float)

        # We'll iterate through each date (could be vectorized further,
        # but loop is clear and safe with the alignment/edge handling).
        for i, date in enumerate(dates):
            # Build price window up to and including this date
            start_idx = max(0, i - lookback)
            # Using iloc to slice by row-position is safe if `prices` rows correspond to same ordering;
            # fall back to date-based slicing if indexes match exactly.
            try:
                prices_for_period = prices.iloc[start_idx : i + 1].copy()
            except Exception:
                # Fallback to label-based slicing
                prices_for_period = prices.loc[dates[start_idx] : date].copy()

            # Need at least one return datapoint to compute std
            if prices_for_period.shape[0] < 2:
                risk.iloc[i] = np.nan
                continue

            # Compute daily returns (fractional). This yields NaN where price was missing.
            returns = prices_for_period.pct_change().iloc[1:]  # drop the first NaN row from pct_change

            # Today's position (dollar notionals). Align columns to ensure consistent ordering.
            current_position: pd.Series = position_history.iloc[i].astype(float)
            # Ensure returns have the same columns as current_position (assets absent historically -> cols missing)
            returns_aligned = returns.reindex(columns=current_position.index)

            # Where price was NaN historically, returns will be NaN -> those contributions should be zero.
            returns_aligned = returns_aligned.fillna(0.0)

            # Compute daily P&L series in dollars: for each day sum(return * today's position)
            # returns_aligned is (days x assets), current_position is (assets,)
            daily_pnl = (returns_aligned * current_position).sum(axis=1)

            # If there's no variation in daily_pnl, set risk to 0 (no volatility)
            std_profit = daily_pnl.std(ddof=0)  # population std is fine for a volatility estimate
            if np.isnan(std_profit) or std_profit == 0.0:
                risk.iloc[i] = 0.0
                continue

            # Annualise
            annualised_vol = std_profit * np.sqrt(252.0)
            risk.iloc[i] = annualised_vol

        # keep the same index / name
        risk.name = "ex_ante_annualised_pnl_vol"
        return risk



