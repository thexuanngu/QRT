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
                 backtester,
                 benchmark, 
                 riskfree_rate=0.0):
        
        assert backtester.backtest_complete, "This backtest has not been run."
        
        self.backtester = backtester
        self.benchmark = benchmark
        self.riskfree_rate = riskfree_rate

        # performance metrics 
        self._performance_metrics = ["net_nav", "cagr", "sharpe", "sortino", "calmar", "max_drawdown", "max_drawdown_duration", "average_recovery_duration", "average_drawdown"]

        # benchmark performance metrics 
        self._strategy_metrics = dict((metric, None) for metric in self._performance_metrics)
        self._benchmark_metrics = dict((metric, None) for metric in self._performance_metrics)
        
    def _calculate_pnl(self):
        # basic performance metrics - cagr and pnl 

        nav = self.backtester.nav_history 
        start_nav, final_nav = nav.iloc[0], nav.iloc[-1]
        net_nav= final_nav - start_nav

        num_years = (nav.index[-1] - nav.index[0]).days / 365.25
        cagr = (final_nav / start_nav) ** (1 / num_years) - 1 
        return dict(net_nav=net_nav, cagr=cagr)


    def _calculate_drawdown(self):

        nav = self.backtester.nav_history
        rolling_max = nav.cummax()
        drawdown = (nav - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Max Drawdown Duration
        end_of_drawdown = drawdown[drawdown == 0].index
        max_duration = pd.Timedelta(0)
        for i in range(1, len(end_of_drawdown)):
            start = end_of_drawdown[i - 1]
            end = end_of_drawdown[i]
            duration = end - start
            if duration > max_duration:
                max_duration = duration
        self._max_drawdown_duration = max_duration

        # Average Drawdown
        self._average_drawdown = drawdown[drawdown < 0].mean()

        return dict(max_drawdown=max_drawdown, max_drawdown_duration=self._max_drawdown_duration, average_drawdown=self._average_drawdown)
    

    def _calculate_performance_metrics(self):
        returns = self.backtester.portfolio_returns
        self._sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized Sharpe Ratio

        # Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std()
        self._sortiono = returns.mean() / downside_deviation * np.sqrt(252)

        # Calmar ratio 
        self._calmar = returns.mean() / abs(self._max_drawdown)

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
        ax_pnl.plot(self.backtester.nav_history.index, self.backtester.nav_history - self.backtester.nav_history.iloc[0], color="blue")
        ax_pnl.fill_between(self.backtester.nav_history.index, 0, self.backtester.nav_history - self.backtester.nav_history.iloc[0], color="lightblue", alpha=0.5)
        ax_pnl.set_title("Total PnL Over Time")

        # Middle: drawdown (row 3, all columns)
        ax_drawdown = fig.add_subplot(gs[3:4, 0:6])
        ax_drawdown.plot(self._drawdown.index, self._drawdown, color="red")
        ax_drawdown.fill_between(self._drawdown.index, 0, self._drawdown, color="salmon", alpha=0.5)
        ax_drawdown.set_title("Drawdown Over Time")
        ax_drawdown.axhline(y=self._max_drawdown, color='black', linestyle='--', label='Max Drawdown')

        # Bottom-right: metrics card (rows 4–5, cols 0–2)
        ax_metrics = fig.add_subplot(gs[4:6, 0:2])
        metrics_text = f'''CAGR: {self._cagr}%
        \nSharpe: {self._sharpe: .4f}
        \nSortino: {self._sortiono: .4f}
        \nCalmar: {self._calmar: .4f}
        \nMax Drawdown: {self._max_drawdown: .4f}
        \nMax Drawdown Duration: {self._max_drawdown_duration}
        \nAverage Drawdown: {self._average_drawdown: .4f}'''

        ax_metrics.text(0.00, 0.98, metrics_text,
        transform=ax_metrics.transAxes,
        va='top',
        family='monospace', 
        linespacing=0.50)
        ax_metrics.axis('off')
        ax_metrics.set_title("Performance Metrics", loc='left')
        
        return 

    def benchmark(self, benchmark_returns, riskfree=0.0):
        
        # Basic error handling - exceptions should be thrown if not completed.
        assert len(benchmark_returns) == len(self.backtester.portfolio_returns), "Benchmark returns length does not match portfolio returns length."
        #assert benchmark is not None, "Benchmark needs to be loaded in before testing can properly be done."
        
        # Straightforward - calculating the portfolio returns 
        portfolio_excess = self.backtester.portfolio_returns - riskfree
        market_excess = benchmark_returns - riskfree

        # Prepares the x variable (market risk-adjusted returns) for regression
        market_excess_regready = sm.add_constant(market_excess)

        # Fit the OLS regression
        model = sm.OLS(portfolio_excess, market_excess_regready).fit()

        # Summarise the findings for alpha (also need to make it intepretable)
        print(f"Constant term of the regression: {model.params['const']} \n")
        print("NB: The alpha value is given in the constant term. \n")
        print(model.summary())

        return
