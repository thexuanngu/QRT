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

    def __init__(self, 
                 backtester):
        
        assert backtester.backtest_complete, "This backtest has not been run."
        self.backtester = backtester

        # performance metrics 
        self._total_pnl = None 
        self._cagr = None 

        # risk-adjusted metrics
        self._sharpe = None 
        self._sortiono = None
        self._calmar = None

        # drawdown
        self._drawdown = None
        self._max_drawdown = None
        self._max_drawdown_duration = None  
        self._average_recovery_duration = None
        self._average_drawdown = None
        
    def _calculate_pnl(self):
        # basic performance metrics - cagr and pnl 

        nav = self.backtester.nav_history 
        start_nav, final_nav = nav.iloc[0], nav.iloc[-1]
        self._total_pnl = final_nav - start_nav

        num_years = (nav.index[-1] - nav.index[0]).days / 365.25
        self._cagr = (final_nav / start_nav) ** (1 / num_years) - 1 

        return 


    def _calculate_drawdown(self):

        nav = self.backtester.nav_history
        rolling_max = nav.cummax()
        drawdown = (nav - rolling_max) / rolling_max
        self._drawdown = drawdown
        self._max_drawdown = drawdown.min()

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

        return
    

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


    # def plot_pnl_curve(self, figsize=(15,5)):
    #     # Calculate the actual profit or loss:
    #     pnl = self.backtester.nav_history - self.backtester.cash_start
    #     pnl_pos = pnl.where(pnl >= 0, 0)
    #     pnl_neg = pnl.where(pnl < 0, 0)

    #     plt.figure(figsize=figsize)
    #     plt.plot(pnl_pos.index, pnl_pos, color="gray", label="Profit")
    #     plt.plot(pnl_neg.index, pnl_neg, color="red", label="Loss")
    #     plt.title(f"PnL Chart for {self.backtester.strategy_name}")
    #     plt.legend()
    #     plt.show()
    #     return
    

    # def pnl_comparison_array(self, target_risk):
    #     """We want an array to plot the pnl curve for comparison with other strategies later."""

    #     assert self.backtester.backtest_complete, ".run() method has not been called."

    #     # Risk-Weighting Portfolio. 
    #     risk_adjusted_returns = target_risk * self.backtester.portfolio_returns / self.backtester.portfolio_returns.std()

    #     return risk_adjusted_returns

    # def performance_metrics(self):
    #     returns = self.backtester.portfolio_returns
    #     sharpe_ratio = returns.mean() / returns.std()
    #     wins = 0
    #     losses = 0
    #     win_rate = 0.0

    #     # win percentage
    #     # what counts as making a trade? its every time the logic is executed such that you make a move. Let's say making a trade is every time you execute the logic. 
    #     for n in range(1, len(self.backtester.trades)):
    #         date_before, date_after = (self.backtester.trades[n - 1]).date, (self.backtester.trades[n]).date
    #         if self.backtester.nav_history[date_before] > self.backtester.nav_history[date_after]: 
    #             wins += 1
    #         else:
    #             losses += 1

    #     win_rate = wins / len(self.backtester.trades)

    #     return dict(sharpe_ratio=sharpe_ratio, win_rate=win_rate)


    # def calculate_risk_metrics(self):
    #     nav_history = self.backtester.nav_history

    #     max_drawdown = 0
    #     max_drawdown_dates = None

    #     max_drawdown_recovery = None

    #     # Step 1 : maximum drawdown 
    #     losses = (nav_history.shift(1) - nav_history).dropna()
    #     current_drawdown = 0
    #     current_drawdown_start = None

    #     for T in range(len(losses)):

    #         # if negative, 
    #         if losses[T] > 0:

    #             if current_drawdown >= max_drawdown:    
    #                 max_drawdown = current_drawdown 
    #                 max_drawdown_dates = (current_drawdown_start, losses.index[T])

    #             current_drawdown = 0
    #             current_drawdown_start = None
    #         else:
    #             current_drawdown += losses[T]
    #             current_drawdown_start = losses.index[T]
                

    #     # Step 2: Time to recover max_drawdown
    #     start_capital = self.backtester.nav_history[max_drawdown_dates[1]] 
    #     # find the first instance where nav goes beyond start_capital + max_drawdown
    #     # if unable to find, return -1
    #     recovery_entries = nav_history[nav_history > start_capital + abs(max_drawdown)] 
    #     recovery_entries = recovery_entries[recovery_entries.index > max_drawdown_dates]
    #     try:
    #         max_drawdown_recovery = (max_drawdown_dates[1], recovery_entries[0].index)
    #     except IndexError:
    #         max_drawdown_recovery = np.nan

    #     return dict(max_drawdown=max_drawdown,
    #                 max_drawdown_dates=max_drawdown_dates,
    #                 max_drawdown_recovery=max_drawdown_recovery
    #                 )
    

