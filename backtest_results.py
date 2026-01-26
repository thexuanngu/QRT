from backtesting import Backtester
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
                 backtester,
                 riskfree):
        
        assert backtester.backtest_complete, "This backtest has not been run."

        self.backtester = backtester
        self.riskfree = riskfree
    

# Thinking of splitting this section into its separate class 
    def calculate_alpha(self, benchmark):

        # Basic error handling - exceptions should be thrown if not completed.
        assert len(benchmark) == len(self.backtester.portfolio_returns) 
        #assert benchmark is not None, "Benchmark needs to be loaded in before testing can properly be done."
        
        # Straightforward - calculating the portfolio returns 
        portfolio_excess = self.backtester.portfolio_returns - self.riskfree
        market_excess = benchmark - self.riskfree

        # Prepares the x variable (market risk-adjusted returns) for regression
        market_excess_regready = sm.add_constant(market_excess)

        # Fit the OLS regression
        model = sm.OLS(portfolio_excess, market_excess_regready).fit()

        # Summarise the findings for alpha (also need to make it intepretable)
        print(f"Alpha of the strategy: {model.params[0]} \n")
        print("NB: The alpha value is given in the constant term. \n")
        model.summary()

        return


    def plot_pnl_curve(self, figsize=(15,5)):
        # Calculate the actual profit or loss:
        pnl = self.backtester.nav_history - self.backtester.cash_start
        pnl_pos = pnl.where(pnl >= 0, 0)
        pnl_neg = pnl.where(pnl < 0, 0)

        plt.figure(figsize=figsize)
        plt.plot(pnl_pos.index, pnl_pos, color="gray", label="Profit")
        plt.plot(pnl_neg.index, pnl_neg, color="red", label="Loss")
        plt.title(f"PnL Chart for {self.backtester.strategy_name}")
        plt.legend()
        plt.show()
        return
    

    def pnl_comparison_array(self, target_risk):
        """We want an array to plot the pnl curve for comparison with other strategies later."""

        assert self.backtester.backtest_complete, ".run() method has not been called."

        # Risk-Weighting Portfolio. 
        risk_adjusted_returns = target_risk * self.backtester.portfolio_returns / self.backtester.portfolio_returns.std()

        return risk_adjusted_returns

    def performance_metrics(self):
        returns = self.backtester.portfolio_returns
        sharpe_ratio = returns.mean() / returns.std()
        wins = 0
        losses = 0
        win_loss_ratio = 0.0

        # win percentage
        # what counts as making a trade? its every time the logic is executed such that you make a move. Let's say making a trade is every time you execute the logic. 
        for n in range(1, len(self.backtester.trades)):
            date_before, date_after = (self.backtester.trades[n - 1]).date, (self.backtester.trades[n]).date
            if self.backtester.nav_history[date_before] > self.backtester.nav_history[date_after]: 
                wins += 1
            else:
                losses += 1

        win_loss_ratio = wins / losses

        return dict(sharpe_ratio=sharpe_ratio, win_loss_ratio=win_loss_ratio)


    def calculate_risk_metrics(self):
        nav_history = self.backtester.nav_history

        max_drawdown = 0
        max_drawdown_dates = None

        max_drawdown_recovery = None

        # Step 1 : maximum drawdown 
        losses = (nav_history.shift(1) - nav_history).dropna()
        current_drawdown = 0
        current_drawdown_start = None

        for T in range(len(losses)):

            # if negative, 
            if losses[T] > 0:

                if current_drawdown >= max_drawdown:    
                    max_drawdown = current_drawdown 
                    max_drawdown_dates = (current_drawdown_start, losses.index[T])

                current_drawdown = 0
                current_drawdown_start = None
            else:
                current_drawdown += losses[T]
                current_drawdown_start = losses.index[T]
                

        # Step 2: Time to recover max_drawdown
        start_capital = self.backtester.nav_history[max_drawdown_dates[1]] 
        # find the first instance where nav goes beyond start_capital + max_drawdown
        # if unable to find, return -1
        recovery_entries = nav_history[nav_history > start_capital + abs(max_drawdown)] 
        recovery_entries = recovery_entries[recovery_entries.index > max_drawdown_dates]
        try:
            max_drawdown_recovery = (max_drawdown_dates[1], recovery_entries[0].index)
        except IndexError:
            max_drawdown_recovery = np.nan

        return dict(max_drawdown=max_drawdown,
                    max_drawdown_dates=max_drawdown_dates,
                    max_drawdown_recovery=max_drawdown_recovery
                    )
    

