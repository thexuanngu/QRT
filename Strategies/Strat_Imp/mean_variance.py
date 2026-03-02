from backtesting_v2 import Strategy
import backtest_results as btr

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class FMMeanVariance(Strategy):

    # Generate docstring here later 
    def __init__(self,
                tradable_assets: List[str], 
                data: pd.DataFrame, # the data that the strategy operates on
                allow_short=True,
                cash_start: float = 100_000.0,
                commission: float = 0.0,  
                slippage: float = 0.0, 
                execute_on_next_tick: bool = True,
                rolling_window: int = 3, 
                target_return: float = 0.01
                ):
        
        super().__init__(tradable_assets, data, allow_short, cash_start, commission, slippage, execute_on_next_tick)
        # Rolling window for the Fama-Macbeth regression
        self.rolling_window = rolling_window
    
    # Calculate the expected returns vector 
    def _calculate_returns(self, index):
        
        # Assuming the y variable in the regression is returns.
        rolling_mean = self.data[self.tradable_assets].iloc[index - self.rolling_window:index].mean()

        z_past = (rolling_mean - rolling_mean.mean()) / rolling_mean.std()
        y_past = self.data[self.tradable_assets].iloc[[index]]

        model = LinearRegression(fit_intercept=True).fit(z_past, y_past)    

        # Get the predictions for the next day 
        x_now = self.data[self.tradable_assets].iloc[index - (self.rolling_window-1):index+1]
        z_now = (x_now - x_now.mean()) / x_now.std()

        signal = model.predict(x_now)

        return pd.Series(signal.flatten(), index=self.tradable_assets)
    
    # Obtain the covariance matrix of returns up to current time-step.
    def _calculate_cov_matrix(self, index):
        return self.data[self.tradable_assets].iloc[:index+1].cov()

    # Minimum Variance Portfolio (MVP) Given a Fixed Expected Return - Method 2
    def _MVP2(self, exp_ret, cov_matrix):
        
        n = exp_ret.shape[0]
        u = np.ones(n, dtype=cov_matrix.dtype)

        # One Cholesky factorization; solve for C^{-1}u and C^{-1}μ in a single pass
        L = np.linalg.cholesky(cov_matrix)
        X = np.linalg.solve(L, np.column_stack((u, exp_ret)))   # solve L Z = [u, μ]
        Y = np.linalg.solve(L.T, X)                        # solve L^T Y = Z
        y_u, y_mu = Y[:, 0], Y[:, 1]                       # C^{-1} u, C^{-1} μ

        # Scalars A, B, Cc and determinant Δ
        A  = float(u @ y_u)            # u^T C^{-1} u
        B  = float(u @ y_mu)           # u^T C^{-1} μ = μ^T C^{-1} u
        Cc = float(exp_ret @ y_mu)          # μ^T C^{-1} μ
        Delta = A * Cc - B * B

        # Lagrange multipliers and weights
        lam = (Cc - B * self.target_ret) / Delta
        gam = (A * self.target_ret - B) / Delta
        w = lam * y_u + gam * y_mu

        # Portfolio stats
        mu_p  = float(exp_ret @ w)                                  # should ≈ mu_target
        var_p = (A * self.target_ret**2 - 2 * B * self.target_ret + Cc) / Delta  # closed form
        std_p = np.sqrt(var_p) 

        return dict(weights=w, exp_ret=mu_p, std=std_p)
    

    def _predict(self, index):
        exp_ret = self._calculate_returns(index)
        cov_matrix = self._calculate_cov_matrix(index)

        return self._MVP2(exp_ret, cov_matrix)["weights"]


    def backtest(self, verbose=False):

        # Vectorised backtest.
         

        # Find all rebalancing dates and then execute on the next tick
        n = len(self.data)

        for i in range(start=0, stop=n, step=)


        # Execution


        # Signal generation



        # Tick execution


        # Performance evaluation



        


