from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as stats

# Time series specific library
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import accor_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

"""
Time-series specific EDA plots. 
"""

def plot_corr(series: pd.Series, lags: int = 40, title_prefix: str = "Original Series", save_path: Optional[str] = None):
    series_to_plot = series.dropna()
    """
    Plots the time series, its ACF, and its PACF.

    :param series: The time series data (pandas Series).
    :param lags: The number of lags to include in the plots.
    :param title_prefix: A string prefix for the plot titles.
    """
    if series.empty:
        print("Error: The input series is empty.")
        return

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))

    # Plot the original series
    axes[0].plot(series_to_plot)
    axes[0].set_title(f"{title_prefix} Time Series")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Value")

    # Plot ACF
    plot_acf(series_to_plot, lags=lags, ax=axes[1])
    axes[1].set_title(f"{title_prefix} Autocorrelation (ACF)")

    # Plot PACF
    plot_pacf(series_to_plot, lags=lags, ax=axes[2])
    axes[2].set_title(f"{title_prefix} Partial Autocorrelation (PACF)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    plt.show()


# For Time Series Hypothesis Testing

def adf_test(y, max_lag=100):
    
    result = adfuller(y)
    
    print(f"ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Used lags:", result[2])
    print("Number of observations:", result[3])
    print("Critical values:")
    for key, value in result[4].items():
        print(f" {key}: {value}")
    return 

def kpss_test(y, trend="c"):
    result = kpss(y, regression=trend)
    print(f"KPSS Statistic:", result[0])
    print("p-value:", result[1])
    print("Used lags:", result[2])
    print("Critical values:")
    for key, value in result[3].items():
        print(f" {key}: {value}")
    return 

def ljbox_test(y, lags=20, figsize=(10,5), savefig=None):
    fig, ax = plt.subplots(figsize=figsize)
    ljbox_test = acorr_ljungbox(y, lags=30) #package version issue

    ax.plot(ljbox_test['lb_pvalue'])
    ax.axhline(y=0.05, color='r', ls='--') 
    ax.set_title("Ljung-Box test")
    ax.set_ylabel("p-value")
    ax.set_xlabel("Lags")

    plt.tight_layout()

    if savefig:
        plt.savefig(savefig, dpi=300)
    
    plt.show()

    return s