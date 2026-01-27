
import backtest.backtesting as backtesting # To backtest the strategy
import heuristic_pf_construct # To test out the weighted strategies.

from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import statsmodels.stats.api as sms
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.stattools import acf, adfuller, kpss, q_stat

# Data preprocessing step.

# Currently first want a high level summary of the data


# 1. Handling Missing Data
# Creating a helper function to check for missing data that can be reused in the future as well
def missing_and_duplicates(df: pd.DataFrame, show_top: int = 20) -> Dict[str, pd.DataFrame]:
    """
    Return dataframes listing missing values and duplicate row counts.
    Justification: Missingness and duplication are common sources of bias/errors.
    """
    total = len(df)
    missing = df.isna().sum().sort_values(ascending=False)
    missing_percent = (missing / total * 100).round(4)
    missing_df = pd.concat([missing, missing_percent], axis=1, keys=["missing_count", "missing_percentage"])
    missing_df = missing_df[missing_df['missing_count'] > 0]
    
    duplicate_count = df.duplicated().sum()
    print(f"Total rows: {total}, Duplicated rows: {duplicate_count}")

    if duplicate_count > 0:
        print("First 5 duplicated rows (by index peek):")
        print(df[df.duplicated(keep=False)].head())
    
    print("\nTop missing columns:")
    print(missing_df.head(show_top))

def eda_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    # Just forward fill the values of the frame
    df_copy = df.ffill()
    df_copy.dropna(inplace=True)
    return df_copy



def plot_autocorrelations(series: pd.Series, lags: int = 40, title_prefix: str = "Original Series"):
    """
    Plots the time series, its ACF, and its PACF.

    :param series: The time series data (pandas Series).
    :param lags: The number of lags to include in the plots.
    :param title_prefix: A string prefix for the plot titles.
    """
    if series.empty:
        print("Error: The input series is empty.")
        return

    # 1. Plot the raw time series data
    plt.figure(figsize=(12, 4))
    plt.plot(series, color='#1f77b4', linewidth=2)
    plt.title(f'Time Plot of {title_prefix}', fontsize=16)
    plt.ylabel('Value')
    plt.xlabel('Time')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # 2. Plot the ACF (Autocorrelation Function)
    # The blue shaded region represents the confidence interval (typically 95%).
    # If a bar extends outside this region, the autocorrelation at that lag is significant.
    fig_acf = plot_acf(series, lags=lags, alpha=0.05, 
                       title=f'ACF for {title_prefix} (Autocorrelation Function)')
    fig_acf.set_size_inches(12, 4)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 3. Plot the PACF (Partial Autocorrelation Function)
    # PACF measures the correlation between the series and its lagged values 
    # after accounting for the correlation at intermediate lags.
    fig_pacf = plot_pacf(series, lags=lags, alpha=0.05, 
                         title=f'PACF for {title_prefix} (Partial Autocorrelation Function)')
    fig_pacf.set_size_inches(12, 4)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Test autocorrelation 
def test_ljung_box(series: pd.Series, lags: int = 20):
    """
    Performs the Ljung-Box Q-test for examining whether a set of autocorrelations 
    of a series are different from zero (i.e., whether the series is white noise).
    
    Null Hypothesis (H0): The data are independently distributed (i.e., the series is white noise).
    We want to FAIL TO REJECT H0 (p-value > 0.05) to confirm white noise.
    
    :param series: The time series data (typically differenced returns or residuals).
    :param lags: The number of lags to test jointly.
    """
    print("\n--- Ljung-Box Q-Test for White Noise ---")
    
    # The q_stat function returns (Ljung-Box Statistic, p-values)
    ljung_box_stat, p_values = q_stat(series.dropna(), nobs=[lags])
    
    # Since we tested only one lag, we extract the first (and only) element
    lb_stat = ljung_box_stat[0]
    p_value = p_values[0]
    
    print(f"Lags Tested: 1 to {lags}")
    print(f"Ljung-Box Statistic (Q): {lb_stat:.4f}")
    print(f"Ljung-Box p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Result: NOT White Noise (Reject H0 - Significant autocorrelation is present)")
    else:
        print("Result: Likely White Noise (Fail to Reject H0 - No significant autocorrelation detected)")

    return 


def test_stationarity(series: pd.Series):
    """
    Performs the Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests.
    """
    print("\n--- Stationarity Tests (Mean) ---")
    
    # 1. ADF Test (Null Hypothesis: Unit Root is Present / Non-stationary)
    # We want a small p-value (< 0.05) to reject H0.
    result_adf = adfuller(series.dropna())
    adf_output = pd.Series(result_adf[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Observations'])
    print("\nAugmented Dickey-Fuller Test (ADF):")
    print(adf_output.to_string())
    print(f"Result: {'STATIONARY (Reject H0)' if result_adf[1] < 0.05 else 'NON-STATIONARY (Fail to Reject H0)'}")
    
    # 2. KPSS Test (Null Hypothesis: Series is Level Stationary)
    # We want a large p-value (> 0.05) to fail to reject H0.
    result_kpss = kpss(series.dropna(), regression='c', nlags='auto') # 'c' is for level stationarity
    kpss_output = pd.Series(result_kpss[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    print("\nKPSS Test (Level Stationarity):")
    print(kpss_output.to_string())
    print(f"Result: {'NON-STATIONARY (Reject H0)' if result_kpss[1] < 0.05 else 'STATIONARY (Fail to Reject H0)'}")

    return
    
def test_heteroskedasticity(series: pd.Series, lags=10):
    """
    Performs the ARCH-LM test for conditional heteroskedasticity.
    Requires data to be mean-stationary (e.g., differenced data or residuals).
    Null Hypothesis: No ARCH effects (Homoskedasticity).
    We want to fail to reject H0 (p-value > 0.05) if variance is constant.
    """
    print("\n--- Conditional Heteroskedasticity Test (ARCH-LM) ---")
    
    try:
        # Running the test
        lm_test = sms.arch_lm_test(series.dropna(), lags=lags)
        
        print(f"Lags Used: {lags}")
        print(f"LM Statistic: {lm_test.lm_value:.4f}")
        print(f"LM p-value: {lm_test.pvalue:.4f}")
        
        if lm_test.pvalue < 0.05:
            print("Result: ARCH effects present (Reject H0 - Series is HETEROSKEDASTIC)")
        else:
            print("Result: No ARCH effects detected (Fail to Reject H0 - Series is HOMOSKEDASTIC)")
            
    except Exception as e:
        print(f"Could not run ARCH-LM test. Check the series length and lags. Error: {e}")

# --- PLOTTING AND DIAGNOSTICS ---

def plot_autocorrelations(series: pd.Series, lags: int = 40, title_prefix: str = "Original Series"):
    """
    Plots the time series, its ACF, and its PACF, and performs diagnostic prints.
    """
    if series.empty:
        print("Error: The input series is empty.")
        return

    # --- Diagnostic Check ---
    print(f"\n--- Diagnostics for {title_prefix} ---")
    print(f"Mean: {series.mean():.4f}")
    print(f"Std Dev: {series.std():.4f}")
    
    autocorr_values = acf(series, nlags=5, fft=False)
    print(f"First 5 ACF Values (Lag 1 to 5):")
    for i, acf_val in enumerate(autocorr_values[1:], start=1):
        print(f"  Lag {i}: {acf_val:.4f}")
    print("---------------------------------")
    # -------------------------

    # 1. Plot the raw time series data
    plt.figure(figsize=(12, 4))
    plt.plot(series, color='#1f77b4', linewidth=2)
    plt.title(f'Time Plot of {title_prefix}', fontsize=16)
    plt.ylabel('Value')
    plt.xlabel('Time')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # 2. Plot the ACF (Autocorrelation Function)
    fig_acf = plot_acf(series, lags=lags, alpha=0.05, 
                       title=f'ACF for {title_prefix} (Autocorrelation Function)')
    fig_acf.set_size_inches(12, 4)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 3. Plot the PACF (Partial Autocorrelation Function)
    fig_pacf = plot_pacf(series, lags=lags, alpha=0.05, 
                         title=f'PACF for {title_prefix} (Partial Autocorrelation Function)')
    fig_pacf.set_size_inches(12, 4)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return