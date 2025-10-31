# File: visualisations.py
# Description: Helper functions for plotting and visualization.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set a consistent, professional theme for all plots
sns.set_theme(style="darkgrid")

# --- 1. EDA PLOTS ---

def plot_price_history(df, price_col='adj_close', title='Price History'):
    """
    Plots the price history for one or more tickers.
    
    Args:
        df (pd.DataFrame): DataFrame with either:
            - A 'ticker' column (for multiple tickers)
            - A single price_col (for one ticker)
        price_col (str): The column name of the price to plot.
    """
    plt.figure(figsize=(14, 7))
    
    if 'ticker' in df.columns or 'ticker' in df.index.names:
        # Use seaborn for easy multi-ticker plotting
        sns.lineplot(data=df, x='date', y=price_col, hue='ticker')
    else:
        # Use simple pandas plot for single ticker
        df[price_col].plot()
        
    plt.title(title, fontsize=16)
    plt.ylabel('Adjusted Close Price')
    plt.xlabel('Date')
    plt.legend()
    plt.show()

def plot_correlation_heatmap(df, title='Feature Correlation Matrix'):
    """
    Plots a heatmap of the correlation matrix for a DataFrame's columns.
    
    Args:
        df (pd.DataFrame): DataFrame with numeric features.
    """
    if df.empty:
        print("Cannot plot correlation matrix for an empty DataFrame.")
        return

    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    
    # Create a mask to hide the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, 
                annot=True,       # Show correlation values
                fmt='.2f',        # Format to 2 decimal places
                cmap='coolwarm',  # Color map
                mask=mask,        # Apply the mask
                vmin=-1, vmax=1)  # Set color bar range
    
    plt.title(title, fontsize=16)
    plt.show()

def plot_return_distribution(returns_series, title='Return Distribution'):
    """
    Plots a histogram and KDE for a series of returns.
    
    Args:
        returns_series (pd.Series): A series of (e.g., daily) returns.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(returns_series, kde=True, bins=100, stat="density")
    
    # Calculate and display mean and std dev
    mean = returns_series.mean()
    std = returns_series.std()
    
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.4f}')
    plt.axvline(mean + std, color='orange', linestyle='--', label=f'Std Dev: {std:.4f}')
    plt.axvline(mean - std, color='orange', linestyle='--')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Returns')
    plt.legend()
    plt.show()

# --- 2. BACKTEST EVALUATION PLOTS ---

def plot_backtest_summary(returns_series, benchmark_returns=None, title='Strategy Backtest Performance'):
    """
    Plots the cumulative returns and drawdown for a strategy's returns.
    
    Args:
        returns_series (pd.Series): Series of strategy's periodic returns.
        benchmark_returns (pd.Series, optional): Series of benchmark returns (e.g., S&P 500).
        title (str): Overall title for the plots.
    """
    
    # --- 1. Calculate Cumulative Returns ---
    cumulative_returns = (1 + returns_series).cumprod() - 1
    
    # --- 2. Calculate Drawdown ---
    # a. Calculate cumulative product
    cumulative_prod = (1 + returns_series).cumprod()
    # b. Find running maximum
    running_max = cumulative_prod.cummax()
    # c. Calculate drawdown
    drawdown = (cumulative_prod - running_max) / running_max
    
    # --- 3. Create the Plots ---
    # Create a figure with 2 subplots (one above the other)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    fig.suptitle(title, fontsize=18, y=1.02)
    
    # --- Plot 1: Cumulative Returns ---
    (cumulative_returns * 100).plot(ax=ax1, label='Strategy', color='blue')
    ax1.set_ylabel('Cumulative Returns (%)')
    ax1.set_title('Cumulative Returns')
    
    if benchmark_returns is not None:
        # Align index just in case
        benchmark_returns = benchmark_returns.reindex(returns_series.index)
        cum_benchmark = (1 + benchmark_returns).cumprod() - 1
        (cum_benchmark * 100).plot(ax=ax1, label='Benchmark', color='gray', linestyle='--')
        
    ax1.legend()
    ax1.grid(True)
    
    # --- Plot 2: Drawdown ---
    (drawdown * 100).plot(ax=ax2, label='Drawdown', color='red')
    ax2.fill_between(drawdown.index, (drawdown * 100), 0, color='red', alpha=0.3)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Drawdown')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# --- Main execution block (for testing) ---

if __name__ == "__main__":
    # This block runs only when you execute `python visualizations.py`
    
    print("Running visualization test plots...")
    
    # 1. Create dummy data for testing
    dates = pd.date_range(start='2020-01-01', periods=500)
    
    # Dummy strategy returns
    strategy_returns = pd.Series(
        np.random.normal(loc=0.0008, scale=0.012, size=500), 
        index=dates
    )
    
    # Dummy benchmark returns
    benchmark_returns = pd.Series(
        np.random.normal(loc=0.0005, scale=0.01, size=500), 
        index=dates
    )
    
    # 2. Test the backtest plot
    print("\nTesting plot_backtest_summary...")
    plot_backtest_summary(strategy_returns, 
                          benchmark_returns=benchmark_returns, 
                          title="Test Backtest Summary")

    # 3. Test the distribution plot
    print("\nTesting plot_return_distribution...")
    plot_return_distribution(strategy_returns, title="Test Strategy Return Distribution")
    
    # 4. Test Correlation plot
    print("\nTesting plot_correlation_heatmap...")
    test_df = pd.DataFrame({
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100) * 0.5,
        'feature_3': np.random.rand(100) * -0.8
    })
    test_df['feature_1'] = test_df['feature_1'] + test_df['feature_2']
    test_df['feature_3'] = test_df['feature_3'] + test_df['feature_1']
    plot_correlation_heatmap(test_df)
    
    print("\nVisualization tests complete.")