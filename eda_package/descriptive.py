from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


"""
Descriptive statistics and visualizations for EDA.
Includes univariate analysis and outlier detection.
"""



# ---------------------------------------------------------------------
# 4) Univariate analysis
# ---------------------------------------------------------------------
def plot_numeric_univariate(df: pd.DataFrame, cols: List[str], max_plots: int = 8, bins: int = 30, save_path: Optional[str] = None):
    """
    Plot histogram + boxplot + QQ for a list of numeric columns.
    Justification: Visualizing distribution, skewness, and potential outliers quickly.
    """
    cols = cols[:max_plots]
    for col in cols:
        series = df[col].dropna()

        # T
        if series.empty:
            print(f"Skipping {col}: no non-null values")
            continue

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
        sns.histplot(series, bins=bins, kde=True, ax=axes[0])
        
        axes[0].set_title(f"Histogram: {col}\n(mean={series.mean():.3g}, med={series.median():.3g}, skew={series.skew():.3g})")
        
        sns.boxplot(x=series, ax=axes[1])
        axes[1].set_title("Boxplot")
        
        stats.probplot(series, dist="norm", plot=axes[2])
        axes[2].set_title("QQ plot (vs normal)")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()


def summarize_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Return numeric summary table: mean, median, std, skew, kurtosis, min, max, percentiles.
    Justification: Quantitative summary to accompany visual checks.
    """
    rows = []
    for c in cols:
        s = df[c].dropna()
        rows.append({
            "column": c,
            "count": s.count(),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "skew": s.skew(),
            "kurtosis": s.kurtosis(),
            "min": s.min(),
            "5%": s.quantile(0.05),
            "25%": s.quantile(0.25),
            "50%": s.quantile(0.50),
            "75%": s.quantile(0.75),
            "95%": s.quantile(0.95),
            "max": s.max()
        })
    return pd.DataFrame(rows).set_index("column")


def plot_categorical_univariate(df: pd.DataFrame, cols: List[str], max_levels: int = 20, max_plots: int = 8, save_prefix: Optional[str] = None):
    """
    Plot barplots for categorical columns; group rare levels into 'OTHER' for readability.
    Justification: Detect label imbalance and cardinality issues early.
    """
    cols = cols[:max_plots]
    for col in cols:
        counts = df[col].value_counts(dropna=False)
        top = counts.head(max_levels)
        others = counts.iloc[max_levels:].sum()
        plot_vals = top.append(pd.Series({'OTHER': others})) if others>0 else top
        plt.figure(figsize=(8,4))
        sns.barplot(x=plot_vals.values, y=plot_vals.index)
        plt.title(f"Value counts: {col} (top {max_levels})")
        plt.xlabel("Count")
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_{col}_cat_counts.png", bbox_inches='tight', dpi=150)
        plt.show()


# ---------------------------------------------------------------------
# 5) Outlier detection helpers
# ---------------------------------------------------------------------
def detect_outliers_iqr(df: pd.DataFrame, cols: List[str], k: float = 1.5) -> pd.DataFrame:
    """
    IQR-based outlier detection. Returns fraction of rows flagged per column.
    Justification: Simple robust method to flag extreme values.
    """
    results = []
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k*iqr, q3 + k*iqr
        mask = (df[c] < lower) | (df[c] > upper)
        results.append({
            "column": c,
            "n_outliers": int(mask.sum()),
            "pct_outliers": (mask.sum() / len(df) * 100)
        })
    return pd.DataFrame(results).set_index("column").sort_values("pct_outliers", ascending=False)


def detect_outliers_zscore(df: pd.DataFrame, cols: List[str], threshold: float = 3.0) -> pd.DataFrame:
    """
    Z-score based outlier detection (assumes approximate normality).
    Justification: Complementary to IQR; detects outliers in bell-shaped features.
    """
    res = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        z = np.abs((s - s.mean()) / s.std())
        n = (z > threshold).sum()
        res.append({"column": c, "n_outliers": int(n), "pct_outliers": n/len(df)*100})
    return pd.DataFrame(res).set_index("column").sort_values("pct_outliers", ascending=False)


# ---------------------------------------------------------------------
# 6) Correlation & association
# ---------------------------------------------------------------------
def correlation_plots(df, cols: List[str], figsize=(10,10), save_path=None):
    """
    Use the seaborn plot.corr method to find the correlation plots between different columns in a df.
    NB: YOU DO NOT NEED TO SLICE THE DF BEFORE WRITING 
    """
    # A bit more encapsulating rather than needing to slice the df before calling
    columns_to_plot = df.loc[:, cols]
    plt.figure(figsize=figsize)
    sns.pairplot(columns_to_plot)
    plt.title(f"Pair Plot for given Columns ({len(cols)} cols)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    return 

def correlation_matrix(df: pd.DataFrame, cols: List[str], method: str = "pearson", annot: bool = True, figsize: Tuple[int,int]=(10,8), save_path: Optional[str] = None):

    """
    Compute and plot a correlation matrix for numeric columns.
    Justification: Correlation helps spot collinearity and groups of related features.
    method: 'pearson', 'spearman', 'kendall'

    """

    corr = df[cols].corr(method=method)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, fmt=".2f", cmap="vlag", center=0)
    plt.title(f"{method.capitalize()} correlation matrix ({len(cols)} cols)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    return corr


# For Time Series ACF and PACF plots 
def autocorr_plots(series: pd.Series, lags: int = 40, title_prefix: str = "Original Series", save_path: Optional[str] = None):
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