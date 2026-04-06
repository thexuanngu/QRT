import sys
sys.path.append(r"C:\Users\trund\QRT")
import eda

from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
import scipy.stats as stats
import yfinance as yf

from get_nasdaq_tickers import get_nasdaq_tickers
from get_MA_warmup_dates import get_rolledback_start

# To repeatedly call data call + splitting into train and test
def obtain_data(start="2015-01-01", end="2025-03-31"):
    tickers = get_nasdaq_tickers()

    ANALYSIS_START, ANALYSIS_END = get_rolledback_start(start, end)

    data = yf.download(tickers, 
                start=ANALYSIS_START, 
                end=ANALYSIS_END, 
                auto_adjust=False
                )
    
    return data

def clean_columns(data: pd.DataFrame) -> pd.DataFrame:
    data.columns = data.columns.set_levels(
            data.columns.levels[0].str.replace(" ", "_"),
            level=0,
            verify_integrity=False,
        )
    data.columns = pd.MultiIndex.from_tuples([(column.lower(), ticker) for column, ticker in data.columns])
    
    return data 

def remove_na_tickers(data: pd.DataFrame) -> pd.DataFrame:
    # Identify tickers where any associated column is all NA,
    # print them, then drop all columns for those tickers.
    flagged_tickers = []
    if isinstance(data.columns, pd.MultiIndex):
        # Assume columns are MultiIndex like (field, ticker)
        ticker_level = 1

        for ticker in data.columns.get_level_values(ticker_level).unique():
            sub = data.xs(ticker, level=ticker_level, axis=1)

            # If any column for this ticker is entirely NA, flag it
            if sub.isna().all().any():
                flagged_tickers.append(ticker)

    else:
        # Single-level columns: treat each column as its own ticker
        for col in data.columns:
            if data[col].isna().all():
                flagged_tickers.append(col)

    # Drop all columns corresponding to the flagged tickers
    if flagged_tickers:
        if isinstance(data.columns, pd.MultiIndex):
            data = data.drop(columns=flagged_tickers, level=ticker_level)
        else:
            data = data.drop(columns=flagged_tickers)

    print("Data shape after dropping flagged tickers:", data.shape)
    return data

def add_columns(data: pd.DataFrame) -> pd.DataFrame:
    adj = data.xs("adj_close", level=0, axis=1)     # same as data["adj_close"] if level-0 label exists
    # column-wise returns (per ticker)
    rets = adj.pct_change()                         # NaN only in the first row for each ticker
    # put returns back as a new level-0 field called "returns"
    rets.columns = pd.MultiIndex.from_product([["returns"], rets.columns])
    # join back onto original data
    out = pd.concat([data, rets], axis=1).sort_index(axis=1)
    out.drop(index=data.index[0], inplace=True)  # drop the first row which has NaN returns
    return out

def train_test_split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_length = int(len(data) * 0.8)
    train_data = data.iloc[:train_length -1 ]
    test_data = data.iloc[train_length -1 :]

    return train_data, test_data
