# File: data_sourcing.py
# Description: Helper functions for fetching and saving market data.

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---

# This creates a robust path to your 'data' folder
# It finds the directory of this script (QRT/) and joins it with 'data'
DATA_DIR = Path(__file__).parent / 'data'
PRICE_DATA_FILE = DATA_DIR / 'price_data.csv'
FUNDAMENTAL_DATA_FILE = DATA_DIR / 'fundamental_data.csv'

# Ensure the data directory exists
DATA_DIR.mkdir(exist_ok=True)

# --- Price Data Functions ---

def get_price_data(tickers, start_date, end_date, auto_adjust=True):
    """
    Fetches historical price data from Yahoo Finance and saves it to a CSV.

    Args:
        tickers (list): List of ticker symbols (e.g., ['AAPL', 'MSFT']).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        auto_adjust (bool): Auto-adjust for splits and dividends ('Adj Close').
    """
    print(f"Fetching price data for {tickers}...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=auto_adjust)
        
        # Format the DataFrame for better processing later
        if len(tickers) > 1:
            # Stack multi-level columns ('Adj Close', 'Close', etc.) into rows
            data = data.stack(future_stack=True)
            # Rename columns to be lowercase and easier to use
            data.index.names = ['date', 'ticker']
        else:
            data.index.name = 'date'
            data['ticker'] = tickers[0]
        
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        
        # Save to the data folder
        data.to_csv(PRICE_DATA_FILE)
        print(f"Successfully saved price data to {PRICE_DATA_FILE}")
        return data
        
    except Exception as e:
        print(f"Error fetching price data: {e}")
        return None

def load_price_data():
    """Loads the saved price data from the data folder."""
    if not PRICE_DATA_FILE.exists():
        print(f"Error: {PRICE_DATA_FILE} not found.")
        print("Run get_price_data() first.")
        return None
    
    print(f"Loading data from {PRICE_DATA_FILE}...")
    # Read with header=0 and index_col=[0,1] if it's a multi-index file
    try:
        data = pd.read_csv(PRICE_DATA_FILE, index_col=['date', 'ticker'], parse_dates=['date'])
    except ValueError:
        # Fallback for single ticker
        data = pd.read_csv(PRICE_DATA_FILE, index_col='date', parse_dates=['date'])
        
    return data

# --- Fundamental Data Functions ---

def get_fundamental_data(tickers):
    """
    Fetches key fundamental ratios for a list of tickers.
    This data is (mostly) static, not time-series.
    
    Note: yfinance also provides .balance_sheet, .financials, .cashflow
    for full-statement data if needed. We focus on ratios for signals.
    """
    
    # Define the key metrics (signals) we want
    # These are keys in the .info dictionary from yfinance
    QUANT_SIGNALS = [
        'trailingPE', 'priceToBook', 'priceToSalesTtm', 'enterpriseToEbitda', # Value
        'returnOnEquity', 'returnOnAssets', 'operatingMargins', 'profitMargins', # Profitability
        'debtToEquity', 'currentRatio', 'quickRatio', # Solvency
        'earningsGrowth', 'revenueGrowth', # Growth
        'beta', 'marketCap', 'sector', 'industry' # Other / Risk
    ]
    
    all_data = []
    print(f"Fetching fundamental data for {len(tickers)} tickers...")
    
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            
            # Build a dictionary for this ticker
            ticker_data = {'ticker': ticker}
            for signal in QUANT_SIGNALS:
                # Use .get() to safely get a value or np.nan if it's missing
                ticker_data[signal] = info.get(signal, np.nan)
            
            all_data.append(ticker_data)
        except Exception as e:
            print(f"Could not get info for {ticker}: {e}")
            
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(all_data).set_index('ticker')
    
    # Save to the data folder
    df.to_csv(FUNDAMENTAL_DATA_FILE)
    print(f"Successfully saved fundamental data to {FUNDAMENTAL_DATA_FILE}")
    return df

def load_fundamental_data():
    """Loads the saved fundamental data from the data folder."""
    if not FUNDAMENTAL_DATA_FILE.exists():
        print(f"Error: {FUNDAMENTAL_DATA_FILE} not found.")
        print("Run get_fundamental_data() first.")
        return None
    
    print(f"Loading data from {FUNDAMENTAL_DATA_FILE}...")
    return pd.read_csv(FUNDAMENTAL_DATA_FILE, index_col='ticker')

# --- Main execution block (for testing) ---

if __name__ == "__main__":
    # This block runs only when you execute `python data_sourcing.py`
    
    my_tickers = ['AAPL', 'MSFT', 'GOOG', 'JPM']
    
    print("--- Fetching Price Data ---")
    get_price_data(my_tickers, start_date='2020-01-01', end_date='2023-12-31')
    
    print("\n--- Fetching Fundamental Data ---")
    get_fundamental_data(my_tickers)
    
    print("\n--- Loading Data (Example) ---")
    prices = load_price_data()
    print("\nLoaded Prices (Head):")
    print(prices.head())
    
    fundamentals = load_fundamental_data()
    print("\nLoaded Fundamentals:")
    print(fundamentals)