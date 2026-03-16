import requests
import pandas as pd


def get_nasdaq_tickers():


    url = "https://en.wikipedia.org/wiki/Nasdaq-100"

    # Add a real browser User-Agent
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    

    response = requests.get(url, headers=headers)
    html = response.text

    import io
    # read_html from raw HTML instead of URL
    tables = pd.read_html(io.StringIO(html))

    # The current constituents of the NASDAQ 100:
    current_const = tables[4]["Ticker"]

    # Access the tickers
    tickers = current_const.tolist()

    # The historical constituents of the NASDAQ 100:
    historical_const = tables[5]

    # When converting strings to datetime objects, pandas defaults to adding a midnight timestamp.
    # To remove the time part, we can use the .dt.date accessor, which returns Python's datetime.date objects.
    historical_const["Date", "Date"] = pd.to_datetime(historical_const["Date", "Date"])
    #print(historical_const.head())

    # Set the date column as the index of the DataFrame
    historical_const = historical_const.set_index(('Date', 'Date'))
    historical_const.index.name = "Date" 
    # The next block of code is to get all the tickers of the historical constituents that have been removed.

    # Convert the string date to a datetime object to allow for comparison
    date_considered = pd.to_datetime("2015-01-01")

    # Filter the DataFrame to include only rows where the date is after the date_considered
    historical_const = historical_const[historical_const.index > date_considered]

    # Because the date is in descending order, the date_considered is at the bottom of the dataframe
    added_tickers = historical_const.loc[:date_considered, ("Added", 'Ticker')].dropna().unique()
    removed_tickers = historical_const.loc[:date_considered, ("Removed", 'Ticker')].dropna().unique().tolist()
    all_tickers = []
    all_tickers.extend(tickers)
    all_tickers.extend(added_tickers)
    all_tickers.extend(removed_tickers)

    # To remove all the duplicates from the list
    all_tickers = list(set(all_tickers))


    return all_tickers