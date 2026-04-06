import io
import re

import pandas as pd
import requests


WIKI_NASDAQ_100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


def _request_wikipedia_tables(url: str = WIKI_NASDAQ_100_URL) -> list[pd.DataFrame]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return pd.read_html(io.StringIO(response.text))


def _clean_ticker(token: str) -> str:
    cleaned = re.sub(r"\[[^\]]*\]", "", str(token).strip().upper())
    cleaned = re.sub(r"[^A-Z0-9.\-]", "", cleaned)
    return cleaned


def _split_ticker_cell(cell_value: object) -> list[str]:
    if pd.isna(cell_value):
        return []

    raw = str(cell_value)
    parts = re.split(r",|;|/|\band\b", raw, flags=re.IGNORECASE)
    tickers = []
    for part in parts:
        ticker = _clean_ticker(part)
        if ticker:
            tickers.append(ticker)
    return tickers


def _flatten_columns(table: pd.DataFrame) -> list[str]:
    if isinstance(table.columns, pd.MultiIndex):
        return [" ".join([str(level) for level in col]).strip().lower() for col in table.columns]
    return [str(col).strip().lower() for col in table.columns]


def _find_current_constituents_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for table in tables:
        flattened = _flatten_columns(table)
        if any("ticker" in col for col in flattened) and not any("removed" in col for col in flattened):
            return table
    raise ValueError("Could not find a current-constituents table with a ticker column on the Wikipedia page.")


def _find_changes_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for table in tables:
        flattened = _flatten_columns(table)
        has_date = any("date" in col for col in flattened)
        has_added = any("added" in col for col in flattened)
        has_removed = any("removed" in col for col in flattened)
        if has_date and has_added and has_removed:
            return table
    raise ValueError("Could not find the historical changes table on the Wikipedia page.")


def _resolve_changes_columns(changes_table: pd.DataFrame) -> tuple[object, object, object]:
    if isinstance(changes_table.columns, pd.MultiIndex):
        date_col = None
        added_col = None
        removed_col = None
        for col in changes_table.columns:
            levels = [str(level).strip().lower() for level in col]
            if date_col is None and any("date" == level or "date" in level for level in levels):
                date_col = col
            if added_col is None and "added" in levels and "ticker" in levels:
                added_col = col
            if removed_col is None and "removed" in levels and "ticker" in levels:
                removed_col = col

        if date_col is None or added_col is None or removed_col is None:
            raise ValueError("Could not resolve Date/Added/Removed ticker columns in the historical table.")
        return date_col, added_col, removed_col

    normalized = {str(c).strip().lower(): c for c in changes_table.columns}
    date_col = next((c for name, c in normalized.items() if "date" in name), None)
    added_col = next((c for name, c in normalized.items() if "added" in name and "ticker" in name), None)
    removed_col = next((c for name, c in normalized.items() if "removed" in name and "ticker" in name), None)

    if date_col is None or added_col is None or removed_col is None:
        raise ValueError("Could not resolve Date/Added/Removed ticker columns in the historical table.")

    return date_col, added_col, removed_col


def _extract_current_tickers(current_table: pd.DataFrame) -> set[str]:
    if isinstance(current_table.columns, pd.MultiIndex):
        ticker_col = None
        for col in current_table.columns:
            levels = [str(level).strip().lower() for level in col]
            if "ticker" in levels:
                ticker_col = col
                break
    else:
        ticker_col = next((c for c in current_table.columns if "ticker" in str(c).strip().lower()), None)

    if ticker_col is None:
        raise ValueError("Could not find ticker column in current constituents table.")

    tickers = {
        _clean_ticker(value)
        for value in current_table[ticker_col].dropna().tolist()
        if _clean_ticker(value)
    }
    return tickers


def _extract_membership_changes(changes_table: pd.DataFrame) -> pd.DataFrame:
    date_col, added_col, removed_col = _resolve_changes_columns(changes_table)

    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(changes_table[date_col], errors="coerce"),
            "added": changes_table[added_col],
            "removed": changes_table[removed_col],
        }
    ).dropna(subset=["date"])

    changes["date"] = changes["date"].dt.tz_localize(None).dt.normalize()
    changes["added"] = changes["added"].apply(_split_ticker_cell)
    changes["removed"] = changes["removed"].apply(_split_ticker_cell)

    return changes


def _get_trading_days(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
    try:
        import pandas_market_calendars as mcal

        nasdaq = mcal.get_calendar("NASDAQ")
        trading_days = nasdaq.valid_days(start_date=start_date, end_date=end_date)
        trading_days = pd.DatetimeIndex(trading_days).tz_localize(None).normalize()
        return trading_days
    except Exception:
        try:
            import exchange_calendars as xcals

            # XNAS is the Nasdaq exchange calendar in exchange_calendars.
            xnas = xcals.get_calendar("XNAS")
            sessions = xnas.sessions_in_range(start_date, end_date)
            return pd.DatetimeIndex(sessions).tz_localize(None).normalize()
        except Exception as exc:
            raise ImportError(
                "A market calendar package is required to build true trading-day indices. "
                "Install pandas_market_calendars or exchange_calendars."
            ) from exc


def get_nasdaq_tickers(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> list[str]:
    """
    Convenience wrapper that returns all tickers that have been in the Nasdaq-100
    at any trading day between start_date and end_date.
    """
    start_ts = pd.to_datetime(start_date).tz_localize(None).normalize()
    end_ts = pd.to_datetime(end_date).tz_localize(None).normalize()

    if end_ts < start_ts:
        raise ValueError("end_date must be on or after start_date")

    tables = _request_wikipedia_tables()
    current_table = _find_current_constituents_table(tables)
    changes_table = _find_changes_table(tables)

    current_members = _extract_current_tickers(current_table)
    changes = _extract_membership_changes(changes_table)

    membership_at_start = set(current_members)
    rewind_events = changes[changes["date"] > start_ts].sort_values("date", ascending=False)
    for _, event in rewind_events.iterrows():
        for added_ticker in event["added"]:
            membership_at_start.discard(added_ticker)
        for removed_ticker in event["removed"]:
            membership_at_start.add(removed_ticker)

    interval_events = changes[(changes["date"] >= start_ts) & (changes["date"] <= end_ts)].sort_values("date")

    universe_since_start = set(membership_at_start)
    for _, event in interval_events.iterrows():
        universe_since_start.update(event["added"])
        universe_since_start.update(event["removed"])

    return sorted(universe_since_start)