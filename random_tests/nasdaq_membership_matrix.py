import pandas as pd

from get_nasdaq_tickers import (
    _extract_current_tickers,
    _extract_membership_changes,
    _find_changes_table,
    _find_current_constituents_table,
    _get_trading_days,
    _request_wikipedia_tables,
    get_nasdaq_tickers,
)


def get_nasdaq_membership_matrix(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    expected_constituents: int | None = 100,
    strict_validation: bool = False,
) -> pd.DataFrame:
    """
    Return a boolean DataFrame indexed by trading day with columns for every ticker
    that has been in the Nasdaq-100 since start_date.

    Each cell is True if the ticker is in the index on that trading day, else False.
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

    trading_days = _get_trading_days(start_ts, end_ts)
    columns = get_nasdaq_tickers(start_date=start_ts, end_date=end_ts)
    membership_df = pd.DataFrame(False, index=trading_days, columns=columns)

    active_members = set(membership_at_start)
    events_by_date: dict[pd.Timestamp, list[tuple[list[str], list[str]]]] = {}
    for _, event in interval_events.iterrows():
        event_date = event["date"]
        events_by_date.setdefault(event_date, []).append((event["added"], event["removed"]))

    for day in membership_df.index:
        day_events = events_by_date.get(day, [])
        for added_list, removed_list in day_events:
            for ticker in added_list:
                active_members.add(ticker)
            for ticker in removed_list:
                active_members.discard(ticker)

        if active_members:
            membership_df.loc[day, list(active_members)] = True

    membership_df.index.name = "Date"

    if expected_constituents is not None:
        constituent_count = membership_df.sum(axis=1)
        invalid_count_mask = constituent_count != expected_constituents
        if strict_validation and invalid_count_mask.any():
            bad = constituent_count[invalid_count_mask]
            sample = bad.head(10).to_dict()
            raise ValueError(
                "Constituent count validation failed. "
                f"Expected {expected_constituents}, found mismatches on {int(invalid_count_mask.sum())} days. "
                f"Sample: {sample}"
            )

    return membership_df
