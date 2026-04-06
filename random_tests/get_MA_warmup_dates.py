from datetime import datetime, timedelta

# Define your actual analysis period (e.g., before COVID - March 2020)
def get_rolledback_start(start, end):
    ANALYSIS_START = start  # Start of 2025 tumble
    ANALYSIS_END = end  # Q1 aftermath

    # Calculate data fetch start date (200 trading days ≈ 285 calendar days to be safe)
    analysis_start_dt = datetime.strptime(ANALYSIS_START, "%Y-%m-%d")
    data_fetch_start = (analysis_start_dt - timedelta(days=285)).strftime("%Y-%m-%d")

    print(f"Fetching data from: {data_fetch_start}")
    print(f"Analysis period: {ANALYSIS_START} to {ANALYSIS_END}")
    print(f"Extra days for 200-day MA warm-up: ~285 days")

    return data_fetch_start, ANALYSIS_END