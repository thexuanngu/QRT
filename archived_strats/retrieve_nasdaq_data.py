import requests
import pandas as pd

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

# read_html from raw HTML instead of URL
tables = pd.read_html(html)

print(f"Loaded {len(tables)} tables")

for i, df in enumerate(tables):
    print("TABLE", i)
    print("shape:", df.shape)
    print("columns:", df.columns.tolist())
    print("\n")
    print(df.head(1).to_dict(orient="records"))
    print("-" * 60)


