import html
import json
import re
from pathlib import Path

import pandas as pd
import requests


URL = "https://www.marketindex.com.au/asx-listed-companies"
DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "asxsmallordinaries.csv"
REQUIRED_COLUMNS = ["Code", "Company", "Sector", "Mkt Cap"]


def extract_companies_json(page_html: str) -> list[dict]:
    match = re.search(r':companies="(.*?)"', page_html)
    if not match:
        raise RuntimeError("Could not find embedded companies data in page HTML.")

    companies_json = html.unescape(match.group(1))
    return json.loads(companies_json)


def fetch_companies() -> pd.DataFrame:
    response = requests.get(URL, timeout=30)
    response.raise_for_status()

    companies = extract_companies_json(response.text)
    if not companies:
        raise RuntimeError("No company records found in embedded page data.")

    records = []
    for company in companies:
        records.append(
            {
                "Code": company.get("code", ""),
                "Company": company.get("title", ""),
                "Sector": (company.get("company_sector") or {}).get("gics_sector", ""),
                "Mkt Cap": ((company.get("formatted") or {}).get("marketCap") or ""),
                "market_cap_numeric": company.get("market_cap") or 0,
            }
        )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("market_cap_numeric", ascending=False).reset_index(drop=True)

    # Rank is 1-indexed, so positions 101-300 map to iloc[100:300].
    small_ords = df.iloc[100:300][REQUIRED_COLUMNS].copy()
    return small_ords


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result = fetch_companies()
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(result)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
