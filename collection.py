from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


DATA_DIR = Path("data")
TICKERS_FILE = DATA_DIR / "asxsmallordinaries.csv"
COMPANIES_OUTPUT_FILE = DATA_DIR / "ohlcv_companies.csv"
INDEX_OUTPUT_FILE = DATA_DIR / "ohclv_index.csv"
INDEX_TICKER = "^AXSO"
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_company_tickers(csv_path: str | Path) -> list[str]:
    tickers_df = pd.read_csv(csv_path)
    if "Code" not in tickers_df.columns:
        raise RuntimeError("Expected 'Code' column in asxsmallordinaries.csv.")

    codes = tickers_df["Code"].dropna().astype(str).str.strip()
    codes = codes[codes != ""]
    # Convert ASX code to Yahoo ticker format.
    return [f"{code}.AX" for code in codes]


def download_companies_ohlcv(tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        raise RuntimeError("No tickers found for company download.")

    data = yf.download(
        tickers=tickers,
        period="max",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=True,
    )

    if data.empty:
        raise RuntimeError("No company OHLCV data returned from Yahoo Finance.")

    long_frames: list[pd.DataFrame] = []
    for ticker in tickers:
        if ticker not in data.columns.get_level_values(0):
            continue

        ticker_df = data[ticker].copy()
        available_cols = [col for col in OHLCV_COLUMNS if col in ticker_df.columns]
        ticker_df = ticker_df[available_cols].dropna(how="all")
        if ticker_df.empty:
            continue

        ticker_df["Ticker"] = ticker
        ticker_df = ticker_df.reset_index()
        if "Date" not in ticker_df.columns:
            # yfinance can name the index "index" when unnamed.
            ticker_df = ticker_df.rename(columns={"index": "Date"})
        long_frames.append(ticker_df)

    if not long_frames:
        raise RuntimeError("Company download succeeded but produced no usable OHLCV rows.")

    companies_df = pd.concat(long_frames, ignore_index=True)
    companies_df["Date"] = pd.to_datetime(companies_df["Date"], errors="coerce")
    companies_df = companies_df.dropna(subset=["Date"])
    companies_df = companies_df.set_index(["Date", "Ticker"]).sort_index()
    return companies_df


def download_index_ohlcv(index_ticker: str) -> pd.DataFrame:
    index_df = yf.download(
        tickers=index_ticker,
        period="max",
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )

    if index_df.empty:
        raise RuntimeError(f"No index OHLCV data returned for {index_ticker}.")

    if isinstance(index_df.columns, pd.MultiIndex):
        # Some yfinance versions return a MultiIndex even for one ticker.
        index_df.columns = index_df.columns.get_level_values(0)

    available_cols = [col for col in OHLCV_COLUMNS if col in index_df.columns]
    index_df = index_df[available_cols].dropna(how="all")
    index_df.index.name = "Date"
    return index_df.sort_index()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_company_tickers(TICKERS_FILE)
    companies_df = download_companies_ohlcv(tickers)
    companies_df.to_csv(COMPANIES_OUTPUT_FILE)

    index_df = download_index_ohlcv(INDEX_TICKER)
    index_df.to_csv(INDEX_OUTPUT_FILE)

    print(f"Saved {len(companies_df)} rows to {COMPANIES_OUTPUT_FILE}")
    print(f"Saved {len(index_df)} rows to {INDEX_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
