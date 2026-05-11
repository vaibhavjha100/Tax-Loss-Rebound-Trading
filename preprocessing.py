"""Preprocessing for the ASX tax-loss-rebound strategy.

Builds per (year, ticker, hyperparameter) feature/target rows for linear
regression across 18 date-window hyperparameter combinations, and a trimmed
daily-close panel (1 Jun - 15 Aug each year) for backtest return calcs.
Writes everything under ``data/``.
"""

from __future__ import annotations

import itertools
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
COMPANIES_FILE = DATA_DIR / "ohlcv_companies.csv"
INDEX_FILE = DATA_DIR / "ohclv_index.csv"

FEATURE_END_MDS = [(5, 31), (6, 15), (6, 30)]
ABNVOL_WINDOWS_MDS = [
    ((6, 1), (6, 15)),
    ((6, 16), (6, 30)),
    ((6, 1), (6, 30)),
]
TARGET_WINDOWS_MDS = [
    ((7, 1), (8, 15)),
    ((7, 15), (8, 15)),
]

TRAIN_TEST_CUTOFF_YEAR = 2020
ABNVOL_BASELINE_DAYS = 60
WINDOW_52W_DAYS = 252


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read company and index OHLCV CSVs into typed, indexed frames."""
    companies = pd.read_csv(COMPANIES_FILE, parse_dates=["Date"])
    companies = companies.set_index(["Date", "Ticker"]).sort_index()

    index_df = pd.read_csv(INDEX_FILE, parse_dates=["Date"])
    index_df = index_df.set_index("Date").sort_index()

    return companies, index_df


def align_and_ffill(
    companies: pd.DataFrame, index_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Truncate companies to the index start date and forward-fill per ticker.

    ``groupby('Ticker').ffill()`` only fills gaps inside each ticker's listed
    period; pre-listing rows stay NaN and get filtered out at feature time.
    """
    index_start = index_df.index.min()
    companies = companies[companies.index.get_level_values("Date") >= index_start]

    companies = (
        companies
        .sort_index()
        .groupby(level="Ticker", group_keys=False)
        .ffill()
    )
    return companies, index_df


def build_hp_grid() -> list[dict]:
    """Cartesian product of feature_end x abnvol_window x target_window."""
    combos = itertools.product(
        FEATURE_END_MDS, ABNVOL_WINDOWS_MDS, TARGET_WINDOWS_MDS
    )
    grid: list[dict] = []
    for i, (feat_end_md, abnvol_md, target_md) in enumerate(combos, start=1):
        grid.append(
            {
                "hp_id": f"v{i:02d}",
                "feature_end_md": feat_end_md,
                "abnvol_start_md": abnvol_md[0],
                "abnvol_end_md": abnvol_md[1],
                "target_start_md": target_md[0],
                "target_end_md": target_md[1],
            }
        )
    return grid


def hp_grid_to_dataframe(hp_grid: list[dict]) -> pd.DataFrame:
    """Flatten the grid into a CSV-friendly table of MM-DD strings."""
    def md(t: tuple[int, int]) -> str:
        return f"{t[0]:02d}-{t[1]:02d}"

    rows = [
        {
            "hp_id": hp["hp_id"],
            "feature_end": md(hp["feature_end_md"]),
            "abnvol_start": md(hp["abnvol_start_md"]),
            "abnvol_end": md(hp["abnvol_end_md"]),
            "target_start": md(hp["target_start_md"]),
            "target_end": md(hp["target_end_md"]),
        }
        for hp in hp_grid
    ]
    return pd.DataFrame(rows)


def _closest_on_or_before(series: pd.Series, target: pd.Timestamp) -> float | None:
    """Return the value at the latest index <= ``target``, or None if missing."""
    sub = series.loc[:target]
    sub = sub.dropna()
    if sub.empty:
        return None
    return float(sub.iloc[-1])


def _mean_in_window(
    series: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> float | None:
    sub = series.loc[start:end].dropna()
    if sub.empty:
        return None
    return float(sub.mean())


def _trailing_mean(
    series: pd.Series, end_exclusive: pd.Timestamp, days: int
) -> float | None:
    """Mean of the last ``days`` observations strictly before ``end_exclusive``."""
    sub = series.loc[: end_exclusive - pd.Timedelta(days=1)].dropna()
    if sub.empty:
        return None
    sub = sub.iloc[-days:]
    if sub.empty:
        return None
    return float(sub.mean())


def _trailing_min(
    series: pd.Series, end_inclusive: pd.Timestamp, days: int
) -> float | None:
    sub = series.loc[:end_inclusive].dropna()
    if sub.empty:
        return None
    sub = sub.iloc[-days:]
    if sub.empty:
        return None
    return float(sub.min())


def build_features_targets(
    companies: pd.DataFrame, index_df: pd.DataFrame, hp_grid: list[dict]
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build x and y dataframes per hp_id.

    Returns a dict ``{hp_id: {"x": DataFrame, "y": DataFrame}}`` where each
    frame is indexed by ``(year, ticker)``.
    """
    today = pd.Timestamp.today().normalize()
    last_complete_year = today.year - 1
    first_year = 2005

    index_close = index_df["Close"].sort_index()

    companies_by_ticker: dict[str, pd.DataFrame] = {
        ticker: frame.reset_index(level="Ticker", drop=True).sort_index()
        for ticker, frame in companies.groupby(level="Ticker", sort=False)
    }

    index_return_cache: dict[tuple[int, tuple, tuple], float | None] = {}

    def index_return(start_md: tuple, end_md: tuple, year: int, year_for_start: int) -> float | None:
        key = (year, start_md, end_md, year_for_start)
        if key in index_return_cache:
            return index_return_cache[key]
        start_ts = pd.Timestamp(date(year_for_start, *start_md))
        end_ts = pd.Timestamp(date(year, *end_md))
        start_val = _closest_on_or_before(index_close, start_ts)
        end_val = _closest_on_or_before(index_close, end_ts)
        result = None
        if start_val is not None and end_val is not None and start_val > 0:
            result = (end_val / start_val) - 1.0
        index_return_cache[key] = result
        return result

    results: dict[str, dict[str, pd.DataFrame]] = {
        hp["hp_id"]: {"x_rows": [], "y_rows": []} for hp in hp_grid
    }

    for ticker, t_df in companies_by_ticker.items():
        close = t_df["Close"]
        low = t_df["Low"]
        volume = t_df["Volume"]

        if close.dropna().empty:
            continue

        ticker_first_date = close.dropna().index.min()
        ticker_last_date = close.dropna().index.max()

        for year in range(first_year, last_complete_year + 1):
            feature_start_ts = pd.Timestamp(date(year - 1, 7, 1))
            if feature_start_ts < ticker_first_date:
                continue

            stock_feat_start = _closest_on_or_before(close, feature_start_ts)
            if stock_feat_start is None or stock_feat_start <= 0:
                continue

            for hp in hp_grid:
                hp_id = hp["hp_id"]
                feature_end_ts = pd.Timestamp(date(year, *hp["feature_end_md"]))
                abnvol_start_ts = pd.Timestamp(date(year, *hp["abnvol_start_md"]))
                abnvol_end_ts = pd.Timestamp(date(year, *hp["abnvol_end_md"]))
                target_start_ts = pd.Timestamp(date(year, *hp["target_start_md"]))
                target_end_ts = pd.Timestamp(date(year, *hp["target_end_md"]))

                if target_end_ts > ticker_last_date:
                    continue

                stock_feat_end = _closest_on_or_before(close, feature_end_ts)
                stock_target_start = _closest_on_or_before(close, target_start_ts)
                stock_target_end = _closest_on_or_before(close, target_end_ts)
                if (
                    stock_feat_end is None
                    or stock_target_start is None
                    or stock_target_end is None
                    or stock_feat_start <= 0
                    or stock_target_start <= 0
                ):
                    continue

                idx_feat_ret = index_return(
                    (7, 1), hp["feature_end_md"], year, year - 1
                )
                idx_target_ret = index_return(
                    hp["target_start_md"], hp["target_end_md"], year, year
                )
                if idx_feat_ret is None or idx_target_ret is None:
                    continue

                stock_feat_ret = (stock_feat_end / stock_feat_start) - 1.0
                underperf = stock_feat_ret - idx_feat_ret

                low_min = _trailing_min(low, feature_end_ts, WINDOW_52W_DAYS)
                if low_min is None or low_min <= 0:
                    continue
                prox_52w_low = (stock_feat_end / low_min) - 1.0

                abnvol_mean = _mean_in_window(volume, abnvol_start_ts, abnvol_end_ts)
                baseline_mean = _trailing_mean(
                    volume, abnvol_start_ts, ABNVOL_BASELINE_DAYS
                )
                if (
                    abnvol_mean is None
                    or baseline_mean is None
                    or baseline_mean <= 0
                ):
                    continue
                abn_vol = (abnvol_mean / baseline_mean) - 1.0

                stock_target_ret = (stock_target_end / stock_target_start) - 1.0
                target_outperf = stock_target_ret - idx_target_ret

                if not all(
                    np.isfinite(v)
                    for v in (underperf, prox_52w_low, abn_vol, target_outperf)
                ):
                    continue

                results[hp_id]["x_rows"].append(
                    {
                        "year": year,
                        "ticker": ticker,
                        "underperf": underperf,
                        "prox_52w_low": prox_52w_low,
                        "abn_vol": abn_vol,
                    }
                )
                results[hp_id]["y_rows"].append(
                    {
                        "year": year,
                        "ticker": ticker,
                        "target_outperf": target_outperf,
                    }
                )

    output: dict[str, dict[str, pd.DataFrame]] = {}
    for hp_id, buckets in results.items():
        x_df = pd.DataFrame(buckets["x_rows"])
        y_df = pd.DataFrame(buckets["y_rows"])
        if not x_df.empty:
            x_df = x_df.set_index(["year", "ticker"]).sort_index()
        if not y_df.empty:
            y_df = y_df.set_index(["year", "ticker"]).sort_index()
        output[hp_id] = {"x": x_df, "y": y_df}
    return output


def split_train_test(
    df: pd.DataFrame, cutoff_year: int = TRAIN_TEST_CUTOFF_YEAR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a (year, ticker)-indexed frame on the observation year."""
    if df.empty:
        return df.copy(), df.copy()
    years = df.index.get_level_values("year")
    train = df[years < cutoff_year]
    test = df[years >= cutoff_year]
    return train, test


def build_daily_close_panel(companies: pd.DataFrame) -> pd.DataFrame:
    """Trim the daily Close panel to 1 Jun - 15 Aug across all years."""
    dates = companies.index.get_level_values("Date")
    months = dates.month
    days = dates.day
    mask = (
        (months == 6)
        | (months == 7)
        | ((months == 8) & (days <= 15))
    )

    trimmed = companies.loc[mask, ["Close"]].copy()
    trimmed = trimmed.dropna(subset=["Close"]).sort_index()
    return trimmed


def split_daily_panel(
    daily: pd.DataFrame, cutoff: pd.Timestamp = pd.Timestamp("2020-01-01")
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = daily.index.get_level_values("Date")
    train = daily[dates < cutoff]
    test = daily[dates >= cutoff]
    return train, test


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading companies and index OHLCV...")
    companies, index_df = load_data()
    print(f"  companies rows: {len(companies):,}")
    print(f"  index rows:     {len(index_df):,}")

    print("Aligning to index start and forward-filling per ticker...")
    companies, index_df = align_and_ffill(companies, index_df)

    print("Building hyperparameter grid...")
    hp_grid = build_hp_grid()
    hp_df = hp_grid_to_dataframe(hp_grid)
    hp_df.to_csv(DATA_DIR / "hyperparameters.csv", index=False)
    print(f"  {len(hp_df)} hyperparameter combos written to data/hyperparameters.csv")

    print("Building features and targets across all hyperparameter combos...")
    feats = build_features_targets(companies, index_df, hp_grid)

    total_x_rows = 0
    for hp_id, frames in feats.items():
        x_df, y_df = frames["x"], frames["y"]
        x_train, x_test = split_train_test(x_df)
        y_train, y_test = split_train_test(y_df)

        x_train.to_csv(DATA_DIR / f"x_{hp_id}_train.csv")
        x_test.to_csv(DATA_DIR / f"x_{hp_id}_test.csv")
        y_train.to_csv(DATA_DIR / f"y_{hp_id}_train.csv")
        y_test.to_csv(DATA_DIR / f"y_{hp_id}_test.csv")
        total_x_rows += len(x_df)

    print(f"  wrote x/y train/test for {len(feats)} hp_ids ({total_x_rows:,} total x rows)")

    print("Building trimmed daily Close panel (1 Jun - 15 Aug)...")
    daily = build_daily_close_panel(companies)
    daily_train, daily_test = split_daily_panel(daily)
    daily_train.to_csv(DATA_DIR / "daily_close_train.csv")
    daily_test.to_csv(DATA_DIR / "daily_close_test.csv")
    print(f"  daily_close_train rows: {len(daily_train):,}")
    print(f"  daily_close_test rows:  {len(daily_test):,}")

    print("Done.")


if __name__ == "__main__":
    main()
