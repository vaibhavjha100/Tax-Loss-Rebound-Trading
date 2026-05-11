"""Backtest the best regression model for the ASX tax-loss-rebound strategy.

Loads ``models/best_model.joblib`` and the matching test features, picks the
top 20% of stocks by predicted excess return each test year, simulates a
50/50 long-stocks / short-index portfolio over the model's target window,
and writes daily returns to ``data/backtest_results.csv``. Frictionless.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
INDEX_FILE = DATA_DIR / "ohclv_index.csv"
OUTPUT_FILE = DATA_DIR / "backtest_results.csv"
TOP_PCT = 0.20
TEST_START = pd.Timestamp("2020-01-01")


def parse_md(md: str) -> tuple[int, int]:
    """Parse 'MM-DD' into (month, day)."""
    month, day = md.split("-")
    return int(month), int(day)


def load_best_model() -> tuple[object, dict]:
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    info = json.loads((MODELS_DIR / "best_model_info.json").read_text())
    return model, info


def load_test_features(hp_id: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"x_{hp_id}_test.csv")


def load_panels() -> tuple[pd.DataFrame, pd.Series]:
    """Load wide (Date x Ticker) close panel for companies and index close."""
    long_df = pd.read_csv(DATA_DIR / "daily_close_test.csv", parse_dates=["Date"])
    companies_wide = (
        long_df.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    )

    index_df = pd.read_csv(INDEX_FILE, parse_dates=["Date"]).set_index("Date").sort_index()
    index_close = index_df.loc[index_df.index >= TEST_START, "Close"]

    return companies_wide, index_close


def select_top(predictions: pd.Series, pct: float = TOP_PCT) -> list[str]:
    """Top-by-rank selection: pick ceil(pct * N) stocks by predicted return."""
    n_top = max(1, int(round(pct * len(predictions))))
    return predictions.sort_values(ascending=False).head(n_top).index.tolist()


def resolve_window(
    panel_index: pd.DatetimeIndex,
    year: int,
    start_md: tuple[int, int],
    end_md: tuple[int, int],
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return (start_td, end_td) trading days bracketing the target window.

    The buy day (``start_td``) is the first trading day on or after
    ``target_start``; daily returns are observed STRICTLY AFTER ``start_td``
    through ``end_td`` so that the chained product equals
    ``close[end_td] / close[start_td] - 1`` (matching the regression target).
    """
    target_start = pd.Timestamp(date(year, *start_md))
    target_end = pd.Timestamp(date(year, *end_md))

    on_or_after = panel_index[panel_index >= target_start]
    on_or_before_end = panel_index[panel_index <= target_end]

    if on_or_after.empty or on_or_before_end.empty:
        return None
    start_td = on_or_after.min()
    end_td = on_or_before_end.max()
    if start_td >= end_td:
        return None
    return start_td, end_td


def backtest_year(
    year: int,
    x_year: pd.DataFrame,
    model: object,
    feature_cols: list[str],
    companies_wide: pd.DataFrame,
    index_close: pd.Series,
    start_md: tuple[int, int],
    end_md: tuple[int, int],
) -> pd.DataFrame | None:
    """Run one year's backtest. Returns per-day rows or None if window invalid."""
    preds = pd.Series(
        model.predict(x_year[feature_cols]),
        index=x_year["ticker"].values,
        name="pred",
    )
    selected = select_top(preds)

    available = [t for t in selected if t in companies_wide.columns]
    if not available:
        return None

    window = resolve_window(companies_wide.index, year, start_md, end_md)
    if window is None:
        return None
    start_td, end_td = window

    stock_slice = companies_wide.loc[start_td:end_td, available]
    stock_daily = stock_slice.pct_change().dropna(how="all")
    long_basket_return = stock_daily.mean(axis=1, skipna=True)

    if start_td not in index_close.index or end_td not in index_close.index:
        return None
    idx_slice = index_close.loc[start_td:end_td]
    idx_ret = idx_slice.pct_change().dropna()

    aligned = long_basket_return.to_frame("long_basket_return").join(
        idx_ret.rename("index_return"), how="inner"
    )
    aligned["strategy_return"] = (
        0.5 * aligned["long_basket_return"] - 0.5 * aligned["index_return"]
    )
    aligned = aligned.dropna(subset=["strategy_return"])

    aligned = aligned.reset_index().rename(columns={"index": "Date"})
    aligned["year"] = year
    aligned["n_selected"] = len(available)
    return aligned[
        ["Date", "year", "n_selected", "long_basket_return", "index_return", "strategy_return"]
    ]


def main() -> None:
    print("Loading best model and hyperparameter windows...")
    model, info = load_best_model()
    hp_id = info["hp_id"]
    feature_cols = info["feature_columns"]
    start_md = parse_md(info["hp_windows"]["target_start"])
    end_md = parse_md(info["hp_windows"]["target_end"])
    print(
        f"  model={info['model_type']} hp_id={hp_id} "
        f"target={info['hp_windows']['target_start']}->{info['hp_windows']['target_end']}"
    )

    print("Loading test features and daily panels...")
    x_test = load_test_features(hp_id)
    companies_wide, index_close = load_panels()
    print(
        f"  x_test rows: {len(x_test):,} "
        f"({x_test['year'].nunique()} years, "
        f"{x_test['ticker'].nunique()} tickers)"
    )

    print("Running per-year backtest...")
    year_frames: list[pd.DataFrame] = []
    for year in sorted(x_test["year"].unique()):
        x_year = x_test[x_test["year"] == year]
        frame = backtest_year(
            year, x_year, model, feature_cols,
            companies_wide, index_close, start_md, end_md,
        )
        if frame is None or frame.empty:
            print(f"  year={year}: skipped (no usable window or stocks)")
            continue
        year_frames.append(frame)
        print(
            f"  year={year}: n_selected={frame['n_selected'].iat[0]:3d} "
            f"days={len(frame):2d} "
            f"period_return={(1 + frame['strategy_return']).prod() - 1:+.4%}"
        )

    if not year_frames:
        raise RuntimeError("No backtest rows produced.")

    all_years = pd.concat(year_frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    all_years["cumulative_strategy_return"] = (
        (1 + all_years["strategy_return"]).cumprod() - 1
    )

    all_years.to_csv(OUTPUT_FILE, index=False)

    total_days = len(all_years)
    mean_daily = all_years["strategy_return"].mean()
    cum = all_years["cumulative_strategy_return"].iat[-1]
    print(
        f"\nWrote {total_days} daily rows across {all_years['year'].nunique()} years "
        f"-> {OUTPUT_FILE}"
    )
    print(
        f"  mean daily strategy return: {mean_daily:+.4%} | "
        f"cumulative: {cum:+.4%}"
    )


if __name__ == "__main__":
    main()
