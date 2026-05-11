"""Evaluate the tax-loss-rebound backtest.

Loads ``data/backtest_results.csv``, computes a comprehensive set of strategy
statistics (CAGR, ann vol, Sharpe, Sortino, drawdown, win rate, skew/kurtosis,
best/worst day/year, in-window benchmark) and a CAPM regression of the
strategy returns against the index. Prints both blocks and saves them under
``data/results/``.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


DATA_DIR = Path("data")
BACKTEST_FILE = DATA_DIR / "backtest_results.csv"
RESULTS_DIR = DATA_DIR / "results"
ANN_FACTOR = 252


def load_backtest() -> pd.DataFrame:
    df = pd.read_csv(BACKTEST_FILE, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def per_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("year", as_index=False)
        .agg(
            n_selected=("n_selected", "first"),
            n_days=("strategy_return", "size"),
            long_period_return=("long_basket_return", lambda r: (1 + r).prod() - 1),
            index_period_return=("index_return", lambda r: (1 + r).prod() - 1),
            strategy_period_return=("strategy_return", lambda r: (1 + r).prod() - 1),
        )
    )


def _max_drawdown(cum_returns: pd.Series) -> tuple[float, int, str, str]:
    """Return (max_drawdown, duration_days, peak_date, trough_date)."""
    wealth = 1.0 + cum_returns
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    trough_pos = int(np.argmin(drawdown.to_numpy()))
    mdd = float(drawdown.iloc[trough_pos])

    if trough_pos > 0:
        peak_pos = int(np.argmax(wealth.iloc[: trough_pos + 1].to_numpy()))
    else:
        peak_pos = trough_pos
    duration = trough_pos - peak_pos
    return mdd, duration, str(cum_returns.index[peak_pos]), str(cum_returns.index[trough_pos])


def compute_stats(df: pd.DataFrame, per_year: pd.DataFrame) -> dict[str, object]:
    returns = df["strategy_return"].astype(float)
    n_obs = len(returns)
    n_years = int(df["year"].nunique())
    mean_active = n_obs / n_years if n_years else float("nan")

    total_return = float(df["cumulative_strategy_return"].iloc[-1])
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else float("nan")

    mean_daily = float(returns.mean())
    median_daily = float(returns.median())
    daily_vol = float(returns.std(ddof=1))
    ann_vol = daily_vol * sqrt(ANN_FACTOR)

    skewness = float(sp_stats.skew(returns, bias=False))
    excess_kurt = float(sp_stats.kurtosis(returns, bias=False))

    sharpe = (mean_daily / daily_vol) * sqrt(ANN_FACTOR) if daily_vol > 0 else float("nan")

    negatives = returns[returns < 0]
    downside_std = float(np.sqrt((negatives ** 2).mean())) if len(negatives) else 0.0
    sortino = (
        (mean_daily / downside_std) * sqrt(ANN_FACTOR)
        if downside_std > 0
        else float("nan")
    )

    cum_indexed = df["cumulative_strategy_return"].copy()
    cum_indexed.index = df["Date"]
    mdd, mdd_days, mdd_peak, mdd_trough = _max_drawdown(cum_indexed)

    calmar = cagr / abs(mdd) if mdd < 0 else float("nan")

    win_rate = float((returns > 0).mean())

    best_idx = int(returns.idxmax())
    worst_idx = int(returns.idxmin())
    best_day_val = float(returns.iloc[best_idx])
    worst_day_val = float(returns.iloc[worst_idx])
    best_day_date = df["Date"].iloc[best_idx].date().isoformat()
    worst_day_date = df["Date"].iloc[worst_idx].date().isoformat()

    best_year_row = per_year.loc[per_year["strategy_period_return"].idxmax()]
    worst_year_row = per_year.loc[per_year["strategy_period_return"].idxmin()]

    index_total_in_window = float((1 + df["index_return"]).prod() - 1)
    long_total_in_window = float((1 + df["long_basket_return"]).prod() - 1)
    excess_total = total_return - index_total_in_window

    return {
        "n_observations": n_obs,
        "n_calendar_years": n_years,
        "mean_active_days_per_year": mean_active,
        "total_return": total_return,
        "cagr_calendar": cagr,
        "mean_daily_return": mean_daily,
        "median_daily_return": median_daily,
        "daily_vol": daily_vol,
        "ann_vol": ann_vol,
        "skewness": skewness,
        "excess_kurtosis": excess_kurt,
        "sharpe_annual": sharpe,
        "sortino_annual": sortino,
        "max_drawdown": mdd,
        "max_drawdown_duration_days": mdd_days,
        "max_drawdown_peak_date": mdd_peak.split(" ")[0],
        "max_drawdown_trough_date": mdd_trough.split(" ")[0],
        "calmar": calmar,
        "win_rate": win_rate,
        "best_day_return": best_day_val,
        "best_day_date": best_day_date,
        "worst_day_return": worst_day_val,
        "worst_day_date": worst_day_date,
        "best_year": int(best_year_row["year"]),
        "best_year_return": float(best_year_row["strategy_period_return"]),
        "worst_year": int(worst_year_row["year"]),
        "worst_year_return": float(worst_year_row["strategy_period_return"]),
        "long_total_return_in_window": long_total_in_window,
        "index_total_return_in_window": index_total_in_window,
        "excess_total_return_vs_index": excess_total,
    }


def run_capm(df: pd.DataFrame) -> dict[str, float]:
    x = df["index_return"].astype(float).to_numpy()
    y = df["strategy_return"].astype(float).to_numpy()
    n = len(y)

    res = sp_stats.linregress(x, y)
    alpha_daily = float(res.intercept)
    alpha_se = float(res.intercept_stderr)
    beta = float(res.slope)
    beta_se = float(res.stderr)
    r_squared = float(res.rvalue) ** 2

    dof = n - 2
    t_alpha = alpha_daily / alpha_se if alpha_se > 0 else float("nan")
    p_alpha = float(2 * sp_stats.t.sf(abs(t_alpha), df=dof)) if dof > 0 else float("nan")
    t_beta = beta / beta_se if beta_se > 0 else float("nan")
    p_beta = float(res.pvalue)

    resid = y - (alpha_daily + beta * x)
    resid_vol_daily = float(np.std(resid, ddof=2)) if n > 2 else float("nan")
    alpha_annual = alpha_daily * ANN_FACTOR
    resid_vol_ann = resid_vol_daily * sqrt(ANN_FACTOR)
    info_ratio = alpha_annual / resid_vol_ann if resid_vol_ann > 0 else float("nan")

    return {
        "n_obs": n,
        "alpha_daily": alpha_daily,
        "alpha_se": alpha_se,
        "alpha_tstat": float(t_alpha),
        "alpha_pvalue": p_alpha,
        "alpha_annualized": alpha_annual,
        "beta": beta,
        "beta_se": beta_se,
        "beta_tstat": float(t_beta),
        "beta_pvalue": p_beta,
        "r_squared": r_squared,
        "residual_vol_daily": resid_vol_daily,
        "residual_vol_annualized": resid_vol_ann,
        "information_ratio": info_ratio,
    }


def _fmt_pct(x: float) -> str:
    return f"{x:+.4%}" if pd.notna(x) else "  n/a   "


def _fmt_num(x: float, prec: int = 4) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:.{prec}f}"


PERCENT_KEYS = {
    "total_return", "cagr_calendar", "mean_daily_return", "median_daily_return",
    "daily_vol", "ann_vol", "max_drawdown", "win_rate",
    "best_day_return", "worst_day_return",
    "best_year_return", "worst_year_return",
    "long_total_return_in_window", "index_total_return_in_window",
    "excess_total_return_vs_index",
    "alpha_daily", "alpha_se", "alpha_annualized",
    "residual_vol_daily", "residual_vol_annualized",
}


def print_per_year(per_year: pd.DataFrame) -> None:
    print("\n== Per-year period returns ==")
    formatted = per_year.copy()
    for col in ("long_period_return", "index_period_return", "strategy_period_return"):
        formatted[col] = formatted[col].map(lambda v: f"{v:+.4%}")
    print(formatted.to_string(index=False))


def _print_block(title: str, items: dict[str, object]) -> None:
    print(f"\n== {title} ==")
    width = max(len(k) for k in items)
    for k, v in items.items():
        if isinstance(v, str) or isinstance(v, int) and k in {
            "n_observations", "n_calendar_years",
            "max_drawdown_duration_days", "best_year", "worst_year", "n_obs",
        }:
            print(f"  {k:<{width}}  {v}")
        elif k in PERCENT_KEYS:
            print(f"  {k:<{width}}  {_fmt_pct(float(v))}")
        else:
            print(f"  {k:<{width}}  {_fmt_num(float(v))}")


def print_stats(stats: dict[str, object]) -> None:
    _print_block("Backtest statistics", stats)


def print_capm(capm: dict[str, float]) -> None:
    _print_block("CAPM regression: strategy ~ alpha + beta * index", capm)
    verdict = "SIGNIFICANT at 5%" if capm["alpha_pvalue"] < 0.05 else "NOT significant at 5%"
    print(
        f"\n  -> alpha = {capm['alpha_annualized']:+.4%} per year (daily t={capm['alpha_tstat']:+.3f}, "
        f"p={capm['alpha_pvalue']:.4f}) - {verdict}"
    )


def save_results(
    per_year: pd.DataFrame,
    stats: dict[str, object],
    capm: dict[str, float],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_year.to_csv(out_dir / "per_year_returns.csv", index=False)
    pd.DataFrame(
        {"metric": list(stats.keys()), "value": list(stats.values())}
    ).to_csv(out_dir / "backtest_stats.csv", index=False)
    pd.DataFrame(
        {"metric": list(capm.keys()), "value": list(capm.values())}
    ).to_csv(out_dir / "capm_results.csv", index=False)


def main() -> None:
    df = load_backtest()
    py = per_year_summary(df)
    stats = compute_stats(df, py)
    capm = run_capm(df)

    print_per_year(py)
    print_stats(stats)
    print_capm(capm)

    save_results(py, stats, capm, RESULTS_DIR)
    print(f"\nResults saved under {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
