"""Train the model zoo for the ASX tax-loss-rebound strategy.

For each of the 18 date-window hyperparameter datasets (v01..v18) under data/,
fits 7 base regression models plus a VotingRegressor ensemble. Scores each by
5-fold CV MSE on the training set, persists every fitted estimator to
models/, writes a leaderboard CSV, and copies the lowest-CV-MSE model to
models/best_model.joblib.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
HP_IDS = [f"v{i:02d}" for i in range(1, 19)]
FEATURE_COLS = ["underperf", "prox_52w_low", "abn_vol"]
TARGET_COL = "target_outperf"
RANDOM_STATE = 42
N_SPLITS = 5


def load_train_data(hp_id: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load (x_train, y_train) for a given hp_id, dropping the (year,ticker) index."""
    x_df = pd.read_csv(DATA_DIR / f"x_{hp_id}_train.csv")
    y_df = pd.read_csv(DATA_DIR / f"y_{hp_id}_train.csv")

    merged = x_df.merge(y_df, on=["year", "ticker"], how="inner")
    X = merged[FEATURE_COLS].copy()
    y = merged[TARGET_COL].copy()
    return X, y


def make_base_models(random_state: int = RANDOM_STATE) -> dict[str, BaseEstimator]:
    """Return the 7 base regressors. Linear-family models are scaled."""
    return {
        "Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("est", LinearRegression()),
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("est", Lasso(alpha=0.001, max_iter=10_000, random_state=random_state)),
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("est", Ridge(alpha=1.0, random_state=random_state)),
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("est", ElasticNet(
                alpha=0.001, l1_ratio=0.5, max_iter=10_000,
                random_state=random_state,
            )),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None,
            n_jobs=-1, random_state=random_state,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1, tree_method="hist",
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300, max_depth=-1, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1, verbose=-1,
        ),
    }


def make_ensemble(base_models: dict[str, BaseEstimator]) -> VotingRegressor:
    """Simple unweighted average of clones of all base models."""
    return VotingRegressor(
        estimators=[(name, clone(model)) for name, model in base_models.items()],
        n_jobs=1,
    )


def evaluate_and_save(
    name: str,
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    hp_id: str,
    models_dir: Path,
    kfold: KFold,
) -> dict:
    """Compute 5-fold CV MSE, fit on full train, persist, return leaderboard row."""
    cv_scores = cross_val_score(
        clone(model), X, y,
        cv=kfold, scoring="neg_mean_squared_error", n_jobs=1,
    )
    cv_mses = -cv_scores

    model.fit(X, y)
    train_preds = model.predict(X)
    train_mse = mean_squared_error(y, train_preds)

    out_path = models_dir / f"{name}_{hp_id}.joblib"
    joblib.dump(model, out_path)

    return {
        "model_type": name,
        "hp_id": hp_id,
        "cv_mse": float(cv_mses.mean()),
        "cv_mse_std": float(cv_mses.std()),
        "train_mse": float(train_mse),
        "n_train": int(len(y)),
    }


def build_leaderboard(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.sort_values("cv_mse", ascending=True).reset_index(drop=True)
    return df


def save_best(
    leaderboard: pd.DataFrame,
    hp_grid_df: pd.DataFrame,
    models_dir: Path,
) -> None:
    """Copy the lowest-cv_mse model to best_model.joblib and write info JSON."""
    best = leaderboard.iloc[0]
    src = models_dir / f"{best.model_type}_{best.hp_id}.joblib"
    dst = models_dir / "best_model.joblib"
    shutil.copyfile(src, dst)

    hp_row = hp_grid_df.loc[hp_grid_df["hp_id"] == best.hp_id].iloc[0]
    hp_windows = {
        "feature_end": hp_row["feature_end"],
        "abnvol_start": hp_row["abnvol_start"],
        "abnvol_end": hp_row["abnvol_end"],
        "target_start": hp_row["target_start"],
        "target_end": hp_row["target_end"],
    }
    info = {
        "model_type": best.model_type,
        "hp_id": best.hp_id,
        "cv_mse": float(best.cv_mse),
        "cv_mse_std": float(best.cv_mse_std),
        "train_mse": float(best.train_mse),
        "n_train": int(best.n_train),
        "feature_columns": FEATURE_COLS,
        "target_column": TARGET_COL,
        "hp_windows": hp_windows,
        "source_model_file": src.name,
    }
    (models_dir / "best_model_info.json").write_text(json.dumps(info, indent=2))


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    hp_grid_df = pd.read_csv(DATA_DIR / "hyperparameters.csv")
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows: list[dict] = []
    total = len(HP_IDS) * 8

    print(f"Training {total} models ({len(HP_IDS)} hp_ids x 8 model types)...")

    idx = 0
    for hp_id in HP_IDS:
        X, y = load_train_data(hp_id)
        base_models = make_base_models()

        for name, model in base_models.items():
            idx += 1
            row = evaluate_and_save(name, model, X, y, hp_id, MODELS_DIR, kfold)
            rows.append(row)
            print(
                f"  [{idx:3d}/{total}] {hp_id} {name:13s} "
                f"cv_mse={row['cv_mse']:.6f} train_mse={row['train_mse']:.6f}"
            )

        idx += 1
        ensemble = make_ensemble(base_models)
        row = evaluate_and_save(
            "Ensemble", ensemble, X, y, hp_id, MODELS_DIR, kfold,
        )
        rows.append(row)
        print(
            f"  [{idx:3d}/{total}] {hp_id} {'Ensemble':13s} "
            f"cv_mse={row['cv_mse']:.6f} train_mse={row['train_mse']:.6f}"
        )

    leaderboard = build_leaderboard(rows)
    leaderboard.to_csv(MODELS_DIR / "leaderboard.csv", index=False)
    save_best(leaderboard, hp_grid_df, MODELS_DIR)

    best = leaderboard.iloc[0]
    print("\nLeaderboard top 5:")
    print(leaderboard.head().to_string(index=False))
    print(
        f"\nBest model: {best.model_type} on {best.hp_id} "
        f"(cv_mse={best.cv_mse:.6f}) -> models/best_model.joblib"
    )


if __name__ == "__main__":
    main()
