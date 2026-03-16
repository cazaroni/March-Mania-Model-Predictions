"""Rolling season cross-validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd

from eval import evaluate_by_regime


@dataclass
class CVResult:
    """Result object for rolling CV."""

    oof: pd.DataFrame
    fold_metrics: pd.DataFrame
    calibration_bins: pd.DataFrame


ModelFactory = Callable[[], object]
PredictFn = Callable[[object, pd.DataFrame], np.ndarray]


def rolling_validation_seasons(seasons: Iterable[int]) -> List[int]:
    """Validation seasons are all seasons after the first in sorted order."""
    uniq = sorted(pd.Series(list(seasons)).dropna().astype(int).unique())
    return uniq[1:]


def default_predict_fn(model: object, x_valid: pd.DataFrame) -> np.ndarray:
    """Predict probabilities from a fitted model."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x_valid)[:, 1], dtype=float)
    pred = np.asarray(model.predict(x_valid), dtype=float)
    return np.clip(pred, 1e-6, 1 - 1e-6)


def run_rolling_cv(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    model_name: str,
    model_factory: ModelFactory,
    predict_fn: PredictFn | None = None,
    season_col: str = "Season",
    y_col: str = "y_true",
    train_target_col: str | None = None,
    meta_cols: list[str] | None = None,
    progress_label: str | None = None,
    progress: bool = True,
) -> CVResult:
    """Run leakage-safe rolling CV (train seasons < t, validate season == t)."""
    if predict_fn is None:
        predict_fn = default_predict_fn
    if meta_cols is None:
        meta_cols = ["Season", "Team1", "Team2", "IsTourney", "Margin"]
    available_meta = [c for c in meta_cols if c in data.columns]

    season_values = data[season_col].astype(int)
    seasons = rolling_validation_seasons(season_values)
    oof_parts: list[pd.DataFrame] = []
    fold_metric_parts: list[pd.DataFrame] = []
    bin_parts: list[pd.DataFrame] = []

    total_folds = len(seasons)
    completed_folds = 0
    elapsed_seconds = 0.0

    for fold_index, valid_season in enumerate(seasons, start=1):
        fold_start = time.perf_counter()
        train_mask = season_values < int(valid_season)
        valid_mask = season_values == int(valid_season)

        train_df = data.loc[train_mask].copy()
        valid_df = data.loc[valid_mask].copy()
        if progress:
            label = progress_label or model_name
            print(
                f"[CV] {label} fold {fold_index}/{total_folds} | season={int(valid_season)} | train={len(train_df):,} | valid={len(valid_df):,}",
                flush=True,
            )
        if train_df.empty or valid_df.empty:
            if progress:
                print(f"[CV] {label} fold {fold_index}/{total_folds} skipped (empty split)", flush=True)
            continue

        model = model_factory()
        if train_target_col and train_target_col in train_df.columns:
            margin_train_mask = train_df["IsTourney"] == 0 if "IsTourney" in train_df.columns else pd.Series(True, index=train_df.index)
            fit_df = train_df.loc[margin_train_mask].copy()
            # Fallback to all training rows if a fold has no regular-season rows.
            if fit_df.empty:
                fit_df = train_df
            fit_target_col = train_target_col
        else:
            fit_df = train_df
            fit_target_col = y_col
        model.fit(fit_df[feature_cols], fit_df[fit_target_col])

        pred = predict_fn(model, valid_df[feature_cols])
        fold_oof = valid_df[available_meta + [y_col]].copy()
        fold_oof["pred"] = np.clip(pred, 1e-6, 1 - 1e-6)
        fold_oof["model"] = model_name
        fold_oof["valid_season"] = int(valid_season)
        oof_parts.append(fold_oof)

        metric_df, bins_df = evaluate_by_regime(fold_oof, y_col=y_col, pred_col="pred", tourney_flag_col="IsTourney")
        metric_df["model"] = model_name
        metric_df["valid_season"] = int(valid_season)
        fold_metric_parts.append(metric_df)

        if not bins_df.empty:
            bins_df["model"] = model_name
            bins_df["valid_season"] = int(valid_season)
            bin_parts.append(bins_df)

        fold_elapsed = time.perf_counter() - fold_start
        completed_folds += 1
        elapsed_seconds += fold_elapsed
        avg_fold = elapsed_seconds / max(completed_folds, 1)
        remaining = total_folds - fold_index
        eta_seconds = avg_fold * max(remaining, 0)
        if progress:
            print(
                f"[CV] {label} fold {fold_index}/{total_folds} done in {fold_elapsed:.1f}s | ETA ~{eta_seconds/60.0:.1f}m",
                flush=True,
            )

    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    fold_metrics_df = pd.concat(fold_metric_parts, ignore_index=True) if fold_metric_parts else pd.DataFrame()
    bins_out_df = pd.concat(bin_parts, ignore_index=True) if bin_parts else pd.DataFrame()
    return CVResult(oof=oof_df, fold_metrics=fold_metrics_df, calibration_bins=bins_out_df)
