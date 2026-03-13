"""Rolling season cross-validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
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
    meta_cols: list[str] | None = None,
) -> CVResult:
    """Run leakage-safe rolling CV (train seasons < t, validate season == t)."""
    if predict_fn is None:
        predict_fn = default_predict_fn
    if meta_cols is None:
        meta_cols = ["Season", "Team1", "Team2", "IsTourney"]

    seasons = rolling_validation_seasons(data[season_col])
    oof_parts: list[pd.DataFrame] = []
    fold_metric_parts: list[pd.DataFrame] = []
    bin_parts: list[pd.DataFrame] = []

    for valid_season in seasons:
        train_mask = data[season_col].astype(int) < int(valid_season)
        valid_mask = data[season_col].astype(int) == int(valid_season)

        train_df = data.loc[train_mask].copy()
        valid_df = data.loc[valid_mask].copy()
        if train_df.empty or valid_df.empty:
            continue

        model = model_factory()
        model.fit(train_df[feature_cols], train_df[y_col].astype(int))

        pred = predict_fn(model, valid_df[feature_cols])
        fold_oof = valid_df[meta_cols + [y_col]].copy()
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

    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    fold_metrics_df = pd.concat(fold_metric_parts, ignore_index=True) if fold_metric_parts else pd.DataFrame()
    bins_out_df = pd.concat(bin_parts, ignore_index=True) if bin_parts else pd.DataFrame()
    return CVResult(oof=oof_df, fold_metrics=fold_metrics_df, calibration_bins=bins_out_df)
