"""Evaluation utilities for season-aware NCAA modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class EvalResult:
    """Container for scalar metrics and calibration bins."""

    metrics: Dict[str, float]
    bins: pd.DataFrame


def _clip_probabilities(pred: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)


def calibration_slope_intercept(y_true: pd.Series, pred: pd.Series) -> tuple[float, float]:
    """Estimate calibration slope/intercept via logistic regression on logit(pred)."""
    p = _clip_probabilities(pred)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    y = np.asarray(y_true, dtype=int)
    model = LogisticRegression(C=1e6, fit_intercept=True, solver="lbfgs", max_iter=2000)
    model.fit(logits, y)
    slope = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    return slope, intercept


def calibration_bins(
    y_true: pd.Series,
    pred: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Create quantile-based calibration table."""
    df = pd.DataFrame({"y_true": y_true, "pred": _clip_probabilities(pred)}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["bin", "pred_mean", "obs_rate", "n"])

    q = min(n_bins, max(2, df["pred"].nunique()))
    df["bin"] = pd.qcut(df["pred"], q=q, duplicates="drop")
    out = (
        df.groupby("bin", observed=False)
        .agg(pred_mean=("pred", "mean"), obs_rate=("y_true", "mean"), n=("y_true", "size"))
        .reset_index()
    )
    out["bin"] = out["bin"].astype(str)
    return out


def evaluate_predictions(
    pred_df: pd.DataFrame,
    *,
    y_col: str = "y_true",
    pred_col: str = "pred",
    n_bins: int = 10,
) -> EvalResult:
    """Compute scalar metrics and calibration bins for predictions."""
    y_true = pred_df[y_col].astype(int)
    pred = pred_df[pred_col].astype(float)
    pred_clip = _clip_probabilities(pred)

    slope, intercept = calibration_slope_intercept(y_true, pred_clip)
    metrics = {
        "brier": float(brier_score_loss(y_true, pred_clip)),
        "log_loss": float(log_loss(y_true, pred_clip)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "n_rows": float(len(pred_df)),
    }
    bins = calibration_bins(y_true, pred_clip, n_bins=n_bins)
    return EvalResult(metrics=metrics, bins=bins)


def evaluate_by_regime(
    pred_df: pd.DataFrame,
    *,
    y_col: str = "y_true",
    pred_col: str = "pred",
    tourney_flag_col: str = "IsTourney",
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate on all rows and tournament-only rows."""
    rows: List[Dict[str, float | str]] = []
    bins_out: List[pd.DataFrame] = []

    all_result = evaluate_predictions(pred_df, y_col=y_col, pred_col=pred_col, n_bins=n_bins)
    rows.append({"regime": "all_games", **all_result.metrics})
    if not all_result.bins.empty:
        b = all_result.bins.copy()
        b["regime"] = "all_games"
        bins_out.append(b)

    if tourney_flag_col in pred_df.columns:
        tourney_df = pred_df[pred_df[tourney_flag_col].astype(int) == 1].copy()
        if len(tourney_df) > 0:
            t_result = evaluate_predictions(tourney_df, y_col=y_col, pred_col=pred_col, n_bins=n_bins)
            rows.append({"regime": "tournament_only", **t_result.metrics})
            if not t_result.bins.empty:
                b = t_result.bins.copy()
                b["regime"] = "tournament_only"
                bins_out.append(b)

    metrics_df = pd.DataFrame(rows)
    bins_df = pd.concat(bins_out, ignore_index=True) if bins_out else pd.DataFrame()
    return metrics_df, bins_df
