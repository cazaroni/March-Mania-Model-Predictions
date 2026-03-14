"""Rolling, leakage-safe probability calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def _clip_prob(p: np.ndarray | pd.Series, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def _to_logit(p: np.ndarray) -> np.ndarray:
    p_clip = _clip_prob(p)
    return np.log(p_clip / (1.0 - p_clip))


def _from_logit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _apply_shrinkage(pred: np.ndarray, shrink: float) -> np.ndarray:
    if shrink <= 0.0:
        return _clip_prob(pred)
    # Shrink toward 0.5 to reduce overconfident tails.
    return _clip_prob((1.0 - shrink) * pred + shrink * 0.5)


@dataclass
class CalibrationModel:
    method: str
    transform: Callable[[np.ndarray], np.ndarray]

    def predict(self, pred: np.ndarray) -> np.ndarray:
        return _clip_prob(self.transform(_clip_prob(pred)))


def fit_platt(pred: np.ndarray, y_true: np.ndarray) -> CalibrationModel:
    x = _to_logit(pred).reshape(-1, 1)
    y = np.asarray(y_true, dtype=int)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    lr.fit(x, y)

    def _predict_fn(p: np.ndarray) -> np.ndarray:
        return lr.predict_proba(_to_logit(p).reshape(-1, 1))[:, 1]

    return CalibrationModel(method="platt", transform=_predict_fn)


def fit_isotonic(pred: np.ndarray, y_true: np.ndarray) -> CalibrationModel:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(_clip_prob(pred), np.asarray(y_true, dtype=float))

    def _predict_fn(p: np.ndarray) -> np.ndarray:
        return np.asarray(iso.predict(_clip_prob(p)), dtype=float)

    return CalibrationModel(method="isotonic", transform=_predict_fn)


def fit_temperature(pred: np.ndarray, y_true: np.ndarray) -> CalibrationModel:
    y = np.asarray(y_true, dtype=int)
    logits = _to_logit(pred)

    # Grid search is deterministic and avoids scipy dependency.
    grid = np.linspace(0.5, 3.0, 101)
    best_t = 1.0
    best_loss = float("inf")
    for t in grid:
        p_t = _from_logit(logits / float(t))
        loss = float(log_loss(y, _clip_prob(p_t)))
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)

    def _predict_fn(p: np.ndarray) -> np.ndarray:
        return _from_logit(_to_logit(p) / best_t)

    return CalibrationModel(method="temperature", transform=_predict_fn)


def fit_calibrator(method: str, pred: np.ndarray, y_true: np.ndarray) -> CalibrationModel:
    method_key = method.strip().lower()
    if method_key == "platt":
        return fit_platt(pred, y_true)
    if method_key == "isotonic":
        return fit_isotonic(pred, y_true)
    if method_key == "temperature":
        return fit_temperature(pred, y_true)
    raise ValueError(f"Unsupported calibration method: {method}")


def rolling_calibrate_oof(
    pred_df: pd.DataFrame,
    *,
    method: str,
    season_col: str = "Season",
    y_col: str = "y_true",
    pred_col: str = "pred",
    tourney_flag_col: str = "IsTourney",
    fit_scope: str = "all",
    shrink: float = 0.0,
    min_rows: int = 200,
) -> pd.DataFrame:
    """Apply leakage-safe rolling calibration to an OOF prediction frame."""
    scope_key = fit_scope.strip().lower()
    if scope_key not in {"all", "tournament_only"}:
        raise ValueError("fit_scope must be 'all' or 'tournament_only'")

    seasons = sorted(pred_df[season_col].astype(int).unique())
    out_parts: list[pd.DataFrame] = []

    for valid_season in seasons:
        tr = pred_df[pred_df[season_col].astype(int) < valid_season].copy()
        va = pred_df[pred_df[season_col].astype(int) == valid_season].copy()
        if va.empty:
            continue

        # For the earliest available OOF season there is no prior calibration history.
        # Keep base probabilities so calibration outputs preserve row parity.
        if tr.empty:
            o = va.copy()
            o[pred_col] = _apply_shrinkage(_clip_prob(va[pred_col].to_numpy()), shrink=shrink)
            o["cal_method"] = method
            o["cal_scope"] = scope_key
            o["cal_shrink"] = float(shrink)
            out_parts.append(o)
            continue

        cal_train = tr
        if scope_key == "tournament_only" and tourney_flag_col in tr.columns:
            cal_train = tr[tr[tourney_flag_col].astype(int) == 1].copy()

        # Fallback to all train rows if tournament-only set is too small or single-class.
        if len(cal_train) < min_rows or cal_train[y_col].nunique(dropna=True) < 2:
            cal_train = tr

        if len(cal_train) < min_rows or cal_train[y_col].nunique(dropna=True) < 2:
            # Not enough data to fit safely; keep original probabilities.
            p_valid = _clip_prob(va[pred_col].to_numpy())
        else:
            calibrator = fit_calibrator(
                method,
                pred=_clip_prob(cal_train[pred_col].to_numpy()),
                y_true=cal_train[y_col].astype(int).to_numpy(),
            )
            p_valid = calibrator.predict(va[pred_col].to_numpy())

        p_valid = _apply_shrinkage(p_valid, shrink=shrink)

        o = va.copy()
        o[pred_col] = p_valid
        o["cal_method"] = method
        o["cal_scope"] = scope_key
        o["cal_shrink"] = float(shrink)
        out_parts.append(o)

    return pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
