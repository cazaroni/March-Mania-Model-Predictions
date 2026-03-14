"""Leakage-safe stacking on rolling OOF base predictions."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_stack_features(base_oof: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pivot model-level OOF rows to wide matrix for meta-model training."""
    key_cols = ["Season", "Team1", "Team2", "y_true"]
    if "GameRowID" in base_oof.columns:
        key_cols.insert(1, "GameRowID")
    if "IsTourney" in base_oof.columns:
        key_cols.append("IsTourney")

    prior_cols = _rating_prior_columns(base_oof)

    wide = (
        base_oof.pivot_table(index=key_cols, columns="model", values="pred", aggfunc="mean")
        .reset_index()
        .rename_axis(columns=None)
    )

    # Preserve prior columns (constant across model rows for the same key).
    if prior_cols:
        prior_frame = base_oof[key_cols + prior_cols].drop_duplicates(subset=key_cols, keep="first")
        wide = wide.merge(prior_frame, on=key_cols, how="left")

    # Meta features should be base model predictions only.
    model_cols = [str(m) for m in sorted(base_oof["model"].astype(str).unique())]
    pred_cols = [c for c in model_cols if c in wide.columns]
    return wide, pred_cols


def _brier(y_true: np.ndarray, pred: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    err = (pred - y_true) ** 2
    if sample_weight is None:
        return float(np.mean(err))
    w = np.asarray(sample_weight, dtype=float)
    if w.sum() <= 0:
        return float(np.mean(err))
    return float(np.average(err, weights=w))


def _fit_convex_blend_weights(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Fit nonnegative linear weights and normalize to a convex blend."""
    lr = LinearRegression(fit_intercept=False, positive=True)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    lr.fit(x_train.fillna(0.5), y_train.astype(float), **fit_kwargs)
    w = np.asarray(lr.coef_, dtype=float)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 1e-12:
        return np.full(x_train.shape[1], 1.0 / max(x_train.shape[1], 1), dtype=float)
    return w / s


def _predict_with_weights(x: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    pred = np.asarray(x.fillna(0.5), dtype=float) @ weights
    return np.clip(pred, 1e-6, 1 - 1e-6)


def _rating_prior_columns(df: pd.DataFrame) -> list[str]:
    """Return stable rating-difference features suitable for prior modeling."""
    preferred = [
        "Diff_Elo",
        "Diff_BT",
        "Diff_AdjNet",
        "Diff_AdjORtg",
        "Diff_AdjDRtg",
        "Diff_PythagWinPct",
        "Diff_LuckResidual",
    ]
    return [c for c in preferred if c in df.columns]


def _fit_rating_prior(
    x: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(C=1.0, max_iter=3000, random_state=42)),
        ]
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["logreg__sample_weight"] = sample_weight
    model.fit(x, y.astype(int), **fit_kwargs)
    return model


def rolling_stack_oof(
    stack_df: pd.DataFrame,
    *,
    season_col: str = "Season",
    y_col: str = "y_true",
    feature_cols: list[str],
    tourney_flag_col: str = "IsTourney",
    tournament_weight: float = 1.0,
    prior_cols: list[str] | None = None,
    return_diagnostics: bool = False,
) -> pd.DataFrame:
    """Train rolling meta-learner with robust fallback strategies."""
    seasons = sorted(stack_df[season_col].astype(int).unique())
    out_parts: List[pd.DataFrame] = []
    diagnostics_parts: List[pd.DataFrame] = []
    if prior_cols is None:
        prior_cols = _rating_prior_columns(stack_df)

    for valid_season in seasons[1:]:
        tr = stack_df[stack_df[season_col].astype(int) < valid_season].copy()
        va = stack_df[stack_df[season_col].astype(int) == valid_season].copy()
        if tr.empty or va.empty:
            continue

        x_tr = tr[feature_cols].fillna(0.5)
        y_tr = tr[y_col].astype(float).to_numpy()
        x_va = va[feature_cols].fillna(0.5)

        tr_weights = np.ones(len(tr), dtype=float)
        if tourney_flag_col in tr.columns:
            tr_weights = np.where(tr[tourney_flag_col].astype(int).to_numpy() == 1, float(tournament_weight), 1.0)

        # Inner selector split for strategy choice (time-aware):
        # selector_train uses older seasons, selector_valid is latest available train season.
        train_seasons = sorted(tr[season_col].astype(int).unique())
        if len(train_seasons) >= 2:
            selector_valid_season = int(train_seasons[-1])
            sel_train = tr[tr[season_col].astype(int) < selector_valid_season].copy()
            sel_valid = tr[tr[season_col].astype(int) == selector_valid_season].copy()
        else:
            selector_valid_season = int(train_seasons[-1])
            sel_train = tr.copy()
            sel_valid = tr.copy()

        x_sel_train = sel_train[feature_cols].fillna(0.5)
        y_sel_train = sel_train[y_col].astype(float).to_numpy()
        x_sel_valid = sel_valid[feature_cols].fillna(0.5)
        y_sel_valid = sel_valid[y_col].astype(float).to_numpy()

        sel_train_weights = np.ones(len(sel_train), dtype=float)
        sel_valid_weights = np.ones(len(sel_valid), dtype=float)
        if tourney_flag_col in sel_train.columns:
            sel_train_weights = np.where(
                sel_train[tourney_flag_col].astype(int).to_numpy() == 1,
                float(tournament_weight),
                1.0,
            )
        if tourney_flag_col in sel_valid.columns:
            sel_valid_weights = np.where(
                sel_valid[tourney_flag_col].astype(int).to_numpy() == 1,
                float(tournament_weight),
                1.0,
            )

        # Candidate 1: ridge meta-learner (selected via selector valid season)
        ridge_sel = Ridge(alpha=1.0, random_state=42)
        ridge_sel.fit(x_sel_train, y_sel_train, sample_weight=sel_train_weights)
        ridge_sel_valid_pred = np.clip(ridge_sel.predict(x_sel_valid), 1e-6, 1 - 1e-6)
        ridge_sel_brier = _brier(y_sel_valid, ridge_sel_valid_pred, sample_weight=sel_valid_weights)

        # Candidate 2: convex nonnegative blend
        convex_w_sel = _fit_convex_blend_weights(x_sel_train, sel_train[y_col], sample_weight=sel_train_weights)
        convex_sel_valid_pred = _predict_with_weights(x_sel_valid, convex_w_sel)
        convex_sel_brier = _brier(y_sel_valid, convex_sel_valid_pred, sample_weight=sel_valid_weights)

        # Candidate 3: best single base model on selector valid season
        base_selector_briers = {
            col: _brier(
                y_sel_valid,
                np.clip(np.asarray(x_sel_valid[col], dtype=float), 1e-6, 1 - 1e-6),
                sample_weight=sel_valid_weights,
            )
            for col in feature_cols
        }
        best_base_col, best_base_sel_brier = min(base_selector_briers.items(), key=lambda kv: kv[1])

        # Candidate 4: rating prior + ML residual correction.
        prior_resid_sel_brier = np.inf
        prior_resid_ready = len(prior_cols) > 0
        if prior_resid_ready:
            x_sel_prior_train = sel_train[prior_cols]
            x_sel_prior_valid = sel_valid[prior_cols]
            prior_sel_model = _fit_rating_prior(
                x_sel_prior_train,
                y_sel_train,
                sample_weight=sel_train_weights,
            )
            prior_sel_train_pred = np.clip(
                np.asarray(prior_sel_model.predict_proba(x_sel_prior_train)[:, 1], dtype=float),
                1e-6,
                1 - 1e-6,
            )
            prior_sel_valid_pred = np.clip(
                np.asarray(prior_sel_model.predict_proba(x_sel_prior_valid)[:, 1], dtype=float),
                1e-6,
                1 - 1e-6,
            )

            resid_sel = Ridge(alpha=1.0, random_state=42)
            resid_sel.fit(
                x_sel_train,
                y_sel_train - prior_sel_train_pred,
                sample_weight=sel_train_weights,
            )
            prior_resid_sel_valid_pred = np.clip(
                prior_sel_valid_pred + resid_sel.predict(x_sel_valid),
                1e-6,
                1 - 1e-6,
            )
            prior_resid_sel_brier = _brier(
                y_sel_valid,
                prior_resid_sel_valid_pred,
                sample_weight=sel_valid_weights,
            )

        selector_rows = [
            ("ridge", ridge_sel_brier),
            ("convex_blend", convex_sel_brier),
            (f"best_base:{best_base_col}", best_base_sel_brier),
        ]
        if prior_resid_ready:
            selector_rows.append(("prior_residual", prior_resid_sel_brier))
        chosen_name, _ = min(selector_rows, key=lambda x: x[1])

        # Refit chosen strategy on full tr and predict current valid season
        if chosen_name == "ridge":
            ridge_full = Ridge(alpha=1.0, random_state=42)
            ridge_full.fit(x_tr, y_tr, sample_weight=tr_weights)
            pred = np.clip(ridge_full.predict(x_va), 1e-6, 1 - 1e-6)
        elif chosen_name == "convex_blend":
            convex_w_full = _fit_convex_blend_weights(x_tr, tr[y_col], sample_weight=tr_weights)
            pred = _predict_with_weights(x_va, convex_w_full)
        elif chosen_name == "prior_residual" and prior_resid_ready:
            x_tr_prior = tr[prior_cols]
            x_va_prior = va[prior_cols]
            prior_full_model = _fit_rating_prior(x_tr_prior, y_tr, sample_weight=tr_weights)
            prior_tr_pred = np.clip(
                np.asarray(prior_full_model.predict_proba(x_tr_prior)[:, 1], dtype=float),
                1e-6,
                1 - 1e-6,
            )
            prior_va_pred = np.clip(
                np.asarray(prior_full_model.predict_proba(x_va_prior)[:, 1], dtype=float),
                1e-6,
                1 - 1e-6,
            )
            resid_full = Ridge(alpha=1.0, random_state=42)
            resid_full.fit(x_tr, y_tr - prior_tr_pred, sample_weight=tr_weights)
            pred = np.clip(prior_va_pred + resid_full.predict(x_va), 1e-6, 1 - 1e-6)
        else:
            pred = np.clip(np.asarray(x_va[best_base_col], dtype=float), 1e-6, 1 - 1e-6)

        oof = va[["Season", "Team1", "Team2", y_col] + (["IsTourney"] if "IsTourney" in va.columns else [])].copy()
        oof["pred"] = pred
        oof["model"] = "stack"
        oof["stack_strategy"] = chosen_name
        out_parts.append(oof)

        diagnostics_parts.append(
            pd.DataFrame(
                {
                    "valid_season": [int(valid_season)] * len(selector_rows),
                    "selector_valid_season": [selector_valid_season] * len(selector_rows),
                    "candidate": [r[0] for r in selector_rows],
                    "selector_brier": [float(r[1]) for r in selector_rows],
                    "chosen_strategy": [chosen_name] * len(selector_rows),
                    "best_base_col": [best_base_col] * len(selector_rows),
                    "ridge_alpha": [1.0] * len(selector_rows),
                    "tournament_weight": [float(tournament_weight)] * len(selector_rows),
                    "prior_cols": ["|".join(prior_cols)] * len(selector_rows),
                }
            )
        )

    out_df = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
    if not return_diagnostics:
        return out_df

    diag_df = pd.concat(diagnostics_parts, ignore_index=True) if diagnostics_parts else pd.DataFrame()
    return out_df, diag_df
